"""moefy_anneal on a tiny dense Qwen3: the all-experts MoE is the dense model, top-k changes it,
and distillation steps lower the gap at a fixed k while training the routers."""

from __future__ import annotations

import torch

import moefy_anneal as ma
from moefy_bench import collect, find_mlps


def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    return Qwen3ForCausalLM(cfg).eval()


def test_lossless_at_all_experts_and_training_lowers_kl():
    teacher, student = tiny(), tiny()
    student.load_state_dict(teacher.state_dict())
    for q in teacher.parameters():
        q.requires_grad_(False)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 128, (4, 24), generator=g)
    mlps = find_mlps(student)
    layout = ma.split_layout(96, 8, 0.125)
    assert layout["shared"] + 8 * layout["expert_size"] == 96 and layout["shared"] == 16
    stats = collect(student, mlps, [(ids, torch.ones_like(ids))], 96)
    Ws = ma.moefy(mlps, stats, layout, "cpu")
    routed = ma.RoutedMLPs(student, mlps, layout, aux_weight=0.1)
    routed.init_routers(Ws)
    with torch.no_grad():
        ref = teacher(ids).logits
        assert torch.allclose(student(ids).logits, ref, atol=1e-5)          # permuted + all on = dense
        routed.k = 2
        cut = student(ids).logits
    assert (cut - ref).abs().max() > 1e-3

    def kl():
        with torch.no_grad():
            return torch.nn.functional.kl_div(student(ids).logits.log_softmax(-1), ref.log_softmax(-1),
                                              log_target=True, reduction="batchmean").item()
    routed.k, routed.k_aux = 4, 2
    before = kl()
    student.train()
    params = [q for n, q in student.named_parameters() if ".mlp." in n]
    for q in student.parameters():
        q.requires_grad_(False)
    for q in params:
        q.requires_grad_(True)
    r0 = [p.detach().clone() for p in routed.routers.parameters()]
    opt = torch.optim.Adam(params + list(routed.routers.parameters()), lr=3e-3)
    for _ in range(30):
        ma.kl_step(teacher, student, ids, 1, chunk=16)
        opt.step()
        opt.zero_grad(set_to_none=True)
    student.eval()
    assert kl() < 0.5 * before
    assert any(not torch.equal(a, b) for a, b in zip(r0, routed.routers.parameters()))
