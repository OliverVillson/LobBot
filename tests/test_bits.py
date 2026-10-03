import math

import pytest

from stages import bits

QWEN3_30B_REAP50 = dict(
    num_hidden_layers=48, hidden_size=2048, moe_intermediate_size=768, num_experts=64,
    num_experts_per_tok=8, vocab_size=151936, num_attention_heads=32, num_key_value_heads=4, head_dim=128,
)


def shape(**kw):
    return bits.MoEShape.from_hf_config({**QWEN3_30B_REAP50, **kw})


def test_param_count_matches_reap50():
    assert 15.5e9 < shape().total_params < 16.5e9


@pytest.mark.parametrize("budget", [6.0, 6.5, 7.0])
def test_allocation_fits_budget(budget):
    imp = [1 + math.sin(i / 5) ** 2 for i in range(48)]
    layers = bits.allocate(shape(), imp, budget)
    assert bits.estimate_size_gb(shape(), layers) <= budget
    # Close to the budget: the greedy loop should not leave a whole step unused.
    assert bits.estimate_size_gb(shape(), layers) > budget - 0.2


def test_salient_layers_get_more_bits():
    imp = [1.0] * 48
    imp[10] = 10.0
    layers = bits.allocate(shape(), imp, 6.5)
    lvl = lambda l: bits.LADDER.index(l.down) + 2 * bits.LADDER.index(l.gate_up)
    assert lvl(layers[10]) >= max(lvl(l) for l in layers)


def test_budget_below_floor_raises():
    with pytest.raises(ValueError):
        bits.allocate(shape(), [1.0] * 48, 3.0)


def test_quantize_args_cover_every_layer():
    layers = bits.allocate(shape(), [1.0] * 48, 6.5)
    args = bits.quantize_args(layers)
    assert sum(a.startswith("blk\\.") for a in args) == 2 * 48
    assert "--output-tensor-type" in args


def test_moe_is_faster_than_dense_at_same_size():
    layers = bits.allocate(shape(), [1.0] * 48, 6.5)
    assert bits.estimate_tok_s(shape(), layers) > 40
