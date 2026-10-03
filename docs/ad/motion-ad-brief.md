# LobBot motion ad: script and fact brief

For derka's Claude. derka designs the ad; this file is everything you need so that
what the ad *says* is true. Read it all before writing copy or storyboards.

- **Look and feel:** follow derka's brand, not this file. The source of truth is
  `UI/BRAND.md` (Dr. Lobbot, the Anatomie palette, the Sillon logo, motion rules), on the
  `ui` branch (PR #19) until it is merged.
- **Words and numbers:** follow this file. Every number below comes from a real run on
  2026-10-03 and says where it came from. If a line you want isn't backed here, leave it
  out or ask Oliver.
- **Tone:** keep the app's goofy vibe. Dr. Lobbot is a funny, nerdy, joyful surgeon for AI
  models, and surgery jokes are the brand (patient, chart, scalpel, the patient goes home).
  Plain English. No pirate speak or any other gimmick voice.

---

## 1. What LobBot is, in plain words

You describe **one job** in one sentence. LobBot takes a big, general AI model and turns
it into a small model that does only that job, then puts it on your laptop, where it runs
offline.

The demo job: *turn a customer support email into a JSON ticket* (category, priority,
summary, customer sentiment).

What happens, step by step:

| # | What really happens | Command / code | Surgery version for the ad |
|---|---|---|---|
| 1 | Gemini drafts a task spec: what goes in, what comes out, 12 example pairs, how to grade it. | `lobbot new "turn support emails into JSON tickets"` | **The chart.** Dr. Lobbot takes the patient's history. |
| 2 | The big "teacher" model (Qwen3-30B-A3B, 61 GB) writes about 2,000 practice examples of the job. | `stages/data.py` | **Practice cases.** The big brain does the job 2,000 times so we know what "good" looks like. |
| 3 | REAP pruning measures which "experts" (sub-networks) the job actually uses and removes the half it uses least: 64 of 128 per layer. | `stages/reap.py` | **The operation.** Out goes the half of the brain this job never uses. Into the jar. |
| 4 | "Heal": the pruned model is fine-tuned on the teacher's answers to recover. | `stages/heal.py` | **Physio.** The patient relearns from the teacher. |
| 5 | Quantization with a different bit width per layer: important layers keep more precision, unimportant ones get squeezed (2 to 6 bits, about 3 bits on average for expert weights). Out comes a 6.52 GB GGUF file. | `stages/quantize.py`, `stages/bits.py` | **Packing for the trip home.** Folded small enough to fit in a laptop. |
| 6 | Gemini writes 100 brand-new test emails the model never saw, then grades the small model's answers and the teacher's answers 0 to 10 against the task's criteria. | `stages/testgen.py`, `stages/eval.py` | **The check-up.** A second doctor (Gemini) writes a surprise exam and grades both. |
| 7 | The file downloads to the Mac, gets checked, and is installed into Ollama, which runs it locally. | `lobbot save <job> --chat` | **Discharge.** The patient goes home, to your laptop. |

The training (steps 2 to 6) runs on one NVIDIA B200 GPU in evroc's cloud. Running the
finished model (step 7) needs no cloud and no internet.

### Read this before using the app's "cut, test, undo" story

The iPhone app in `UI/` currently describes LobBot as a loop: *cut a block, test the skill,
keep the cut or undo it, repeat*. **That is not how the real pipeline works, so the ad must
not show it as fact.** The real pipeline plans one operation from measurements (step 3),
then recovers, packs and tests the model once at the end (steps 4 to 6). Nothing is undone
cut by cut.

Ways to keep the surgery joke and stay accurate:
- "Dr. Lobbot scans the brain *while it does your job*, then removes what it never uses."
- "Then he checks the patient against the original with a surprise exam."

### Where the model lives: the Mac, not the phone

The finished model is a 6.52 GB file that runs in Ollama on a 16 GB Mac. The iPhone app is
the face of LobBot: Dr. Lobbot, the chart, the surgery. The plan is for the phone to chat
with the model on the Mac over the network, but that isn't wired up yet. So:
- Show the model's answers coming from the MacBook (scene 8).
- If you show the phone chatting with the model, make it visibly a remote for the Mac (for
  example, a line or beam from the phone to the MacBook), never the model inside the phone.
  Check with Oliver that the phone link actually works before the ad shows it.

Also from the app, keep these out of the ad as claims: its simulated numbers
(70B / 32B / 8B models, "x× lighter", capability percentages), the "phone" and "edge
board" targets, and the "Simulation" runs. The app's screens and Dr. Lobbot are fine to
show as the product's look; just put the real numbers from section 3 on top of them, not
the simulated ones.

---

## 2. Script (about 50 seconds)

Nine scenes. Times are a guide; stretch or squeeze to the music. "VO" is the voice-over
(Dr. Lobbot himself works well). On-screen text uses the exact numbers from section 3.

| # | Time | What's shown | On-screen text | VO |
|---|---|---|---|---|
| 1 | 0:00–0:05 | A giant brain on a hospital trolley, wedged in a laptop-sized doorway. A "61 GB" tag on its toe. | **61 GB of brain.** | "Big AI models know everything. Poetry. Tax law. The capital of Mongolia." |
| 2 | 0:05–0:09 | Dr. Lobbot bursts in, head mirror flashes. Sillon logo draws on. | **lobbot** · Cut the model. Keep the capability. | "Your support inbox needs none of that. Hi, I'm Dr. Lobbot." |
| 3 | 0:09–0:14 | Terminal: `lobbot new "turn support emails into JSON tickets"`. Dr. Lobbot scribbles on his clipboard. | **1. Describe one job.** | "Tell me the one job you need done. One sentence is plenty." |
| 4 | 0:14–0:19 | The big brain sweats out a stack of practice emails and tickets; a counter runs to 2,030. | **2,030 practice examples** written by the big model | "First the big model does your job a couple of thousand times, so we know what good looks like." |
| 5 | 0:19–0:26 | Operating room. Rows of little "expert" blocks per layer; the dim half lift out into a jar. Small "evroc B200" badge on the theatre lamp. | **Half the experts removed.** 64 of 128 per layer, the ones your job uses least | "Then, scalpel. Out goes the half of the brain your job never uses." |
| 6 | 0:26–0:31 | Quick physio montage (the model doing reps on teacher answers), then layers fold into a suitcase, some thicker than others. | **Healed, then packed:** 2 to 6 bits per layer | "A little physio, then I pack it tight. Important parts get more room." |
| 7 | 0:31–0:38 | Check-up: Gemini hands over a surprise exam of 100 new emails. Scoreboard flips in. | **Big model 9.8/10 · LobBot 9.7/10** on 100 new emails, graded by Gemini · **99% of the score at 1/9 the size** | "Surprise exam! A hundred emails it has never seen. Graded against the original: ninety-nine percent of its score." |
| 8 | 0:38–0:45 | Discharge: the jar rolls into a MacBook. Wi-Fi icon switches off. A messy email (the real one from section 4) goes in; the JSON ticket types out live. Badge: "offline". | **6.52 GB · ~50 tokens/s on a 16 GB MacBook · offline** | "And the patient goes home. To your laptop. Fast, small, and it works with the Wi-Fi off." |
| 9 | 0:45–0:50 | End card: Dr. Lobbot waves from the stage. Sponsor row. | **lobbot** · Your one job, on your laptop. · Built on evroc · Tested with Google DeepMind's Gemini · Prompts trimmed by condense.chat | "lobbot. Cut the model, keep the job." |

### 15-second cut

| Time | Shown | On-screen | VO |
|---|---|---|---|
| 0:00–0:04 | Giant brain stuck in a laptop doorway | **61 GB** | "Your AI is too big for your laptop." |
| 0:04–0:09 | Dr. Lobbot operates; half the experts go in a jar | **One job. Half the experts.** | "Dr. Lobbot keeps only what your one job needs." |
| 0:09–0:13 | Scoreboard, then the jar rolls into a MacBook | **6.52 GB · 99% of the score · offline** | "Ninety-nine percent of the score, running on your laptop." |
| 0:13–0:15 | End card with logo | **lobbot** | "lobbot." |

### Copy lines that are safe to use

- "Describe one job. Get a model that does it on your laptop."
- "From 61 GB to 6.52 GB."
- "99% of the big model's score on its job, graded by Gemini." (quality run)
- "About 50 tokens per second on a 16 GB MacBook."
- "Runs offline once it's on your laptop."
- "Built in about 25 minutes on one GPU." (quality run; see section 3 for what that covers)
- "Half the experts removed: the ones your job uses least."
- "A specialist, not a know-it-all."

---

## 3. Fact sheet

Use these numbers exactly as written in the "Say it as" column.

| Claim | Say it as | Exact value | Source |
|---|---|---|---|
| Teacher model | "the big model", or Qwen3-30B-A3B | Qwen3-30B-A3B-Instruct-2507, 30B parameters, mixture of experts | `stages/_util.py` (`Config.teacher`) |
| Teacher size | **61 GB** | 61.1 GB | `eval.json` of the runs; results report |
| LobBot model size | **6.52 GB** (or "about 6.5 GB") | 6,523,292,032 bytes, GGUF | `out/model.gguf` on the VM; `tests/data/vm_quality_facts.json` (`gguf_bytes`) |
| How much smaller | **about 9× smaller** / **1/9 the size** | 61.1 / 6.52 = 9.4 | computed. Don't say "1/10": it isn't. |
| Quality score (quality run) | **9.7/10 vs 9.8/10**, **99% of the big model's score** | 0.974 vs 0.981; 0.974 / 0.981 = 99.3% | `lobbot results quality`; `/mnt/project-files/vm-run-results.txt` |
| Test set (quality run) | **100 new emails it never saw** | 100 held-out inputs written by gemini-3.8-flash | same |
| Quality score (fast run) | only if you need it: 9.5/10, 97% of the big model's score | 0.953 vs 0.987, 30 held-out tests | `lobbot results fast` |
| How it's scored | "graded by Gemini" | Gemini grades each answer 0 to 10 against the task's criteria; average / 10. The teacher is graded the same way. No reference answers. | `stages/eval.py` |
| Training data | **2,030 practice examples** | 2,030 train + 100 held-out, written by the teacher from 15 seed examples | results report; `vm-run-results.txt` |
| Pruning | **half the experts removed**, 64 of 128 per layer | REAP at 50% sparsity, calibrated on the task data | `stages/reap.py` (`kept 64 experts per layer`), `Config.reap_sparsity = 0.5` |
| Bit widths | **2 to 6 bits per layer**, about 3 on average | expert weights: q2_k 77%, q4_k 12%, q3_k 8%, q5_k 2%, avg 3.00 bits/weight; floor q2_k, ceiling q6_k | results report; `Config.bit_floor/bit_ceiling` |
| Speed on a laptop | **about 50 tokens/s on a 16 GB MacBook** | 56.7 tok/s measured in Ollama on Oliver's MacBook (Apple M5, 16 GB), fast-run model; 49 tok/s estimated for an M4 Air (from memory bandwidth); 54.4 tok/s measured on the VM for the quality model | Oliver's run, 2026-10-03; results report |
| Build time | **about 25 minutes on one GPU** | quality run: 25m26s of stage time (data 2m29s, reap 8m39s, heal 3m53s, quantize 9m09s, eval 1m11s, package 5s). First fast run: 20m02s. | results reports |
| GPU | **one NVIDIA B200 on evroc** | evroc cloud VM | project setup |
| Runs offline | **yes, once it's downloaded** | Ollama runs the GGUF locally | `lobbot save --chat` |
| Target it was built for | "fits a 16 GB laptop" | TaskSpec target: max 7 GB, min 40 tok/s, 16 GB RAM | `examples/support-tickets.taskspec.json` |
| Why the pruned model won over Gemma | "the fastest one that's good enough" | Gemma 4 E4B scored 9.8/10 but ran at about 26 tok/s estimated, below the 40 tok/s target; the pruned model meets the target and wins within 0.02 | results report "Why" line; `stages/eval.py` |

### Sponsors: what each one really does

| Sponsor | What it does in LobBot | Safe line |
|---|---|---|
| **Google DeepMind** | Gemini (gemini-3.8-flash) drafts the task spec in `lobbot new`, writes the 100 surprise test emails, and grades every answer. Gemma 4 E4B is the backup "dense" candidate LobBot also trains and compares; in the demo it lost on speed. | "Tested and graded by Gemini." "Gemma 4 competes as the backup model." |
| **evroc** | The cloud GPU (NVIDIA B200) where the training, pruning and testing run. | "Built on an evroc B200." |
| **condense.chat** | Compresses the example emails Gemini reads when it writes the test problems (in a demo, 44 words down to 24). It does not compress the model, and it is not used for grading. | "Prompts trimmed by condense.chat." |

Matrix OS is not used. Don't mention it.

---

## 4. Real output to animate

All of this is real output or rebuilt from the code with real numbers. Animate it as is;
typing it out character by character works well. Shorten with "…" if needed, but don't
change values.

### The chart: `lobbot new`

Format from `lobbot/app.py` (`cmd_new`). Gemini drafts different examples each time, so
these values come from the repo's example spec (`examples/support-tickets.taskspec.json`):

```
$ lobbot new "turn support emails into JSON tickets"
✓ wrote support-email-to-ticket.taskspec.json  (support-email-to-ticket, 12 seed examples)
  e.g. Hi, I was charged twice for my Pro subscription this month. Can you refund the extra charge? Order #  →  {"category": "billing", "priority": "high", "summary": "Customer was double-charged for the Pro subs
Edit it if you like, then: lobbot run support-email-to-ticket.taskspec.json --fast
```

### The operation: `lobbot run` live board (quality run, finished)

Rebuilt with the CLI's own board renderer from the quality run's real stage timings; the
stage messages are the ones each stage prints, with that run's numbers. In the live CLI
the bars are cyan, ✓ is green and the bars fill up as each stage runs.

```
job quality  (Ctrl-C stops watching; the job keeps running)
 ✓ data     ████████████████████ 100.0%    2m29s  2030 train, 100 held out
 ✓ reap     ████████████████████ 100.0%    8m39s  kept 64 experts per layer
 ✓ heal     ████████████████████ 100.0%    3m53s  healed + dense student
 ✓ quantize ████████████████████ 100.0%    9m09s  lobbot-moe 6.52 GB, dense 5.3 GB
 ✓ eval     ████████████████████ 100.0%    1m11s  winner lobbot-moe: score 0.974 vs teacher 0.981 (judge)
 ✓ package  ████████████████████ 100.0%    0m05s  lobbot-moe ready at out/model.gguf
  elapsed 25m31s
✓ pipeline finished
```

### The check-up: `lobbot results quality` (real, verbatim)

```
━━ LobBot results · support-email-to-ticket · job quality ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Turn a customer support email into a structured JSON ticket for a SaaS helpdesk.

  Winner   lobbot-moe  6.52 GB  ~49 tok/s on a 16 GB laptop  score 9.7/10 (99% of the teacher)
  Why      best judge score among candidates that meet the 7 GB / 40 tok/s target; dense scores
           9.8/10 but misses it (26 tok/s)

  Model                         Size  Laptop tok/s  VM tok/s   Judge  vs teacher  Target
  Qwen3-30B-A3B (teacher)    61.1 GB             –         –  9.8/10        100%
  lobbot-moe (pruned MoE) ★  6.52 GB            49        54  9.7/10         99%  ✓
  dense (gemma-4-E4B-it)     5.30 GB            26        53  9.8/10        100%  ✗ 26 tok/s

  Tests    100 held-out examples, written by gemini-3.8-flash, never seen in training
  Scoring  gemini-3.8-flash grades each answer 0-10 against the task's criteria
  Data     2030 training examples written by Qwen3-30B-A3B from 15 seeds
  Time     25m26s on the GPU: data 2m29s · reap 8m39s · heal 3m53s · quantize 9m09s · eval 1m11s ·
           package 5s
  Bits     expert weights: q2_k 77%, q4_k 12%, q3_k 8%, q5_k 2% (avg 3.00 bits/weight)
  File     out/model.gguf, 6.52 GB
```

Note the "dense" row: Gemma scores 9.8 too, and it loses because it's too slow for the
target. That's a nice honest beat if you want one ("the backup was smart, but slow").

### Discharge: the model on the laptop

`lobbot save quality --chat` downloads the file, checks its sha256 against the VM's, and
installs it into Ollama (from `cmd_save` in `lobbot/app.py`):

```
Verifying sha256 ...
Importing into Ollama as lobbot-support-email-to-ticket ...
✓ saved in ~/lobbot-models/support-email-to-ticket-quality and imported into Ollama. Chat with: lobbot chat lobbot-support-email-to-ticket
```

The hero moment. The email and the answer are a real held-out test from the quality run
(the small model's actual answer, generated from the same GGUF file on the VM during the
check-up):

```
>>> helo... cancel subscritpion immediately!! do not charge me again! also why was i billed $49 instead of $29 last month?? wait actually keep my acount active until the 15th becos my accountant needs the reciepts then delete everything thx -Sent from my iPhone

{"category": "billing", "priority": "urgent", "summary": "Customer requests immediate subscription cancellation, questions a $49 charge instead of $29, and asks to keep account active until the 15th for invoice purposes.", "customer_sentiment": "negative"}
```

Two more real ones from the quality run, if you need variety:

```
>>> Hello, I'm reaching out because we're running into a bizarre permission inheritance bug after upgrading to the Enterprise workspace tier yesterday. … one contractor accidentally purged an entire sprint retro document earlier this morning! …

{"category": "bug", "priority": "urgent", "summary": "Enterprise upgrade caused SAML group mapping to incorrectly grant edit/delete permissions to external contractors, despite read-only settings.", "customer_sentiment": "negative"}
```

From the fast run (the cat is a gift for a goofy ad):

```
>>> Hey there, I've been using your platform for a few months now and overall it's great, but lately my dashboard is taking forever to load—like 45 seconds—especially when I'm viewing data from my Q3 sales reports. Also, my cat knocked over my coffee yesterday and I'm not sure if that's related, but I thought I'd mention it just in case 😅

{"category": "bug", "priority": "high", "summary": "Dashboard loading time is excessively slow, especially when viewing Q3 sales reports.", "customer_sentiment": "neutral"}
```

The cat did not make it into the ticket. Correctly.

---

## 5. Claims to avoid

| Don't say or show | Why | Say instead |
|---|---|---|
| It runs on a phone. | The model is 6.52 GB and runs in Ollama on a 16 GB Mac. The iPhone app is the face of the product and (once wired) a remote for the Mac, not where the model runs. | "Runs on your laptop." |
| Each cut is tested and undone if it hurts. | Not how the pipeline works (see section 1). | "Removes what your job never uses, then checks it against the original." |
| It's as smart as ChatGPT / a general assistant. | It's a specialist for one job. Ask it about anything else and it will try to make a ticket out of it. | "A specialist, not a know-it-all." |
| No quality lost / 100% / lossless. | It's 99% of the teacher's score on its own task (97% in the fast run). | "99% of the big model's score on its job." |
| Benchmark-proven, human-verified, accuracy. | The score is an AI judge (Gemini) grading AI-written test emails against the task's criteria. No standard benchmark, no human grading. | "Graded by Gemini on 100 new emails." |
| 10× smaller / a tenth of the size. | It's 9.4×. | "About 9× smaller" or "61 GB to 6.52 GB." |
| Built in 5 minutes / instantly. | 20 to 25 minutes of GPU stage time, plus a one-time VM setup and model download not counted. | "About 25 minutes on one GPU." |
| 56.7 tok/s on any laptop, or measured on an M4. | 56.7 was measured on an M5 MacBook; 49 on an M4 Air is an estimate. | "About 50 tokens per second on a 16 GB MacBook." |
| Your data never leaves your computer. | Building it uses the cloud GPU and Gemini sees the task description and examples. Only *running* the finished model is fully local. | "Runs offline once it's on your laptop." |
| Free / no subscription. | Nobody has decided pricing, and the GPU isn't free. | Leave pricing out. |
| It writes code / does any task. | Only the support-ticket job has verified results. Long-output jobs such as code need `lobbot run --long`, and no code run has a final score yet. | Show the support-email demo only. |
| A parameter count for the small model ("a 15B model", "3B"). | Not measured or reported anywhere. | Use the file size: 6.52 GB. |
| The shipped model is Gemma. | The winner is the pruned Qwen model; Gemma 4 is the backup candidate that lost on speed. | "Gemma 4 competes as the backup model." |
| Matrix OS, or condense.chat compressing the model. | Matrix OS isn't used; condense.chat only shortens the examples Gemini reads when writing tests. | See the sponsor table in section 3. |
| The app's simulated numbers ("70B → 8B", "4.2× lighter", capability %). | They're placeholders in the app's simulation mode. | The numbers in section 3. |

---

## 6. Glossary for the voice-over

- **Teacher / big model:** Qwen3-30B-A3B, the 61 GB model LobBot starts from.
- **Experts:** a mixture-of-experts model is built from many small sub-networks ("experts");
  each word only uses a few of them. LobBot measures which ones your job relies on and
  removes the rest.
- **REAP:** the pruning method that scores experts by how much they actually contribute on
  your task's data.
- **Heal:** fine-tuning the pruned model on the teacher's answers so it recovers.
- **Bits / quantization:** storing each number in fewer bits. LobBot gives important layers
  more bits and squeezes the rest.
- **GGUF:** the model file format Ollama and llama.cpp run.
- **Ollama:** the free app that runs models locally; the LobBot app wraps it as the
  "LobBot engine".
- **Tokens per second:** how fast it writes. About 50 tokens/s is faster than you can read.
