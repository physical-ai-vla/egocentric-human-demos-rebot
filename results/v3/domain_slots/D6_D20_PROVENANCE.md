# D6 vs D20 domain-slot provenance (X-VLA, read-only audit, 2026-10-02)

## Short answer

**D6 (slot 6)**
- soft prompt: never trained in xvla-base; untouched N(0, 0.02) init (shrunk ×0.82, see §5). Carried bit-exact through every widening.
- action encoder: action rows 0:20 and time rows are the original untrained Xavier init (bit-exact); action rows 20:32 and proprio rows are new random rows added by `widen_v3.py` (seed 0). Bias exactly 0.
- action decoder: outputs 0:20 are the original untrained Xavier init (bit-exact); outputs 20:32 are new random columns from `widen_v3.py`. Bias exactly 0.

**D20 (slot 20)**
- soft prompt, action encoder, action decoder: **the same provenance as slot 6, block for block.** It is an untrained slot that already existed in `lerobot/xvla-base`, kept bit-exact through `widen_v3` and `widen_cart20`. It is not a copy of slot 6 or of any other slot.
- Answer to §9: **A) a genuinely unused slot that was already in the checkpoint.** We did not make new weights. We started using slot 20 instead of slot 6.

**Main implication**
- D6 and D20 start from statistically equivalent, independent random draws of the same init distribution. The cosine similarity between them is about 0 in every block.
- So D6 vs D20 is a comparison of **init-draw / seed variance** only. Neither slot carries pretrained specialization.
- Neither does slot 0, which all the earlier runs used.
- Only slots 10–17 were trained in xvla-base.

## Sanity check (requested one-liner)

Changing `policy_preprocessor.json` 6 → 20 did **not** copy or initialize any weights.

| | inode | links | sha256 (model.safetensors) | processor domain_id |
|---|---|---|---|---|
| xvla_base_cart20v3 | 24548715 | 3 | 7e9c63322c516eeb… | 0 |
| xvla_base_cart20v3_d6 | 24548715 | 3 | 7e9c63322c516eeb… | 6 |
| xvla_base_cart20v3_d20 | 24548715 | 3 | 7e9c63322c516eeb… | 20 |

All three are **the same file on disk** (one inode, three hard links). The d6 and d20 folders differ from cart20 only in `policy_preprocessor.json` and `DOMAIN.md`. The processor only puts `domain_id` into the batch, and `DomainAwareLinear` / `soft_prompt_hub` then *select* that row. Training then updates the selected rows.

## 1. Lineage (actual, from scripts and WIDENING.md)

```
lerobot/xvla-base  HF snapshot cdb7964e (sha f05bc0fa…)        encoder in 72 = [action 20 | proprio 20 | time 32], decoder out 20
  └ umi76/widen_v3.py --seed 0 → xvla_base_rel16v3 (64225139…)   encoder in 140 = [action 32 | proprio 76 | time 32], decoder out 32
  └ umi76/widen_cart20.py      → xvla_base_cart20v3 (7e9c6332…)  encoder in 84  = [action 32 | proprio 20 | time 32], decoder out 32
  └ hard links + processor json → xvla_base_cart20v3_d6 (domain 6), xvla_base_cart20v3_d20 (domain 20)
```
No script copies one domain slot into another. Every widening op is applied to all 30 slots identically (`E1[:, ...]` over the slot axis).

## 2. Dimensions

| | num_domains | hidden | len_soft_prompts | dim_time | action / proprio |
|---|---|---|---|---|---|
| all checkpoints | 30 | 1024 | 32 | 32 | orig 20/20 → v3 32/76 → cart20/d6/d20 32/20 |

Full key shapes. `DomainAwareLinear.fc` is `Embedding(nd, in*out)`, read as `view(nd, in, out)`, which is input-major.

| key | orig | v3 | cart20 = d6 = d20 |
|---|---|---|---|
| model.transformer.action_encoder.fc.weight | (30, 73728) = 72×1024 | (30, 143360) = 140×1024 | (30, 86016) = 84×1024 |
| model.transformer.action_encoder.bias.weight | (30, 1024) | (30, 1024) | (30, 1024) |
| model.transformer.action_decoder.fc.weight | (30, 20480) = 1024×20 | (30, 32768) = 1024×32 | (30, 32768) |
| model.transformer.action_decoder.bias.weight | (30, 20) | (30, 32) | (30, 32) |
| model.transformer.soft_prompt_hub.weight | (30, 32768) = 32×1024 | same | same |

## 3. Slot-vs-slot similarity (within the d6/d20 file; full table `domain_similarity.csv`)

Best match among all other slots, per sub-block:

| sub-block | D6 best other slot (cos) | D20 best other slot (cos) | D6 vs D20 |
|---|---|---|---|
| enc.action_old0:20 | 26 (+0.012) | 1 (+0.016) | cos −0.003, not equal |
| enc.action_new20:32 | 13 (+0.018) | 23 (+0.023) | cos −0.006 |
| enc.proprio | 26 (+0.015) | 28 (+0.011) | cos +0.0001 |
| enc.time | 9 (+0.013) | 26 (+0.008) | cos −0.005 |
| dec.out_old0:20 | 27 (+0.015) | 21 (+0.011) | cos −0.002 |
| dec.out_new20:32 | 10 (+0.018) | 3 (+0.012) | cos +0.006 |
| soft_prompt | 1 (+0.013) | 27 (+0.011) | cos −0.007 |
| enc.bias, dec.bias | all-zero, so equal to every other untrained slot (trivially) | same | equal (both 0) |

No weight block of D6 or D20 equals or nearly equals any other slot. All cosines are within ±0.025 and relative L2 is about 1.40, which is what two independent random vectors give. Exact equality occurs only for the all-zero biases.

## 4. Where each block first changed (slot 6 and slot 20 identical pattern; also checked 0 and 15)

| block | orig → v3 | v3 → cart20 | cart20 → d6 → d20 |
|---|---|---|---|
| enc.action rows 0:20 | **exact copy** | exact | exact (same file) |
| enc.action rows 20:32 | **new** N(0, sd(base action rows)), seed 0 | exact | exact |
| enc.proprio | **replaced**: v3 proprio[0:20] ≠ orig proprio (cos ≈ 0), new N(0, sd) | **v3 proprio[0:20] kept exactly**, 20:76 dropped | exact |
| enc.time | exact copy (orig rows 40:72) | exact | exact |
| enc.bias | exact (0 for slots 6/20) | exact | exact |
| dec.out 0:20 | exact copy | exact | exact |
| dec.out 20:32 | **new** N(0, sd(base decoder)), seed 0 | exact | exact |
| dec.bias | 0:20 exact (0), 20:32 = 0 | exact | exact |
| soft_prompt | exact | exact | exact |

Partial-copy behaviour (§5 of the request), stated as facts:
- Widening was "old sub-block copied + new rows/cols random". There was no full re-initialisation.
  - Encoder: inherited action[0:20] and time; new random action[20:32] and proprio.
  - Decoder: inherited outputs 0:20; new random outputs 20:32.
- The **proprio rows were not inherited for any slot**. That includes the trained slots 10–17, whose original proprio rows were learned (std 0.033–0.077). `widen_v3.py` documents this choice ("base proprio was slim20, unrelated meaning").

## 5. Random-init fingerprint

Original init (lerobot `soft_transformer.py`):
- `DomainAwareLinear`: `xavier_uniform_(fc)`, `zeros_(bias)`.
- `soft_prompt_hub`: `normal_(std=0.02)`.

Per-slot std, d20 file (identical to orig for inherited blocks):

| block | expected init std | slot 6 | slot 20 | 20 other untrained slots (0–9, 18–29 except 6, 20) | trained slots 10–17 |
|---|---|---|---|---|---|
| enc.action 0:20 | 0.00521 | 0.00426 | 0.00425 | 0.00423–0.00428 | 0.0171–0.0270 |
| enc.time | 0.00521 | 0.00424 | 0.00423 | 0.00422–0.00427 | 0.0112–0.0181 |
| dec.out 0:20 | 0.00987 | 0.00809 | 0.00809 | 0.00798–0.00810 | 0.081–0.180 |
| soft_prompt | 0.02 | 0.01624 | 0.01638 | 0.01626–0.01648 | 0.0367–0.0412 |
| enc.bias / dec.bias 0:20 | 0 | **exactly 0** | **exactly 0** | exactly 0 | non-zero (0.004–0.05) |
| enc.proprio (new) | – | 0.0286 | 0.0286 | 0.0282–0.0288 | 0.0286 (also new random) |
| enc.action 20:32 / dec.out 20:32 (new) | – | 0.0122 / 0.0689 | 0.0121 / 0.0690 | 0.0121–0.0123 / 0.0677–0.0695 | same (also new) |

- **Fact:** biases of slots 6 and 20 are exactly zero. That is their init value, and any gradient step would have moved them. So these slots never received a task gradient during xvla-base pretraining.
- **Fact:** slots 6, 20 and the other 20 untrained slots share mean ≈ 0 and the same std to 3 significant digits in every block. Yet they are mutually uncorrelated (cos ≈ 0). They are independent draws of the same distribution.
- **Inference:** the uniform ×0.82 ratio to the init std (Xavier, and soft prompt 0.02) is consistent with decoupled AdamW weight decay. That decay would act on the dense-zero-gradient rows during pretraining. It shrinks values but does not specialise them.
- The trained slots 10–17 differ clearly: weights 4–20× larger and non-zero biases. By which action dims 10:20 are trained, 10/15/16/17 look bimanual and 11–14 single-arm (inference from weights only).

## 6. Scripts that created D6 / D20

- `widen_v3.py`, `widen_cart20.py`: per-block ops over all slots, no slot indexing, and a functional gate across all domains.
- D6 (2026-10-01) and D20 (2026-10-02) were created by hard-linking every file of `xvla_base_cart20v3` except `policy_preprocessor.json`, then rewriting only `xvla_add_domain_id.domain_id`. No Python touched `model.safetensors`; the shared inode confirms it.
- Nothing in the lineage contains a "copy domain" operation.

## 7. Semantic meaning of domain 6

- **Unknown from local evidence.** Neither the xvla-base snapshot (config, processor, README) nor the lerobot fork's X-VLA code contains a domain-id → dataset table.
- The earlier label "6 = robotwin2 / AgileX" came from the LeRobot X-VLA docs page, not from this checkpoint.
- Whatever that table means, **slot 6 of this checkpoint has no learned values.**

## 8. Implications for the running experiments

- `R312C-RELCART20-RELONLY-D20-S600K` vs `R312C-RELCART20-RELONLY-D6-S600K`: identical data, recipe and base file. The only difference is which never-trained random slot is selected, i.e. a different init draw for the soft prompt and the per-domain encoder/decoder.
  - A gap between them measures init/seed sensitivity, **not** a domain effect.
  - The same holds for domain 0, used by all runs before 10-01.
- `EGO-CART20V2-RELONLY-D20-100K-V2B` vs the D6 ego run: the same holds.
- Shared-trunk weights are bit-identical across D0/D6/D20 bases. Those are the VLM, the transformer, and the action-encoder action rows and decoder rows that matter for pretrained behaviour.
- The actually different option would be to copy a trained slot (10/15/16/17) into a free slot. It was measured but not chosen. Initial REL loss on R312c:

  | slot | initial REL loss |
  |---|---|
  | 0 | 1.11 |
  | 6 | 1.12 |
  | 15 | 1.61 |
  | 10 | 2.35 |
  | 16 | 2.65 |
  | 17 | 4.07 |

  Even then the proprio rows would be random, because `widen_v3` replaced them for every slot.

## Files
- `domain_similarity.csv`: every (group, ref slot ∈ {6, 20}, slot 0..29) pair with exact / max_abs / l2 / rel_l2 / cos.
- `lineage_comparison.json`: file identity (inode, sha256, link count), dims and shapes, per-slot fingerprint stats (orig and d20), per-block lineage steps for slots 0/6/15/20.
- `domain_slot_audit.py`: the read-only script (ran on the 4090 CPU; it opens safetensors read-only and writes only to /tmp/dsa).
  - Known artifact: its lineage rows for `enc.proprio` orig→v3 and v3→cart20 compare tensors of different shapes and print "changed". The matched-shape check in §4 is a separate run. Result: v3 proprio[0:20] == cart20 proprio exactly, and orig proprio vs v3 proprio[0:20] has cos ≈ 0 and is not equal.
