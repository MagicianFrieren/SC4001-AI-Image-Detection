# Round 7 — fixed-protocol evaluation and whole-image study

**Author:** Ruixuan · **Completed:** 26 October 2026 · **Status:** finished, including the one-time reserved-test evaluation.

This round builds on Rounds 1–6 (frozen CLIP global/local experts, fusion and gates). **No Round 1–6 file was changed:** every Round 7 file is new. `README.md` is left as it was because the Round 7 data verifier checks that pre-existing files stay byte-identical.

## In short

- Re-ran the existing global, local, fusion and gate models under one fixed protocol, with **3 data splits**, a **separate development set** and a **locked test set** of generators never used for development.
- Added a **fine-tuned ResNet-50** whole-image baseline.
- Found that the whole-image detector partly relies on a **JPEG/PNG shortcut** (all real images are JPEG, all fakes PNG): JPEG-compressing ADM fakes pushes AUROC **below chance (~0.36)**.
- Tested a fix, **format-matched training (W1)**: it removes the ADM flip and halves the JPEG drop on unseen generators, but costs clean accuracy on GLIDE and BigGAN. It failed the success criteria fixed in advance, so it is reported as a trade-off, not an improvement.
- **Simple averaging of global + local is the best combiner.** The learned gates, including the quality-aware gate, never beat it.

## ⚠ Important for everyone

- **The reserved test set has been used (once, on 26 Oct).** Any model or setting change from now on cannot be evaluated on it as a fair test. Please don't develop the local branch or gates further and then report test numbers.
- Development and test claims must keep the exposure labels below (untouched / previously inspected / in-domain).
- Verification needs the **pinned Pillow 12.2.0** (older Pillow changes JPEG pixel hashes).

## Protocol

| Item | Setting |
|---|---|
| Source | `TheKernel01/Tiny-GenImage`, revision `89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69`, all 14 train shards (28,000 images, SHA-256 verified). Licence reported as CC BY-NC-SA 4.0; images are not redistributed. |
| Training generator | **SD1.5.** SD1.4 has 0 images in this source, so SD1.5 replaces the official GenImage SD1.4 protocol. |
| Exclusions | All 1,600 images of the Round 2 pool (+4 near-duplicates by dHash ≤ 3) and 25 further duplicates. |
| Fitting pool | 1,250 real + 1,250 SD15, split per seed into expert-fit / gate-fit / calibration / selection = 750 / 250 / 125 / 125 per class. |
| Seeds | 20261101, 20261102, 20261103 (same pool, different splits). |
| Development set | 300 ADM + 300 BigGAN + 300 real, disjoint from fitting and test. |
| Reserved test | 3,750 images: 250 SD15, 500 each ADM / BigGAN / GLIDE / Midjourney / VQDM / Wukong, one shared 500-real pool. |
| Exposure labels | **Untouched (main evidence):** GLIDE, Midjourney, VQDM, Wukong. **Previously inspected:** ADM, BigGAN (Rounds 1–6 and Round 7 development). **In-domain:** SD15. |
| Models / settings | Inherited unchanged from Round 6: frozen CLIP ViT-B/32 global and PatchShuffle local experts, average / fixed fusion, score and quality gates, temperature calibration, `original` and `common128` profiles, conditions clean / jpeg70 / jpeg50 / resize05. |

Disclosure: 116 of the 3,750 test images (and 94 fitting images) also appear in a separate, independent team-member pilot drawn from the same source. That pilot informed no Round 7 setting.

## Steps

| Step | What | Main scripts | Results | Docs |
|---|---|---|---|---|
| A | Inspect the source; build splits and the test manifest | `round7_inspect_source.py`, `round7_prepare_data.py`, `round7_verify_data.py` | `results/round7/data_*.json`, `assignments_seed{1,2,3}.json`, `*_manifest.jsonl` | `docs/ROUND7_SOURCE_INSPECTION.md`, `docs/ROUND7_DATA.md` |
| B | Development set (ADM / BigGAN) | `round7_add_development.py`, `round7_verify_development.py` | `results/round7/development_manifest.jsonl` | `docs/ROUND7_DEVELOPMENT.md` |
| C | Baselines under one protocol + ResNet-50, 3 seeds | `round7_run_baselines.py`, `round7_baseline_*.py`, `round7_verify_baselines.py`, `round7_summarize_seeds.py` | `results/round7/baselines_seed{1,2,3}/`, `seeds_summary/` | `docs/ROUND7_BASELINES.md`, `docs/ROUND7_REPEATED_SEEDS.md` |
| D | Whole-image study: W1 format-matched CLIP, W2 format-matched ResNet-50 | `round7_stepD_*.py` | `results/round7/stepD_protocol.json`, `stepD_whole_image/` | `docs/ROUND7_STEPD.md` |
| E | Freeze, then one reserved-test evaluation of all 138 saved model instances | `round7_final_*.py` | `results/round7/final_freeze.json`, `final_execution_addendum.json`, `final_test/` | `docs/ROUND7_FINAL.md`, `docs/ROUND7_FINAL_PRECISION_NOTE.md` |

Each step wrote its protocol before running and has an independent verifier. Step E's freeze (code, settings and model hashes) was sealed before the test manifest was opened. One post-freeze reporting-precision correction (float32 → float64 before sigmoid) is documented in the addendum; it changed no model, threshold or prediction.

## Final test results (untouched generators)

AUROC, mean of 3 seeds, `original` profile, raw scores. Equal-weight average over GLIDE, Midjourney, VQDM and Wukong.

| Model | Clean | JPEG70 | JPEG50 | Resize ½ |
|---|---:|---:|---:|---:|
| Global CLIP (W0) | 0.854 | 0.734 | 0.734 | 0.780 |
| Global CLIP, format-matched (W1) | 0.819 | 0.749 | 0.753 | 0.788 |
| Local (PatchShuffle) | 0.813 | 0.717 | 0.712 | 0.711 |
| **Average fusion** | **0.860** | **0.756** | **0.755** | 0.782 |
| Quality gate | 0.851 | 0.748 | 0.743 | 0.769 |
| ResNet-50, fine-tuned | 0.748 | 0.658 | 0.640 | 0.686 |
| ResNet-50, format-matched (W2) | 0.722 | 0.665 | 0.650 | 0.679 |

Per generator, global CLIP (W0) on clean images: Midjourney 0.93, Wukong 0.92, GLIDE 0.91, VQDM 0.66. ADM (previously inspected): W0 0.60 clean / 0.39 JPEG70; W1 0.67 / 0.59. Full tables, each seed, AP, balanced accuracy and real-image FPR: `results/round7/final_test/SUMMARY.md` and CSVs.

## Findings

1. **Frozen CLIP transfers to unseen generators, unevenly:** roughly 0.9 AUROC on Midjourney, Wukong and GLIDE, but only 0.66 on VQDM.
2. **JPEG is the main weakness.** It costs about 0.12 AUROC on untouched generators and flips ADM below chance. Development and test agree.
3. **Format-matched training (W1) trades clean accuracy for JPEG robustness.** It improves ADM in every condition and halves the JPEG drop, but lowers clean GLIDE (0.91 → 0.82) and BigGAN. Its pre-registered criteria failed on development (clean BigGAN). Remaining confound: re-encoded real images are double-compressed, fakes single-compressed.
4. **Fine-tuning ResNet-50 generalises worse than frozen CLIP**, except on Wukong (0.96), which is Stable-Diffusion-like. This suggests fine-tuning helps mainly for generators similar to SD15.
5. **Average fusion ≥ every learned gate.** The quality gate does react to JPEG (on seed-1 development data its global weight falls from about 0.75 on clean images to about 0.15 at JPEG50), but this does not improve ranking.
6. `common128` preprocessing is worse than `original` for the global branch in this protocol.

## Reproduce or check

Data, feature caches, checkpoints and the Pillow 12.2 install live under `artifacts/` and are not in Git. **The per-image prediction CSVs (~370 MB) are also not in Git**: `results/round7/final_test/predictions/`, `baselines_seed*/*_predictions.csv` and `stepD_whole_image/**/*predictions.csv`. All summaries, metrics, protocols, manifests, freeze and verification reports are committed, so every reported number can be read without them. Ask Ruixuan for the predictions archive if you need to rerun the verifiers that recompute metrics from predictions, or to make new plots. The 14 source shards go in `artifacts/round7/source/data/shards/` (see `docs/ROUND7_SOURCE_INSPECTION.md`). The notebook guides contain the `run_script` setup cell; use the same Anaconda GPU environment as the original runs.

Read-only checks, run from the repo root:

```text
python scripts/round7_verify_data.py
python scripts/round7_verify_development.py
python scripts/round7_verify_baselines.py --seed 20261101   (also 20261102, 20261103)
python scripts/round7_summarize_seeds.py --verify-only
python scripts/round7_stepD_verify.py --summary --preservation
python scripts/round7_final_verify.py
python scripts/verify_shared_results.py
```

Do not rerun `round7_final_run.py`: it refuses once `results/round7/final_test/complete.json` exists, by design.

## Limitations

- Development domains (ADM, BigGAN) and SD15 selection results are not blind. SD15 numbers are optimistic.
- The three seeds share one fitting pool and one development set. The SD describes run-to-run variation, not a confidence interval.
- The JPEG-vs-PNG and generator-specific size confounds are inherent to the source data. Harmonisation and format matching control them but do not remove them.
- Tiny-GenImage generator labels are inherited from the third-party source and not independently authenticated.

## AI assistance

Implementation was carried out with OpenAI Codex (ChatGPT), and planning and result review with Anthropic Claude, under the author's direction. Results were checked with the independent verifier scripts listed above.
