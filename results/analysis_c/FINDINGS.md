# Task C: failure cases, W0 vs W1 disagreement and per-generator plots

Post-hoc analysis of the saved Round 8 test predictions. No model was trained, refitted or reselected, and no threshold was changed. A decision is "fake" if sigmoid(logit) ≥ the model's saved threshold (`scripts/round7_baseline_common.py`, `threshold()`; W1 and R8A matched avg were tuned on format-matched selection views, `docs/ROUND7_STEPD.md`, `config/round8_a.json`).

**Data.** Round 8 test: 500 images from each of 6 generators plus one shared pool of 500 real images, 3 run seeds. **Main macro** = GLIDE, Midjourney, VQDM, Wukong (no model use in Round 7 development, but inspected before Round 8; `results/round8_final/SUMMARY.md`). **ADM and BigGAN** = development generators, marked "(dev)".

**Models.** **W0** = Round 7 global CLIP expert. **W1** = W0 retrained on format-matched images (`docs/ROUND7_STEPD.md`). **Local** = PatchShuffle local expert. **R7 average** = average fusion of global and local. **R8A matched avg** = average fusion of the format-matched global and local experts. **ResNet-50** = fine-tuned CNN. Values are means over 3 seeds; ± and error bars are the sample SD over seeds.

**Check.** The script stops unless its AUROC, balanced accuracy and real false-positive rate match `results/round8_final/metrics_mean_sd.csv` and `metrics_each_seed.csv`.

## How to run

### Environment
Python 3.12, CPU.

### Dependencies
numpy 1.26.4, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.9.2, pandas >= 2.2. Fig. 4 also needs pyarrow 25.0.1 and Pillow 12.2.0.

### Commands
Run from the repository root.
```
python -m zipfile -e results/handoff/round8_final_test_predictions.zip .
python scripts/analysis_c.py
```
Check the zip first against `results/handoff/SHA256SUMS.txt`.

## Figures

| Figure | Content | Numbers |
|---|---|---|
| `fig1_per_generator_auroc.png` | AUROC per generator and condition | `per_generator_auroc.csv` |
| `fig2_w0_w1_disagreement.png` | Images where only W0 or only W1 is correct | `w0_w1_disagreement.csv` |
| `fig3_failure_score_distributions.png` | R7 average: P(fake) per source vs each seed's threshold | `fusion_error_rates.csv`, `persistent_failures_counts.csv`, `always_wrong_images.csv` |
| `fig4_hard_examples.png` | Clean images all six models get wrong (first 4 per source by image index) | `clean_images_all_models_wrong.csv` |

Fig. 4 is built only when Tiny-GenImage shards (revision `89c4fe9`, SHA-256 checked against `results/round7/source_inventory.json`) are in `artifacts/round7/source/data/shards/`. Full coverage needs shards 0, 1, 2, 3, 5, 8 and 10. The PNG is not committed because it contains dataset images (`ROUND7.md`); the report version was built from shards 0 and 1 and credits Tiny-GenImage (CC BY-NC-SA 4.0).

## Findings

**1. Per generator (Fig. 1).** On clean images, W0 and the R7 average reach 0.90–0.96 AUROC on GLIDE, Midjourney and Wukong; on VQDM and ADM all four models shown are at or below 0.68. Under JPEG70/JPEG50, W0 and the R7 average fall below chance on ADM (0.37–0.41). W1 and R8A matched avg stay above chance there (0.54–0.59) but are lower than W0 on clean GLIDE (0.80 and 0.83 vs 0.91) and clean BigGAN (0.62 and 0.67 vs 0.76).

**2. W0 vs W1 (Fig. 2).** The two models make different decisions on 3–31 % of images.
- On clean images, W1 alone is right on more ADM (30 ± 9 %) and VQDM (22 ± 11 %) fakes, while W0 alone is right on more real images (16 ± 7 % vs 2.5 ± 1.7 %).
- Under JPEG70/JPEG50, W0 alone is right on more Wukong (24 % / 23 %), Midjourney (15 % / 16 %) and GLIDE (12 % / 13 %) fakes than W1 alone (5–9 %). On ADM, VQDM and BigGAN both are wrong on 85–96 % of fakes.
- On the main macro, W1's AUROC under JPEG70 is slightly higher than W0's (0.743 vs 0.734), but its balanced accuracy is lower (0.599 vs 0.629; `results/round8_final/metrics_mean_sd.csv`).

**3. Failure cases of the R7 average (Fig. 3).**
- Under JPEG50 the fusion misses 51–92 % of main-macro fakes and 96–98 % of ADM and BigGAN fakes, while flagging 4 % of real images (`fusion_error_rates.csv`).
- Under resize, 30 % of real images are flagged, against 12 % on clean images; 84 real images are flagged in all 3 seeds (26 on clean images).
- The saved threshold differs a lot between seeds (0.34, 0.56, 0.06; Fig. 3).
- 688 fakes are missed in all seeds and all four conditions (ADM 205, BigGAN 208, VQDM 187, GLIDE 60, Wukong 16, Midjourney 12), against 1 real image (`always_wrong_images.csv`).

**4. Images every model misses (Fig. 4).** On clean images, 367 images are wrong for all six models in all seeds: 340 are ADM, BigGAN or VQDM fakes and 2 are real (`clean_images_all_models_wrong.csv`). The 16 examples from shards 0 and 1 (not curated, no Wukong) are mostly close-ups of animals, 128–1024 px; some look natural, others are blurred or distorted (e.g. VQDM train:39, BigGAN train:969).

## Limits

- Post-hoc analysis of a test set that has already been used; nothing here may be used to change models or thresholds.
- The 3 seeds share one fitting pool, so the SD is run-to-run variation, not a confidence interval.
- One shared pool of 500 real images is reused across generators, so comparisons between generators are not independent.
