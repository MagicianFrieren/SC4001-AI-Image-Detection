# Experiment guide

Use this guide to connect each research question to its code and saved evidence.
The experiments were completed through Round 6 on October 22, 2026; this guide was updated on October 23.

## Experiment sequence

| Round | Question | Runner | Results and interpretation |
| --- | --- | --- | --- |
| Pilot | Does a frozen CLIP classifier transfer beyond SD1.5? | [run_pilot.py](../scripts/run_pilot.py) | [Initial English report](../output/pdf/sc4001_j_preliminary_report_en.pdf) |
| 2 | How do preprocessing and generator diversity affect transfer? | [run_round2.py](../scripts/run_round2.py) | [Summary](../results/round2/summary_zh.txt), [protocol](../results/round2/protocol.json) |
| 3 | Do global and PatchShuffle features offer complementary information? | [run_round3.py](../scripts/run_round3.py) | [Interpretation](../results/round3/interpretation_zh.txt), [error overlap](../results/round3/error_overlap.csv) |
| 4 | Can ordinary or quality-aware gating use that information? | [run_round4.py](../scripts/run_round4.py) | [Interpretation](../results/round4/interpretation_zh.txt), [gate weights](../results/round4/gate_weight_summary.csv) |
| 5 | Do expert augmentation and score calibration improve gating? | [run_round5.py](../scripts/run_round5.py) | [Interpretation](../results/round5/interpretation_zh.txt), [calibration results](../results/round5/heldout_calibration.csv) |
| 6 | Which augmentation contributes to the performance loss? | [run_round6.py](../scripts/run_round6.py) | [Interpretation](../results/round6/interpretation_zh.txt), [macro contrasts](../results/round6/macro_contrasts.csv) |

Compare models within a round's controlled protocol.
Rounds change training partitions, validation conditions, and selection procedures; their headline scores are not one continuous leaderboard.

## Model components

The global branch uses the processed image and a frozen CLIP ViT-B/32 encoder.
The local branch rearranges sixteen intact 56 × 56 tiles in a 224 × 224 image.
It uses four fixed, label-independent permutations and averages the four normalized image embeddings.
Both branches fit separate standardized logistic regression heads.
All image degradations happen before PatchShuffle.

The gates share an 8 → 32 → 1 network with 321 parameters.
Their output is the global branch weight in a convex combination of expert logits.
Both receive expert probabilities, their absolute gap, mean predictive entropy, and local-view logit variance.
Quality-aware gating also receives Laplacian sharpness, high-frequency energy, and an 8-pixel block-boundary ratio.
Ordinary gating zeros those three standardized quality inputs while retaining the same architecture.

We compare global-only, local-only, average fusion, validation-selected fixed fusion, ordinary gating, and quality-aware gating.
The gates use paired initialization seeds: 20261022, 20261023, and 20261024.
The [gate implementation](../scripts/gate_models.py) contains optimization and early-stopping settings.

## Partitions in Rounds 5–6

All partitions are disjoint at the original-image level.
Different degraded views of one image stay within its partition.

| Purpose | Real | SD1.5 | BigGAN | ADM | Total originals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Expert fitting | 200 | 100 | 100 | 0 | 400 |
| Gate fitting | 100 | 50 | 50 | 0 | 200 |
| Temperature calibration | 50 | 25 | 25 | 0 | 100 |
| Model and threshold selection | 50 | 25 | 25 | 0 | 100 |
| Exploratory development | 100 | 100 | 100 | 100 | 400 |

Each generator's development metric compares its 100 fake images against the same 100 real images.
See [assignments.json](../results/round6/assignments.json) for exact source IDs.
The remaining images in the manifest are unused in these partitions.

Gate fitting, calibration, and model selection use clean, JPEG70, and half-resolution views.
Development uses clean, JPEG50, and half-resolution views.
Therefore half-resolution is a seen degradation, while JPEG50 changes the compression strength.

Temperature calibration fits one positive scalar per expert on calibration data only.
It changes confidence and relative logit scale, while preserving each expert's ranking.
Model selection uses separate validation images; thresholds remain fixed during development evaluation.

## Round 6 controls

| Training arm | Clean weight | JPEG70 weight | Half-resolution weight |
| --- | ---: | ---: | ---: |
| `clean` | 1 | 0 | 0 |
| `jpeg_only` | 2/3 | 1/3 | 0 |
| `resize_only` | 2/3 | 0 | 1/3 |
| `augmented` | 1/3 | 1/3 | 1/3 |

Each original contributes total weight 1, with the same classifier regularization setting.
Single-augmentation arms replace the missing degradation with another clean view.
They do not use a 50/50 clean/augmented mixture.
This keeps each degradation's dose matched between the single and joint arms.

We cross four training arms with two preprocessing profiles and raw/temperature-calibrated scores.
The resulting 16 configurations contain 96 fitted gates and 1,440 development metric cells.
The clean and joint controls reproduce all 96,000 matching Round 5 development predictions exactly in the recorded run.

## Evidence and verification

| File | How to use it |
| --- | --- |
| `protocol.json` | Read the question, controls, selection rules, and limits before comparing scores |
| `predictions.csv` | Inspect development logits, probabilities, and fusion weights by source ID |
| `validation_predictions.csv` | Trace model selection and fixed decision thresholds |
| `metrics.csv` | Compare individual models, initializations, generators, and conditions |
| `aggregate_metrics.csv` | Read condition-level results averaged across gate initializations |
| `macro_summary.csv` | Read equal-weight averages across the three development conditions |
| `paired_contrasts.csv` | Compare paired models within a development condition |
| `macro_contrasts.csv` | Compare augmentation recipes across all three conditions; available in Round 6 |
| `verification.json` | Check what the round-specific verifier actually checked |

The full Round 6 verifier refits experts, independently checks temperatures, reconstructs gate outputs, and recomputes metrics.
It does not independently repeat gate training.
Bootstrap draws keep each original image paired across all three conditions when calculating macro differences.
Intervals remain conditional on the fitted models and this small development sample.

The [lightweight shared-result check](../scripts/verify_shared_results.py) has a narrower scope.
It needs no images or caches and checks source/protocol hashes plus Round 6 point metrics.
Use the [reproduction guide](REPRODUCING.md) for both verification paths.
