# Round 7 Step E: one frozen reserved-test evaluation

Open **`docs/ROUND7_FINAL.ipynb`** in Jupyter using the same Anaconda GPU kernel as Steps C/D. This is an actual notebook file, including visible child-process progress, safe interruption, incomplete-run resume and read-only verification/display cells.

## Freeze before test access

`results/round7/final_freeze.json` is created exclusively before the test manifest is opened. It declares all 138 model instances: every Step C model/profile/calibration for three run seeds, including all three gate initializations, plus Step D W1 and W2 (original/raw) for three seeds. The declaration contains the saved thresholds, temperatures, fixed weights, checkpoint paths and initialization identifiers. The selected checkpoint is read from the saved settings; no epoch/threshold/weight is selected again.

The freeze pins all existing Round 7 result metadata, protocols and manifests, saved Step C/D heads, gates, checkpoints and caches, scripts, documentation and available dependency/config files by SHA-256. Notebook **source cells** are hashed separately, excluding mutable execution counts and outputs. The reserved manifest hash is copied from the already-pinned data protocol and three agreeing assignments, without opening that manifest. Its actual bytes are checked only after the freeze exists. The freeze cannot be overwritten.

## Evaluation and isolation

Earlier scripts and their isolation guards stay byte-identical. Step E has its own exact reserved-ID allowlist, and rejects fitting/development IDs. Source payloads are selected from the local pinned shards before conversion to Python image bytes; original payload and Pillow 12.2 pixel hashes are verified. The freeze is required before any reserved-manifest read or source-image access.

All conditions use original evaluation images with unchanged Step C geometry: original or common128, followed by clean/jpeg70/jpeg50/resize05. Step D receives original-profile evaluation images without format matching. Shared CLIP features use the original frozen encoder and four PatchShuffle views. Logistic heads, gate scaler/weights, fusion weights, temperatures and thresholds are restored; no fit, training, calibration or selection occurs. ResNet loads the exact saved selected checkpoint in evaluation mode and FP32, at its saved inference batch size.

Reserved composition: 250 SD15 fakes, 500 each from ADM/BigGAN/GLIDE/Midjourney/VQDM/Wukong, and one shared pool of 500 real originals, totaling 3,750. Real images are predicted once per model/condition and reused across generator comparisons; they are not duplicated in a pooled metric.

## Run once, resume only incomplete work

There are 72 immutable prediction tasks. Shared CLIP feature batches and ResNet inference batches are saved atomically. Completed prediction CSVs are validated and reused, including recovery after an interruption between CSV publication and receipt creation. An operating-system lock prevents concurrent evaluators. An interrupted batch that was never atomically saved may need to be recomputed; completed batches/chunks are not rescored.

When `results/round7/final_test/complete.json` exists, the evaluator refuses to run again **even with `--resume`**. The notebook detects completion and skips evaluation. The verifier remains read-only and can be run repeatedly. Do not delete result files or edit frozen settings to restart the experiment.

Commands (the notebook streams these for you):

```python
run_script("round7_final_run.py", "--resume")
run_script("round7_final_verify.py", "--report")
```

Freeze creation, only if not already present:

```python
run_script("round7_final_run.py", "--freeze")
```

## Reporting

`results/round7/final_test/SUMMARY.md` presents every declared model, profile, calibration, generator and condition, with AUROC, AP, balanced accuracy and real FPR as mean ± sample SD over three run seeds. CSV files retain each seed and gate-initialization-level metrics. Gate metrics are averaged within a run before calculating the three-run SD; nine gate initializations are not treated as nine independent experiments.

Generator groups are frozen:

- **Untouched / main evidence:** GLIDE, Midjourney, VQDM, Wukong.
- **Previously inspected:** ADM, BigGAN.
- **In-domain:** SD15.

“Untouched” means unused for Round 7 development. The existing independent-pilot disclosure remains: 116 of the 3,750 reserved images overlapped a separate pilot that informed no Round 7 setting. This is not a claim that the domains/images were universally blind.

The untouched macro averages four generator metrics equally for each condition. Paired W1−W0 uses original/raw W1 versus Step C global. Average−global is reported for both Step C profiles/calibrations. Differences are computed within seed before mean/SD, for every generator and the untouched macro. Positive AUROC/AP/BA differences favor the left model; positive FPR differences are worse. No post-test model selection or extra condition is introduced.

Outputs include:

- `predictions/` and `receipts/`: immutable per-task original-level predictions and coverage/hash records.
- `metrics_initializations.csv`: 3,864 model-instance/generator/condition cells.
- `metrics_each_seed.csv`: 2,520 cells after within-seed gate averaging.
- `untouched_macro_each_seed.csv`: 360 cells.
- `metrics_mean_sd.csv`: 3,840 model/profile/calibration/domain/condition/metric rows, including untouched macro and each seed value.
- `paired_differences.csv`: 640 contrast/domain/condition/metric rows.
- `SUMMARY.md`, `run.json`, `complete.json`, and optional `verification.json`.

The verifier checks the freeze hashes, completion hashes, exact source/model/condition coverage and frozen thresholds, independently recomputes all four metrics from saved predictions, and checks macro/paired/three-seed tables. It does not load images or score models again.

Pre-freeze tests in `scripts/round7_final_test.py` use synthetic fixtures and already-existing development feature caches only. They check exact restoration of Step C heads/gates and W1, full declaration coverage, cache identity, once-only behavior, test-isolation rejection and aggregation arithmetic. No commit or push is performed.
