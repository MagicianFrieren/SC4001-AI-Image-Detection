# Round 7 Step D: whole-image format matching

The pre-training protocol is `results/round7/stepD_protocol.json`. Its exact question, variants and success criteria were written before any training, including synthetic training. This implementation adds new files only; it does not edit earlier rounds, Step C scripts, manifests, predictions, checkpoints or summaries. Do not change the protocol or implementation after starting real runs.

## What runs

- **W0:** reuse Step C's saved **raw**, original-profile global CLIP predictions and thresholds for each seed. No retraining or new W0 feature extraction.
- **W1:** same frozen CLIP ViT-B/32 encoder and standardized logistic head as Step C. Re-encode native-size decoded RGB expert-fit images as JPEG once, using an integer quality uniformly drawn from 60–95 by a deterministic SHA-256 rejection sampler keyed by seed and source ID. Labels, epoch, batch order and variant do not enter the draw. Then apply the unchanged original 224-pixel profile.
- **W2:** use those same deterministic matched inputs with the unchanged Step C ResNet training implementation: all layers trainable, ImageNet V2 weights, AdamW 1e-4, weight decay 1e-4, batch 32, ten epochs, cosine schedule and selection-based epoch choice. Only CUDA OOM allows the inherited batch-halving/epoch-rollback behavior.

For W1 and W2, **tuning selection views are format-matched** at native resolution, followed by original preprocessing and inherited clean/jpeg70/resize05 conditions. Thresholds use pooled selection Youden J. W2 selects the epoch maximizing mean AUROC across these same three conditions, retaining the earliest exact tie. JPEG50 remains report-only.

**Reporting selection and development views start from the original images.** They are not format-matched. They receive unchanged original preprocessing and clean/jpeg70/jpeg50/resize05 conditions. Selection therefore has two explicitly separate views: matched tuning and original reporting. SD15 selection, ADM development and BigGAN development are reported separately. Step D uses raw scores only; it does not fit temperatures or use gate-fit/calibration partitions.

Only the same expert-fit images train each run. The same three seed assignments, 20261101/20261102/20261103, remain unchanged. All image/feature access uses the isolation guard and a narrowed allowlist of expert-fit, selection and development IDs. Original images and pretrained weights are read from the existing verified Step C caches; original caches are never rewritten. The reserved-test manifest is never opened and reserved-test images are never decoded, featurized or scored.

## Frozen success criteria

For **each of W1 and W2 separately**:

1. Mean of JPEG70/JPEG50 ADM AUROC >= 0.50.
2. Clean ADM and clean BigGAN AUROC drop <= 0.02 versus W0.
3. Both requirements hold **in every seed**.

The report prints PASS/FAIL for the JPEG criterion, each clean-domain criterion, each seed jointly and each variant across all seeds. The inequalities are inclusive; only a fixed 1e-12 numerical tolerance is allowed. A good mean across seeds cannot rescue a failing seed. These criteria are not changed after results.

The clean SD15 cost is reported, not used as an additional pass criterion. Paired W1−W0, W2−W0 and W2−W1 differences are calculated within each seed for every condition, domain and metric. Reports contain each seed, mean and sample SD (`ddof=1`), not pooled predictions or population confidence intervals.

## Limitations recorded in advance

Real images become double-compressed, fakes single-compressed. Native image sizes and compression history are not eliminated. Original reporting JPEG conditions add compression to the original evaluation images, not to training-style matched copies. The experiment can show sensitivity to the format-matching intervention; it cannot prove the format confound is the sole cause. W1−W0 holds architecture fixed; W2−W0 also changes architecture.

Original SD15 reporting images share identities with the matched selection images used for tuning, so they are not independent test evidence. ADM and BigGAN remain previously inspected development domains. The seed runs share the fitting pool and development images.

## Jupyter setup

Use the same GPU-enabled Anaconda kernel as the three Step C runs. No dependency changes are needed. The scripts select repository-pinned Pillow 12.2.0 before importing image code and reject runtime version drift from Step C. Run the loops sequentially on the RTX 4050.

```python
from pathlib import Path
import os, subprocess, sys

project = Path(r"C:\sc4001\prjJ\SC4001-AI-Image-Detection")
env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["PYTHONIOENCODING"] = "utf-8"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def run_script(script, *args):
    command = [sys.executable, "-u", str(project / "scripts" / script), *args]
    process = subprocess.Popen(command, cwd=project, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    finally:
        process.stdout.close()
    if code:
        raise subprocess.CalledProcessError(code, command)

subprocess.run([sys.executable, "-c",
               "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
               "print(torch.cuda.get_device_name(0))"], cwd=project, env=env, check=True)
run_script("round7_stepD_run.py", "--init")  # validates the pre-training protocol/implementation lock
```

### W1: all three seeds

```python
for seed in (20261101, 20261102, 20261103):
    run_script("round7_stepD_run.py", "--variant", "W1", "--seed", str(seed), "--resume")
    run_script("round7_stepD_verify.py", "--variant", "W1", "--seed", str(seed))
```

### W2: all three seeds

```python
for seed in (20261101, 20261102, 20261103):
    run_script("round7_stepD_run.py", "--variant", "W2", "--seed", str(seed), "--resume")
    run_script("round7_stepD_verify.py", "--variant", "W2", "--seed", str(seed))
```

### Summary and verification

```python
run_script("round7_stepD_report.py", "--resume")
run_script("round7_stepD_verify.py", "--summary", "--preservation")

from IPython.display import display, Markdown
display(Markdown((project / "results/round7/stepD_whole_image/summary/SUMMARY.md")
                 .read_text(encoding="utf-8")))
```

Rerun the same cells after interruption. Completed runs are hash-checked and skipped. W1 resumes validated feature caches and its saved logistic head. W2 resumes its last completed epoch, optimizer, scheduler and RNG state; an incomplete epoch is rerun. Final predictions/settings are stored in an atomic result bundle before publication, so partial final publication can also resume by checking existing bytes and adding missing files. The summary similarly refuses to overwrite changed files and requires all six W1/W2 runs to be complete before writing any output. Verification is read-only.

## Outputs

`results/round7/stepD_whole_image/seed{1,2,3}/W{1,2}/` contains original-view predictions, matched-view tuning predictions, per-condition metrics, quality draws, settings/history, an isolation audit, a run protocol and completion hashes. `artifacts/round7/stepD_whole_image/` contains only new caches, heads, checkpoints, bundles and access logs.

`results/round7/stepD_whole_image/summary/` contains:

- `W0_predictions.csv`: exact reused raw global logits/probabilities/thresholds, with Step D variant and run-seed identifiers.
- `metrics_each_seed.csv`: 108 variant/seed/domain/condition cells, each with AUROC, balanced accuracy and real FPR.
- `metrics_mean_sd.csv`: 108 variant/domain/condition/metric cells, each seed plus mean ± SD.
- `paired_differences.csv`: 108 contrast/domain/condition/metric cells, each seed plus mean ± SD.
- `criteria.csv`, `success.json`: the frozen per-seed and all-seed PASS/FAIL checks.
- `clean_SD15_cost.csv`: six variant/metric cells. Positive cost means worse than W0: AUROC/BA use W0 minus variant; FPR uses variant minus W0.
- `SUMMARY.md`, `provenance.json`, `complete.json`: readable tables and immutable input/output hashes.

`stepD_preservation.json` records the pre-change hashes of 10,606 existing files. The verifier can independently check that they remain unchanged. It never reads the reserved-test manifest. The saved Step D metrics are recomputed with scikit-learn from predictions; matched-selection threshold fitting, source/condition coverage, label identity, quality draws, epoch choice/schedule and isolation logs are also checked.

## Implementation validation

Synthetic tests are in `scripts/round7_stepD_test.py`; they do not use source images or pretrained networks. Real W1/W2 GPU runs are left to the notebook cells. No commit or push is performed.

Completed implementation checks:

- 11 Step D synthetic tests passed, including native-size matching order, original reporting views, label-independent deterministic qualities, isolation rejection, W1 head resume, exact ResNet checkpoint resume, partial result-publication recovery, independent metric recomputation and frozen criterion boundaries/every-seed logic.
- All 14 existing baseline/seed regression tests passed. Their temporary files were kept outside existing Step C output/artifact folders.
- W0 reuse matched all 36 original/raw/global Step C metric cells across three seeds, using 4,600 saved predictions per seed and no new model/image access.
- The historical shared-results verifier passed; all 10,606 snapshotted existing files remained byte-identical.
- Protocol and implementation initialization passed without source-image access or real training. Summary generation correctly refused missing W1/W2 runs before creating any summary output.

These checks validate the implementation, not the experimental hypothesis. PASS/FAIL scientific results will only be available after the six GPU runs finish.
