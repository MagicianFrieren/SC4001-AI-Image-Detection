# Round 7 Step C: repeat seeds 2 and 3

Seed 1 is complete and verified. This extension repeats its protocol using assignment seeds **20261102** and **20261103**, with separate result and artifact folders. No real training was run while implementing this extension. Run the cells below to produce the remaining measurements; the three-seed summary deliberately refuses incomplete runs.

## What stays fixed

The model, preprocessing, feature extraction, losses, hyperparameters, tuning conditions, calibration, threshold selection and bootstrap code remain byte-identical to seed 1. Both original/common128 profiles and clean/jpeg70/jpeg50/resize05 reporting conditions remain unchanged. JPEG50 is still report-only; tuning still uses clean/jpeg70/resize05. Only the assignment/run seed and output paths change. ResNet initialization and epoch shuffling follow the selected run seed. CLIP logistic-head, PatchShuffle, gate initialization and bootstrap seeds retain their original fixed values.

Each assignment uses the same fitting pool: 750+750 expert-fit, 250+250 gate-fit, 125+125 calibration, and 125+125 selection. The 900-image ADM/BigGAN/Real development set stays fixed. Existing manifests and assignments are unchanged. Step C reads only the fitting/development records and ID-only isolation metadata; it does not open the reserved-test manifest or decode, extract features from, or score reserved-test images.

The original runner/verifier were archived byte-for-byte under `artifacts/round7/seed1_frozen_entrypoints/` before adding `--seed`. `results/round7/seed_repetition_provenance.json` records all 3,488 seed-1 result/artifact files, checked against 3,476 saved hashes. The seed-1 protocol is not rewritten: verification checks those original entry points through their archives and checks all unchanged model modules directly. New protocols additionally bind the current routing code. Completed seed-1 outputs are read-only, including verification reports: verify seed 1 without `--report`.

Run in the **same Anaconda GPU environment as seed 1**, including repository-pinned Pillow 12.2.0. The runner rejects package-version drift before starting. No dependencies were added or changed.

## Notebook setup

Run this setup once in the notebook. Output and errors stream into the cell. Interrupting a cell stops its child process; rerun with `--resume` afterwards.

```python
from pathlib import Path
import os, subprocess, sys

project = Path(r"C:\sc4001\prjJ\SC4001-AI-Image-Detection")
env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def run_script(script, *args):
    command = [sys.executable, "-u", str(project / "scripts" / script), *args]
    process = subprocess.Popen(command, cwd=project, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, encoding="utf-8", errors="replace")
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
```

### Seed 2

```python
run_script("round7_run_baselines.py", "--seed", "20261102", "--stage", "all", "--resume")
run_script("round7_verify_baselines.py", "--seed", "20261102", "--bootstrap")
```

Outputs: `results/round7/baselines_seed2/` and `artifacts/round7/baselines_seed2/`.

### Seed 3

Run after seed 2, not concurrently on the 6 GB GPU.

```python
run_script("round7_run_baselines.py", "--seed", "20261103", "--stage", "all", "--resume")
run_script("round7_verify_baselines.py", "--seed", "20261103", "--bootstrap")
```

Outputs: `results/round7/baselines_seed3/` and `artifacts/round7/baselines_seed3/`.

Completed stages are hash-checked and skipped. ResNet restores the latest completed epoch, optimizer, scheduler and RNG state, as in seed 1. Interrupted feature extraction reuses validated feature caches. Gate fitting restarts an unfinished fit. The inherited runner refuses to overwrite partially published stage result files; if interrupted during final publication, preserve those files and inspect the failure before proceeding. No resume option silently replaces an existing artifact. Verifier cells are read-only and can be rerun freely.

### Three-seed summary

```python
run_script("round7_summarize_seeds.py", "--resume")
run_script("round7_summarize_seeds.py", "--verify-only")

from IPython.display import display, Markdown
display(Markdown((project / "results/round7/seeds_summary/SUMMARY.md").read_text(encoding="utf-8")))
```

Summary publication can resume: existing files must match exactly, and only missing files are added. `--verify-only` recomputes the tables in memory and compares them without writing. All three completed runs and their hashes are required; the script never silently summarizes fewer seeds.

## Summary interpretation

Every CSV contains each of the three run values, their mean, sample SD (`ddof=1`), and a formatted mean ± SD. A gate's three initialization-specific values are averaged **within each run first**, so the sample size remains three, not nine. AUROCs are averaged; predictions are not pooled across initializations or runs.

| Output under `results/round7/seeds_summary/` | Coverage |
|---|---|
| `per_condition_auroc.csv` | 336 cells: seven models, two profiles, two calibration modes, three domains, four conditions |
| `macro_auroc.csv` | 84 cells: equal-weight mean of all four conditions, separately for SD15 selection, ADM development and BigGAN development |
| `paired_differences.csv` | 510 contrasts, per condition and four-condition macro: average − global; score_gate − average; quality_gate − average; local − global; resnet50 − global; common128 − original |
| `gate_global_weight_means.csv` | 96 cells: both gates, profiles, calibrations, domains and conditions |
| `SUMMARY.md` | Readable macro and per-condition tables |
| `provenance.json`, `complete.json` | Input/code hashes, aggregation rules, seed-1 preservation and summary hashes |

Paired differences are calculated **within each seed** before computing their mean and SD. Positive differences favour the left model/profile. Gate means cover the same real+fake comparison population as the per-run saved gate summaries; they are not test-domain estimates.

SD describes variation across the three runs; it is not a confidence interval or a significance test. Fitting partitions overlap across seeds and the development images are shared. SD15 selection results remain optimistic because that partition also selects settings/epochs. ADM and BigGAN remain previously inspected development domains. No reserved-test evidence is introduced.

## Verification

New synthetic tests: `scripts/round7_test_seeds.py`. They check partition routing, settings preservation, no reserved-manifest access, complete model/initialization coverage, within-run gate aggregation, paired SD, real seed-1 table compatibility and exclusive/resumable summary publication. Existing baseline synthetic tests also exercise checkpoint resume, OOM rollback and test isolation using temporary test directories outside seed 1.

Implementation checks completed on 25 October 2026:

- All six new seed/summary tests and all eight existing baseline tests passed.
- The updated seed-1 verifier passed: 202,400 saved predictions, 528 metric cells, 84 macro cells and six isolation logs; zero test source IDs touched. This rerun checked point metrics and existing CI bounds; bootstrap CIs were not replayed in this implementation check. The notebook verifier commands above do replay them.
- `verify_shared_results.py` passed its historical provenance and Round 6 metric checks.
- Final byte check: all 3,488 seed-1 result/artifact files unchanged; all 3,476 previously saved hashes matched.
- Seed-2/3 protocol initialization and repeated initialization passed without changing any existing protocol bytes. No images or model weights were loaded for initialization.
- The summary correctly refused incomplete seed 2 before creating the summary folder. Actual three-seed tables await the two GPU runs.

No commit or push is performed by this extension.
