# Reproduce or inspect the experiments

Reading the committed CSVs, plots, and interpretations requires no local setup.
Use a separate checkout for full reruns: the scripts write into their existing result folders.

## Check published results without a GPU

Run from the repository root in PowerShell with Python 3.12:

```powershell
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install numpy==1.26.4 scipy==1.17.1 scikit-learn==1.8.0
./.venv/Scripts/python.exe scripts/verify_shared_results.py
```

This read-only check validates provenance for Rounds 2–6.
It also recomputes 1,440 Round 6 point-metric cells and 288 macro AUROC values from the published predictions.
It does not verify images, refit models, reselect thresholds, or recompute bootstrap intervals.

## Set up a full rerun

The recorded environment used Windows, Python 3.12.5, and an NVIDIA RTX 4050 Laptop GPU with 6 GB memory.
The pinned requirements include PyTorch 2.2.0 with CUDA 12.1.
Use a compatible NVIDIA driver and keep feature-extraction batches at 32 for comparison with the recorded run.

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
./.venv/Scripts/python.exe -u scripts/restore_data.py
./.venv/Scripts/python.exe scripts/restore_data.py --check-only
```

The restoration script downloads missing pinned shards and recreates only the 1,600 manifest images.
It verifies source metadata and PNG hashes, preserves the saved partitions, and leaves result files unchanged.
Existing images with mismatched hashes cause an error instead of being overwritten.
The four source shards total about 1.90 GB; the pilot later downloads approximately 605 MB of CLIP weights.
Decoded images and feature caches require additional disk space.

The original `prepare_data.py` creates the pilot sample from scratch and rewrites its manifest and preparation metadata.
For this published experiment snapshot, use `restore_data.py` to recover the existing sample.
Round 2 alone cannot restore missing image files when its manifest already exists.

## Run in dependency order

Use the following commands in a separate checkout after restoring the images.
Each verifier follows its runner; later rounds consume earlier assignments and feature caches.
Stop at the first nonzero exit code and inspect the error before continuing.

```powershell
./.venv/Scripts/python.exe -u scripts/run_pilot.py --batch-size 32
./.venv/Scripts/python.exe scripts/summarize_results.py

./.venv/Scripts/python.exe -u scripts/run_round2.py --batch-size 32
./.venv/Scripts/python.exe scripts/verify_round2.py

./.venv/Scripts/python.exe -u scripts/run_round3.py --batch-size 32
./.venv/Scripts/python.exe scripts/verify_round3.py

./.venv/Scripts/python.exe -u scripts/run_round4.py
./.venv/Scripts/python.exe scripts/verify_round4.py
./.venv/Scripts/python.exe scripts/diagnose_round4.py

./.venv/Scripts/python.exe -u scripts/run_round5.py
./.venv/Scripts/python.exe scripts/verify_round5.py
./.venv/Scripts/python.exe scripts/summarize_round5.py

./.venv/Scripts/python.exe -u scripts/run_round6.py
./.venv/Scripts/python.exe scripts/verify_round6.py
./.venv/Scripts/python.exe scripts/summarize_round6.py
```

The saved `interpretation_zh.txt` files are written analyses of the published run; runners do not regenerate them.
Review those interpretations before reusing them with different hardware or changed experiments.

## Files that stay local

| Directory | Contents |
| --- | --- |
| `data/` | Pinned parquet shards and restored images |
| `.cache/` | Frozen CLIP configuration and weights |
| `artifacts/` | Feature arrays and fitted joblib experts |
| `.venv/` | Python environment |

These directories are ignored by Git.
Predictions, metrics, gate JSON parameters, protocols, manifests, and plots are committed under `results/`.
Load joblib files only when you trust their source.

## Reproducibility boundaries

Source hashes and cache identities bind experiments to their implementations.
The repository preserves script and result bytes through `.gitattributes`, including existing line endings.
Avoid reformatting the historical experiment scripts when checking the published snapshot.
Use a new experiment version and output directory when changing its protocol.

Fresh feature caches include new timing metadata, so their file hashes need not match the original caches.
Runners write provenance for those regenerated files; verify each round immediately after running it.
Do not combine published downstream metadata with independently regenerated upstream caches.

Different hardware, floating-point precision, library versions, or batch shapes can change predictions.
Some verifiers enforce strict cross-round control equivalence and will report those differences.
The lightweight check verifies the published numbers without making a cross-hardware reproducibility claim.

For the recorded full checks, see [Round 6 verification](../results/round6/verification.json) and its stated scope.
