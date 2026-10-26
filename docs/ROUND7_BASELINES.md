# Round 7 Step C: seed 20261101, clean expert training

Step B is committed separately as `1b83d2a`. Step C is not committed and has not run on the real images. There are no measured Step C AUROCs yet. The synthetic tests are implementation checks, not experiment results.

## Frozen decisions

The frozen base protocol is extended by `results/round7/data_protocol_step_b.json` and `results/round7/baselines_seed1/protocol.json`; no earlier file is rewritten. The same seed-1 fitting assignments are used: expert-fit 750+750, gate-fit 250+250, calibration 125+125, selection 125+125 Real/SD15. Development contains 300 Real, 300 ADM and 300 BigGAN. Development cannot influence fitting, calibration, epochs, thresholds or fixed weights.

- CLIP ViT-B/32 stays frozen. Global and four-view 56px PatchShuffle local heads use the unchanged Round 6 standardized logistic-regression settings, clean expert-fit originals only. The four local view seeds, encoder revision, weight hash, normalization, FP16 CUDA encoder and FP32 embedding normalization are fixed.
- Global, local, average, selected fixed fusion, ordinary gate and quality gate are reported with raw and temperature-calibrated logits, both original/common128 profiles. Gate architecture, optimizer, early stopping and the three initialization seeds are inherited through the unchanged `gate_models.py`. The data partition seed is distinct from those gate initialization seeds.
- **JPEG50 stays report-only.** Gate fitting, calibration and all tuning use clean/JPEG70/resize05 exactly as Round 6. Selection and development predictions/metrics are reported under all four conditions. Fixed fusion uses selection SD15 macro AUROC across the three tuning conditions, alpha 0..1 by .1, ties nearest .5 then smaller. One pooled-selection Youden threshold per fitted model is frozen across all reporting conditions. Thus the inherited selection rule is preserved while the requested reporting coverage grows to four conditions.
- ResNet-50 uses torchvision `IMAGENET1K_V2` weights, all layers trainable, clean expert-fit only. Fixed AdamW lr=1e-4, weight decay=1e-4, batch=32, 10 epochs, cosine decay to zero, seed=20261101, FP32 and deterministic Torch algorithms. Best epoch maximizes mean selection AUROC over clean/JPEG70/resize05; ties retain the earliest epoch. Raw and positive-temperature ResNet results are both reported; its temperature uses calibration only, and its thresholds use selection only.
- Both architectures receive the same decoded RGB geometry/degradation: original centre-fit224, or centre-fit128 then resize224, bicubic; JPEG uses the inherited quality and subsampling=2; resize05 is 224->112->224. Encoder-specific channel normalization remains CLIP vs ImageNet. No new augmentation/harmonisation is introduced. The source JPEG/PNG and resolution-history confounds remain.
- ADM/BigGAN are previously inspected development domains. Their eventual reserved-test results must be identified as previously inspected domains. GLIDE/VQDM/Midjourney/Wukong have no model use before final evaluation. The earlier independent-pilot exposure disclosure remains unchanged.

Algorithm/settings inheritance does not imply bitwise equivalence across different Torch/scikit-learn versions. Each completed stage records its runtime package versions and device. The current Anaconda has Torch 2.5.1, torchvision 0.20.1 and scikit-learn 1.5.1, whereas the historical environment used Torch 2.2.0, torchvision 0.17.0 and scikit-learn 1.8.0. Within-Step-C comparisons use the same environment; keep it fixed throughout the run. Pillow **12.2.0 is mandatory** for matching original JPEG pixel hashes. No dependency file was changed; transformers was already a project dependency but is missing from this Anaconda environment.

## Test isolation

Step C reads `step_b_isolation.json`, an ID-only denylist plus an exact 3,400-original fitting/development allowlist. It never opens `reserved_test_manifest.jsonl`. Guards run before raw payload selection, cache/image access, decoding, feature use, fitting and scoring. Every stage saves an access log; forbidden IDs raise an assertion before image access. Data caches and all new weights live under `artifacts/round7/baselines_seed1/`.

Parquet pages can physically contain compressed neighbouring source rows. The reader selects allowed Arrow rows before converting payloads to Python/PIL; no forbidden image is materialized as a Python image, decoded, exported or scored. The baseline verifier reads saved predictions, hashes and ID-only audit logs, not images, model weights or the reserved-test manifest.

## Jupyter cells (RTX 4050, 6 GB)

Use your GPU-enabled Anaconda kernel. If transformers is missing, run this optional setup cell. These are existing repository dependency pins, not new dependencies. On this checkout the scripts automatically use the isolated Pillow 12.2 install; installing the pin into another device's environment is also supported.

```python
%pip install "transformers==4.39.3" "Pillow==12.2.0"
```

Set up the runner. Child-process output is streamed so progress and errors stay visible in Jupyter.

```python
from pathlib import Path
import os, subprocess, sys

project = Path(r"C:\sc4001\prjJ\SC4001-AI-Image-Detection")
env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def run_script(script, *args):
    command = [sys.executable, "-u", str(project / "scripts" / script), *args]
    with subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)

subprocess.run([sys.executable, "-c",
               "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
               "print(torch.cuda.get_device_name(0))"], cwd=project, env=env, check=True)
run_script("round7_run_baselines.py", "--stage", "init")
```

Run the real experiment (not run by Codex). This downloads the pinned CLIP and torchvision weights into the new Round 7 cache, prepares only permitted images, runs both profiles, and writes the bootstrap summary. Keep this cell running; it is a full experiment, not a two-epoch smoke run.

```python
run_script("round7_run_baselines.py", "--stage", "all", "--resume")
```

For separate sessions, run these instead of the previous cell:

```python
run_script("round7_run_baselines.py", "--stage", "prepare", "--resume")
run_script("round7_run_baselines.py", "--stage", "clip", "--resume")
run_script("round7_run_baselines.py", "--stage", "resnet", "--resume")
run_script("round7_run_baselines.py", "--stage", "summarize", "--resume")
```

After interruption, rerun the same cell with `--resume`. Completed stages are checked and skipped; ResNet resumes its latest completed epoch with optimizer, scheduler and RNG state restored. Every epoch has its own immutable checkpoint, including initialization. Only CUDA OOM can halve batch size; the entire failed epoch is rolled back before retry. No other exception silently changes the batch. Exact resume assumes the same code, software and device; changing the protocol/code stops reuse. Completed outputs are never overwritten. A crash during final file publication can leave an incomplete stage; the runner refuses to overwrite those published partial files rather than silently replace them.

Verify and display the measured table after completion:

```python
from datetime import datetime
from IPython.display import display, Markdown

report = f"verification_{datetime.now():%Y%m%d_%H%M%S}.json"
run_script("round7_verify_baselines.py", "--bootstrap", "--report", report)
display(Markdown((project / "results/round7/baselines_seed1/SUMMARY.md")
                 .read_text(encoding="utf-8")))
```

## Saved outputs and interpretation

All experiment results live in `results/round7/baselines_seed1/`:

- Per-image prediction CSVs for each architecture/profile, with calibration, initialization, source ID, partition, condition, label, logit, probability, threshold and gate weight.
- `metrics.csv`: AUROC, AP, balanced accuracy, real FPR and 95% class-stratified percentile bootstrap CIs for each fitted model/profile/calibration/condition/generator.
- `macro_summary.csv` and `SUMMARY.md`: mean AUROC across all four conditions, separately for SD15 selection, ADM development and BigGAN development. Gates average metrics across three initializations, not predictions.
- Gate parameters, histories, temperatures, fixed-weight searches, epoch histories and selected thresholds; gate-weight summaries per condition/domain/initialization.
- Resource JSONs: shared encoder feature extraction, per-condition head/inference costs, model fitting and peak allocated CUDA memory. Frozen features and heads are reused across fusions/calibrations, so their shared costs must not be summed repeatedly. Gates run on CPU as in Round 6. ResNet reports all epoch times, per-condition inference, checkpoint hashes and any OOM batch changes.
- Protocol, stage seals, exact software/device records and references to immutable access logs/caches/checkpoints under `artifacts/round7/baselines_seed1/`.

Bootstrap uses 2,000 class-stratified draws, seed 20261022, paired across models, gate initializations and conditions by original-image positions; development real positions are shared across ADM/BigGAN comparisons. CIs condition on fitted models and selected settings. Selection metrics are optimistic after threshold/epoch selection and are not unbiased in-domain test estimates. The verifier independently recomputes point metrics, Youden thresholds, macro means and weight summaries; `--bootstrap` also replays all paired intervals. No held-out final claim is made here.

Pending measured table (both raw and temperature results will be produced):

| Model | Original: SD15 / ADM / BigGAN macro AUROC | Common128: SD15 / ADM / BigGAN macro AUROC |
|---|---|---|
| Global | Pending GPU run | Pending GPU run |
| Local | Pending GPU run | Pending GPU run |
| Average | Pending GPU run | Pending GPU run |
| Fixed | Pending GPU run | Pending GPU run |
| Ordinary gate (3 initializations) | Pending GPU run | Pending GPU run |
| Quality gate (3 initializations) | Pending GPU run | Pending GPU run |
| ResNet-50 | Pending GPU run | Pending GPU run |

Step C is intentionally uncommitted until results are reviewed. Nothing is pushed.
