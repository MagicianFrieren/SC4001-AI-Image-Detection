# Round 8 final evaluation — user-run notebook

Open `docs/ROUND8_FINAL.ipynb` in Jupyter and run its cells in order. The notebook has not been executed by Codex. No model loading, development reproduction, fresh-test preparation or final inference was performed during implementation. The ZIP's outer SHA-256 was inspected and matches `c023b09ca182df1b7a97e7d2b68c6172d594ada2f6e10e582e9bf926f8605a85`; complete archive/member/receipt verification remains a notebook step.

Keep your existing Anaconda/Jupyter kernel. The notebook's environment setup cell creates a separate Conda environment at `artifacts/round8_final/environments/a_recorded_runtime` when you run it, with Python and packages matching A's saved protocol. The original worker scripts run in that environment and require CUDA and Pillow 12.2.0. No packages are installed by Codex or into your current Anaconda environment. Outputs stream into the notebook. Run from this repository only; no files from another project are used.

There is one active notebook and one set of original scripts. Active results stay in `results/round8_final/`; active caches stay in `artifacts/round8_final/`. Reopen the notebook from disk and restart its kernel after this update.

The original development reproduction failed. Before rebuilding it, the notebook's preservation cell runs `round8_final_common.py --archive-failed`. This moves the failed report and associated cache files into `archive/failed_development/` under their respective results/artifacts roots, keeping their bytes and recording original paths, archived paths and SHA-256 hashes. It does not overwrite failed evidence, relax criteria or clear a final evaluation. An interrupted move resumes from its manifest; a completed archive is verified without moving active files again. Archival has not been run by Codex.

## Order of operations

1. **Setup:** parse the seven original Python files without importing them, create/check the separate recorded Conda environment, and preserve failed development evidence before reusing the original paths.
2. **Restore handoff:** snapshot historical research-file hashes, verify the model ZIP against both external checksums and its internal checksum list, and check all twelve members against completion receipts. Restore only missing files at the original `artifacts/round8_a/seedN/<arm>/` paths. Existing identical files are accepted; differing files stop the process. The internal checksum list is checked in memory, not written over a root file. Verify the predictions archive and all six CSV checksums against its metadata and receipts, without overwriting prior prediction files.
3. **Development reproduction:** load the heads, extract explicit scaler means/scales and logistic coefficients/intercepts, and write portable JSON parameters only in `artifacts/round8_final/portable/`. If a trusted sklearn pickle cannot load normally, a state-only joblib unpickler restores sklearn parameter containers without invoking estimator methods. All final CLIP heads use the portable arithmetic. No fitting is performed.
4. **Recreate original development views** for all 900 development originals and all four conditions, using the pinned CLIP weights. Compare all six A runs, every model and all gate initializations with the archived predictions; also compare global/local branch logits with the saved bundles. Reused development images stay in a separate allowlisted cache. This is required because A's handoff does not include feature caches.
5. **Build and verify fresh test hashes.** Only after reproduction passes, exclude the Round 2 pool plus Round 7 fitting/development/reserved pools by source ID, pixel hash and dHash distance <=3. Greedily deduplicate all remaining eligible generators together before deterministic sampling. Target 500 each of six fake generators plus 500 shared real images. A shortfall stops the script; it never silently shrinks targets or reuses excluded images. There is no SD15 in-domain test.
6. **Freeze before evaluation.** Pin code, configuration, notebook source, manifest, original model files, portable parameters, gates/settings, selected ResNet checkpoints, reproduction evidence and pinned CLIP weights. Nothing is retrained, recalibrated or reselected.
7. **Final evaluation:** original profile, raw scores, clean/JPEG70/JPEG50/resize05, original evaluation images, all three run seeds. A separate exact test allowlist is enforced. Earlier isolation guards stay unchanged. Immutable feature batches, raw ResNet logits, prediction CSVs and receipts support incomplete-run recovery.
8. **Independent verifier and summary:** recheck frozen hashes, historical preservation, seeded sampling, exclusions, exact prediction coverage and saved thresholds. Recompute AUROC, AP, balanced accuracy and real-FPR directly from predictions, then reproduce all summary/paired tables. No inference is performed by the verifier.

## Reproduction criteria fixed in advance

`results/round8_final/handoff_reproduction.json` records every numerical difference and loading warning. Reproduction requires:

- Maximum absolute logit difference <= 0.001, for both branch logits and fused model logits.
- Maximum absolute fake-probability difference <= 0.00025.
- Maximum absolute global-weight difference <= 0.0001.
- **Zero** binary-decision disagreements at each saved threshold.
- Where native sklearn prediction is available, portable/native head logit differences <= 1e-10 on actual regenerated development features.

These allow small runtime numerical drift, not model changes; bit identity is reported if observed, never assumed. A failure blocks test preparation and freeze. Do not relax tolerances or alter thresholds to force a pass. Inspect the failure in the development-only stage and resolve the runtime/artifact issue before proceeding. Existing evidence is immutable.

The portable scaler now follows sklearn 1.8's explicit cast of mean/scale to the feature dtype before each arithmetic operation. This fixes a mismatch with the earlier implementation, which used sklearn 1.5-style float64 parameters with float32 inputs. The native-head tolerance remains 1e-10. For the specific interrupted pre-test run, `head_arithmetic_correction.json` records both changed source hashes and the original feature-cache identity. Image/encoder code, inputs, runtime and criteria must match the saved plan; completed feature batches are reused with their existing hash checks. Rerun the same reproduction cell; no environment reinstall or cache deletion is required. This correction has received static source review only; its numerical result still requires the user's run.

## Declared models

- Round 7: global/W0, W1, local, average, fixed, score gate, quality gate and ResNet-50; original/raw, three seeds.
- Round 8A: unmatched and matched global, local, average, fixed, score gate and quality gate; three seeds.
- Each gate keeps all three saved initializations: **96 individual model instances**, representing 20 model/arm configurations across three run seeds.
- W2, common128 and temperature calibration are not in this final experiment.

Expected complete coverage is 3,500 unique originals, 1,344,000 prediction rows and 2,304 initialization-level generator/condition metric cells. These are targets, not measured results at implementation time.

## Reporting

The main macro equally weights GLIDE, Midjourney, VQDM and Wukong. ADM and BigGAN are also reported individually. The same 500 real originals are predicted once per model/condition and reused for each generator comparison. No pooled statistic duplicates the real pool.

Gate-initialization metrics are averaged within a run seed, then the mean and sample SD (ddof=1) are calculated over the three seeds. Each seed remains visible. Three-seed SD is descriptive, not a population confidence interval.

Within-seed contrasts include W1−W0, matched−unmatched local, matched−unmatched average, and matched quality gate−matched average. All four metrics are reported per generator/condition and for the main macro. Positive FPR differences are worse.

`cross_test_differences.csv` compares each **same frozen Round 7 model** on the new Round 8 test versus its saved Round 7 test results. The images differ: this is paired by fitted model/run seed, not by image, and it is not evidence of an intervention effect. Round 8A models have no earlier Round 7 test measurement; similarly named refitted models are not substituted.

## Freshness and limitations

“Fresh” is verified relative to the four repository exclusion pools and their pixel/dHash neighbours. The separate independent teammate pilot is outside this repository and is not accessed; this implementation cannot certify non-membership in that external pilot. Generator domains have already been inspected. Do not describe them as universally blind or previously unseen domains.

Sampling reuses the complete 28,000-row Pillow 12.2 hash inventory after checking its content/provenance hashes and historical hash agreement. It performs no new decoding. Final inference checks each native byte hash and decoded RGB hash before the inherited image transform. Source shards must already exist locally; no downloads are started automatically.

Real JPEG/fake PNG history, generator-specific native sizes and content differences remain confounds. Matched training does not erase real-image double compression. No test-time format matching is introduced.

## Resume and output locations

The notebook skips completed stages and verifies existing files. Final inference refuses a second run after `results/round8_final/complete.json` exists, even with `--resume`. Completed batches and prediction files are reused during incomplete-run recovery; work interrupted before atomic publication can repeat. An OS lock rejects concurrent executions. Do not delete completion files to bypass these controls.

Final files are written under `results/round8_final/`:

- `handoff_integrity.json`, `handoff_reproduction.json`: restoration and numerical reproduction evidence.
- `test_manifest.jsonl`, `data_audit.json`: hash-only manifest, exclusion counts and source provenance.
- `freeze.json`: pre-inference model/code/data lock.
- `evaluation/predictions/`, `evaluation/receipts/`: per-image scores and provenance.
- `metrics_initializations.csv`, `metrics_each_seed.csv`, `main_macro_each_seed.csv`, `metrics_mean_sd.csv`.
- `paired_differences.csv`, `cross_test_differences.csv`, `SUMMARY.md`.
- `complete.json`, `verification.json`: completion and independent verification.

All caches and portable model exports live in `artifacts/round8_final/`. Historical files are read-only, except that missing checksum-verified handoff files are restored at the original paths explicitly requested by the protocol owner. No commits or pushes are made.

## Implementation status

New scripts and notebook received static source review only. Python scripts, reproduction checks, tests and final evaluation were **not run** by Codex. The first notebook cell performs a syntax parse in the user's environment; subsequent cells perform the actual checks. Numerical differences and fresh-test counts remain pending until those cells succeed.
