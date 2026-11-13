# Final submission

The final report and submission package were frozen on **14 November 2026**. The report has 10 body pages and 14 pages in total. Its final layout, editable equations and report contents have been checked.

## Download

- [Final report PDF](../reports/final_20261114/SC4001_Report_Final_20261114.pdf)
- [Editable Word report](../reports/final_20261114/SC4001_Report_Final_20261114.docx)
- [Complete submission ZIP](https://github.com/MagicianFrieren/SC4001-AI-Image-Detection/releases/download/final-20261114/SC4001_Submission_Final_20261114.zip)
- [Release page](https://github.com/MagicianFrieren/SC4001-AI-Image-Detection/releases/tag/final-20261114)
- [Checksums](../reports/final_20261114/SHA256SUMS.txt)

The complete ZIP contains the final report, 765 source files, November analysis inputs and scripts, evidence, reproduction commands and file manifests. Its 971 payload files were checked against their SHA-256 values, and the ZIP passed a CRC check.

The complete ZIP has SHA-256:

```text
9ef16d6548a306a089a7e43b62d030e30e0c4e3c8a6b0d977e22241b812c5335
```

Use the named submission ZIP for the workflow below. GitHub's generated source-code archives contain the Git checkout layout instead.

## Run the verified package workflow

Extract the ZIP into a new directory and run these commands from that directory. The recorded environment uses Python 3.12; exact analysis dependencies are in `reproduction/requirements_analysis.txt`.

```powershell
python VERIFY_PACKAGE.py
python -m pip install -r reproduction/requirements_analysis.txt
python source_tree/scripts/submission_revision_20261114/reproduce.py --package . list
python source_tree/scripts/submission_revision_20261114/reproduce.py --package . run report-tables --workspace ../rebuild-report-tables
python source_tree/scripts/submission_revision_20261114/reproduce.py --package . run report-figures --workspace ../rebuild-report-figures
python source_tree/scripts/submission_revision_20261114/reproduce.py --package . run mechanism-predictions --workspace ../rebuild-mechanism-predictions
python source_tree/scripts/submission_revision_20261114/reproduce.py --package . run mechanism-bootstrap --workspace ../rebuild-mechanism-bootstrap
```

Use fresh workspace paths for each run. The wrapper preserves input bytes and records the selected inputs, environment, command and outputs. All four commands were executed successfully from the package, with 30 archived numerical or image comparisons matching exactly.

The package's `reproduction/REPRODUCING_en.txt` and `REPRODUCING_zh.txt` explain the full workflow. Its top-level `START_HERE` files identify the final report. Older evidence notes remain preserved as historical records.

## Rebuild tables and a figure from a Git checkout

From the repository root, install the analysis dependencies and point the builders at the committed evidence snapshots:

```powershell
python -m pip install -r reports/final_20261114/requirements_analysis.txt
python scripts/report_evidence_20261111/build_evidence.py --root output/analysis/report_evidence_20261111/sources --output ../sc4001-table-rebuild
python reports/final_20261114/build_compact_figure.py --source output/analysis/report_evidence_20261111/figures/figure2_transfer_data.csv --output-dir ../sc4001-figure-rebuild
```

The explicit `--root` selects the preserved source snapshots. The explicit `--source` handles the figure script's location in this repository. The original figure notes alongside the script describe its layout inside the Release ZIP.

`reproduce.py --package .` expects the extracted package's `source_tree/`, `evidence/` and `reproduction/` directories. Use the Release ZIP for that command.

## Evidence and provenance

- [Improvement comparisons](../output/analysis/report_evidence_20261111/tables/improvement_comparisons.csv) identify each modification's comparator and outcome.
- [Claim-to-evidence mapping](../output/analysis/report_evidence_20261111/claim_evidence.csv) links report claims to experimental artifacts.
- [Saved-analysis verification](../reports/final_20261114/verification/saved_analysis_reproduction.json) records the four completed reconstruction runs.
- [Release manifest](../reports/final_20261114/RELEASE_MANIFEST.json) lists the 237 additional November and analysis-c files published with their original relative paths and hashes.
- [Source manifest](../reports/final_20261114/verification/source_tree_manifest.json) describes the immutable source snapshot inside the complete Release ZIP.

The submission source snapshot starts from main commit `42059e3d0eabcb6393efcb9dd3b879092963af96`, includes relevant analysis-c material from `e498d0ed997e679a5db22b4b3ce05db34e6b31a0`, and adds the actual November scripts and dependencies. The final Git update preserves existing tracked results and publishes the additional files without replacing historical state snapshots.

November diagnostics are exploratory development analyses. Earlier final-evaluation predictions and thresholds remain frozen. This publication adds no model fitting, threshold selection or final-test scoring.

Historical training and feature extraction require the indexed images, CLIP weights, feature caches and runtime bundles. The complete ZIP provides exact input references and commands; its verified reconstruction scope is the four saved-result analyses above. The course-platform upload is performed separately by the team.
