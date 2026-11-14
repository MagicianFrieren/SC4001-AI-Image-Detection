# Reproducing the results

Use Python 3.12. The saved-analysis commands run on a CPU and use the included predictions and result tables.

## From this repository

Install the analysis dependencies, then rebuild the tables or the transfer figure from the repository root:

```powershell
python -m pip install -r reports/final_20261114/requirements_analysis.txt
python scripts/report_evidence_20261111/build_evidence.py --root output/analysis/report_evidence_20261111/sources --output ../sc4001-table-rebuild
python reports/final_20261114/build_compact_figure.py --source output/analysis/report_evidence_20261111/figures/figure2_transfer_data.csv --output-dir ../sc4001-figure-rebuild
```

Choose fresh output directories. The table command reads the preserved evidence snapshots; the figure command reads the saved transfer summary.
Both have been checked against the report's archived outputs.

## From the complete submission

Download [SC4001_Submission_Final_20261114_v2.zip](https://github.com/MagicianFrieren/SC4001-AI-Image-Detection/releases/download/final-20261114-v2/SC4001_Submission_Final_20261114_v2.zip) and extract it into a new directory.
It includes the report, source snapshot, saved inputs and analysis runner.
Run the following commands from the extracted root:

```powershell
python VERIFY_PACKAGE.py
python -m pip install -r reproduction/requirements_analysis.txt
python source_tree/scripts/submission_revision_20261114/reproduce.py --package . list
python source_tree/scripts/submission_revision_20261114/reproduce.py --package . run report-tables --workspace ../rebuild-report-tables
python source_tree/scripts/submission_revision_20261114/reproduce.py --package . run report-figures --workspace ../rebuild-report-figures
python source_tree/scripts/submission_revision_20261114/reproduce.py --package . run mechanism-predictions --workspace ../rebuild-mechanism-predictions
python source_tree/scripts/submission_revision_20261114/reproduce.py --package . run mechanism-bootstrap --workspace ../rebuild-mechanism-bootstrap
```

The runner expects the ZIP's `source_tree/`, `evidence/` and `reproduction/` directories.
GitHub's generated source-code archives have the ordinary repository layout.
All four commands passed, with 30 numerical or image comparisons matching the archived analyses.
The [execution record](../reports/final_20261114/verification/saved_analysis_reproduction.json) lists the checks.

## Historical training

The November studies use development data. The earlier final evaluations retain their recorded models and thresholds.
Training and feature extraction require additional images, CLIP weights, feature caches and runtime bundles.
Exact input references, hashes and commands are in the ZIP's `reproduction/REPRODUCING_en.txt` and `review/input_artifact_manifest.json`.

Historical verifiers compare files with hashes recorded at the time of each experiment.
Use the source revisions and restoration instructions in the submission package when replaying those checks.
The current `main` branch contains the reader-facing documentation and analysis entry points; its documentation can change after the experiments.

## Report and file checks

The [final PDF](../reports/final_20261114/SC4001_Report_Final_20261114.pdf) has 10 body pages and 14 pages overall.
The [Word report](../reports/final_20261114/SC4001_Report_Final_20261114.docx) is editable.
The `final-20261114-v2` [release](https://github.com/MagicianFrieren/SC4001-AI-Image-Detection/releases/tag/final-20261114-v2) preserves the submitted files and source layout.

<details>
<summary>Checksums and source manifests</summary>

The submission ZIP has SHA-256:

```text
85138853a7f018b136e242ecbeccd06a234178c375f777dccead7428b983d35c
```

[Download checksums](../reports/final_20261114/SHA256SUMS.txt) · [Package file manifest](../reports/final_20261114/verification/release_package_manifest.json) · [Source manifest](../reports/final_20261114/verification/source_tree_manifest.json)

The source snapshot starts from main commit `42059e3d0eabcb6393efcb9dd3b879092963af96` and includes analysis-c material from `e498d0ed997e679a5db22b4b3ce05db34e6b31a0`.
The [release manifest](../reports/final_20261114/RELEASE_MANIFEST.json) lists the additional November files.
These manifests describe the frozen release package.

</details>
