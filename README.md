# SC4001 AI Image Detection

**Final submission — 14 November 2026.**

We study frozen-CLIP global and patch-shuffled local experts for AI-generated image detection across generators and image degradations. The final report covers the Round 7 and Round 8 evaluations, seven improvement attempts, and November development diagnostics.

In this setup, format matching improves ranking under JPEG compression but introduces clean-image costs. Quality-aware gating does not consistently outperform simple averaging.

## Final report and submission

- [Read the final PDF](reports/final_20261114/SC4001_Report_Final_20261114.pdf) — 10 body pages; 14 pages including the cover, references and appendices.
- [Download the editable Word report](reports/final_20261114/SC4001_Report_Final_20261114.docx).
- [Download the complete submission ZIP](https://github.com/MagicianFrieren/SC4001-AI-Image-Detection/releases/download/final-20261114/SC4001_Submission_Final_20261114.zip) — report, code, saved inputs, evidence and reproduction instructions.
- [Release notes](https://github.com/MagicianFrieren/SC4001-AI-Image-Detection/releases/tag/final-20261114), [checksums](reports/final_20261114/SHA256SUMS.txt) and [reproduction guide](docs/FINAL_SUBMISSION.md).

The Release ZIP is the complete submission package. GitHub's automatically generated source-code ZIP contains the repository checkout and has a different layout.

## Findings

- On the Round 8 main-generator comparison, format matching raises average-fusion JPEG70 AUROC from **0.7556 to 0.7749**, while clean AUROC falls from **0.8711 to 0.8268**.
- Clean-image ranking loss accompanies a threshold tradeoff: fewer missed fakes and more false alarms at the original frozen thresholds.
- The seven modifications address checkpoint selection, gate inputs, learning objectives, condition routing, training-generator coverage, recompression cues and paired-view training. Each has an explicit comparator in the report.
- Development diagnostics suggest that generator-dependent expert preferences and quality signals contribute to JPEG gating failures. Resizing retains a positive result and follows a different pattern.

The report separates November exploratory development analyses from the earlier frozen final evaluation. It includes the costs and benefits of each modification alongside its experimental evidence.

## Reproduce the saved analyses

Download and extract the complete submission ZIP, then run these commands **from its extracted root**:

```powershell
python VERIFY_PACKAGE.py
python -m pip install -r reproduction/requirements_analysis.txt
python source_tree/scripts/submission_revision_20261114/reproduce.py --package . list
python source_tree/scripts/submission_revision_20261114/reproduce.py --package . run report-tables --workspace ../rebuild-report-tables
```

The four supported commands are `report-tables`, `report-figures`, `mechanism-predictions` and `mechanism-bootstrap`. All four were run from the supplied package; 30 archived numerical or image comparisons matched exactly. See the [verification record](reports/final_20261114/verification/saved_analysis_reproduction.json).

The [final reproduction guide](docs/FINAL_SUBMISSION.md) also gives commands for a normal Git checkout. Historical training requires the separately indexed images, weights and feature caches; the verified submission workflow reconstructs saved-result analyses.

## Repository map

| Path | Contents |
| --- | --- |
| [reports/final_20261114](reports/final_20261114/) | Final PDF and Word report, manuscript figures, checksums and verification records |
| [output/analysis/report_evidence_20261111](output/analysis/report_evidence_20261111/) | Seven-modification comparison, claim-to-evidence mapping, mechanism analyses and source snapshots |
| [scripts/report_evidence_20261111](scripts/report_evidence_20261111/) | Table, figure and bootstrap analysis scripts |
| [scripts/submission_revision_20261114](scripts/submission_revision_20261114/) | Submission source assembler and saved-analysis runner |
| [docs/ROUND7_FINAL.md](docs/ROUND7_FINAL.md) | Round 7 protocol and final evaluation |
| [docs/ROUND8_FINAL.md](docs/ROUND8_FINAL.md) | Round 8 fresh-image evaluation |
| [results/round8_final](results/round8_final/) | Frozen final metrics, contrasts, manifests and verification records |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Earlier pilot and Rounds 2–6 experiment history |

The tracked November `tmp/gate_diagnosis/`, `tmp/gate_failure_mechanism_20261108/`, `tmp/multigen_gate_20261108/`, `tmp/paired_view_global_20261108/` and `tmp/recompression_reliability_20261108/` paths preserve the original experiment source layout. They are archived project-relative paths. Their scripts, configurations and dependencies are also included in the Release ZIP.

## Scope and attribution

The November diagnostics use repeatedly inspected development data, with shared image pools across seeds. Their loss decomposition is descriptive. The final report discusses these limits together with the observed JPEG losses and resize gains.

Our experts use frozen OpenAI CLIP ViT-B/32 features and separate logistic regression heads. The local expert averages four PatchShuffle views. [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) and [SFLD](https://arxiv.org/abs/2502.17105) inform the design; this implementation uses its own backbone and experimental protocol.

Data come from a pinned subset of [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage), a third-party subset of [GenImage](https://github.com/GenImage-Dataset/GenImage). The dataset card declares CC BY-NC-SA 4.0; underlying image rights still apply. Codex assisted with scripts, documentation and analysis under the project owner's direction.
