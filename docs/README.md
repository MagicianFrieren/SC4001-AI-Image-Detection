# Experiment notes

The [final report](../reports/final_20261114/SC4001_Report_Final_20261114.pdf) brings together the methods, results and failure analysis.
Use the [reproduction guide](FINAL_SUBMISSION.md) for the current analysis commands.
The notes below record individual experiments and their original protocols.

## Study sequence

| Stage | Question | Notes and results |
| --- | --- | --- |
| Pilot and Rounds 2–6 | Do local features, gating, calibration or augmentation help transfer? | [Experiment guide](EXPERIMENTS.md) |
| Round 7 | How do the models compare under a fixed protocol and three data splits? | [Final evaluation](ROUND7_FINAL.md), [whole-image study](ROUND7_STEPD.md) |
| Round 8 A | What happens when both branches use format-matched training? | [Runbook](ROUND8_A_RUNBOOK.txt), [development results](../results/round8_a/) |
| Round 8 final | Do the observed differences recur on new images? | [Evaluation](ROUND8_FINAL.md), [metrics](../results/round8_final/) |
| November diagnostics | Why does the quality-aware gate underperform, and which changes help? | [Seven modifications](../output/analysis/report_evidence_20261111/tables/improvement_comparisons.csv), [evidence index](../output/analysis/report_evidence_20261111/claim_evidence.csv) |

## November source code

| Study | Source |
| --- | --- |
| Checkpoints, gate inputs, ranking and condition routing | [gate_diagnosis](../tmp/gate_diagnosis/) |
| Generator transfer, quality cues and loss decomposition | [gate_failure_mechanism_20261108](../tmp/gate_failure_mechanism_20261108/) |
| Multiple training generators | [multigen_gate_20261108](../tmp/multigen_gate_20261108/) |
| Paired-view global training | [paired_view_global_20261108](../tmp/paired_view_global_20261108/) |
| Recompression reliability | [recompression_reliability_20261108](../tmp/recompression_reliability_20261108/) |

These project-relative `tmp/` directories contain archived experiment code, configurations and results.
Their relative paths are used by the original scripts.

<details>
<summary>Data preparation and earlier reproduction instructions</summary>

- [Source inspection](ROUND7_SOURCE_INSPECTION.md) and [data partitions](ROUND7_DATA.md)
- [Development set](ROUND7_DEVELOPMENT.md) and [baseline implementation](ROUND7_BASELINES.md)
- [Repeated data splits](ROUND7_REPEATED_SEEDS.md) and [reporting precision](ROUND7_FINAL_PRECISION_NOTE.md)
- [Pilot and Rounds 2–6 reproduction](REPRODUCING.md)

Some notes describe the implementation stage at which they were written.
For the final interpretation, use the report linked above.

</details>
