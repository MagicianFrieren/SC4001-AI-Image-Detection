Prediction archives for error analysis and plots (per-image predictions of the frozen models on the two reserved test sets).
These CSVs are kept out of Git as loose files because of their size; the zips restore them to their original paths.

  round7_final_test_predictions.zip  -> results/round7/final_test/predictions/   (Round 7 reserved test, 72 CSVs)
  round8_final_test_predictions.zip  -> results/round8_final/evaluation/predictions/  (Round 8 fresh test)

Unpack from the repository root (PowerShell):
  Expand-Archive -LiteralPath results/handoff/round7_final_test_predictions.zip -DestinationPath .
  Expand-Archive -LiteralPath results/handoff/round8_final_test_predictions.zip -DestinationPath .

Check integrity first: Get-FileHash -Algorithm SHA256 <zip>  and compare with SHA256SUMS.txt.
Round 8A development predictions are in results/round8_a/predictions.zip.
Analysis of these predictions is post-hoc: do not change, refit or reselect any model based on it.
