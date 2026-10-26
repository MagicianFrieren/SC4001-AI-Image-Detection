# Execution note

The original freeze is unchanged. A numerical-reporting correction was recorded after the first ResNet task stopped before CSV publication: raw FP32 logits are promoted to float64 for saved-temperature division and sigmoid, matching Step C reporting arithmetic. No model, threshold, preprocessing, condition, metric or selection setting changed. Completed predictions and raw-logit batches were reused without rescoring.

See `../final_execution_addendum.json` and `docs/ROUND7_FINAL_PRECISION_NOTE.md`. Verify with `scripts/round7_final_precision_resume.py --verify`. This is a documented post-freeze technical correction, not a replacement freeze or another experiment.
