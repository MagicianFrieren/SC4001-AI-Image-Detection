# Step E execution addendum: reporting precision only

The original `results/round7/final_freeze.json`, frozen evaluator scripts, saved models, settings, development results and completed test predictions remain unchanged.

The first final-evaluation invocation stopped at the first ResNet task, before publishing its prediction CSV. NumPy/SciPy retained float32 for a sigmoid computed from a float32 scalar, while the frozen validator correctly recomputed the probability from a Python float (float64) with tolerance 1e-14. The observed difference was about 5.1e-8 in the first cached batch. This was a serialization/reporting-precision failure, not a model or dataset failure.

`scripts/round7_final_precision_resume.py` supplies one explicit numerical adapter. It preserves FP32 model forward passes and raw cached logits. It promotes those raw logits to float64 before dividing by the already-saved positive temperature and applying sigmoid, matching the Python-float reporting arithmetic in Step C. No weights, thresholds, temperatures, epochs, preprocessing, conditions, generators or metrics change. No fitting, calibration, model selection, threshold selection or new scoring of completed batches is performed.

At the interruption, all eight shared CLIP feature passes, four CLIP prediction tasks and all 118 raw-logit batches for the first ResNet condition were complete. Their hashes are recorded in `results/round7/final_execution_addendum.json` before continuation. The original freeze is retained as the pre-test scientific declaration; the addendum openly records this post-freeze technical correction and its code/notebook hashes. It is not a replacement freeze or a new experiment.

Use **`docs/ROUND7_FINAL_RESUME.ipynb`** for continuation and verification. It skips evaluation when the original final-completion marker exists. The adapter also refuses a completed run through the original evaluator's guard. Verification checks both the original freeze and this addendum, independently recomputing metrics without loading images or scoring models again.

This addendum must be retained with the report. It records an execution correction after the test was opened; the final evaluation should not be described as having proceeded without any technical correction.
