# Round 7 Step B: development addendum

This additive protocol leaves all prior files unchanged. `data_protocol_step_b.json` extends the frozen base protocol.

Development has 300 Real + 300 ADM + 300 BigGAN originals, sampled with seed 20261104 from unused eligible source rows. It is disjoint from the fitting pool, reserved test, and historical 1,600-image pool by pixel SHA256 and dHash distance <=3. The original global deduplication remains in force.

ADM/BigGAN are already heavily inspected. Development is diagnostic only: it cannot fit heads or gates, calibrate temperatures, select epochs/fusion weights, or fit thresholds. Their future reserved-test results remain previously inspected-domain results. GLIDE/VQDM/Midjourney/Wukong have no model use in Round 7 before final evaluation; historical independent-pilot exposure remains disclosed, and audit hashing is allowed.

`step_b_isolation.json` contains an ID-only reserved-test denylist and the exact fitting-plus-development allowlist. Step C reads this file, never the reserved-test manifest or test images.

Run `python scripts/round7_verify_development.py --full` with pinned Pillow 12.2. Preparation is exclusive-create; reruns refuse existing outputs. The verifier is read-only and extends the existing data verifier.
