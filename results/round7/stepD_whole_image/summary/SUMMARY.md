# Round 7 Step D: whole-image format matching

Raw scores; original profile. W0 is reused Step C global CLIP. W1/W2 tune on format-matched selection views and report on original images. All values below are mean ± sample SD over three seeds.

| Variant | Domain | Condition | AUROC | Balanced accuracy | Real FPR |
|---|---|---|---:|---:|---:|
| W0 | SD15 | clean | 0.971925 ± 0.016375 | 0.926667 ± 0.012858 | 0.109333 ± 0.020133 |
| W0 | SD15 | jpeg70 | 0.894485 ± 0.031090 | 0.808000 ± 0.041569 | 0.074667 ± 0.032332 |
| W0 | SD15 | jpeg50 | 0.899541 ± 0.039198 | 0.784000 ± 0.040596 | 0.074667 ± 0.044061 |
| W0 | SD15 | resize05 | 0.929557 ± 0.017324 | 0.821333 ± 0.035852 | 0.298667 ± 0.127583 |
| W0 | ADM | clean | 0.605556 ± 0.014775 | 0.547778 ± 0.002546 | 0.134444 ± 0.028350 |
| W0 | ADM | jpeg70 | 0.364585 ± 0.008677 | 0.460000 ± 0.002887 | 0.105556 ± 0.005092 |
| W0 | ADM | jpeg50 | 0.353711 ± 0.005039 | 0.464444 ± 0.006310 | 0.090000 ± 0.010000 |
| W0 | ADM | resize05 | 0.575230 ± 0.011332 | 0.538333 ± 0.019650 | 0.325556 ± 0.149530 |
| W0 | BigGAN | clean | 0.750322 ± 0.073994 | 0.623889 ± 0.079063 | 0.134444 ± 0.028350 |
| W0 | BigGAN | jpeg70 | 0.526019 ± 0.044720 | 0.486667 ± 0.007638 | 0.105556 ± 0.005092 |
| W0 | BigGAN | jpeg50 | 0.538630 ± 0.038061 | 0.492222 ± 0.001925 | 0.090000 ± 0.010000 |
| W0 | BigGAN | resize05 | 0.568611 ± 0.021551 | 0.524444 ± 0.045348 | 0.325556 ± 0.149530 |
| W1 | SD15 | clean | 0.952789 ± 0.019266 | 0.874667 ± 0.043879 | 0.218667 ± 0.089918 |
| W1 | SD15 | jpeg70 | 0.898645 ± 0.026256 | 0.725333 ± 0.055185 | 0.040000 ± 0.028844 |
| W1 | SD15 | jpeg50 | 0.896469 ± 0.032520 | 0.702667 ± 0.052205 | 0.037333 ± 0.028095 |
| W1 | SD15 | resize05 | 0.923584 ± 0.019904 | 0.821333 ± 0.028095 | 0.282667 ± 0.053267 |
| W1 | ADM | clean | 0.676733 ± 0.022475 | 0.603889 ± 0.015486 | 0.275556 ± 0.096398 |
| W1 | ADM | jpeg70 | 0.575563 ± 0.020492 | 0.502778 ± 0.006939 | 0.047778 ± 0.011706 |
| W1 | ADM | jpeg50 | 0.514041 ± 0.021360 | 0.490556 ± 0.003469 | 0.041111 ± 0.018359 |
| W1 | ADM | resize05 | 0.704244 ± 0.015116 | 0.633889 ± 0.015396 | 0.290000 ± 0.070000 |
| W1 | BigGAN | clean | 0.648844 ± 0.076000 | 0.565556 ± 0.038237 | 0.275556 ± 0.096398 |
| W1 | BigGAN | jpeg70 | 0.636681 ± 0.007132 | 0.518333 ± 0.013229 | 0.047778 ± 0.011706 |
| W1 | BigGAN | jpeg50 | 0.629159 ± 0.003254 | 0.518889 ± 0.013472 | 0.041111 ± 0.018359 |
| W1 | BigGAN | resize05 | 0.633611 ± 0.010267 | 0.580556 ± 0.018733 | 0.290000 ± 0.070000 |
| W2 | SD15 | clean | 0.972565 ± 0.010292 | 0.813333 ± 0.010066 | 0.354667 ± 0.036074 |
| W2 | SD15 | jpeg70 | 0.940139 ± 0.015426 | 0.820000 ± 0.006928 | 0.021333 ± 0.004619 |
| W2 | SD15 | jpeg50 | 0.924480 ± 0.024779 | 0.741333 ± 0.016166 | 0.018667 ± 0.004619 |
| W2 | SD15 | resize05 | 0.943275 ± 0.011863 | 0.860000 ± 0.006928 | 0.138667 ± 0.052051 |
| W2 | ADM | clean | 0.526741 ± 0.019038 | 0.508889 ± 0.018434 | 0.325556 ± 0.048800 |
| W2 | ADM | jpeg70 | 0.485696 ± 0.008051 | 0.493889 ± 0.004194 | 0.026667 ± 0.016667 |
| W2 | ADM | jpeg50 | 0.485319 ± 0.001183 | 0.492778 ± 0.003849 | 0.017778 ± 0.013472 |
| W2 | ADM | resize05 | 0.522730 ± 0.026062 | 0.495556 ± 0.006939 | 0.113333 ± 0.055678 |
| W2 | BigGAN | clean | 0.337174 ± 0.062051 | 0.388889 ± 0.021495 | 0.325556 ± 0.048800 |
| W2 | BigGAN | jpeg70 | 0.555496 ± 0.059563 | 0.497778 ± 0.001925 | 0.026667 ± 0.016667 |
| W2 | BigGAN | jpeg50 | 0.558619 ± 0.049244 | 0.497222 ± 0.002546 | 0.017778 ± 0.013472 |
| W2 | BigGAN | resize05 | 0.503330 ± 0.015274 | 0.495000 ± 0.008333 | 0.113333 ± 0.055678 |

## Frozen success criteria

| Variant | Seed | Criterion | Value | Requirement | Result |
|---|---|---|---:|---|---|
| W1 | 20261101 | ADM JPEG70/JPEG50 mean AUROC | 0.535072 | >= 0.5 | PASS |
| W1 | 20261101 | clean ADM AUROC drop vs W0 | -0.084289 | <= 0.02 | PASS |
| W1 | 20261101 | clean BigGAN AUROC drop vs W0 | 0.111500 | <= 0.02 | FAIL |
| W1 | 20261102 | ADM JPEG70/JPEG50 mean AUROC | 0.567722 | >= 0.5 | PASS |
| W1 | 20261102 | clean ADM AUROC drop vs W0 | -0.076467 | <= 0.02 | PASS |
| W1 | 20261102 | clean BigGAN AUROC drop vs W0 | 0.077656 | <= 0.02 | FAIL |
| W1 | 20261103 | ADM JPEG70/JPEG50 mean AUROC | 0.531611 | >= 0.5 | PASS |
| W1 | 20261103 | clean ADM AUROC drop vs W0 | -0.052778 | <= 0.02 | PASS |
| W1 | 20261103 | clean BigGAN AUROC drop vs W0 | 0.115278 | <= 0.02 | FAIL |
| W2 | 20261101 | ADM JPEG70/JPEG50 mean AUROC | 0.485411 | >= 0.5 | FAIL |
| W2 | 20261101 | clean ADM AUROC drop vs W0 | 0.062811 | <= 0.02 | FAIL |
| W2 | 20261101 | clean BigGAN AUROC drop vs W0 | 0.431767 | <= 0.02 | FAIL |
| W2 | 20261102 | ADM JPEG70/JPEG50 mean AUROC | 0.481250 | >= 0.5 | FAIL |
| W2 | 20261102 | clean ADM AUROC drop vs W0 | 0.116122 | <= 0.02 | FAIL |
| W2 | 20261102 | clean BigGAN AUROC drop vs W0 | 0.465389 | <= 0.02 | FAIL |
| W2 | 20261103 | ADM JPEG70/JPEG50 mean AUROC | 0.489861 | >= 0.5 | FAIL |
| W2 | 20261103 | clean ADM AUROC drop vs W0 | 0.057511 | <= 0.02 | FAIL |
| W2 | 20261103 | clean BigGAN AUROC drop vs W0 | 0.342289 | <= 0.02 | FAIL |

**W1: FAIL** (all criteria in every seed).

**W2: FAIL** (all criteria in every seed).

## Clean SD15 cost

Positive means worse than W0: W0 minus variant for AUROC/BA, variant minus W0 for FPR.

| Variant | Metric | Cost, mean ± SD |
|---|---|---:|
| W1 | auroc | 0.019136 ± 0.002891 |
| W1 | balanced_accuracy | 0.052000 ± 0.044000 |
| W1 | real_false_positive_rate | 0.109333 ± 0.100027 |
| W2 | auroc | -0.000640 ± 0.010483 |
| W2 | balanced_accuracy | 0.113333 ± 0.014048 |
| W2 | real_false_positive_rate | 0.245333 ± 0.030288 |

Paired W1−W0, W2−W0 and W2−W1 differences for every condition/domain/metric and each seed are in `paired_differences.csv`. Positive AUROC/BA differences favour the left variant; positive FPR differences are worse.

Real images become double-compressed, fakes single-compressed. Native resolution and previous compression history remain confounded. W1−W0 holds architecture fixed; W2 also changes architecture. These results assess the intervention, not definitive proof of sole causation.

SD15 reporting uses selection identities already used for tuning. ADM/BigGAN are inspected development domains. The seeds share fitting/development pools; sample SD is descriptive, not a confidence interval. No reserved-test images or predictions were used.
