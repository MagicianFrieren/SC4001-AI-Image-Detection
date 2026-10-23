"""Post-hoc confidence diagnostics; never use label-oracle results as models."""
import csv
import numpy as np

from run_round2 import dump_csv
from run_round4 import OUT, PROFILES


def main():
    with (OUT / 'branch_inputs.csv').open(encoding='utf-8') as handle:
        inputs = list(csv.DictReader(handle))
    output = []
    for profile in PROFILES:
        for partition in ['gate_train', 'validation', 'development']:
            for condition in ['clean', 'jpeg70', 'jpeg50', 'resize05']:
                rows = [r for r in inputs if r['profile'] == profile and r['partition'] == partition
                        and r['condition'] == condition and (partition != 'development' or r['generator'] in ['Real', 'ADM'])]
                if not rows:
                    continue
                y = np.array([int(r['label']) for r in rows])
                zg, zl = [np.array([float(r[k]) for r in rows]) for k in ['global_logit', 'local_logit']]
                def loss(z):
                    return float(np.mean(np.logaddexp(0, z) - y * z))
                # BCE is monotone in the logit for a fixed binary label, so the
                # best attainable convex blend for each labeled example is one endpoint.
                oracle = np.where(y == 1, np.maximum(zg, zl), np.minimum(zg, zl))
                assert loss(oracle) <= min(loss(zg), loss(zl), loss((zg + zl) / 2)) + 1e-12
                both_wrong = ((zg >= 0) != y) & ((zl >= 0) != y)
                output.append(dict(profile=profile, partition=partition, condition=condition, n=len(rows),
                                   global_bce=loss(zg), local_bce=loss(zl), average_bce=loss((zg + zl) / 2),
                                   label_oracle_endpoint_bce=loss(oracle), both_wrong_at_probability_half=int(both_wrong.sum()),
                                   note='Label oracle is an unattainable diagnostic lower bound on BCE for convex logit fusion; no deployable model.'))
    dump_csv(OUT / 'expert_confidence_diagnostic.csv', output)
    print(f'Saved {len(output)} descriptive confidence diagnostics; no models changed.')


if __name__ == '__main__':
    main()
