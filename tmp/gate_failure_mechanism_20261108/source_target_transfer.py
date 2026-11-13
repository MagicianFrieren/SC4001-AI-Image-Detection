"""Summarize existing tuning and held-out metrics; no fitting or selection."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'output/analysis/multigen_gate_20261108/run_v1_codefix1'
OUT = ROOT / 'output/analysis/gate_failure_mechanism_20261108'


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    source_path = OUT / 'objective_source_summary.csv'
    target_path = RUN / 'metrics_seed_condition.csv'
    source, target = read_csv(source_path), read_csv(target_path)
    complete = json.loads((RUN / 'complete.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(target_path.read_bytes()).hexdigest() == complete['outputs_sha256'][target_path.name]
    rows = []
    for donor, heldout in [('ADM', 'BigGAN'), ('BigGAN', 'ADM')]:
        scenario = 'mixed_' + donor
        for condition in ['clean', 'jpeg70', 'resize05', 'jpeg50']:
            for domain, generator in [('donor_tuning', donor), ('heldout_development', heldout)]:
                if condition == 'jpeg50' and domain == 'donor_tuning':
                    continue
                output = dict(donor=donor, domain=domain, generator=generator, condition=condition)
                for model in ['global', 'local', 'average', 'fixed', 'score_gate', 'quality_gate']:
                    cp = 'auroc' if 'gate' in model else 'fixed'
                    if domain == 'donor_tuning':
                        found = [r for r in source if r['scenario'] == scenario and r['partition'] == 'tune'
                                 and r['generator'] == generator and r['condition'] == condition
                                 and r['model'] == model and r['checkpoint'] == cp]
                        assert len(found) == 1, (output, model, len(found))
                    else:
                        found = [r for r in target if r['scenario'] == scenario and r['heldout_generator'] == generator
                                 and r['condition'] == condition and r['model'] == model and r['checkpoint'] == cp]
                        assert len(found) == 3, (output, model, len(found))
                    output[model + '_auroc'] = sum(float(r['auroc']) for r in found) / len(found)
                    if 'gate' in model:
                        output[model + '_alpha_global'] = sum(float(r['alpha_mean']) for r in found) / len(found)
                output['better_single_expert'] = 'global' if output['global_auroc'] > output['local_auroc'] else 'local'
                rows.append(output)
    with (OUT / 'source_target_transfer.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps([r for r in rows if r['condition'] == 'jpeg70'], indent=2))


if __name__ == '__main__':
    main()
