"""Seed routing and aggregation tests; synthetic scores, no real model execution."""
import sys
sys.dont_write_bytecode = True
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import round7_baseline_common as c
import round7_seed_support as support
import round7_summarize_seeds as summary


def fixture(seed):
    metrics, weights = [], []
    for key in sorted(summary.expected_keys()):
        row = dict(zip(summary.KEYS, key))
        model = row['model']
        inits = c.GATE_SEEDS if model.endswith('gate') else [seed] if model == 'resnet50' else [-1]
        for j, init in enumerate(inits):
            # Model-specific slopes make paired SD differ from either marginal SD.
            mi = summary.MODELS.index(model)
            value = .55 + mi * .01 + support.number(seed) * .001 * (mi + 1) + j * .002
            value += .03 * (row['profile'] == 'common128') + .005 * c.CONDITIONS.index(row['condition'])
            metrics.append(dict(row, seed=init, auroc=value))
            if model.endswith('gate'):
                weights.append(dict(row, seed=init, mean=.2 + j * .1 + support.number(seed) * .01))
    macro = []
    for key in sorted({k[:-1] for k in summary.expected_keys()}):
        values = [r['auroc'] for r in metrics if tuple(r[k] for k in summary.KEYS[:-1]) == key]
        macro.append(dict(zip(summary.KEYS[:-1], key), macro_auroc=float(np.mean(values))))
    return metrics, macro, weights


class SeedTests(unittest.TestCase):
    def tearDown(self):
        support.configure(20261101)

    def test_routing_fixed_pool_and_test_isolation(self):
        original_open = Path.open
        def deny_test(path, *args, **kwargs):
            if path.name == 'reserved_test_manifest.jsonl':
                raise AssertionError('Reserved test manifest opened')
            return original_open(path, *args, **kwargs)
        pools, selections = [], []
        with patch.object(Path, 'open', deny_test):
            for seed in support.SEEDS:
                support.configure(seed)
                records, parts, deny = c.context()
                self.assertEqual(len(records), 3400)
                self.assertEqual([len(parts[p]) for p in ('expert_fit','gate_fit','calibration','selection','development')], [1500,500,250,250,900])
                self.assertFalse(set(records) & deny)
                self.assertTrue(str(c.OUT).endswith(f'baselines_seed{support.number(seed)}'))
                self.assertTrue(str(c.ART).endswith(f'baselines_seed{support.number(seed)}'))
                pools.append(set(records)); selections.append(set(parts['selection']))
        self.assertTrue(all(p == pools[0] for p in pools))
        self.assertEqual(len({frozenset(p) for p in selections}), 3)

    def test_all_settings_preserved_and_drift_rejected(self):
        for seed in support.SEEDS:
            protocol = support.expected_protocol(seed)
            support.assert_identical_settings(protocol, seed)
            bad = copy.deepcopy(protocol)
            bad['resnet']['lr'] = .01
            with self.assertRaises(AssertionError):
                support.assert_identical_settings(bad, seed)
        self.assertEqual(support.expected_protocol(20261101), c.read(support.REFERENCE / 'protocol.json'))

    def test_three_run_summary_and_paired_sd(self):
        tables = summary.build_tables({s: fixture(s) for s in support.SEEDS})
        self.assertEqual({k: len(v) for k,v in tables.items()}, {'per_condition_auroc.csv':336, 'macro_auroc.csv':84, 'paired_differences.csv':510, 'gate_global_weight_means.csv':96})
        row = next(r for r in tables['per_condition_auroc.csv'] if r['model']=='score_gate')
        self.assertAlmostEqual(row['sd'], .005)
        self.assertEqual(row['n_seeds'], 3)
        pair = next(r for r in tables['paired_differences.csv'] if r['contrast']=='average - global')
        np.testing.assert_allclose([pair[f'seed_{s}'] for s in support.SEEDS], [.022,.024,.026], atol=1e-12)
        self.assertAlmostEqual(pair['sd'], .002)
        profile = next(r for r in tables['paired_differences.csv'] if r['contrast']=='common128 - original')
        self.assertAlmostEqual(profile['mean'], .03)
        self.assertAlmostEqual(profile['sd'], 0)
        weight = tables['gate_global_weight_means.csv'][0]
        self.assertAlmostEqual(weight['mean'], .32)
        self.assertAlmostEqual(weight['sd'], .01)

    def test_missing_duplicate_wrong_seed_and_bad_macro_rejected(self):
        with self.assertRaises(AssertionError):
            summary.build_tables({20261101: fixture(20261101)})
        original = fixture(20261102)
        variants = []
        missing = copy.deepcopy(original); missing[0].pop(); variants.append(missing)
        duplicate = copy.deepcopy(original); duplicate[0].append(duplicate[0][0]); variants.append(duplicate)
        macro = copy.deepcopy(original); macro[1][0]['macro_auroc'] += .01; variants.append(macro)
        seed = copy.deepcopy(original)
        next(r for r in seed[0] if r['model']=='resnet50')['seed'] = 20261101
        variants.append(seed)
        weight = copy.deepcopy(original); weight[2].pop(); variants.append(weight)
        for variant in variants:
            with self.assertRaises(AssertionError):
                summary.aggregate_seed(*variant, 20261102)

    def test_actual_seed1_table_coverage_read_only(self):
        inputs = [c.read_csv(support.REFERENCE / name) for name in ('metrics.csv','macro_summary.csv','gate_weight_summary.csv')]
        collapsed = summary.aggregate_seed(*inputs, 20261101)
        self.assertEqual(tuple(map(len, collapsed)), (336,84,96))

    def test_publication_is_resumable_and_never_overwrites(self):
        folder = c.ROOT / 'artifacts/round7/seed_repetition_tests'
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            path = Path(temp) / 'output.csv'
            summary.publish(path, b'original', False)
            summary.publish(path, b'original', True)
            with self.assertRaises(FileExistsError):
                summary.publish(path, b'original', False)
            with self.assertRaises(AssertionError):
                summary.publish(path, b'changed', True)
            self.assertEqual(path.read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main(verbosity=2)
