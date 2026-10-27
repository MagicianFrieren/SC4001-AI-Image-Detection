"""Synthetic contract/integration checks. No downloaded images or CLIP inference."""
import sys
sys.dont_write_bytecode = True
import copy
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import round8_a_common as a
import round8_a_run as runner
import round8_a_verify as verify
from round8_a_common import c, images, seeds, w1
import gate_models as gates


def fixture():
    records, parts = {}, {}
    for part in a.PARTS:
        generators = ['Real'] * 6 + (['ADM'] * 6 + ['BigGAN'] * 6 if part == 'development' else ['SD15'] * 6)
        parts[part] = []
        for gen in generators:
            identity = f'train:{len(records):04d}'
            parts[part].append(identity)
            records[identity] = dict(source_id=identity, generator=gen, label=int(gen != 'Real'))
    return records, parts, {'train:9999'}


class FakeFeatures:
    def __init__(self, seed, records, parts, guard, protocol=None, signature=None):
        self.records, self.parts, self.guard, self.seed = records, parts, guard, seed
        self.hashes = {}

    def get(self, part, condition, matched):
        ids = self.parts[part]
        self.guard.touch(ids, 'features/' + part)
        if matched:
            self.guard.touch(ids, 'format_match')
        rng_seed = int.from_bytes(hashlib.sha256(f'{self.seed}|{part}|{condition}|{matched}'.encode()).digest()[:4], 'big')
        rng = np.random.default_rng(rng_seed)
        y = np.array([self.records[i]['label'] for i in ids]) * 2 - 1
        return dict(g=rng.normal(size=(len(ids), 12)) + y[:, None] * .3,
                    l=rng.normal(size=(len(ids), 4, 12)) + y[:, None, None] * .15,
                    q=rng.normal(size=(len(ids), 3)))


class Round8Tests(unittest.TestCase):
    def test_native_jpeg_precedes_resize_and_patchshuffle(self):
        records, parts, deny = fixture()
        guard = a.Guard(records, parts, deny)
        native = Image.fromarray(np.random.default_rng(3).integers(0, 256, (301, 497, 3), dtype=np.uint8))
        identity = parts['expert_fit'][0]
        spec = dict(quality_min=60, quality_max=95)
        with patch.object(a, 'load_original', return_value=native):
            result = a.view(identity, 'jpeg70', True, seeds.SEEDS[0], records, guard, spec)
            reference = images.view(w1.match_format(native, seeds.SEEDS[0], identity, spec), 'original', 'jpeg70')
            self.assertEqual(result.tobytes(), reference.tobytes())
            wrong = w1.match_format(images.view(native, 'original', 'jpeg70'), seeds.SEEDS[0], identity, spec)
            self.assertNotEqual(result.tobytes(), wrong.tobytes())
            for seed in images.VIEW_SEEDS:
                self.assertEqual(images.shuffled(result, seed).tobytes(), images.shuffled(reference, seed).tobytes())

    def test_reporting_never_matches_and_forbidden_access_precedes_decode(self):
        records, parts, deny = fixture()
        guard = a.Guard(records, parts, deny)
        native = Image.new('RGB', (320, 250), color=(50, 70, 90))
        with patch.object(a, 'load_original', return_value=native) as loader:
            identity = parts['development'][0]
            result = a.view(identity, 'clean', False, seeds.SEEDS[0], records, guard, {})
            self.assertEqual(result.tobytes(), images.view(native, 'original', 'clean').tobytes())
            loader.reset_mock()
            with self.assertRaises(ValueError):
                a.view(identity, 'clean', True, seeds.SEEDS[0], records, guard, {})
            loader.assert_not_called()
            with self.assertRaises(AssertionError):
                a.view('train:9999', 'clean', False, seeds.SEEDS[0], records, guard, {})
            loader.assert_not_called()

    def test_partition_roles_and_quality_draws(self):
        records, parts, deny = fixture()
        guard = a.Guard(records, parts, deny)
        for action, invalid in [('fit_experts', 'gate_fit'), ('fit_gate', 'selection'), ('select', 'development')]:
            with self.assertRaises(ValueError):
                guard.touch(parts[invalid], action)
        draws = {i: w1.quality(seeds.SEEDS[0], i) for i in records}
        self.assertEqual(draws, {i: w1.quality(seeds.SEEDS[0], i) for i in reversed(records)})
        self.assertTrue(all(60 <= q <= 95 for q in draws.values()))
        self.assertTrue(any(q != w1.quality(seeds.SEEDS[1], i) for i, q in draws.items()))

    def test_quality_ablation_reconstruction(self):
        records, parts, deny = fixture()
        guard = a.Guard(records, parts, deny)
        features = FakeFeatures(seeds.SEEDS[0], records, parts, guard)
        with patch.object(gates, 'MAX_EPOCHS', 8), patch.object(gates, 'PATIENCE', 4):
            predictions, settings, _, raw = runner.fit_arm(seeds.SEEDS[0], 'matched', records, parts, guard, features.get)
        batch = raw['development']
        changed = copy.deepcopy(batch)
        changed['q'] = (np.asarray(changed['q']) + 100).tolist()
        for init in gates.SEEDS:
            self.assertTrue(np.array_equal(verify.alpha('score_gate', init, batch, settings),
                                           verify.alpha('score_gate', init, changed, settings)))
            for name in ('score_gate', 'quality_gate'):
                cell = [r for r in predictions if r['model'] == name and r['gate_seed'] == init and r['partition'] == 'development']
                self.assertTrue(np.allclose([r['alpha_global'] for r in cell], verify.alpha(name, init, batch, settings), atol=2e-6, rtol=0))
        self.assertEqual(len(verify.metrics(predictions)), 120)
        self.assertFalse(set(parts['development']) & guard.events['format_match'])

    def test_exclusive_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'output.json'
            a.publish(path, b'original')
            a.publish(path, b'original')
            with self.assertRaises(AssertionError):
                a.publish(path, b'changed')
            self.assertEqual(path.read_bytes(), b'original')

    def test_cache_rejects_cross_condition_before_model_load(self):
        records, parts, deny = fixture()
        with tempfile.TemporaryDirectory() as temporary, patch.object(a, 'ART', Path(temporary)):
            guard = a.Guard(records, parts, deny)
            features = runner.Features(seeds.SEEDS[0], records, parts, guard, {}, 'signature')
            path = a.ART / 'seed1/features/expert_fit_clean_0/00000.npz'
            path.parent.mkdir(parents=True)
            np.savez(path, identity=np.array('wrong condition'), ids=np.array(parts['expert_fit']), signature=np.array('signature'))
            with self.assertRaises(ValueError):
                features.get('expert_fit', 'clean', False)
            self.assertIsNone(features.model)

    def test_complete_resume_and_tamper_rejection(self):
        records, parts, deny = fixture()
        (a.ROOT / 'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=a.ROOT / 'tmp') as temporary:
            base = Path(temporary)
            with patch.object(a, 'OUT', base / 'results'), patch.object(a, 'ART', base / 'artifacts'), \
                 patch.object(a, 'context', return_value=(records, parts, deny)), \
                 patch.object(runner, 'Features', FakeFeatures), \
                 patch.object(gates, 'MAX_EPOCHS', 8), patch.object(gates, 'PATIENCE', 4):
                protocol = dict(code_sha256={}, input_sha256={}, plan_sha256=c.sha(a.PLAN))
                a.write_json(a.OUT / 'protocol.json', protocol)
                signature = c.sha(a.OUT / 'protocol.json')
                runner.execute(seeds.SEEDS[0], 'matched', protocol, signature)
                check = verify.verify_run(seeds.SEEDS[0], 'matched')
                self.assertEqual(check['status'], 'passed')
                with patch.object(runner, 'fit_arm', side_effect=AssertionError('Must not retrain')):
                    runner.execute(seeds.SEEDS[0], 'matched', protocol, signature)
                path = a.OUT / 'seed1/matched/predictions.csv'
                path.write_bytes(path.read_bytes() + b'\n')
                with self.assertRaises(ValueError):
                    verify.verify_run(seeds.SEEDS[0], 'matched')

    def test_threshold_ties_and_average_is_logit_fusion(self):
        y = np.array([0, 0, 1, 1])
        z = np.array([-.5, .5, -.5, .5])
        self.assertEqual(verify.threshold(y, z), c.threshold(y, verify.expit(z)))
        batch = dict(y=y.tolist(), zg=[-4., 1., 5., 2.], zl=[1., 2., -1., 3.])
        weight = verify.alpha('average', -1, batch, {'settings': {'average/-1': {'fixed_alpha': .7}}})
        np.testing.assert_array_equal(weight, np.full(4, .5))
        fused = verify.expit(weight * batch['zg'] + (1 - weight) * batch['zl'])
        self.assertFalse(np.allclose(fused, (verify.expit(batch['zg']) + verify.expit(batch['zl'])) / 2))

    def test_summary_averages_initializations_within_data_seed(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(a, 'OUT', Path(temporary)):
            for index, seed in enumerate(seeds.SEEDS):
                for arm in a.ARMS:
                    rows = []
                    for model, init in verify.INSTANCES:
                        init_offset = (c.GATE_SEEDS.index(init) + 1) * .01 if init != -1 else 0
                        score = .5 + index * .1 + init_offset + (.05 if arm == 'matched' else 0)
                        for generator in ('SD15', 'ADM', 'BigGAN'):
                            for condition in c.CONDITIONS:
                                rows.append(dict(model=model, generator=generator, condition=condition,
                                                 **{metric: score for metric in verify.METRICS}))
                    a.write_csv(a.OUT / f'seed{index + 1}' / arm / 'metrics.csv', rows)
            with patch.object(verify, 'verify_run', return_value={'status': 'fixture'}), patch('builtins.print'):
                result = verify.summarize()
            self.assertEqual(result['summary_rows'], 576)
            self.assertEqual(result['contrast_rows'], 192)
            summary = c.read_csv(a.OUT / 'metrics_mean_sd.csv')
            row = next(r for r in summary if r['arm'] == 'matched' and r['model'] == 'quality_gate')
            self.assertAlmostEqual(float(row['mean']), .67)
            self.assertAlmostEqual(float(row['sd']), .1)
            contrast = c.read_csv(a.OUT / 'paired_differences.csv')
            row = next(r for r in contrast if r['contrast'] == 'matched_quality_minus_matched_average')
            self.assertAlmostEqual(float(row['mean']), .02)
            self.assertAlmostEqual(float(row['sd']), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
