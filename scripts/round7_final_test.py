"""Pre-freeze tests: synthetic fixtures and read-only DEVELOPMENT-cache parity."""
import sys
sys.dont_write_bytecode = True
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import round7_final_common as f
from round7_final_common import c
import round7_final_models as models
import round7_final_report as report


class FinalTests(unittest.TestCase):
    def test_declaration_covers_all_development_models_and_three_gate_inits(self):
        declared=f.declarations()
        self.assertEqual(len(declared),138)
        self.assertEqual(len(f.task_groups(declared)),72)
        self.assertEqual(sum(r['model']=='W1' for r in declared),3)
        self.assertEqual(sum(r['model']=='W2' for r in declared),3)
        for r in declared:
            self.assertTrue(0<=r['threshold']<=1)

    def test_frozen_gate_and_head_restoration_matches_stepC_development_cache(self):
        import joblib
        import torch
        torch.set_num_threads(2)
        declared=f.declarations()
        for seed in f.seeds.SEEDS:
            for profile in c.PROFILES:
                n=f.seeds.number(seed)
                selected=[r for r in declared if r['stage']=='C' and r['seed']==seed and r['profile']==profile and r['family']=='clip']
                heads=joblib.load(f.ROOT/selected[0]['head'])
                settings=c.read(f.ROOT/selected[0]['settings_path'])
                # Only existing selection features/predictions, no original/test image access.
                old=c.read_csv(f.ROOT/f'results/round7/baselines_seed{n}/clip_{profile}_predictions.csv')
                lookup={(r['model'],r['calibration'],int(r['seed']),r['condition'],r['source_id']):float(r['logit']) for r in old if r['partition']=='selection'}
                batches=[]; identities=[]
                for condition in c.CONDITIONS:
                    path=f.ROOT/f'artifacts/round7/baselines_seed{n}/features/{profile}_selection_{condition}.npz'
                    with np.load(path,allow_pickle=False) as data:
                        ids=data['ids'].tolist(); features={k:data[k] for k in ('g','l','q')}
                    batches.append(features); identities.extend((condition,i) for i in ids)
                # Step C applied each gate to all four selection conditions together.
                # Match that batch shape for the exact restoration parity check.
                features={k:np.concatenate([b[k] for b in batches]) for k in ('g','l','q')}
                restored=models.clip_scores(selected,features,heads,settings)
                for row in selected:
                    expected=[lookup[row['model'],row['calibration'],row['initialization'],condition,i] for condition,i in identities]
                    np.testing.assert_allclose(restored[row['id']][0],expected,rtol=0,atol=1e-12)

    def test_W1_restoration_matches_original_selection_reporting_cache(self):
        import joblib
        for seed in f.seeds.SEEDS:
            n=f.seeds.number(seed)
            saved=joblib.load(f.ROOT/f'artifacts/round7/stepD_whole_image/seed{n}/W1/global_head.joblib')
            rows=c.read_csv(f.ROOT/f'results/round7/stepD_whole_image/seed{n}/W1/predictions.csv')
            lookup={(r['condition'],r['source_id']):float(r['logit']) for r in rows if r['partition']=='selection'}
            for condition in c.CONDITIONS:
                with np.load(f.ROOT/f'artifacts/round7/stepD_whole_image/seed{n}/W1/features/selection_reporting_{condition}.npz',allow_pickle=False) as data:
                    ids=data['ids'].tolist(); x=data['x']
                np.testing.assert_allclose(saved['head'].decision_function(x),[lookup[condition,i] for i in ids],rtol=0,atol=1e-12)

    def test_test_access_requires_freeze_and_rejects_non_reserved(self):
        with patch.object(f,'FREEZE',f.ROOT/'artifacts/round7/final_test_synthetic_nonexistent.json'):
            with self.assertRaisesRegex(AssertionError,'freeze'):
                f.TestGuard({'test:a':{}})
            with self.assertRaises(AssertionError):
                f.records({})
        # A synthetic guard test does not open the real reserved manifest.
        with patch.object(f,'FREEZE',f.ROOT/'results/round7/data_protocol.json'):
            guard=f.TestGuard({'test:a':{}})
            guard.touch(['test:a'],'predict')
            with self.assertRaisesRegex(AssertionError,'non-reserved'):
                guard.touch(['train:b'],'predict')

    def test_atomic_cache_identity_and_once_only_gate(self):
        from round7_final_run import assert_runnable
        folder=f.ROOT/'artifacts/round7/final_synthetic_tests'; folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            path=Path(temp)/'cache.npz'
            f.npz_publish(path,['a'],{'z':np.array([.3])},'freeze1')
            np.testing.assert_array_equal(f.npz_read(path,['a'],'freeze1')['z'],[.3])
            with self.assertRaises(AssertionError):
                f.npz_read(path,['a'],'changed')
            with self.assertRaises(FileExistsError):
                f.npz_publish(path,['a'],{'z':np.array([.3])},'freeze1')
            with patch.object(f,'OUT',Path(temp)/'results'):
                assert_runnable(False)
                f.publish(f.OUT/'run.json',b'{}')
                with self.assertRaises(FileExistsError):
                    assert_runnable(False)
                assert_runnable(True)
                f.publish(f.OUT/'complete.json',b'{}')
                for resume in (False,True):
                    with self.assertRaises(FileExistsError):
                        assert_runnable(resume)

    def test_notebook_hash_ignores_runtime_outputs_but_pins_source(self):
        import json
        folder=f.ROOT/'artifacts/round7/final_synthetic_tests'; folder.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            p=Path(temp)/'book.ipynb'
            value={'cells':[{'cell_type':'code','source':['print(1)'],'outputs':[],'execution_count':None}]}
            p.write_text(json.dumps(value),encoding='utf-8'); original=f.notebook_source_hash(p)
            value['cells'][0].update(outputs=[{'text':'1'}],execution_count=1)
            p.write_text(json.dumps(value),encoding='utf-8')
            self.assertEqual(original,f.notebook_source_hash(p))
            value['cells'][0]['source']=['print(2)']
            p.write_text(json.dumps(value),encoding='utf-8')
            self.assertNotEqual(original,f.notebook_source_hash(p))

    def test_saved_resnet_checkpoint_restores_without_training_or_images(self):
        declared=f.declarations()
        for stage,profile in [('C','common128'),('D','original')]:
            row=next(r for r in declared if r['stage']==stage and r['profile']==profile and r['family']=='resnet')
            proto=(f.ROOT/row['settings_path']).parent/'protocol.json'
            paths=[f.ROOT/row['checkpoint'],proto]
            frozen={'files_sha256':{p.relative_to(f.ROOT).as_posix():c.sha(p) for p in paths}}
            model=models.restore_resnet(row,frozen,'cpu')
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertEqual(model.fc.out_features,1)
            del model

    def test_aggregation_gate_mean_paired_sd_and_untouched_macro(self):
        declared=f.declarations(); raw=[]
        for m in declared:
            for gi,(gen,exposure) in enumerate(f.GROUPS.items()):
                for condition in f.CONDITIONS:
                    slope=.02 if m['model']=='W1' else .01
                    value=.5+slope*f.seeds.number(m['seed'])+gi*.001
                    if m['model'].endswith('gate'):
                        value+=(m['initialization']-20261022)*.002
                    raw.append({**{k:m[k] for k in ('stage','seed','profile','model','calibration','initialization')},'model_id':m['id'],'generator':gen,'exposure':exposure,'condition':condition,
                                'n_real':500,'n_fake':250 if gen=='SD15' else 500,**{metric:value for metric in f.METRICS}})
        tables=report.summarize(raw,declared)
        self.assertEqual(len(tables['metrics_initializations.csv']),3864)
        self.assertEqual(len(tables['metrics_each_seed.csv']),2520)
        self.assertEqual(len(tables['untouched_macro_each_seed.csv']),360)
        self.assertEqual(len(tables['metrics_mean_sd.csv']),3840)
        self.assertEqual(len(tables['paired_differences.csv']),640)
        pair=next(r for r in tables['paired_differences.csv'] if r['contrast']=='W1 - W0' and r['generator']=='untouched_macro')
        self.assertAlmostEqual(pair['mean'],.02); self.assertAlmostEqual(pair['sd'],.01)
        macro=next(r for r in tables['untouched_macro_each_seed.csv'] if r['model']=='global')
        self.assertAlmostEqual(macro['auroc'],.5+.01*f.seeds.number(macro['seed'])+.0045)
        with self.assertRaises(AssertionError):
            report.summarize(raw[:-1],declared)

    def test_prediction_coverage_threshold_and_id_rejection(self):
        from scipy.special import expit
        records={'a':{'generator':'Real','label':0},'b':{'generator':'ADM','label':1}}
        declaration=[dict(id='m',threshold=.5)]
        rows=[dict(model_id='m',source_id=i,condition='clean',**r,logit=2*r['label']-1,probability_fake=float(expit(2*r['label']-1)),threshold=.5) for i,r in records.items()]
        report.validate_rows(rows,declaration,'clean',records)
        for key,value in [('source_id','forbidden'),('threshold',.4),('condition','jpeg50')]:
            bad=[dict(r) for r in rows]; bad[0][key]=value
            with self.assertRaises(AssertionError):
                report.validate_rows(bad,declaration,'clean',records)


if __name__=='__main__':
    unittest.main(verbosity=2)
