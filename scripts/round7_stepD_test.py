"""Synthetic Step D checks. No source images, pretrained models, or real training."""
import sys
sys.dont_write_bytecode = True
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import round7_stepD_common as d
from round7_stepD_common import c
from PIL import Image
import round7_stepD_report as report
import round7_stepD_verify as verifier


def fixture():
    records={}; parts={}
    for part,gens in [('expert_fit',['Real','Real','SD15','SD15']),('selection',['Real','Real','SD15','SD15']),('development',['Real','Real','ADM','ADM','BigGAN','BigGAN'])]:
        parts[part]=[]
        for gen in gens:
            identity=f'train:{len(records)}'; parts[part].append(identity)
            records[identity]=dict(generator=gen,label=int(gen!='Real'),source_id=identity)
    return records,parts,{'train:999'}


def predicted(variant='W1',seed=20261101,tuning=False):
    records,parts,deny=fixture(); rows=[]
    for part in (['selection'] if tuning else ['selection','development']):
        for cond in (c.TUNE if tuning else c.CONDITIONS):
            for i in parts[part]:
                r=records[i]; z=2*r['label']-1.
                if tuning:
                    rows.append(dict(variant=variant,seed=seed,partition=part,condition=cond,source_id=i,generator=r['generator'],label=r['label'],logit=z,view='format_matched'))
                else:
                    rows+=d.prediction_rows([dict(condition=cond,source_id=i,generator=r['generator'],label=r['label'],logit=z)],variant,seed,part,float(d.expit(1.)))
    return rows


class StepDTests(unittest.TestCase):
    def setUp(self):
        self.previous=(c.SEED,c.OUT,c.ART)
        self.protocol=c.read(d.PROTOCOL)

    def tearDown(self):
        c.SEED,c.OUT,c.ART=self.previous

    def test_quality_label_epoch_order_independent_and_uniform_range(self):
        ids=[f'train:{i}' for i in range(7200)]
        first={i:d.quality(20261101,i) for i in ids}
        self.assertEqual(first,{i:d.quality(20261101,i) for i in reversed(ids)})
        self.assertEqual(set(first.values()),set(range(60,96)))
        # A loose deterministic balance check detects a constant or truncated draw.
        counts=np.bincount(list(first.values()),minlength=96)[60:]
        self.assertTrue(np.all((counts>140)&(counts<260)))
        self.assertNotEqual(list(first.values()),[d.quality(20261102,i) for i in ids])
        image=Image.fromarray(np.random.default_rng(5).integers(0,256,(70,90,3),dtype=np.uint8))
        a=d.match_format(image,20261101,ids[0],self.protocol['format_match'])
        b=d.match_format(image,20261101,ids[0],self.protocol['format_match'])
        self.assertEqual(a.tobytes(),b.tobytes())
        self.assertEqual(a.size,image.size)

    def test_native_matching_precedes_geometry_and_reporting_is_original(self):
        records,parts,deny=fixture(); guard=d.Guard(records,parts,deny)
        im=Image.fromarray(np.random.default_rng(17).integers(0,256,(301,497,3),dtype=np.uint8))
        identity=parts['selection'][0]
        with patch.object(d,'load_original',return_value=im):
            for cond in c.CONDITIONS:
                original=d.view(identity,cond,False,records,guard,self.protocol)
                expected=d.images.view(im,'original',cond)
                self.assertEqual(original.tobytes(),expected.tobytes())
                matched=d.view(identity,cond,True,records,guard,self.protocol)
                expected=d.images.view(d.match_format(im,c.SEED,identity,self.protocol['format_match']),'original',cond)
                self.assertEqual(matched.tobytes(),expected.tobytes())
            with patch.object(d,'match_format',side_effect=AssertionError('Reporting must not match')):
                d.view(parts['development'][0],'clean',False,records,guard,self.protocol)
            with self.assertRaisesRegex(AssertionError,'development'):
                d.view(parts['development'][0],'clean',True,records,guard,self.protocol)

    def test_forbidden_access_fails_before_file_or_image_read(self):
        records,parts,deny=fixture(); guard=d.Guard(records,parts,deny)
        with patch.object(Image,'open',side_effect=RuntimeError('Must not decode')),patch.object(c,'sha',side_effect=RuntimeError('Must not read')):
            with self.assertRaisesRegex(AssertionError,'TEST ISOLATION'):
                d.load_original('train:999',records,guard)
        with self.assertRaisesRegex(AssertionError,'Wrong partition'):
            d.Guard(records,parts,deny).touch(parts['development'],'tune_threshold')

    def test_seed_context_excludes_unused_pools_and_never_opens_test_manifest(self):
        original=Path.open
        def check(path,*args,**kwargs):
            if path.name=='reserved_test_manifest.jsonl':
                raise AssertionError('Reserved test manifest opened')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',check):
            for seed in d.seeds.SEEDS:
                records,parts,deny=d.configure(seed,'W1')
                self.assertEqual(len(records),2650)
                self.assertEqual(set(parts),{'expert_fit','selection','development'})
                self.assertFalse(set(records)&deny)

    def test_independent_metrics_and_bad_rows_rejected(self):
        records,parts,deny=fixture(); rows=predicted(); tune=predicted(tuning=True)
        verifier.check_rows(rows,records,parts,deny,'W1',20261101)
        verifier.check_rows(tune,records,parts,deny,'W1',20261101,True)
        verifier.compare_metrics(d.metrics(rows),verifier.recompute(rows))
        self.assertTrue(all(r['auroc']==1 and r['balanced_accuracy']==1 and r['real_false_positive_rate']==0 for r in d.metrics(rows)))
        for field,value in [('source_id','train:999'),('view','format_matched'),('seed',20261102),('probability_fake',.5)]:
            bad=copy.deepcopy(rows); bad[0][field]=value
            with self.assertRaises(AssertionError):
                verifier.check_rows(bad,records,parts,deny,'W1',20261101)
        with self.assertRaises(AssertionError):
            verifier.check_rows(rows[:-1],records,parts,deny,'W1',20261101)

    def test_success_boundaries_and_every_seed_requirement(self):
        rows=[]
        for v in ('W0','W1','W2'):
            for seed in d.seeds.SEEDS:
                for g,part in d.DOMAINS.items():
                    for cond in c.CONDITIONS:
                        auc=.7 if v=='W0' else .68 if cond=='clean' else .5
                        rows.append(dict(variant=v,seed=seed,generator=g,partition=part,condition=cond,auroc=auc,balanced_accuracy=.6,real_false_positive_rate=.4))
        tables,success=report.tables(rows)
        self.assertTrue(all(s['all_criteria_every_seed']=='PASS' for s in success.values()))
        self.assertEqual([len(tables[k]) for k in ('metrics_each_seed.csv','metrics_mean_sd.csv','paired_differences.csv','criteria.csv','clean_SD15_cost.csv')],[108,108,108,18,6])
        next(r for r in rows if r['variant']=='W1' and r['seed']==20261103 and r['generator']=='ADM' and r['condition']=='jpeg50')['auroc']=.49
        _,success=report.tables(rows)
        self.assertEqual(success['W1']['all_criteria_every_seed'],'FAIL')
        self.assertEqual(success['W1']['every_seed']['20261101'],'PASS')
        self.assertEqual(success['W2']['all_criteria_every_seed'],'PASS')

    def test_summary_sd_paired_differences_and_cost_sign(self):
        rows=[]
        for v in ('W0','W1','W2'):
            for seed in d.seeds.SEEDS:
                values=d.metrics(predicted(v,seed))
                for r in values:
                    slope={'W0':.01,'W1':.03,'W2':.02}[v]
                    r.update(auroc=.6+slope*d.seeds.number(seed),balanced_accuracy=.5,real_false_positive_rate=.1+slope*d.seeds.number(seed))
                rows+=values
        tables,_=report.tables(rows)
        pair=next(r for r in tables['paired_differences.csv'] if r['contrast']=='W1 - W0' and r['metric']=='auroc')
        self.assertAlmostEqual(pair['mean'],.04); self.assertAlmostEqual(pair['sd'],.02)
        cost=next(r for r in tables['clean_SD15_cost.csv'] if r['variant']=='W1' and r['metric']=='auroc')
        self.assertAlmostEqual(cost['mean'],-.04)
        fpr=next(r for r in tables['clean_SD15_cost.csv'] if r['variant']=='W1' and r['metric']=='real_false_positive_rate')
        self.assertAlmostEqual(fpr['mean'],.04)
        with self.assertRaises(AssertionError):
            report.tables(rows[:-1])

    def test_atomic_resume_refuses_changed_outputs(self):
        d.ART.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='synthetic_',dir=d.ART) as temp:
            path=Path(temp)/'predictions.csv'
            d.publish(path,b'first')
            d.publish(path,b'first',True)
            with self.assertRaises(FileExistsError):
                d.publish(path,b'first')
            with self.assertRaises(AssertionError):
                d.publish(path,b'changed',True)
            self.assertEqual(path.read_bytes(),b'first')

    def test_w1_head_resume_and_separate_tuning_reporting_paths(self):
        import round7_stepD_run as run
        records,parts,deny=fixture(); calls=[]
        def features(self_,name,ids,condition,matched):
            calls.append((name,matched))
            # Reporting features deliberately differ: threshold must still come from tuning.
            return np.array([[records[i]['label']-(0 if matched else .4), .1] for i in sorted(ids)],dtype=np.float64)
        d.ART.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='synthetic_w1_',dir=d.ART) as temp:
            c.OUT=Path(temp)/'results'; c.ART=Path(temp)/'artifacts'
            c.write(c.OUT/'protocol.json',{'synthetic_w1':True})
            with patch.object(run.GlobalFeatures,'get',features):
                a,tuning,settings=run.w1(records,parts,d.Guard(records,parts,deny),self.protocol,False)
                self.assertTrue(all(m for name,m in calls if name=='expert_fit' or name=='selection_tuning'))
                self.assertTrue(all(not m for name,m in calls if name.endswith('_reporting')))
                calls.clear()
                b,_,again=run.w1(records,parts,d.Guard(records,parts,deny),self.protocol,True)
            self.assertNotIn(('expert_fit',True),calls)
            self.assertEqual(a,b); self.assertEqual(settings,again)
            cut=verifier.independently_threshold([r['label'] for r in tuning],d.expit([r['logit'] for r in tuning]))
            original=[r for r in a if r['partition']=='selection' and r['condition'] in c.TUNE]
            wrong=verifier.independently_threshold([r['label'] for r in original],[r['probability_fake'] for r in original])
            self.assertEqual(settings['threshold'],cut)
            self.assertNotAlmostEqual(cut,wrong)

    def test_runner_partial_publication_resumes_and_verifies(self):
        import round7_stepD_run as run
        records,parts,deny=fixture()
        d.ART.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='synthetic_run_',dir=d.ART) as temp:
            base=Path(temp)
            def configure(seed,variant):
                c.SEED=seed; c.OUT=base/'results/seed1'/variant; c.ART=base/'artifacts/seed1'/variant
                return records,parts,deny
            def fit(*args):
                guard=args[2]
                guard.touch(parts['expert_fit'],'fit_clip_expert')
                guard.touch(parts['selection'],'tune_threshold')
                for part in ('selection','development'):
                    guard.touch(parts[part],'report/'+part)
                return predicted(),predicted(tuning=True),{'threshold':float(d.expit(1.)),'tuning_view':'format_matched'}
            original_publish=d.publish
            interrupted=[False]
            def interrupt(path,payload,resume=False):
                if Path(path).name=='tuning_predictions.csv' and not interrupted[0]:
                    interrupted[0]=True
                    raise RuntimeError('synthetic publication interruption')
                original_publish(path,payload,resume)
            with patch.object(d,'OUT',base/'results'),patch.object(d,'configure',side_effect=configure),patch.object(d,'lock_implementation',return_value={}):
                d.publish(d.OUT/'implementation.json',d.json_bytes({}))
                with patch.object(run,'w1',side_effect=fit),patch.object(d,'publish',side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError,'publication interruption'):
                        run.execute(20261101,'W1',True)
                existing=(c.OUT/'predictions.csv').read_bytes()
                with patch.object(run,'w1',side_effect=AssertionError('Must resume bundle without fitting')):
                    run.execute(20261101,'W1',True)
                    self.assertEqual(existing,(c.OUT/'predictions.csv').read_bytes())
                    verifier.verify_run(20261101,'W1')
                    run.execute(20261101,'W1',True)

    def test_resnet_resume_and_matched_tuning_batch(self):
        import torch
        import round7_baseline_resnet as rn
        records,parts,deny=fixture()
        def image(identity,*args):
            return Image.fromarray(np.full((40,60,3),40+records[identity]['label']*100,dtype=np.uint8))
        def factory():
            return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1,1)),torch.nn.Flatten(),torch.nn.Dropout(.2),torch.nn.Linear(3,1))
        d.ART.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='synthetic_resume_',dir=d.ART) as temp:
            c.OUT=Path(temp); c.SEED=20261102
            c.write(c.OUT/'protocol.json',{'synthetic':True})
            spec={'lr':1e-4,'weight_decay':1e-4,'epochs':2,'batch_size':4}
            batch=d.tensor_batch(True,self.protocol)
            with patch.object(d,'load_original',side_effect=image):
                g=d.Guard(records,parts,deny)
                a,sa=rn.fit('original',records,parts,g,spec,factory=factory,batch_fn=batch,device='cpu',checkpoint_dir=c.OUT/'full')
                rn.fit('original',records,parts,d.Guard(records,parts,deny),spec,factory=factory,batch_fn=batch,device='cpu',checkpoint_dir=c.OUT/'resume',stop_after=1)
                b,sb=rn.fit('original',records,parts,d.Guard(records,parts,deny),spec,factory=factory,batch_fn=batch,device='cpu',checkpoint_dir=c.OUT/'resume',resume=True)
                self.assertEqual(set(g.events['format_match']),set(parts['expert_fit']+parts['selection']))
                self.assertFalse(set(parts['development']) & set(g.events['format_match']))
            for key in a.state_dict():
                torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key],rtol=0,atol=0)
            self.assertEqual(sa['best_epoch'],sb['best_epoch'])
            self.assertEqual([r['train_loss'] for r in sa['history']],[r['train_loss'] for r in sb['history']])


if __name__=='__main__':
    unittest.main(verbosity=2)
