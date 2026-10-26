"""Synthetic Step C tests: no real images, weights, features or long training."""
import sys
sys.dont_write_bytecode = True
import ast
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import round7_baseline_common as c
import round7_baseline_data as data
import round7_baseline_clip as clip
import round7_baseline_resnet as rn
import round7_verify_baselines as verifier
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score
from scipy.special import expit
from scipy.ndimage import laplace


def fixture():
    records, parts = {}, {}
    index = 0
    for part, n in [("expert_fit", 8), ("gate_fit", 4), ("calibration", 4), ("selection", 4), ("development", 6)]:
        parts[part] = []
        for j in range(n):
            identity = f"train:{index}"
            generator = ("Real" if j < 2 else "ADM" if j < 4 else "BigGAN") if part == "development" else ("Real" if j % 2 == 0 else "SD15")
            records[identity] = dict(source_id=identity, generator=generator, label=int(generator != "Real"))
            parts[part].append(identity); index += 1
    return records, parts, {"train:999"}


def old_functions(path, names, globals_):
    module = ast.parse(path.read_text(encoding="utf-8"))
    selected = ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    exec(compile(selected, str(path), "exec"), globals_)
    return globals_


def batch(ids, profile, condition, records, guard, device):
    guard.touch(ids, "synthetic_resnet_batch")
    x = torch.tensor([[int(i.split(':')[1]) / 30, records[i]["label"], .3] for i in ids], dtype=torch.float32)
    return x, torch.tensor([records[i]["label"] for i in ids], dtype=torch.float32)


def tiny_model():
    return torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.ReLU(), torch.nn.Dropout(.2), torch.nn.Linear(5, 1))


class BaselineTests(unittest.TestCase):
    def test_geometry_degradation_quality_and_shuffle_match_round6(self):
        rng = np.random.default_rng(42)
        im = Image.fromarray(rng.integers(0, 256, (300, 500, 3), dtype=np.uint8))
        old = old_functions(c.ROOT / "scripts/run_pilot.py", ["degrade", "quality"], {"Image": Image, "np": np, "io": io, "laplace": laplace})
        local = old_functions(c.ROOT / "scripts/local_branch.py", ["permutation", "shuffle_image"], {"np": np, "Image": Image, "PATCH_SIZE": 56})
        for profile in c.PROFILES:
            base = ImageOps.fit(im, (224,224) if profile == "original" else (128,128), method=Image.Resampling.BICUBIC)
            if profile == "common128":
                base = base.resize((224,224), Image.Resampling.BICUBIC)
            for condition in c.CONDITIONS:
                actual = data.view(im, profile, condition)
                expected = old["degrade"](base, condition)
                self.assertEqual(actual.tobytes(), expected.tobytes())
                np.testing.assert_array_equal(data.quality(actual), old["quality"](expected))
                for seed in data.VIEW_SEEDS:
                    self.assertEqual(data.shuffled(actual, seed).tobytes(), local["shuffle_image"](expected, local["permutation"](seed)).tobytes())

    def test_guard_fails_before_any_image_access(self):
        records, parts, deny = fixture()
        guard = c.Guard(records, deny)
        with patch.object(Image, "open", side_effect=RuntimeError("Image must never be opened")):
            with self.assertRaisesRegex(AssertionError, "TEST ISOLATION"):
                data.load_image("train:999", records, guard)
            with self.assertRaisesRegex(AssertionError, "TEST ISOLATION"):
                data.load_image("train:123456", records, guard)

    def test_context_never_opens_reserved_test_manifest(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.name == "reserved_test_manifest.jsonl":
                raise AssertionError("Reserved manifest opened")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            records, parts, deny = c.context()
        self.assertEqual(len(records), 3400)
        self.assertFalse(set(records) & deny)

    def test_vectorized_bootstrap_matches_sklearn_with_ties(self):
        y = np.repeat([0,1], 4)
        z = np.array([.1, .1, -.3, .7, .1, .7, .7, -.3])
        actual = c.bootstrap(y, z, .5, 80)
        rng = np.random.default_rng(20261022)
        draws = np.concatenate([rng.integers(0,4,(80,4)), rng.integers(4,8,(80,4))], axis=1)
        expected = {k: [] for k in actual}
        for ix in draws:
            expected["auroc"].append(roc_auc_score(y[ix],z[ix]))
            expected["average_precision"].append(average_precision_score(y[ix],z[ix]))
            expected["balanced_accuracy"].append(balanced_accuracy_score(y[ix],expit(z[ix])>=.5))
            expected["real_false_positive_rate"].append(np.mean(expit(z[ix[:4]])>=.5))
        for key in actual:
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-14, rtol=0)

    def test_clip_heads_gates_calibration_synthetic_smoke(self):
        records, parts, deny = fixture()
        rng = np.random.default_rng(12)
        features = {}
        for part, ids in parts.items():
            conditions = ["clean"] if part == "expert_fit" else c.TUNE if part in ("gate_fit", "calibration") else c.CONDITIONS
            y = np.array([records[i]["label"] for i in sorted(ids)])
            for condition in conditions:
                features[part,condition] = dict(g=rng.normal(size=(len(ids),5)) + y[:,None], l=rng.normal(size=(len(ids),4,5)) + y[:,None,None], q=rng.normal(size=(len(ids),3)))
        guard = c.Guard(records, deny)
        with patch.object(clip.gates, "MAX_EPOCHS", 2), patch.object(clip.gates, "PATIENCE", 1):
            pred, settings, runtime, heads = clip.fit_predict("original", features, records, parts, guard)
        self.assertEqual(len(pred), 800)
        self.assertEqual(len(settings["gates"]), 12)
        self.assertFalse(guard.violations)
        self.assertEqual(guard.events["fit_clip_experts"], set(parts["expert_fit"]))
        self.assertEqual(guard.events["fit_clip_temperatures"], set(parts["calibration"]))
        self.assertEqual(guard.events["fit_quality_gate"], set(parts["gate_fit"]))
        self.assertTrue(all(np.isfinite(r["logit"]) for r in pred))

    def test_resnet_checkpoint_resume_matches_uninterrupted(self):
        records, parts, deny = fixture()
        c.ART.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="synthetic_", dir=c.ART) as temp:
            root = Path(temp)
            with patch.object(c, "OUT", root):
                c.write(root/"protocol.json", {"synthetic_test": True})
                spec=dict(lr=1e-4,weight_decay=1e-4,epochs=2,batch_size=4)
                a, sa = rn.fit("original",records,parts,c.Guard(records,deny),spec,factory=tiny_model,batch_fn=batch,device="cpu",checkpoint_dir=root/"full")
                rn.fit("original",records,parts,c.Guard(records,deny),spec,factory=tiny_model,batch_fn=batch,device="cpu",checkpoint_dir=root/"resumed",stop_after=1)
                b, sb = rn.fit("original",records,parts,c.Guard(records,deny),spec,resume=True,factory=tiny_model,batch_fn=batch,device="cpu",checkpoint_dir=root/"resumed")
                for key in a.state_dict():
                    torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], atol=0,rtol=0)
                self.assertEqual(sa["best_epoch"],sb["best_epoch"])
                self.assertEqual([h["train_loss"] for h in sa["history"]],[h["train_loss"] for h in sb["history"]])
                self.assertEqual(len(list((root/"resumed").glob('epoch_*.pt'))),3)
                with self.assertRaises(FileExistsError):
                    rn.fit("original",records,parts,c.Guard(records,deny),spec,factory=tiny_model,batch_fn=batch,device="cpu",checkpoint_dir=root/"resumed")

    def test_metrics_coverage_and_forbidden_prediction_rejected(self):
        records, parts, deny = fixture()
        predictions = []
        for profile in c.PROFILES:
            for model in ["global","local","average","fixed","score_gate","quality_gate","resnet50"]:
                seeds = c.GATE_SEEDS if model.endswith("gate") else [c.SEED] if model=="resnet50" else [-1]
                for cal in ["raw","temperature"]:
                    for seed in seeds:
                        for part in ["selection","development"]:
                            for cond in c.CONDITIONS:
                                for identity in parts[part]:
                                    row=records[identity]; z=2*row["label"]-1.
                                    predictions.append(dict(profile=profile,model=model,calibration=cal,seed=seed,partition=part,condition=cond,**row,logit=z,probability_fake=float(expit(z)),threshold=float(expit(1.)),alpha_global=.5))
        verifier.verify_predictions(predictions,records,parts,deny)
        metrics,macro,weights=c.summarize(predictions,draws=20)
        self.assertEqual((len(metrics),len(macro),len(weights)),(528,84,288))
        self.assertTrue(all(r["macro_auroc"]==1 for r in macro))
        bad=copy.deepcopy(predictions);bad[0]["source_id"]="train:999"
        with self.assertRaisesRegex(AssertionError,"forbidden"):
            verifier.verify_predictions(bad,records,parts,deny)

    def test_oom_rolls_back_entire_epoch_before_halving_batch(self):
        records, parts, deny = fixture()
        c.ART.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="synthetic_oom_", dir=c.ART) as temp:
            root=Path(temp)
            with patch.object(c,"OUT",root):
                c.write(root/"protocol.json",{"synthetic_oom_test":True})
                spec=dict(lr=1e-4,weight_decay=1e-4,epochs=2,batch_size=2)
                expected,state_expected=rn.fit("original",records,parts,c.Guard(records,deny),spec,factory=tiny_model,batch_fn=batch,device="cpu",checkpoint_dir=root/"expected")
                def pretend_cuda_factory():
                    model=tiny_model()
                    # Emulate the CUDA OOM branch while keeping this fixture on CPU.
                    model.to=lambda *args,**kwargs:model
                    return model
                original=rn.train_epoch
                calls=[0]
                def fail_after_mutation(model,*args,**kwargs):
                    calls[0]+=1
                    if calls[0]==1:
                        with torch.no_grad():
                            next(model.parameters()).add_(123)
                        raise torch.cuda.OutOfMemoryError("Synthetic OOM after partial epoch mutation")
                    return original(model,*args,**kwargs)
                with patch.object(rn,"train_epoch",fail_after_mutation):
                    actual,state_actual=rn.fit("original",records,parts,c.Guard(records,deny),dict(spec,batch_size=4),factory=pretend_cuda_factory,batch_fn=batch,device="cuda:0",checkpoint_dir=root/"oom")
                self.assertEqual(state_actual["oom_events"],[dict(epoch=1,**{"from":4,"to":2},reason="CUDA OOM",rollback_to_epoch=0)])
                self.assertEqual([h["train_loss"] for h in state_actual["history"]],[h["train_loss"] for h in state_expected["history"]])
                for key in expected.state_dict():
                    torch.testing.assert_close(actual.state_dict()[key],expected.state_dict()[key],atol=0,rtol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
