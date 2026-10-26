"""Frozen Round 6 CLIP experts/fusions on Step C allowlisted partitions."""
import sys
sys.dont_write_bytecode = True
import os
import time
import numpy as np
import round7_baseline_common as c
import round7_baseline_data as images
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import gate_models as gates
import score_calibration as calibration


def encode(model, pictures, device):
    import torch
    array = np.stack([images.normalized(im, "clip") for im in pictures])
    x = torch.from_numpy(array).to(device=device, dtype=next(model.parameters()).dtype)
    with torch.inference_mode():
        z = model(pixel_values=x).image_embeds.float()
        return torch.nn.functional.normalize(z, dim=-1).cpu().numpy()


def extract(profile, records, parts, guard, protocol):
    import torch
    os.environ.setdefault("HF_HOME", str(c.ART / "hf"))
    try:
        from transformers import CLIPVisionModelWithProjection
    except ImportError as error:
        raise RuntimeError("Install repository-pinned transformers==4.39.3 using the notebook setup cell") from error
    weights = c.ART / "weights/clip"
    spec = protocol["clip"]
    for name in ("config.json", "pytorch_model.bin"):
        images.download(f"https://huggingface.co/{spec['model']}/resolve/{spec['revision']}/{name}", weights / name,
                        digest=spec["weights_sha256"] if name.endswith(".bin") else None)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(c.SEED); torch.set_num_threads(4)
    model = None
    features, provenance, condition_runtime = {}, {}, []
    with c.Timer() as timer:
        for part, identities in parts.items():
            conditions = ["clean"] if part == "expert_fit" else c.TUNE if part in ("gate_fit", "calibration") else c.CONDITIONS
            ids = sorted(identities)
            for condition in conditions:
                condition_start = time.perf_counter()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                guard.touch(ids, "clip_features/" + part)
                path = c.ART / "features" / f"{profile}_{part}_{condition}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    with np.load(path, allow_pickle=False) as cache:
                        c.require(cache["ids"].tolist() == ids and str(cache["protocol"]) == c.sha(c.OUT / "protocol.json"), "Feature cache provenance mismatch")
                        features[part, condition] = {k: cache[k].copy() for k in ("g", "l", "q")}
                else:
                    if model is None:
                        model = CLIPVisionModelWithProjection.from_pretrained(weights, local_files_only=True).to(device).eval()
                        model.requires_grad_(False)
                        if device.startswith("cuda"):
                            model.half()
                    gg, ll, qq = [], [], []
                    for start in range(0, len(ids), 32):
                        batch = ids[start:start+32]
                        guard.touch(batch, "clip_encoder_input")
                        pictures = [images.view(images.load_image(i, records, guard), profile, condition) for i in batch]
                        gg.append(encode(model, pictures, device))
                        ll.append(np.stack([encode(model, [images.shuffled(im, seed) for im in pictures], device) for seed in images.VIEW_SEEDS], axis=1))
                        qq.extend(images.quality(im) for im in pictures)
                    features[part, condition] = {"g": np.concatenate(gg), "l": np.concatenate(ll), "q": np.asarray(qq)}
                    with path.open("xb") as f:
                        np.savez_compressed(f, ids=np.array(ids), protocol=np.array(c.sha(c.OUT / "protocol.json")), **features[part, condition])
                provenance[str(path.relative_to(c.ROOT)).replace('\\', '/')] = c.sha(path)
                condition_runtime.append(dict(partition=part, condition=condition, seconds=time.perf_counter()-condition_start,
                    peak_gpu_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                    shared_by="global/local/all fusions; both calibration modes and gate initializations"))
                print(f"CLIP {profile}: {part}/{condition} ready ({len(ids)} images)", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    extraction = dict(seconds=timer.seconds, peak_gpu_bytes=max(r["peak_gpu_bytes"] for r in condition_runtime), by_partition_condition=condition_runtime)
    return features, provenance, extraction


def head(x, y):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., solver="liblinear", max_iter=2000, random_state=20261018))
    model.fit(x, y, logisticregression__sample_weight=np.ones(len(y)))
    return model


def bundle(part, conditions, features, records, parts, gh, lh, guard, runtime=None):
    ids = sorted(parts[part])
    guard.touch(ids, "clip_head_input/" + part)
    rows, zs, ls, variances, qualities = [], [], [], [], []
    for condition in conditions:
        started = time.perf_counter()
        f = features[part, condition]
        zg = gh.decision_function(f["g"])
        zl = lh.decision_function(f["l"].mean(axis=1))
        width = f["l"].shape[-1]
        variance = lh.decision_function(f["l"].reshape(-1, width)).reshape(-1, 4).var(axis=1)
        rows += [dict(source_id=i, generator=records[i]["generator"], label=records[i]["label"], condition=condition) for i in ids]
        zs.extend(zg); ls.extend(zl); variances.extend(variance); qualities.extend(f["q"])
        if runtime is not None:
            runtime.append(dict(partition=part, condition=condition, seconds=time.perf_counter()-started, peak_gpu_bytes=0,
                                shared_by="global and local head logits; all fusions/calibrations/initializations"))
    return {"rows": rows, "zg": np.array(zs), "zl": np.array(ls), "variance": np.array(variances),
            "q": np.array(qualities), "y": np.array([r["label"] for r in rows], dtype=float)}


def transform(batch, tg, tl):
    result = dict(batch)
    result["zg"], result["zl"] = batch["zg"] / tg, batch["zl"] / tl
    result["variance"] = batch["variance"] / tl ** 2
    result["x"] = gates.inputs(result["zg"], result["zl"], result["variance"], batch["q"])
    return result


def fixed_alpha(batch):
    candidates = []
    for a in np.arange(11) / 10:
        z = a * batch["zg"] + (1-a) * batch["zl"]
        scores = []
        for cond in c.TUNE:
            mask = np.array([r["condition"] == cond for r in batch["rows"]])
            scores.append(roc_auc_score(batch["y"][mask], z[mask]))
        candidates.append((float(a), float(np.mean(scores))))
    best = max(score for _, score in candidates)
    return min((a for a, score in candidates if np.isclose(score, best)), key=lambda a: (abs(a-.5), a)), candidates


def fit_predict(profile, features, records, parts, guard):
    ids = sorted(parts["expert_fit"])
    guard.touch(ids, "fit_clip_experts")
    y = np.array([records[i]["label"] for i in ids])
    with c.Timer() as head_timer:
        f = features["expert_fit", "clean"]
        gh, lh = head(f["g"], y), head(f["l"].mean(axis=1), y)
    head_inference = []
    raw = {part: bundle(part, c.TUNE if part in ("gate_fit", "calibration") else c.CONDITIONS, features, records, parts, gh, lh, guard, head_inference)
           for part in ("gate_fit", "calibration", "selection", "development")}
    # Tuning uses only the inherited three conditions. JPEG50 is report-only.
    mask = np.array([r["condition"] in c.TUNE for r in raw["selection"]["rows"]])
    raw["tuning"] = {k: [v[i] for i in np.flatnonzero(mask)] if k == "rows" else v[mask] for k, v in raw["selection"].items()}
    guard.touch(parts["calibration"], "fit_clip_temperatures")
    temperatures = {name: calibration.fit(raw["calibration"][field], raw["calibration"]["y"])
                    for name, field in (("global", "zg"), ("local", "zl"))}
    predictions, settings, artifacts, runtime = [], {}, {}, []
    for cal in ("raw", "temperature"):
        tg, tl = [temperatures[name]["temperature"] if cal == "temperature" else 1. for name in ("global", "local")]
        batches = {name: transform(batch, tg, tl) for name, batch in raw.items()}
        tune = batches["tuning"]
        guard.touch(parts["selection"], "select_clip_fusion_threshold_epoch")
        fixed, search = fixed_alpha(tune)
        fitted = [(name, -1, {part: np.full(len(batch["y"]), a) for part, batch in batches.items()})
                  for name, a in (("global", 1.), ("local", 0.), ("average", .5), ("fixed", fixed))]
        for name, seed, _ in fitted:
            runtime.append(dict(model=name, calibration=cal, seed=seed, fit_seconds=head_timer.seconds, peak_gpu_bytes=head_timer.peak_gpu_bytes, note="Expert fitting cost shared across four simple fusions; feature extraction reported separately"))
        for seed in gates.SEEDS:
            for name in ("score_gate", "quality_gate"):
                guard.touch(parts["gate_fit"], "fit_" + name)
                with c.Timer() as timer:
                    model, scaler, artifact, history = gates.fit(batches["gate_fit"], tune, name, seed)
                    alpha = {part: gates.predict(model, scaler, batch["x"], name) for part, batch in batches.items()}
                artifacts[f"{cal}/{name}/{seed}"] = dict(artifact=artifact, history=history)
                fitted.append((name, seed, alpha))
                runtime.append(dict(model=name, calibration=cal, seed=seed, fit_seconds=timer.seconds, peak_gpu_bytes=timer.peak_gpu_bytes, note="Unchanged Round 6 full-batch CPU gate; feature/expert costs shared"))
        for name, seed, alpha in fitted:
            tz = alpha["tuning"] * tune["zg"] + (1-alpha["tuning"]) * tune["zl"]
            cut = c.threshold(tune["y"], expit(tz))
            settings[f"{cal}/{name}/{seed}"] = {"threshold": cut, "global_temperature": tg, "local_temperature": tl, "fixed_alpha": fixed, "fixed_search": search}
            for part in ("selection", "development"):
                guard.touch(parts[part], "predict/" + part)
                batch, a = batches[part], alpha[part]
                z = a * batch["zg"] + (1-a) * batch["zl"]
                for r, zz, aa in zip(batch["rows"], z, a):
                    predictions.append(dict(profile=profile, model=name, calibration=cal, seed=seed, partition=part, **r,
                                            logit=float(zz), probability_fake=float(expit(zz)), threshold=cut, alpha_global=float(aa)))
    return predictions, dict(settings=settings, temperatures=temperatures, gates=artifacts, shared_head_inference_by_condition=head_inference), runtime, {"global": gh, "local": lh}


def run(profile, records, parts, guard, protocol):
    import joblib
    started = time.perf_counter()
    with c.Timer() as timer:
        features, provenance, extraction = extract(profile, records, parts, guard, protocol)
        predictions, settings, runtime, experts = fit_predict(profile, features, records, parts, guard)
        path = c.ART / f"clip_{profile}_experts.joblib"
        with path.open("xb") as f:
            joblib.dump(experts, f)
    c.write_csv(c.OUT / f"clip_{profile}_predictions.csv", predictions)
    c.write(c.OUT / f"clip_{profile}_settings.json", settings)
    c.write(c.OUT / f"clip_{profile}_resources.json", {"total": {"seconds": time.perf_counter()-started, "peak_gpu_bytes": max([extraction["peak_gpu_bytes"]] + [r["peak_gpu_bytes"] for r in runtime])}, "shared_feature_extraction": extraction, "per_model": runtime,
             "shared_head_inference_by_condition": settings["shared_head_inference_by_condition"], "feature_sha256": provenance,
             "experts_sha256": c.sha(path)})
