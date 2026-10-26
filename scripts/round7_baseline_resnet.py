"""Fully fine-tuned ResNet50 with immutable epoch checkpoints and exact resume."""
import sys
sys.dont_write_bytecode = True
import gc
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import random
import time
import numpy as np
import round7_baseline_common as c
import round7_baseline_data as images
import score_calibration as calibration
from scipy.special import expit
from sklearn.metrics import roc_auc_score


def seed_everything(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def rng_state():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import torch
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def model_factory():
    import torch
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    path = c.ART / "weights/resnet50-11ad3fa6.pth"
    images.download(weights.url, path, prefix="11ad3fa6")
    model = resnet50(weights=None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.requires_grad_(True)
    return model


def tensor_batch(ids, profile, condition, records, guard, device):
    import torch
    guard.touch(ids, "resnet_input")
    x = np.stack([images.normalized(images.view(images.load_image(i, records, guard), profile, condition), "resnet") for i in ids])
    y = np.array([records[i]["label"] for i in ids], dtype=np.float32)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def train_epoch(model, optimizer, ids, epoch, batch_size, profile, records, guard, device, batch_fn=tensor_batch):
    import torch
    guard.touch(ids, "fit_resnet_expert_only")
    order = np.random.default_rng(c.SEED + epoch).permutation(len(ids))
    model.train()
    total, correct = 0., 0
    for start in range(0, len(ids), batch_size):
        chosen = [ids[i] for i in order[start:start+batch_size]]
        x, y = batch_fn(chosen, profile, "clean", records, guard, device)
        optimizer.zero_grad(set_to_none=True)
        z = model(x).flatten()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
        c.require(bool(torch.isfinite(loss)), "Nonfinite ResNet loss")
        loss.backward(); optimizer.step()
        total += float(loss.detach()) * len(y)
        correct += int(((z.detach() >= 0) == (y >= .5)).sum())
    return total / len(ids), correct / len(ids)


def predict(model, ids, conditions, batch_size, profile, records, guard, device, batch_fn=tensor_batch, timings=None, partition=None):
    import torch
    guard.touch(ids, "resnet_predict_input")
    model.eval(); rows = []
    with torch.inference_mode():
        for condition in conditions:
            started = time.perf_counter()
            if timings is not None and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            for start in range(0, len(ids), batch_size):
                chosen = ids[start:start+batch_size]
                x, _ = batch_fn(chosen, profile, condition, records, guard, device)
                z = model(x).flatten().cpu().numpy()
                c.require(np.isfinite(z).all(), "Nonfinite ResNet logits")
                rows.extend(dict(source_id=i, generator=records[i]["generator"], label=records[i]["label"], condition=condition, logit=float(v)) for i, v in zip(chosen, z))
            if timings is not None:
                timings.append(dict(model="resnet50", profile=profile, partition=partition, condition=condition, seconds=time.perf_counter()-started,
                                    peak_gpu_bytes=int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                                    shared_across_calibration_modes=True))
    return rows


def selection_score(rows):
    scores = []
    for condition in c.TUNE:
        cell = [r for r in rows if r["condition"] == condition]
        scores.append(roc_auc_score([r["label"] for r in cell], [r["logit"] for r in cell]))
    return float(np.mean(scores))


def load_checkpoint(path, model, optimizer, scheduler, profile, identity):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)  # own local checkpoint only
    c.require(state["profile"] == profile and state["protocol_sha256"] == identity, "Checkpoint protocol mismatch")
    model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"]); restore_rng(state["rng"])
    return state


def fit(profile, records, parts, guard, spec, resume=False, factory=model_factory, batch_fn=tensor_batch, device=None, checkpoint_dir=None, stop_after=None):
    """stop_after/factory/batch_fn are synthetic-test hooks, not CLI overrides."""
    import torch
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    seed_everything(c.SEED)
    model = factory().to(device)
    c.require(all(p.requires_grad for p in model.parameters()), "All ResNet layers must be fine-tuned")
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=spec["epochs"], eta_min=0.)
    folder = checkpoint_dir or c.ART / "checkpoints" / profile
    folder.mkdir(parents=True, exist_ok=True)
    identity = c.sha(c.OUT / "protocol.json")
    existing = sorted(folder.glob("epoch_*.pt"))
    if existing and not resume:
        raise FileExistsError("Checkpoints already exist; use --resume")
    state = {"epoch": 0, "batch_size": spec["batch_size"], "best_epoch": 0, "best_score": -1., "history": [], "oom_events": []}
    def save(epoch):
        c.new_checkpoint(folder / f"epoch_{epoch:03d}.pt", dict(state, profile=profile, protocol_sha256=identity,
                         model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state()))
    if existing:
        state = load_checkpoint(existing[-1], model, optimizer, scheduler, profile, identity)
        # Do not retain full model/optimizer tensors twice during training.
        state = {k: v for k, v in state.items() if k not in ("model", "optimizer", "scheduler", "rng", "profile", "protocol_sha256")}
        guard.touch(parts["expert_fit"], "resnet_resume_expert_partition")
        guard.touch(parts["selection"], "resnet_resume_selection_partition")
    else:
        save(0)
    for epoch in range(state["epoch"] + 1, spec["epochs"] + 1):
        batch_size = state["batch_size"]
        events = list(state["oom_events"])
        while True:
            try:
                with c.Timer() as timer:
                    loss, accuracy = train_epoch(model, optimizer, sorted(parts["expert_fit"]), epoch, batch_size, profile, records, guard, device, batch_fn)
                    guard.touch(parts["selection"], "select_resnet_epoch")
                    selected = predict(model, sorted(parts["selection"]), c.TUNE, batch_size, profile, records, guard, device, batch_fn)
                    score = selection_score(selected)
                    scheduler.step()
                break
            except torch.cuda.OutOfMemoryError:
                if not str(device).startswith("cuda") or batch_size <= 1:
                    raise
                new_size = max(1, batch_size // 2)
                events.append({"epoch": epoch, "from": batch_size, "to": new_size, "reason": "CUDA OOM", "rollback_to_epoch": epoch-1})
                batch_size = new_size
            optimizer.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
            restored = load_checkpoint(folder / f"epoch_{epoch-1:03d}.pt", model, optimizer, scheduler, profile, identity)
            state = {k: v for k, v in restored.items() if k not in ("model", "optimizer", "scheduler", "rng", "profile", "protocol_sha256")}
            del restored
        state.update(epoch=epoch, batch_size=batch_size, oom_events=events)
        if score > state["best_score"]:
            state.update(best_score=score, best_epoch=epoch)
        state["history"].append(dict(epoch=epoch, train_loss=loss, train_accuracy=accuracy, selection_macro_auroc=score,
                                     batch_size=batch_size, next_lr=optimizer.param_groups[0]["lr"], **timer.result()))
        save(epoch)
        print(f"ResNet {profile} epoch {epoch}/{spec['epochs']}: loss={loss:.5f}, selection AUC={score:.4f}, best={state['best_epoch']}, batch={batch_size}", flush=True)
        if stop_after is not None and epoch >= stop_after:
            return model, state
    best = torch.load(folder / f"epoch_{state['best_epoch']:03d}.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    return model, state


def run(profile, records, parts, guard, protocol, resume):
    import torch
    start = time.perf_counter()
    model, state = fit(profile, records, parts, guard, protocol["resnet"], resume=resume)
    device = str(next(model.parameters()).device)
    timings = []
    with c.Timer() as evaluation:
        guard.touch(parts["calibration"], "fit_resnet_temperature")
        cal = predict(model, sorted(parts["calibration"]), c.TUNE, state["batch_size"], profile, records, guard, device, timings=timings, partition="calibration")
        temperature = calibration.fit([r["logit"] for r in cal], [r["label"] for r in cal])
        predictions, settings = [], {}
        all_rows = {part: predict(model, sorted(parts[part]), c.CONDITIONS, state["batch_size"], profile, records, guard, device, timings=timings, partition=part) for part in ("selection", "development")}
        tuning = [r for r in all_rows["selection"] if r["condition"] in c.TUNE]
        guard.touch(parts["selection"], "select_resnet_threshold")
        for cal_mode in ("raw", "temperature"):
            t = temperature["temperature"] if cal_mode == "temperature" else 1.
            cut = c.threshold([r["label"] for r in tuning], expit(np.array([r["logit"] for r in tuning]) / t))
            settings[cal_mode] = dict(threshold=cut, temperature=t)
            for part, items in all_rows.items():
                guard.touch(parts[part], "predict/" + part)
                for row in items:
                    z = row["logit"] / t
                    predictions.append(dict(profile=profile, model="resnet50", calibration=cal_mode, seed=c.SEED, partition=part,
                                            **{k: v for k, v in row.items() if k != "logit"}, logit=float(z), probability_fake=float(expit(z)), threshold=cut, alpha_global=""))
    c.write_csv(c.OUT / f"resnet_{profile}_predictions.csv", predictions)
    c.write(c.OUT / f"resnet_{profile}_settings.json", dict(settings=settings, temperature=temperature, best_epoch=state["best_epoch"], best_selection_macro_auroc=state["best_score"], history=state["history"], oom_events=state["oom_events"]))
    checkpoints = {str(p.relative_to(c.ROOT)).replace('\\', '/'): c.sha(p) for p in sorted((c.ART / "checkpoints" / profile).glob("epoch_*.pt"))}
    c.write(c.OUT / f"resnet_{profile}_resources.json", {"seconds_this_invocation": time.perf_counter()-start, "evaluation": {"seconds": evaluation.seconds, "peak_gpu_bytes": max(r["peak_gpu_bytes"] for r in timings)},
             "training_seconds_all_epochs": sum(r["seconds"] for r in state["history"]), "peak_gpu_bytes": max([r["peak_gpu_bytes"] for r in state["history"]] + [r["peak_gpu_bytes"] for r in timings]),
             "per_condition_inference": timings,
             "checkpoint_sha256": checkpoints, "weights_sha256": c.sha(c.ART / "weights/resnet50-11ad3fa6.pth")})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
