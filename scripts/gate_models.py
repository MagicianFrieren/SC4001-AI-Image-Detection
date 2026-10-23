"""Small paired-initialization gates for frozen global/local experts."""
import copy

import numpy as np
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

SEEDS = [20261022, 20261023, 20261024]
MAX_EPOCHS = 400
PATIENCE = 40
LEARNING_RATE = .003


def inputs(zg, zl, variance, quality):
    pg, pl = expit(zg), expit(zl)
    def entropy(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -p * np.log(p) - (1 - p) * np.log1p(-p)
    return np.column_stack([pg, pl, np.abs(pg - pl), (entropy(pg) + entropy(pl)) / 2,
                            np.log1p(variance), quality])


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(8, 32)
        self.output = nn.Linear(32, 1)
        # Both variants begin at identical average fusion for every image.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return torch.sigmoid(self.output(torch.relu(self.hidden(x)))).flatten()


def fit(train, validation, kind, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = Gate()
    scaler = StandardScaler().fit(train['x'])
    def tensors(batch):
        x = scaler.transform(batch['x']).astype(np.float32)
        if kind == 'score_gate':
            x[:, 5:] = 0
        return [torch.from_numpy(a.astype(np.float32)) for a in [x, batch['zg'], batch['zl'], batch['y']]]
    tx, tg, tl, ty = tensors(train)
    vx, vg, vl, vy = tensors(validation)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_loss, best_epoch, best_state = float('inf'), 0, None
    history = []
    for epoch in range(MAX_EPOCHS + 1):
        if epoch:
            model.train()
            optimizer.zero_grad()
            a = model(tx)
            loss = criterion(a * tg + (1 - a) * tl, ty)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            ta, va = model(tx), model(vx)
            train_loss = float(criterion(ta * tg + (1 - ta) * tl, ty))
            val_loss = float(criterion(va * vg + (1 - va) * vl, vy))
        if not np.isfinite(train_loss + val_loss):
            raise ValueError('Non-finite gate loss')
        history.append({'epoch': epoch, 'train_bce': train_loss, 'validation_bce': val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    artifact = {'kind': kind, 'seed': seed, 'best_epoch': best_epoch, 'stopped_epoch': history[-1]['epoch'],
                'best_validation_bce': best_loss, 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                'state': {k: v.numpy().tolist() for k, v in best_state.items()}}
    return model, scaler, artifact, history


def predict(model, scaler, x, kind):
    scaled = scaler.transform(x).astype(np.float32)
    if kind == 'score_gate':
        scaled[:, 5:] = 0
    with torch.inference_mode():
        return model(torch.from_numpy(scaled)).numpy().astype(float)
