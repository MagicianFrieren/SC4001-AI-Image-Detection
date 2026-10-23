"""Positive per-expert temperature scaling, fitted on disjoint originals."""
import numpy as np
from scipy.optimize import minimize_scalar

MIN_T, MAX_T = .05, 100.


def nll(logits, labels):
    return float(np.mean(np.logaddexp(0, logits) - labels * logits))


def fit(logits, labels):
    logits, labels = np.asarray(logits), np.asarray(labels)
    assert np.isfinite(logits).all() and set(np.unique(labels)) == {0, 1}
    result = minimize_scalar(lambda log_t: nll(logits / np.exp(log_t), labels),
                             bounds=(np.log(MIN_T), np.log(MAX_T)), method='bounded',
                             options={'xatol': 1e-12})
    if not result.success:
        raise RuntimeError(f'Temperature optimization failed: {result.message}')
    candidates = [1., MIN_T, MAX_T, float(np.exp(result.x))]
    temperature = min(candidates, key=lambda t: nll(logits / t, labels))
    return {'temperature': temperature, 'fit_n': len(labels), 'fit_bce_before': nll(logits, labels),
            'fit_bce_after': nll(logits / temperature, labels),
            'at_boundary': bool(np.isclose(temperature, MIN_T) or np.isclose(temperature, MAX_T))}
