"""Replication-front smoothing."""
import numpy as np
from src.replication_simulation.front_smoothing import smooth_front, replication_times

def demo(N=60, T=40, seed=0):
    rng = np.random.default_rng(seed); tau = np.sort(rng.integers(0, T, N)); tau[0], tau[-1] = 0, T - 1; tau = rng.permutation(tau)
    f = (np.arange(T)[None] >= tau[:, None]).astype(int); lf = np.zeros_like(f); rf = np.zeros_like(f)
    lf[np.arange(N), np.clip(tau + 1, 0, T - 1)] = 1; rf[np.arange(N), np.clip(tau + 1, 0, T - 1)] = 1; return f, lf, rf, tau

def test_sigma_zero_is_identity():
    f, lf, rf, _ = demo(); g, a, b = smooth_front(f, lf, rf, 0); assert g is f and a is lf and b is rf

def test_smoothing_is_monotone_and_keeps_span():
    f, lf, rf, tau = demo(); g, a, b = smooth_front(f, lf, rf, 2.0)
    assert (np.diff(g, axis=1) >= 0).all() and g.shape == f.shape                    # each bead flips 0 -> 1 once
    t2 = replication_times(g); assert t2.min() == tau.min() and t2.max() == tau.max()
    assert np.abs(np.diff(t2)).mean() < np.abs(np.diff(tau)).mean()                     # neighbours are closer in time
    assert (a.sum(1) == lf.sum(1)).all() and (b.sum(1) == rf.sum(1)).all()             # one marker per row is kept
