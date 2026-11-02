"""Statistics of the replication simulation: timing, origins, S-phase duration."""
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import spearmanr

def timing_profile(rep_frac):
    """Replication time of every bead in [0,1] (first time point at which it is replicated)."""
    T = rep_frac.shape[1]; rep = rep_frac > 0
    t = np.where(rep.any(axis=1), rep.argmax(axis=1), T - 1)
    return t / max(T - 1, 1)

def global_fraction(rep_frac):
    return (rep_frac > 0).mean(axis=0)

def quantile_time(frac, q):
    """Normalised time at which the replicated fraction first reaches q."""
    return float(np.argmax(frac >= q) / max(len(frac) - 1, 1))

def infer_origins(timing, prominence=0.01):
    """Origins: local minima of the replication time."""
    pk, _ = find_peaks(-np.r_[timing[0] + 1, timing, timing[-1] + 1], prominence=prominence)
    return pk - 1

def rep_summary(rep_frac, input_timing, speed, bead_kb, rep_duration):
    """Rows (quantity, value) describing the replication simulation."""
    f = global_fraction(rep_frac); t = timing_profile(rep_frac); org = infer_origins(t)
    rho = spearmanr(t, 1 - np.asarray(input_timing)).statistic if input_timing is not None and np.std(t) > 0 else np.nan
    t10, t50, t90 = (quantile_time(f, q) for q in (.1, .5, .9))
    return [("beads x time points", f"{rep_frac.shape[0]} x {rep_frac.shape[1]}"), ("replicated at the end", f"{f[-1] * 100:.1f} %"),
            ("T10 / T50 / T90 (fraction of S)", f"{t10:.2f} / {t50:.2f} / {t90:.2f}"), ("S-phase span T10-T90", f"{(t90 - t10):.2f} of S"),
            ("inferred origins", len(org)),
            ("median origin spacing", f"{np.median(np.diff(org)) * bead_kb:.0f} kb" if len(org) > 1 else "n/a"),
            ("mean fork speed (sim. units)", f"{speed:.3f}"), ("S duration (MC steps)", rep_duration),
            ("timing vs input, Spearman rho", f"{rho:.2f}" if rho == rho else "n/a")]
