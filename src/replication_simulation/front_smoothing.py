"""Smoothing of the replication front: a sharp front becomes a gently varying replication-time profile."""
import numpy as np
from scipy.ndimage import gaussian_filter1d

def replication_times(rep_frac):
    """First time column where each bead is replicated (T = never)."""
    r = np.asarray(rep_frac) > 0
    return np.where(r.any(1), r.argmax(1), r.shape[1]).astype(float)

def smooth_front(rep_frac, l_forks, r_forks, sigma):
    """Gaussian-smooth the replication time of each bead (sigma in beads) and rebuild the matrices.

    The S-phase start and end are kept; sigma = 0 returns the inputs unchanged.
    """
    if not sigma or sigma <= 0: return rep_frac, l_forks, r_forks
    T = rep_frac.shape[1]; tau = replication_times(rep_frac)
    new = gaussian_filter1d(tau, sigma, mode="nearest")
    if np.ptp(new) > 0: new = tau.min() + (new - new.min()) * np.ptp(tau) / np.ptp(new)       # keep the S-phase span
    new = np.clip(np.round(new), 0, T).astype(int)
    f = (np.arange(T)[None, :] >= new[:, None]).astype(rep_frac.dtype)                         # monotone 0 -> 1 per bead
    shift = new - np.round(tau).astype(int)
    out = []
    for m in (l_forks, r_forks):                                                                  # move each fork marker with its bead
        m2 = np.zeros_like(m)
        for x in np.flatnonzero(np.asarray(m).any(1)):
            j = int(np.argmax(m[x] > 0)); m2[x, int(np.clip(j + shift[x], 0, T - 1))] = m[x, j]
        out.append(m2)
    return f, out[0], out[1]
