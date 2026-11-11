"""Hi-C style features of a contact map: P(s), O/E, insulation (TADs), A/B compartments."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import eigsh
from scipy.signal import find_peaks

def ps_curve(mat):
    """Mean contact at bead separation s."""
    return np.array([np.diagonal(mat, k).mean() for k in range(mat.shape[0])])

def fill_masked(mat):
    """NaN (masked beads) -> mean of their diagonal, so they are neutral in O/E, P(s) and insulation."""
    m = np.array(mat, float)
    if np.isfinite(m).all(): return m
    n = len(m)
    for k in range(n):
        d = np.diagonal(m, k); bad = ~np.isfinite(d)
        if bad.any(): ix = np.arange(n - k)[bad]; m[ix, ix + k] = m[ix + k, ix] = np.nanmean(d) if (~bad).any() else 0.0
    return m

def smooth_track(x, sigma):
    """Gaussian smoothing of a 1D track (sigma in beads; 0 = unchanged)."""
    x = np.asarray(x, float); return gaussian_filter1d(x, sigma, mode="nearest") if sigma > 0 else x

def observed_over_expected(mat):
    """Every diagonal divided by its mean."""
    n = mat.shape[0]; out = np.ones_like(mat, dtype=float)
    for k in range(n):
        d = np.diagonal(mat, k); m = d.mean(); idx = np.arange(n - k)
        if m > 0: out[idx, idx + k] = out[idx + k, idx] = d / m
    return out

def oe_correlation(mat):
    """Pearson correlation matrix of the rows of ln(O/E + 0.2) (compartments show as a checkerboard)."""
    return np.nan_to_num(np.corrcoef(np.log(observed_over_expected(np.asarray(mat, float)) + 0.2)))

def insulation_score(mat, window=5, sigma=0.0):
    """Diamond insulation score: mean contact across each bead (ends stay 0), optionally smoothed."""
    n = mat.shape[0]; ins = np.zeros(n)
    for i in range(window, n - window): ins[i] = mat[i - window:i, i + 1:i + window + 1].mean()
    if sigma > 0 and n > 2 * window: ins[window:n - window] = smooth_track(ins[window:n - window], sigma)
    return ins

def insulation_window(bp_per_bead, bp=100_000):
    """Diamond half-width in beads for a span of `bp` (at least 5 beads)."""
    return max(5, int(round(bp / bp_per_bead)))

def pc1(mat, smooth=0.0):
    """Leading eigenvector of the O/E correlation map (sign arbitrary), scaled to [-1, 1]."""
    c = np.nan_to_num(np.corrcoef(np.log2(observed_over_expected(np.asarray(mat, float)) + 1e-3)))
    v = smooth_track(np.linalg.eigh(c)[1][:, -1], smooth); return v / (np.abs(v).max() or 1.0)

def compartment_from_hic(hic, min_sep=2, smooth=0.0):
    """A/B track in [-1, 1] from a Hi-C map (NaN = masked): leading eigenvector of the O/E correlation, B (negative) = denser short-range contacts."""
    n = len(hic); seen = np.isfinite(hic).sum(0) > n / 2
    h = np.clip(fill_masked(hic)[np.ix_(seen, seen)], 0.0, None); m = len(h)
    sep = np.abs(np.subtract.outer(np.arange(m), np.arange(m))); oe = observed_over_expected(h); oe[sep < min_sep] = 1.0
    c = np.nan_to_num(np.corrcoef(np.log2(np.clip(oe, .125, 8.0))))
    v = np.linalg.eigh(c)[1][:, -1] if m <= 300 else eigsh(c, k=1, which="LA", v0=np.ones(m))[1][:, 0]
    band = (sep >= min_sep) & (sep <= max(m // 8, 10)); dens = np.log((h * band).sum(1) / np.maximum(band.sum(1), 1) + 1e-12)
    if np.corrcoef(v, dens)[0, 1] > 0: v = -v
    lim = 3 * 1.4826 * np.median(np.abs(v - np.median(v))) or np.abs(v).max()          # robust cap before scaling
    v = smooth_track(np.clip(v, -lim, lim), smooth)
    return np.interp(np.arange(n), np.flatnonzero(seen), v / (np.abs(v).max() or 1.0))

def boundaries(ins, sigma=1.5, prominence=.5):
    """Beads at local minima of the z-scored, smoothed insulation score."""
    z = smooth_track(ins, sigma); z = (z - z.mean()) / (z.std() or 1.0); return find_peaks(-z, prominence=prominence)[0]

def boundary_overlap(a, b, tol=2):
    """Share of boundaries in `a` with one of `b` within `tol` beads (NaN if empty)."""
    return float(np.mean([np.abs(b - i).min() <= tol for i in a])) if len(a) and len(b) else float("nan")
