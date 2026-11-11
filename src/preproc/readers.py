"""Input readers: BEDPE loops, BED / narrowPeak CTCF sites -> per-bead CTCF binding profiles (L, R).

Smoothing follows LoopSage: Gaussian, sigma = max(1.5, N_beads / 300) beads by default.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, binary_dilation
from ..logger import log
from ..motif_finder import motif_orientation

FORMATS = {".bedpe": "bedpe", ".bed": "bed", ".narrowpeak": "narrowpeak"}

@dataclass
class Binding:
    """Binding profiles at bead resolution. L, R: model input (smoothed). L0, R0: raw, before smoothing."""
    L: np.ndarray
    R: np.ndarray
    L0: np.ndarray
    R0: np.ndarray
    n_sites: int                 # beads carrying an anchor in L or R (anchors in one bead count once)
    fmt: str
    sigma: float = 0.0           # smoothing width, beads (0 = off)
    has_orientation: bool = False
    stats: dict = field(default_factory=dict)
    anchors: dict = field(default_factory=dict)   # per anchor: bp, w (strength), p (P(reverse), -1 unknown), loop (BEDPE loop index or None)

    def site_masks(self):
        """Boolean per-bead masks of the CTCF sites (raw sites widened by the smoothing width)."""
        k = int(round(self.sigma))
        return tuple(binary_dilation(v > 0, iterations=k) if k > 0 else v > 0 for v in (self.L0, self.R0))

def detect_format(path):
    """Input format from the file extension (.gz ignored): bedpe, bed or narrowpeak."""
    low = str(path).lower()
    low = low[:-3] if low.endswith(".gz") else low
    for ext, fmt in FORMATS.items():
        if low.endswith(ext): return fmt
    raise ValueError(f"Unsupported input file '{path}': use a .bedpe, .bed or .narrowPeak file.")

def sigma_default(n_beads):
    """LoopSage default smoothing width, beads."""
    return max(1.5, n_beads / 300)

def _same_chrom(col, chrom):
    """Chromosome match that ignores the 'chr' prefix."""
    strip = lambda s: s.astype(str).str.replace(r"^chr", "", regex=True, case=False)
    return strip(col) == str(chrom).removeprefix("chr")

def load_region(path, region, chrom, paired=True):
    """Rows of a BEDPE (paired: both anchors) or BED/narrowPeak table lying inside the region (bp)."""
    df = pd.read_csv(path, sep='\t', header=None, comment='#')
    cols = [1, 2, 4, 5] if paired else [1, 2]
    inside = _same_chrom(df[0], chrom) & (df[cols] >= region[0]).all(axis=1) & (df[cols] < region[1]).all(axis=1)
    df = df[inside].reset_index(drop=True)
    if df.empty: raise ValueError("The selected region or chromosome is invalid: the final set does not contain any loops or peaks.")
    return df

def numeric_col(df, col, default):
    """Column as floats; a missing column or bad cells become `default`."""
    if col >= df.shape[1]: return np.full(len(df), default, float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(float)

def valid_prob(p):
    """True if every value is -1 (unknown) or a probability in [0, 1]."""
    p = np.asarray(p, float); return bool(np.all(np.isfinite(p) & ((p == -1) | ((p >= 0) & (p <= 1)))))

def smooth_sparse(v, sigma):
    """Gaussian smoothing with sigma capped, so that sparse peaks are not erased (as in LoopSage)."""
    nz = np.flatnonzero(v > 0)
    cap = max(3.0, 3.0 * np.median(np.diff(nz))) if len(nz) > 2 else 5.0
    return gaussian_filter1d(v, min(sigma, cap), mode="nearest")

def to_beads(pos, region, n_beads):
    """Genomic positions (bp) -> bead index."""
    res = (region[1] - region[0]) // n_beads
    return np.clip((np.asarray(pos) - region[0]) // res, 0, n_beads - 1).astype(int)

def _split(w, p):
    """Weight of one anchor split into (left, right) parts by its reverse-orientation probability p (<0: unknown, even split)."""
    known = p >= 0
    return np.where(known, w * (1 - p), w * .5), np.where(known, w * p, w * .5)

def _fill_motif(p, starts, ends, chrom, motif):
    """Unknown orientations (p < 0 or None) filled from the motif scan; known ones are kept."""
    p = np.full(len(starts), -1.0) if p is None else np.asarray(p, float).copy(); need = p < 0
    if need.any(): p[need] = motif_orientation(chrom, np.asarray(starts)[need], np.asarray(ends)[need], **motif)
    return p

def _bedpe_raw(df, region, n, chrom=None, motif=None):
    """Raw (L, R, has_orientation, anchors) of a BEDPE table: weight = column 7, reverse probability = columns 8-9."""
    x, y = to_beads((df[1] + df[2]) // 2, region, n), to_beads((df[4] + df[5]) // 2, region, n)
    w = numeric_col(df, 6, 1.0); w = np.where(w < 0, 0.0, w)               # missing weight -> 1, negative -> no signal
    p1 = p2 = None
    if df.shape[1] > 8:
        a, b = numeric_col(df, 7, -1), numeric_col(df, 8, -1)
        if valid_prob(a) and valid_prob(b): p1, p2 = a, b
        else: log.alert("BEDPE columns 8-9 are not valid orientation probabilities: ignored.")
    if motif: p1, p2 = _fill_motif(p1, df[1], df[2], chrom, motif), _fill_motif(p2, df[4], df[5], chrom, motif)
    L, R = np.zeros(n), np.zeros(n)
    for v, p in ((x, p1), (y, p2)):
        wl, wr = _split(w, p) if p is not None else (w, w)                 # no orientation: both sides get the full weight
        np.add.at(L, v, wl); np.add.at(R, v, wr)
    k = np.full(len(df), -1.0); pp = np.r_[k if p1 is None else p1, k if p2 is None else p2]
    anc = dict(bp=np.r_[(df[1] + df[2]) / 2, (df[4] + df[5]) / 2].astype(float), w=np.r_[w, w], p=pp, loop=np.r_[np.arange(len(df)), np.arange(len(df))])
    return L, R, p1 is not None and bool(np.any(p1 >= 0) or np.any(p2 >= 0)), anc

def _peak_raw(df, region, n, fmt, chrom=None, motif=None):
    """Raw (L, R, has_orientation, anchors) of a BED / narrowPeak table; each peak sits on one bead (narrowPeak: its summit)."""
    if fmt == "bed":
        w = numeric_col(df, 4, 1.0); mid = (df[1] + df[2]) // 2; prob = numeric_col(df, 6, -1.0)
        if df.shape[1] > 5:                                                # '+' / '-' strand column -> P(reverse) = 0 / 1
            s = df[5].astype(str); prob = np.where(prob >= 0, prob, np.where(s == "+", 0.0, np.where(s == "-", 1.0, -1.0)))
        if not valid_prob(prob): log.alert("BED column 7 is not a valid probability: orientation ignored."); prob = np.full(len(df), -1.0)
    else:
        w = numeric_col(df, 6, 0.0); w = w if np.any(w > 0) else numeric_col(df, 4, 1.0)   # signalValue, else score
        off = numeric_col(df, 9, -1); mid = np.where(off >= 0, df[1].to_numpy() + off, (df[1] + df[2]) // 2)
        prob = np.full(len(df), -1.0)
        for c in (10, 5):                                                  # appended forward-probability column
            cand = numeric_col(df, c, -1.0) if df.shape[1] > c else None
            if cand is not None and valid_prob(cand) and np.any(cand >= 0): prob = np.where(cand >= 0, 1 - cand, -1.0); break
    if motif: prob = _fill_motif(prob, df[1], df[2], chrom, motif)
    w = np.where(w < 0, 0.0, w); wl, wr = _split(w, prob); b = to_beads(mid, region, n)
    L, R = np.zeros(n), np.zeros(n); np.add.at(L, b, wl); np.add.at(R, b, wr)
    return L, R, bool(np.any(prob >= 0)), dict(bp=np.asarray(mid, float), w=w, p=np.asarray(prob, float), loop=None)

def read_binding(path, region, chrom, n_beads, smooth=True, sigma=None, motif=None):
    """Read a BEDPE / BED / narrowPeak file -> Binding (sigma in beads; None = LoopSage default).

    motif: dict(genome=, pfm=, pvalue=, pseudocount=, pad=) fills the unknown orientations from CTCF motifs; None = off.
    """
    fmt = detect_format(path); df = load_region(path, region, chrom, paired=fmt == "bedpe")
    L0, R0, has_o, anc = _bedpe_raw(df, region, n_beads, chrom, motif) if fmt == "bedpe" else _peak_raw(df, region, n_beads, fmt, chrom, motif)
    s = (sigma_default(n_beads) if sigma is None else float(sigma)) if smooth else 0.0
    f = (lambda v: gaussian_filter1d(v, s, mode="nearest")) if fmt == "bedpe" else (lambda v: smooth_sparse(v, s))
    L, R = (f(L0), f(R0)) if s > 0 else (L0.copy(), R0.copy())
    n = int(np.count_nonzero((L0 > 0) | (R0 > 0)))
    return Binding(L, R, L0, R0, n, fmt, s, has_o, dict(rows=len(df), motif=bool(motif)), anc)
