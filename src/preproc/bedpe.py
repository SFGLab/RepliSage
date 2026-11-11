"""Loop (BEDPE) preprocessing: CTCF binding potentials, loop statistics."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..logger import log
from .readers import load_region, read_binding, detect_format

MOTIF_FINDER = "https://github.com/BlackPianoCat/motif_finder"
ORIENT_WARN = ("The input has no CTCF orientation (BEDPE columns 8-9, BED column 7 or a narrowPeak probability column). The model still runs, assuming every orientation blocks LEFs with the same "
               f"probability, but it is a bit less accurate. To detect CTCF orientation from motifs give a reference genome (--motif-genome genome.fa, built in), or see {MOTIF_FINDER}")

def bead_loops(bedpe_file, N_beads, region, chrom):
    """Loops of the region with anchor coordinates converted from bp to simulation beads."""
    df = load_region(bedpe_file, region, chrom); res = (region[1] - region[0]) // N_beads
    for c in (1, 2, 4, 5): df[c] = (df[c] - region[0]) // res
    return df

def occupancy_profile(bedpe_file, N_beads, region, chrom):
    """Loop occupancy per bead in [0, 1]: every loop adds its strength to all beads between its anchors, so nested loops stack up."""
    df = bead_loops(bedpe_file, N_beads, region, chrom); w = df[6].astype(float).where(df[6] >= 0, 0).fillna(1.0).values if df.shape[1] > 6 else np.ones(len(df))
    a = np.minimum((df[1] + df[2]) // 2, N_beads - 1).values; b = np.minimum((df[4] + df[5]) // 2, N_beads - 1).values
    lo, hi = np.minimum(a, b), np.maximum(a, b); d = np.zeros(N_beads + 1)
    np.add.at(d, lo, w); np.add.at(d, hi + 1, -w)                  # difference array -> cumulative sum
    O = np.cumsum(d)[:N_beads]; return O / O.max() if O.max() > 0 else O

def loading_weights(occ, floor=.05, power=1.0):
    """Relative LEF loading weight per bead: floor for loop-naked beads, 1 for the most occupied."""
    return floor + (1 - floor) * np.asarray(occ, float) ** power

def orientation_confidence(L, R):
    """Per bead: orientation score in [-1 (left-ward), +1 (right-ward)] and confidence |score| (0 = unknown or mixed)."""
    t = L + R; s = np.where(t > 0, (R - L) / np.where(t > 0, t, 1), 0.0); return s, np.abs(s)

def binding_vectors_from_bedpe(bedpe_file,N_beads,region,chrom,normalization=False,viz=False):
    '''
    Definition of left and right CTCF binding potential.

    Input:
    bedpe_file (str): path with bepde file with loops
    region (list): a list with two integers [start,end], which represent the start and end point of the region of interest.
    chrom (str): chromosome of interest.
    normalization (bool): in case that it is needed to normalize to numpy arrays that represent left and right CTCF binding potential.
    viz (bool): If True is vizualizes the distribution of distances of loops from the diagonal and the binding potentials as functions of simulated polymer distance.

    Output:
    L (numpy array): left CTCF binding potential.
    R (numpy array): right CTCF binding potential.
    '''
    df = bead_loops(bedpe_file, N_beads, region, chrom)
    
    # Check if columns 7 and 8 exist
    has_col_7_8 = df.shape[1] > 8
    if has_col_7_8:
        log.info("The input bedpe file contains CTCF orientation: it is taken into account.")
    else:
        log.alert(ORIENT_WARN)

    # Compute the matrix
    distances = list()
    L, R = np.zeros(N_beads, dtype=np.float64), np.zeros(N_beads, dtype=np.float64)
    for i in range(len(df)):
        x, y = min((df[1][i] + df[2][i]) // 2, N_beads - 1), min((df[4][i] + df[5][i]) // 2, N_beads - 1)
        distances.append(distance_point_line(x, y))
        strength = df[6][i]
        # If strength is None or negative, skip adding signal
        if pd.isna(strength) or strength is None or strength < 0:
            continue
        if has_col_7_8:
            if df[7][i] >= 0:
                L[x] += strength * (1 - df[7][i])
            if df[8][i] >= 0:
                L[y] += strength * (1 - df[8][i])
            if df[7][i] >= 0:
                R[x] += strength * df[7][i]
            if df[8][i] >= 0:
                R[y] += strength * df[8][i]
        else:
            L[x] += strength
            L[y] += strength
            R[x] += strength
            R[y] += strength
    
    # Normalize (if neccesary): it means to convert values to probabilities
    if normalization:
        L, R = L/np.sum(L), R/np.sum(R)

    if viz:
        fig, axs = plt.subplots(2, figsize=(15, 10))
        axs[0].plot(L,'g-')
        axs[0].set_ylabel('Left potential',fontsize=16)
        axs[1].plot(R,'r-')
        axs[1].set_ylabel('Right potential',fontsize=16)
        axs[1].set_xlabel('Genomic Distance (with simumation beads as a unit)',fontsize=16)
        plt.close(fig)

    return L, R

def distance_point_line(x0,y0,a=1,b=-1,c=0):
    return np.abs(a*x0+b*y0+c)/np.sqrt(a**2+b**2)


def preprocessing(path: str, region: list, chrom: str, N_beads: int, smooth: bool = False, sigma=None):
    """Return left/right CTCF binding potentials and the number of CTCF sites (BEDPE, BED or narrowPeak; smoothing is optional)."""
    b = read_binding(path, region, chrom, N_beads, smooth, sigma)
    return b.L, b.R, b.n_sites

def loop_stats(L, R):
    """Rows for a statistics table of the binding potentials (per bead)."""
    return [("beads with a left CTCF site", int(np.count_nonzero(L))), ("beads with a right CTCF site", int(np.count_nonzero(R))),
            ("total left strength", f"{L.sum():.1f}"), ("total right strength", f"{R.sum():.1f}")]

def _q(v, f="{:,.0f}"):
    """min / median / mean / max of an array as one string."""
    v = np.asarray(v, float); return " / ".join(f.format(x) for x in (v.min(), np.median(v), v.mean(), v.max()))

def input_stats(bedpe_file, region, chrom, n_beads):
    """Rows (bp / kb units) describing the loop input inside the modelled region."""
    if detect_format(bedpe_file) != "bedpe": return peak_stats(bedpe_file, region, chrom, n_beads)
    df = load_region(bedpe_file, region, chrom); size = region[1] - region[0]; bead = size / n_beads
    c1, c2 = (df[1] + df[2]) / 2, (df[4] + df[5]) / 2; length = np.abs(c2 - c1) / 1e3; w = df[6].dropna() if df.shape[1] > 6 else pd.Series(dtype=float)
    rows = [("region", f"{chrom}:{region[0]:,}-{region[1]:,}"), ("region size", f"{size / 1e6:.2f} Mb"), ("bead size", f"{bead / 1e3:.2f} kb ({int(bead):,} bp)"),
            ("loops in the region", len(df)), ("loop length, kb (min / median / mean / max)", _q(length)),
            ("loop length, beads (median)", f"{np.median(length) * 1e3 / bead:.1f}"), ("anchor width, bp (median)", f"{np.median(np.r_[df[2] - df[1], df[5] - df[4]]):,.0f}")]
    if len(w): rows.append(("loop strength (min / median / mean / max)", _q(w, "{:.2f}")))
    rows.append(("CTCF orientation", "present (columns 8-9)" if df.shape[1] > 8 else "absent: all orientations block equally"))
    return rows

def peak_stats(path, region, chrom, n_beads):
    """Rows describing a BED / narrowPeak input inside the modelled region."""
    df = load_region(path, region, chrom, paired=False); size = region[1] - region[0]; bead = size / n_beads
    w = pd.to_numeric(df[6] if detect_format(path) == "narrowpeak" and df.shape[1] > 6 else df[4] if df.shape[1] > 4 else pd.Series(dtype=float), errors="coerce").dropna()
    rows = [("region", f"{chrom}:{region[0]:,}-{region[1]:,}"), ("region size", f"{size / 1e6:.2f} Mb"), ("bead size", f"{bead / 1e3:.2f} kb ({int(bead):,} bp)"),
            ("peaks in the region", len(df)), ("peak width, bp (median)", f"{np.median(df[2] - df[1]):,.0f}")]
    if len(w): rows.append(("peak strength (min / median / mean / max)", _q(w, "{:.2f}")))
    return rows
