"""Clear errors for replication-timing input that cannot be modelled (instead of a failure deep in the simulation)."""
import numpy as np

MIN_POINTS = 4   # cubic resampling of the timing curve onto the beads needs at least 4 data points

def check_sc(rep):
    """Single-cell parquet: the chromosome and region must exist in the data."""
    if len(rep.gen_windows) == 0:
        have = sorted({str(c) for c in rep.data["chromosome"]})
        raise ValueError(f"The single-cell timing data has no chromosome '{rep.chrom}' (available: {', '.join(have)}).")
    if rep.is_region:
        lo, hi = rep.gen_windows[:, 0].min(), rep.gen_windows[:, 1].max()
        if rep.coords[1] <= lo or rep.coords[0] >= hi:
            raise ValueError(f"Region {rep.chrom}:{rep.coords[0]:,}-{rep.coords[1]:,} lies outside the replication-timing data ({lo:,}-{hi:,} bp).")

def check_bigwig(rep):
    """bigWig: the chromosome must be present and the region inside it."""
    chroms = rep.data.chroms()
    if rep.chrom not in chroms:
        raise ValueError(f"The bigWig file has no chromosome '{rep.chrom}' (available: {', '.join(list(chroms)[:12])}...).")
    if rep.is_region and rep.coords[1] > chroms[rep.chrom]:
        raise ValueError(f"Region end {rep.coords[1]:,} is beyond the end of {rep.chrom} in the bigWig file ({chroms[rep.chrom]:,} bp).")

def check_txt(rep):
    """.txt table: the chromosome must be in column 1."""
    if not (rep.data[0] == rep.chrom).any():
        raise ValueError(f"The timing .txt file has no rows for chromosome '{rep.chrom}'.")

def check_signal(rep):
    """After mapping onto the beads: finite, varying signal with enough points; usable fork speed."""
    x = np.asarray(rep.avg_fx, float)
    if not np.isfinite(x).all():
        raise ValueError("The replication timing signal contains NaN after processing: the region probably has no data (all empty/zero).")
    if np.ptp(x) < 1e-9:
        raise ValueError("The replication timing is constant in this region, so there are no early/late origins to simulate. Use a longer region or another dataset.")
    if not np.isfinite(rep.speed_avg) or rep.speed_avg <= 0:
        raise ValueError("No fork speed could be estimated: the timing curve has no peaks or dips at this resolution. "
                         "Use a longer region or fewer beads, or set REP_SPEED_SCALE.")
