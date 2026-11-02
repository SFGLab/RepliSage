"""Time-major replication matrix for the MC kernel."""
import numpy as np

def to_time_major(rep_frac, rep_duration):
    """(N_beads, T1) replication matrix -> (rep_duration, N_beads) int8.

    Same values as utils.arrays.expand_columns(rep_frac, rep_duration).T (each column repeated
    rep_duration // T1 times, remaining columns zero) but built without the huge int64 intermediate.
    """
    N, T1 = rep_frac.shape
    if rep_duration <= T1:
        raise ValueError("REP_TIME_DURATION must be greater than the number of replication time points (REP_SIM_T).")
    rf = rep_duration // T1
    idx = np.arange(rep_duration) // rf
    ok = idx < T1
    out = np.zeros((rep_duration, N), dtype=np.int8)
    out[ok] = rep_frac.T[idx[ok]]
    return np.ascontiguousarray(out)
