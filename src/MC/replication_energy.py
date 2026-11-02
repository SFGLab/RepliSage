"""Replication-fork energy terms. f is the replicated-DNA state at one time point (1D, per bead).

The replication matrix is stored time-major, f_rep_T[t, bead] (int8, C-contiguous),
so f_rep_T[t] is a contiguous row: identical values as the old f_rep[:, t], faster access.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def _any_zero(f, a, b):
    '''True if any f[a:b] == 0 (no temporary arrays).'''
    for k in range(a, b):
        if f[k] == 0: return True
    return False

@njit(cache=True)
def _count_ones(f, a, b):
    '''Number of f[a:b] == 1.'''
    c = 0
    for k in range(a, b):
        if f[k] == 1: c += 1
    return c

@njit(cache=True)
def violation_check(m, n, f):
    '''Return (violation_flag, violation_strength).'''
    if m < 0 or n < 0:
        return 0.0, 0.0
    cond1 = f[m] != f[n]                                       # ends in different replication states
    cond2 = (f[m] == 1 and f[n] == 1 and _any_zero(f, m, n))  # replicated ends, unreplicated inside
    if cond1 or cond2:
        replicated = _count_ones(f, m, n)  # f is binary: sum == count of ones
        total = n - m
        strength = min(replicated, total - replicated) + 1
        return 1.0, strength
    return 0.0, 0.0

@njit(cache=True)
def compute_violations(f_rep_T, ms, ns, t):
    N_viols = 0.0
    viol_strength = 0.0
    f = f_rep_T[t]
    for i in range(len(ms)):
        v, s = violation_check(ms[i], ns[i], f)
        N_viols += v
        viol_strength += s
    return N_viols, viol_strength / (N_viols + 1)

@njit(cache=True)
def Rep_Penalty(m, n, f):
    '''Penalty for a LEF crossing replication fork boundaries.'''
    r = 0.0
    if m >= 0 and n >= 0:
        if f[m] != f[n]:
            r += 1.0
        if (f[m] == 1 and f[n] == 1) and _any_zero(f, m, n):
            r += 1.0
        if (f[m] == 0 and f[n] == 0) and n > m + 1:
            replicated_in_between = _count_ones(f, m + 1, n)
            if replicated_in_between > 0:
                r += replicated_in_between / (n - m)
    return r

@njit(cache=True)
def E_rep(f_rep_T, ms, ns, t, rep_norm):
    '''Penalty of the replication energy.'''
    f = f_rep_T[t]
    E_penalty = 0.0
    for i in range(len(ms)):
        E_penalty += Rep_Penalty(ms[i], ns[i], f)
    return rep_norm * E_penalty

@njit(cache=True)
def get_dE_rep(f_rep_T, rep_norm, ms, ns, m_new, n_new, t, idx):
    '''Energy difference for replication energy.'''
    dE_rep = Rep_Penalty(m_new, n_new, f_rep_T[t]) - Rep_Penalty(ms[idx], ns[idx], f_rep_T[t - 1])
    return rep_norm * dE_rep
