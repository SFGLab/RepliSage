"""Potts (epigenetic state) energy with a SPARSE coupling J.

The old code stored J as a dense N x N matrix. J only contains
  - the polymer backbone (J[i,i+1] = 1) and
  - the loops of the LEFs (J[m,n] += 1 for every bound LEF),
so every J lookup / row-sum is computed directly from (ms, ns): O(N_lef) instead of O(N),
without allocating temporaries. The energy values are mathematically identical.
A bound LEF is one with ms >= 0 and ns > ms (same rule as the old J updates).
"""
import numpy as np
from numba import njit

@njit(cache=True)
def J_pair(ms, ns, m, n):
    '''Coupling J[m,n]: backbone neighbour (1) + number of LEFs forming loop (m,n).'''
    lo, hi = min(m, n), max(m, n)
    w = 1 if hi == lo + 1 else 0
    for k in range(ms.shape[0]):
        if ms[k] == lo and ns[k] == hi and lo >= 0 and hi > lo:
            w += 1
    return w

@njit(cache=True)
def E_potts(spins, ms, ns, h, ht, potts_norm1, potts_norm2, t, rep_fork_organizers):
    '''Potts model energy: field term + sum_{i<j} J_ij |s_i - s_j|.'''
    N_beads = spins.shape[0]
    h_dot = 0.0
    for i in range(N_beads):
        h_dot += h[i] * spins[i]
    ht_dot = 0.0
    if t > 0:
        for i in range(N_beads):
            ht_dot += ht[i] * spins[i]

    E1 = h_dot / 2 + h_dot / 2 * (1 - int(rep_fork_organizers))
    if t > 0:
        E1 += ht_dot / 2 * int(rep_fork_organizers)

    E2 = 0.0
    for i in range(N_beads - 1):                 # backbone
        E2 += np.abs(spins[i] - spins[i + 1])
    for k in range(ms.shape[0]):                 # loops
        if ms[k] >= 0 and ns[k] > ms[k]:
            E2 += np.abs(spins[ms[k]] - spins[ns[k]])
    return potts_norm1 * E1 + potts_norm2 * E2

@njit(cache=True)
def get_dE_node(spins, i, s_new, ms, ns, h, ht, potts_norm1, potts_norm2, t, rep_fork_organizers=True):
    '''Energy difference of flipping spin i to s_new.'''
    s_old = spins[i]
    d = s_new - s_old
    dE1 = h[i] * d / 2 + h[i] * d / 2 * (1 - int(rep_fork_organizers))
    # time-dependent field: ht_old is the same array as ht in the update scheme, so only bead i changes
    if t > 0: dE1 += (ht[i] * d / 2) * int(rep_fork_organizers)

    dE2 = 0.0
    N = spins.shape[0]
    if i > 0:
        dE2 += np.abs(s_new - spins[i - 1]) - np.abs(s_old - spins[i - 1])
    if i < N - 1:
        dE2 += np.abs(s_new - spins[i + 1]) - np.abs(s_old - spins[i + 1])
    for k in range(ms.shape[0]):
        m, n = ms[k], ns[k]
        if m >= 0 and n > m:
            if m == i:
                dE2 += np.abs(s_new - spins[n]) - np.abs(s_old - spins[n])
            elif n == i:
                dE2 += np.abs(s_new - spins[m]) - np.abs(s_old - spins[m])
    return potts_norm1 * dE1 + potts_norm2 * dE2

@njit(cache=True)
def get_dE_potts_link(spins, m_new, n_new, m_old, n_old, potts_norm2=0.0):
    '''Energy difference of one LEF moving from (m_old, n_old) to (m_new, n_new): its own loop term |s_m - s_n| (bound = 0 <= m < n, as in E_potts).'''
    old = np.abs(spins[m_old] - spins[n_old]) if m_old >= 0 and n_old > m_old else 0
    new = np.abs(spins[m_new] - spins[n_new]) if m_new >= 0 and n_new > m_new else 0
    return potts_norm2 * (new - old)
