"""Monte Carlo moves: LEF (re)binding, sliding, and initial state."""
import numpy as np
import random as rd
from numba import njit

@njit(cache=True)
def unbind_bind(N_beads):
    '''Rebinding Monte-Carlo step.'''
    m_new = rd.randint(0, N_beads - 2)
    return m_new, m_new

@njit(cache=True)
def slide(m_old, n_old, N_beads, f_T, t, rw=True, push_condition=True):
    '''Sliding Monte-Carlo step. f_T[t, bead] is the replication state (used for fork pushing).'''
    if n_old > m_old + 1:
        r1 = np.random.randint(3) - 1 if rw else -1
        r2 = np.random.randint(3) - 1 if rw else 1
    else:  # just bound: it has to extrude
        r1 = np.random.randint(2) - 1 if rw else -1
        r2 = np.random.randint(2) if rw else 1
    m_new = m_old + r1 if m_old + r1 >= 0 else 0
    n_new = n_old + r2 if n_old + r2 < N_beads else N_beads - 1

    # replication forks push the LEFs (disabled when push_condition is False)
    if push_condition:
        if f_T[t, m_new] != f_T[max(t - 1, 0), m_old] and n_old > m_old:
            m_new = m_old + 1
        if f_T[t, n_new] != f_T[max(t - 1, 0), n_old] and n_old > m_old:
            n_new = n_old - 1

    if n_new < m_new: n_new = m_new
    if n_new >= N_beads: n_new = N_beads - 1
    if m_new >= N_beads: m_new = N_beads - 1
    if m_new < 0: m_new = 0
    if n_new < 0: n_new = 0
    return int(m_new), int(n_new)

@njit(cache=True)
def initialize(N_lef, N_lef2, N_beads, random_init_spins=True):
    '''Random initial condition of the simulation.'''
    ms = np.full(N_lef + N_lef2, -5, dtype=np.int64)
    ns = np.full(N_lef + N_lef2, -5, dtype=np.int64)
    for j in range(N_lef):
        ms[j], ns[j] = unbind_bind(N_beads)
    if random_init_spins:
        state = np.random.randint(0, 2, size=N_beads) * 4 - 2
    else:
        state = np.zeros(N_beads, dtype=np.int64)
    return ms, ns, state
