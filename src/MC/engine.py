"""Numba Metropolis kernel of RepliSage (loop extrusion + replication + Potts states).

The kernel is nogil, so several independent chains (replicas) can run in threads.
`progress[0]` is updated every step so the caller can drive a progress bar.
"""
import numpy as np
import random as rd
from numba import njit
from .moves import unbind_bind, load_bind, slide, initialize
from .total_energy import get_E, get_dE_rewiring
from .potts_energy import E_potts, get_dE_node
from .loops_energy import E_fold, E_bind
from .replication_energy import compute_violations

@njit(cache=True, nogil=True)
def run_energy_minimization(
    N_steps, N_sweep, N_lef, N_lef2, N_beads, MC_step, T,
    L, R, k_norm, fold_norm, fold_norm2, bind_norm,
    rep_norm, t_rep, rep_duration, f_rep_T, has_rep,
    potts_norm1, potts_norm2, h, rw, p_rew, rep_fork_organizers,
    cohesin_blocks_condensin, random_spins, repfork_push, seed, progress, bind_cdf
):
    '''
    Monte Carlo (Metropolis) simulation of LEFs and Potts states.
    f_rep_T: replication matrix, time-major (T_rep, N_beads), int8. If has_rep is False it is a dummy.
    seed < 0: random seed. progress: int64 array of size 1 (current step).
    Returns Ms, Ns, Es, Es_potts, Fs, Bs, spin_traj, mags, acceptance_rate, N_viols, S_viols.
    '''
    if seed >= 0:
        np.random.seed(seed)
        rd.seed(seed)       # LEF rebinding uses python's random
    ht = np.zeros(N_beads, dtype=np.float64)   # time-dependent field (Potts)
    n_accepted = 0
    use_push = repfork_push and has_rep
    use_pref = bind_cdf.shape[0] > 1                # preferential loading of LEFs
    is_potts = potts_norm1 != 0.0 or potts_norm2 != 0.0

    spin_choices = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    lef_idx_choices = np.arange(N_lef, dtype=np.int64)

    ms, ns, spins = initialize(N_lef, N_lef2, N_beads, bind_cdf, random_spins)
    n_rec = N_steps // MC_step
    spin_traj = np.zeros((N_beads, n_rec), dtype=np.int32)

    E = get_E(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns, 0, f_rep_T, has_rep,
              spins, h, ht, potts_norm1, potts_norm2, rep_fork_organizers, cohesin_blocks_condensin)

    Es = np.zeros(n_rec, dtype=np.float64)
    Es_potts = np.zeros(n_rec, dtype=np.float64)
    mags = np.zeros(n_rec, dtype=np.float64)
    Fs = np.zeros(n_rec, dtype=np.float64)
    Bs = np.zeros(n_rec, dtype=np.float64)
    N_viols = np.zeros(n_rec, dtype=np.float64)
    S_viols = np.zeros(n_rec, dtype=np.float64)
    Ms = np.zeros((N_lef + N_lef2, n_rec), dtype=np.int64)
    Ns = np.zeros((N_lef + N_lef2, n_rec), dtype=np.int64)
    Ms[:, 0], Ns[:, 0] = ms, ns

    inv_rep_duration = 1.0 / rep_duration if rep_duration > 0 else 0.0
    rep_dur_i = int(rep_duration)

    for i in range(N_steps):
        progress[0] = i

        # current replication time index
        if i < t_rep:
            rt = 0
        elif i < t_rep + rep_duration:
            rt = int(i - t_rep)
        else:
            rt = rep_dur_i - 1
        if rt == (rep_dur_i - 1):                     # after replication all LEFs move
            lef_idx_choices = np.arange(N_lef + N_lef2, dtype=np.int64)
        # time-dependent field during replication (a bead's field is set once, when it first replicates)
        if has_rep and rt > 0 and rt < rep_dur_i:
            mag_field = (1 - 2 * rt * inv_rep_duration)
            for b in range(N_beads):
                if ht[b] == 0:
                    ht[b] += mag_field * f_rep_T[rt, b]

        for j in range(N_sweep):
            if np.random.rand() < p_rew:               # LEF rewiring move
                lef_idx = lef_idx_choices[np.random.randint(lef_idx_choices.shape[0])]
                m_old, n_old = ms[lef_idx], ns[lef_idx]
                r = np.random.randint(2)
                if m_old <= 0 or n_old <= 0:           # unbound LEF: force rebinding
                    r = 0
                if r == 0:
                    m_new, n_new = load_bind(bind_cdf) if use_pref else unbind_bind(N_beads)
                else:
                    m_new, n_new = slide(m_old, n_old, N_beads, f_rep_T, rt, rw, use_push)
                dE = get_dE_rewiring(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns,
                                     m_new, n_new, lef_idx, rt, f_rep_T, has_rep, spins, potts_norm2, cohesin_blocks_condensin)
                if dE <= 0 or np.exp(-dE / T) > np.random.rand():
                    E += dE
                    ms[lef_idx], ns[lef_idx] = m_new, n_new
                    n_accepted += 1
            else:                                      # Potts spin flip
                spin_idx = np.random.randint(N_beads)
                r = np.random.randint(4)               # uniform over the 4 other states
                if r >= spins[spin_idx] + 2: r += 1
                s = spin_choices[r]
                dE = 0.0   # without Potts terms every flip has dE = 0 (skip the computation)
                if is_potts:
                    dE = get_dE_node(spins, spin_idx, s, ms, ns, h, ht, potts_norm1, potts_norm2, rt, rep_fork_organizers)
                if dE <= 0 or np.exp(-dE / T) > np.random.rand():
                    E += dE
                    spins[spin_idx] = s
                    n_accepted += 1

        if i % MC_step == 0:                           # record observables
            idx = i // MC_step
            Es[idx] = E
            mags[idx] = np.average(spins)
            Ms[:, idx], Ns[:, idx] = ms, ns
            spin_traj[:, idx] = spins
            Es_potts[idx] = E_potts(spins, ms, ns, h, ht, potts_norm1, potts_norm2, rt, rep_fork_organizers)
            Fs[idx] = E_fold(ms, ns, fold_norm)
            Bs[idx] = E_bind(L, R, ms, ns, bind_norm)
            if has_rep:
                n, s = compute_violations(f_rep_T, ms, ns, rt)
                N_viols[idx] = n
                S_viols[idx] = s

    progress[0] = N_steps
    acceptance_rate = n_accepted / (N_steps * N_sweep)
    return Ms, Ns, Es, Es_potts, Fs, Bs, spin_traj, mags, acceptance_rate, N_viols, S_viols
