from numba import njit
import numpy as np
import random as rd
from .preproc import *

def preprocessing(bedpe_file: str, region: list, chrom: str, N_beads: int):
    '''
    It computes the binding potential and the number of CTCF motifs.
    ----------------------------------------------------------------
    Input:
    bedpe_file: the path of the bedpe file.
    region: the coordinates of region in genomic distance units, in format of list [start,end].
    chrom: the chromosome of interest.
    N_beads: the number of simulation beads.
    '''
    L, R, J = binding_vectors_from_bedpe(bedpe_file, N_beads, region, chrom, False, False)
    N_CTCF = np.max(np.array([np.count_nonzero(L), np.count_nonzero(R)]))
    return L, R, J, N_CTCF

@njit
def closest_opposite(f, m):
    # Find the indices of the opposite value
    target_value = f[m]
    opposite_indices = np.where(f != target_value)[0]  # Indices of opposite value

    # Calculate distances and find the closest one
    closest_index = opposite_indices[np.argmin(np.abs(opposite_indices - m))]
    return closest_index

@njit
def Kappa(mi, ni, mj, nj):
    '''
    Computes the crossing function of LoopSage.
    '''
    k = 0.0
    if mi >= 0 and ni >= 0 and mj >= 0 and nj >= 0:
        if mi < mj < ni < nj: k += 1
        if mj < mi < nj < ni: k += 1
        if mj == ni or mi == nj or ni == nj or mi == mj: k += 1
    return k

@njit
def Rep_Penalty(m, n, f):
    r = 0.0
    
    # The case that cohesin crosses a replication fork: for sure penalized
    if m >= 0 and n >= 0:
        if f[m] != f[n]: r += 1.0
        if (f[m] == 1 and f[n] == 1) and np.any(f[m:n] == 0): r += 1.0
    
    return r

@njit
def E_bind(L, R, ms, ns, bind_norm):
    '''
    The binding energy.
    '''
    binding = np.sum(L[ms[ms >= 0]] + R[ns[ns >= 0]])
    E_b = bind_norm * binding
    return E_b

@njit
def E_rep(f_rep, ms, ns, t, rep_norm):
    '''
    Penalty of the replication energy.
    '''
    E_penalty = 0.0
    for i in range(len(ms)):
        E_penalty += Rep_Penalty(ms[i], ns[i], f_rep[:, t])
    return rep_norm * E_penalty

@njit
def E_cross(ms, ns, k_norm, cohesin_blocks_condensin=False):
    '''
    The crossing energy.
    '''
    crossing = 0.0
    N_lef = len(ms)
    for i in range(N_lef):
        for j in range(i + 1, N_lef):
            if cohesin_blocks_condensin or (i < N_lef and j < N_lef) or (i >= N_lef and j >= N_lef):
                crossing += Kappa(ms[i], ns[i], ms[j], ns[j])
    return k_norm * crossing

@njit
def E_fold(ms, ns, fold_norm):
    ''''
    The folding energy.
    '''
    folding = np.sum(np.log(ns - ms))
    return fold_norm * folding

@njit
def E_potts(spins, J, h, ht, potts_norm1, potts_norm2, t, rep_fork_organizers):
    N_beads = spins.shape[0]
    # Precompute h*spins and ht*spins
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

    # Only sum upper triangle (i < j) for J and |spins[i] - spins[j]|
    E2 = 0.0
    for i in range(N_beads - 1):
        for j in range(i + 1, N_beads):
            diff = abs(spins[i] - spins[j])
            E2 += J[i, j] * diff

    return potts_norm1 * E1 + potts_norm2 * E2

@njit
def get_E(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns, t, f_rep, spins, J, h, ht, potts_norm1=0.0, potts_norm2=0.0, rep_fork_organizers=True, cohesin_blocks_condensin=False):
    '''
    The total energy.
    '''
    energy = E_bind(L, R, ms, ns, bind_norm) + E_cross(ms, ns, k_norm, cohesin_blocks_condensin) + E_fold(ms, ns, fold_norm)
    if fold_norm2 != 0: 
        energy += E_fold(ms[N_lef:N_lef + N_lef2], ns[N_lef:N_lef + N_lef2], fold_norm2)
    if rep_norm != 0.0 and f_rep is not None: 
        energy += E_rep(f_rep, ms, ns, t, rep_norm)
    if potts_norm1 != 0.0 or potts_norm2 != 0.0: 
        energy += E_potts(spins, J, h, ht, potts_norm1, potts_norm2, t, rep_fork_organizers)
    return energy

@njit
def get_dE_bind(L, R, bind_norm, ms, ns, m_new, n_new, idx):
    '''
    Energy difference for binding energy.
    '''
    B_new = L[m_new] + R[n_new] if m_new >= 0 and n_new >= 0 else 0
    B_old = L[ms[idx]] + R[ns[idx]] if ms[idx] >= 0 and ns[idx] >= 0 else 0
    return bind_norm * (B_new - B_old)

@njit
def get_dE_fold(fold_norm, ms, ns, m_new, n_new, idx):
    '''
    Energy difference for folding energy.
    '''
    return fold_norm * (np.log(n_new - m_new) - np.log(ns[idx] - ms[idx]))

@njit
def get_dE_rep(f_rep, rep_norm, ms, ns, m_new, n_new, t, idx):
    '''
    Energy difference for replication energy.
    '''
    dE_rep = Rep_Penalty(m_new, n_new, f_rep[:, t]) - Rep_Penalty(ms[idx], ns[idx], f_rep[:, t - 1])
    return rep_norm * dE_rep

@njit(fastmath=True, cache=True)
def get_dE_cross(ms, ns, m_new, n_new, idx, k_norm, cohesin_blocks_condensin=False):
    '''
    Optimized energy difference for crossing energy.
    '''
    K1 = 0.0
    K2 = 0.0
    N_lef = ms.shape[0]

    ms_idx = ms[idx]
    ns_idx = ns[idx]

    # Precompute block type for idx
    idx_block = idx < N_lef

    for i in range(N_lef):
        if i == idx:
            continue
        i_block = i < N_lef
        if cohesin_blocks_condensin or (idx_block and i_block) or (not idx_block and not i_block):
            ms_i = ms[i]
            ns_i = ns[i]
            K1 += Kappa(ms_idx, ns_idx, ms_i, ns_i)
            K2 += Kappa(m_new, n_new, ms_i, ns_i)
    return k_norm * (K2 - K1)

@njit(fastmath=True, cache=True, inline='always')
def get_dE_node(spins, spin_idx, spin_val, J, h, ht_new, ht_old, potts_norm1, potts_norm2, t, rep_fork_organizers=True):
    '''
    Optimized energy difference for node state change.
    '''
    old_spin = spins[spin_idx]
    delta_spin = spin_val - old_spin

    # h term
    dE1 = h[spin_idx] * delta_spin
    if not rep_fork_organizers:
        dE1 *= 1.0
    else:
        dE1 *= 0.5

    # ht term (only if t > 0 and rep_fork_organizers)
    if t > 0 and rep_fork_organizers:
        # Only the changed spin contributes to the difference
        dE1 += 0.5 * (ht_new[spin_idx] - ht_old[spin_idx]) * delta_spin

    # J term: only neighbors affected
    dE2 = 0.0
    N = spins.shape[0]
    for j in range(N):
        if j == spin_idx:
            continue
        dE2 += J[spin_idx, j] * (abs(spin_val - spins[j]) - abs(old_spin - spins[j]))

    return potts_norm1 * dE1 + potts_norm2 * dE2

@njit(fastmath=True, cache=True, inline='always')
def get_dE_potts_link(spins, J, m_new, n_new, m_old, n_old, potts_norm2=0.0):
    '''
    Optimized energy difference for Potts link energy.
    '''
    dE = 0.0
    # Avoid branching by using masks and direct computation
    if m_new >= 0 and n_new >= 0:
        dE += J[m_new, n_new] * (spins[m_new] == spins[n_new])
    if m_old >= 0 and n_old >= 0:
        dE -= J[m_old, n_old] * (spins[m_old] == spins[n_old])
    return potts_norm2 * dE

@njit
def get_dE_rewiring(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns, m_new, n_new, idx, t, f_rep, spins, J, potts_norm2=0.0, cohesin_blocks_condensin=False):
    '''
    Total energy difference for rewiring.
    '''
    dE = 0.0
    if idx < N_lef:
        dE += get_dE_fold(fold_norm, ms[:N_lef], ns[:N_lef], m_new, n_new, idx)
    else:
        dE += get_dE_fold(fold_norm2, ms[N_lef:N_lef + N_lef2], ns[N_lef:N_lef + N_lef2], m_new, n_new, idx - N_lef)
    dE += get_dE_bind(L, R, bind_norm, ms, ns, m_new, n_new, idx)
    dE += get_dE_cross(ms, ns, m_new, n_new, idx, k_norm, cohesin_blocks_condensin)
    
    if rep_norm > 0.0 and f_rep is not None:
        dE += get_dE_rep(f_rep, rep_norm, ms, ns, m_new, n_new, t, idx)
    
    if potts_norm2 > 0.0:
        dE += get_dE_potts_link(spins, J, m_new, n_new, ms[idx], ns[idx], potts_norm2)
    
    return dE

@njit
def unbind_bind(N_beads):
    '''
    Rebinding Monte-Carlo step.
    '''
    m_new = rd.randint(0, N_beads - 3)
    n_new = m_new + 2  # Ensure n_new - m_new >= 1
    return m_new, n_new

@njit(fastmath=True, cache=True, inline='always')
def slide(m_old, n_old, N_beads, f=None, t=0, rw=True):
    '''
    Optimized sliding Monte-Carlo step.
    '''
    # Use integer random for speed
    r1 = -1 if not rw else (1 if np.random.randint(2) else -1)
    r2 = 1 if not rw else (1 if np.random.randint(2) else -1)

    m_new = m_old + r1
    n_new = n_old + r2

    # Clamp to valid range
    if m_new < 0:
        m_new = 0
    if n_new > N_beads - 1:
        n_new = N_beads - 1

    # Ensure n_new - m_new >= 2
    if n_new - m_new < 2:
        if m_new > 0:
            m_new = n_new - 2
            if m_new < 0:
                m_new = 0
        if n_new < N_beads - 1:
            n_new = m_new + 2
            if n_new > N_beads - 1:
                n_new = N_beads - 1

    # Replication fork logic (minimize np.any and slicing)
    if f is not None:
        t_prev = t - 1 if t > 0 else 0
        # Only check if there are any zeros in f[:, t]
        has_zero = False
        for i in range(f.shape[0]):
            if f[i, t] == 0:
                has_zero = True
                break
        if has_zero:
            if f[m_new, t] != f[m_old, t_prev]:
                m_new = closest_opposite(f[:, t], m_new)
            if f[n_new, t] != f[n_old, t_prev]:
                n_new = closest_opposite(f[:, t], n_new)

    return m_new, n_new

@njit
def initialize(N_lef, N_lef2, N_beads):
    '''
    Random initial condition of the simulation.
    '''
    ms = np.full(N_lef + N_lef2, -5, dtype=np.int64)
    ns = np.full(N_lef + N_lef2, -4, dtype=np.int64)
    for j in range(N_lef):
        ms[j], ns[j] = unbind_bind(N_beads)
    state = np.random.randint(0, 2, size=N_beads) * 4 - 2
    return ms, ns, state

@njit
def initialize_J(N_beads, J, ms, ns):
    for i in range(N_beads - 1):
        J[i, i + 1] += 1
        J[i + 1, i] += 1
    for idx in range(len(ms)):
        m, n = ms[idx], ns[idx]
        if m >= 0 and n >= 0:  # Ensure valid indices
            J[m, n] += 1
            J[n, m] += 1
    return J

@njit(fastmath=True, cache=True)
def run_energy_minimization(
    N_steps, N_sweep, N_lef, N_lef2, N_beads, MC_step, T, T_min, mode,
    L, R, k_norm, fold_norm, fold_norm2, bind_norm,
    rep_norm=0.0, t_rep=np.inf, rep_duration=np.inf, f_rep=None,
    potts_norm1=0.0, potts_norm2=0.0, J=None, h=None, rw=True, spins=None,
    p_rew=0.5, rep_fork_organizers=True, cohesin_blocks_condensin=False
):
    '''
    Runs a Monte Carlo or simulated annealing energy minimization for a chromatin simulation.
    [docstring omitted for brevity]
    '''
    Ti = T  # Current temperature
    ht = np.zeros(N_beads, dtype=np.float64)      # Time-dependent field (for Potts)
    ht_old = np.zeros(N_beads, dtype=np.float64)  # Previous time-dependent field
    mask = (ht_old == 0)  # Mask for updating ht

    # Possible spin values and indices
    spin_choices = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    lef_idx_choices = np.arange(N_lef, dtype=np.int64)

    # Initialize LEF positions and Potts spins
    ms, ns, spins = initialize(N_lef, N_lef2, N_beads)
    spin_traj = np.zeros((N_beads, N_steps // MC_step), dtype=np.int32)

    # Initialize coupling matrix J with current LEF positions
    J = initialize_J(N_beads, J, ms, ns)

    # Compute initial energy
    E = get_E(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns, 0, f_rep, spins, J, h, ht, potts_norm1, potts_norm2, rep_fork_organizers, cohesin_blocks_condensin)

    # Allocate arrays for observables
    Es = np.zeros(N_steps // MC_step, dtype=np.float64)
    Es_potts = np.zeros(N_steps // MC_step, dtype=np.float64)
    mags = np.zeros(N_steps // MC_step, dtype=np.float64)
    Fs = np.zeros(N_steps // MC_step, dtype=np.float64)
    Bs = np.zeros(N_steps // MC_step, dtype=np.float64)
    Rs = np.zeros(N_steps // MC_step, dtype=np.float64)
    Ms = np.zeros((N_lef + N_lef2, N_steps // MC_step), dtype=np.int64)
    Ns = np.zeros((N_lef + N_lef2, N_steps // MC_step), dtype=np.int64)
    Ms[:, 0], Ns[:, 0] = ms, ns

    # Precompute reciprocal for replication duration
    inv_rep_duration = 1.0 / rep_duration if rep_duration != np.inf else 0.0

    # Progress bar setup
    last_percent = -1

    for i in range(N_steps):
        # Print progress every 5%
        percent = int(100 * i / N_steps)
        if percent % 5 == 0 and percent != last_percent:
            # Numba can't use print with flush, so just print
            print(percent, "% completed")
            last_percent = percent

        # Determine current replication time index (rt)
        if rep_norm == 0.0 or f_rep is None:
            rt = 0
        else:
            if i < t_rep:
                rt = 0
            elif i >= t_rep and i < t_rep + rep_duration:
                rt = int(i - t_rep)
            else:
                rt = int(rep_duration) - 1
            # After replication, allow all LEFs to move
            if rt == (int(rep_duration) - 1):
                lef_idx_choices = np.arange(N_lef + N_lef2, dtype=np.int64)
            # Update time-dependent field ht during replication
            if rt > 0 and rt < int(rep_duration):
                mag_field = (1 - 2 * rt * inv_rep_duration)
                ht += mask * mag_field * f_rep[:, rt]

        # Update temperature for annealing
        if mode == 'Annealing':
            Ti = T - (T - T_min) * i / N_steps
        else:
            Ti = T

        for j in range(N_sweep):
            # With probability p_rew, propose a LEF rewiring move
            if np.random.rand() < p_rew:
                lef_idx = np.random.randint(lef_idx_choices.shape[0])
                lef_idx = lef_idx_choices[lef_idx]
                m_old, n_old = ms[lef_idx], ns[lef_idx]
                r = np.random.randint(2)
                # If LEF is unbound, force unbinding move
                if m_old <= 0 or n_old <= 0:
                    r = 0
                if r == 0:
                    # Unbind and rebind at random
                    m_new, n_new = unbind_bind(N_beads)
                else:
                    # Slide LEF along the polymer
                    m_new, n_new = slide(ms[lef_idx], ns[lef_idx], N_beads, f_rep, rt, rw)
                # Compute energy difference for move
                dE = get_dE_rewiring(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns, m_new, n_new, lef_idx, rt, f_rep, spins, J, potts_norm2, cohesin_blocks_condensin)
                # Metropolis criterion
                if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                    E += dE
                    # Update J matrix for LEF move
                    if m_old >= 0:
                        J[m_old, n_old] -= 1
                        J[n_old, m_old] -= 1
                    if m_new >= 0:
                        J[m_new, n_new] += 1
                        J[n_new, m_new] += 1
                    ms[lef_idx], ns[lef_idx] = m_new, n_new
            else:
                # Propose a Potts spin flip
                spin_idx = np.random.randint(N_beads)
                s_choices = spin_choices[spin_choices != spins[spin_idx]]
                s = s_choices[np.random.randint(s_choices.shape[0])]
                dE = get_dE_node(spins, spin_idx, s, J, h, ht, ht_old, potts_norm1, potts_norm2, rt, rep_fork_organizers)
                # Metropolis criterion
                if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                    E += dE
                    spins[spin_idx] = s

        # Update previous time-dependent field and mask
        ht_old = ht
        mask = (ht_old == 0)

        # Record observables every MC_step
        if i % MC_step == 0:
            idx = i // MC_step
            Es[idx] = E
            mags[idx] = np.average(spins)
            Ms[:, idx], Ns[:, idx] = ms, ns
            spin_traj[:, idx] = spins
            Es_potts[idx] = E_potts(spins, J, h, ht, potts_norm1, potts_norm2, rt, rep_fork_organizers)
            Fs[idx] = E_fold(ms, ns, fold_norm)
            Bs[idx] = E_bind(L, R, ms, ns, bind_norm)
            if rep_norm != 0.0 and f_rep is not None:
                Rs[idx] = E_rep(f_rep, ms, ns, rt, rep_norm)

    return Ms, Ns, Es, Es_potts, Fs, Bs, spin_traj, mags
