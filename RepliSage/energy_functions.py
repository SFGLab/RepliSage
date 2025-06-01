from numba import njit
import numpy as np
import random as rd
from .utils import *

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
    # Computes penalty for cohesin crossing replication fork boundaries
    r = 0.0

    # Only consider valid indices
    if m >= 0 and n >= 0:
        # Penalize if cohesin crosses a replication fork boundary
        if f[m] != f[n]:
            r += 1.0

        # Penalize if both ends are replicated but there is an unreplicated region in between
        if (f[m] == 1 and f[n] == 1) and np.any(f[m:n] == 0):
            r += 1.0

        # Penalize if one end is replicated and the other is not (already covered above)
        if (f[m] == 1 and f[n] == 0) or (f[m] == 0 and f[n] == 1):
            r += 1.0

        # Additional penalty: both ends unreplicated, but replicated region(s) in between
        if (f[m] == 0 and f[n] == 0) and n > m + 1:
            replicated_in_between = np.sum(f[m+1:n] == 1)
            if replicated_in_between > 0:
                r += replicated_in_between/(n-m)  # Penalty increases with size of replicated region

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
def E_cross(ms, ns, N_lef, N_lef2, k_norm, cohesin_blocks_condensin=False):
    '''
    The crossing energy.
    '''
    crossing = 0.0
    for i in range(N_lef+N_lef2):
        for j in range(i + 1, N_lef+N_lef2):
            # Only count crossing for allowed pairs
            if cohesin_blocks_condensin or (i < N_lef and j < N_lef) or (i >= N_lef and j >= N_lef):
                crossing += Kappa(ms[i], ns[i], ms[j], ns[j])
    return k_norm * crossing

@njit
def E_fold(ms, ns, fold_norm):
    ''''
    The folding energy.
    '''
    folding = np.sum(np.log(ns-ms+1))
    return fold_norm * folding

@njit
def E_potts(spins, J, h, ht, potts_norm1, potts_norm2, t, rep_fork_organizers):
    # Potts model energy calculation
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
            diff = np.abs(spins[i] - spins[j])
            E2 += J[i, j] * diff

    return potts_norm1 * E1 + potts_norm2 * E2

@njit
def get_E(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns, t, f_rep, spins, J, h, ht, potts_norm1=0.0, potts_norm2=0.0, rep_fork_organizers=True, cohesin_blocks_condensin=False):
    '''
    The total energy.
    '''
    energy = E_bind(L, R, ms, ns, bind_norm) + E_cross(ms, ns, N_lef, N_lef2, k_norm, cohesin_blocks_condensin) + E_fold(ms, ns, fold_norm)
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
def get_dE_fold(fold_norm,ms,ns,m_new,n_new,idx):
    '''
    Energy difference for folding energy.
    '''
    return fold_norm*(np.log(n_new-m_new+1)-np.log(ns[idx]-ms[idx]+1))

@njit
def get_dE_rep(f_rep, rep_norm, ms, ns, m_new, n_new, t, idx):
    '''
    Energy difference for replication energy.
    '''
    dE_rep = Rep_Penalty(m_new, n_new, f_rep[:, t]) - Rep_Penalty(ms[idx], ns[idx], f_rep[:, t - 1])
    return rep_norm * dE_rep

@njit
def get_dE_cross(ms, ns, m_new, n_new, idx, N_lef, N_lef2, k_norm, cohesin_blocks_condensin=False):
    '''
    Energy difference for crossing energy.
    '''
    K1, K2 = 0, 0
    
    for i in range(N_lef+N_lef2):
        if i != idx:
            # Only count crossing for allowed pairs
            if cohesin_blocks_condensin or (idx < N_lef and i < N_lef) or (idx >= N_lef and i >= N_lef):
                K1 += Kappa(ms[idx], ns[idx], ms[i], ns[i])
                K2 += Kappa(m_new, n_new, ms[i], ns[i])
    return k_norm*(K2 - K1)

@njit
def get_dE_node(spins,spin_idx,spin_val,J,h,ht_new,ht_old,potts_norm1,potts_norm2,t,rep_fork_organizers=True):
    # Energy difference for Potts node flip
    dE1 = h[spin_idx]*(spin_val-spins[spin_idx])/2+h[spin_idx]*(spin_val-spins[spin_idx])/2*(1-int(rep_fork_organizers))
    if t>0: dE1 += ((np.sum(ht_new*spins) - ht_new[spin_idx]*(spins[spin_idx]-spin_val) - np.sum(ht_old*spins))/2)*int(rep_fork_organizers)
    dE2 = np.sum(J[spin_idx, :] * (np.abs(spin_val - spins) - np.abs(spins[spin_idx] - spins)))
    return potts_norm1 * dE1 + potts_norm2 * dE2

@njit
def get_dE_potts_link(spins,J,m_new,n_new,m_old,n_old,potts_norm2=0.0):
    # Energy difference for Potts link update
    if m_new>=0 and m_old>=0:
        dE = J[m_new,n_new]*np.abs(spins[m_new]-spins[n_new])-J[m_old,n_old]*np.abs(spins[m_old]-spins[n_old])
    elif m_new<0 and m_old>=0:
        dE = -J[m_old,n_old]*np.abs(spins[m_old]-spins[n_old])
    elif m_new>=0 and m_old<0:
        dE = J[m_new,n_new]*np.abs(spins[m_new]-spins[n_new])
    else:
        dE = 0
    return potts_norm2*dE

@njit
def get_dE_rewiring(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns, m_new, n_new, idx, t, f_rep, spins, J, potts_norm2=0.0, cohesin_blocks_condensin=False):
    '''
    Total energy difference.
    '''
    dE = 0.0
    if idx < N_lef:
        dE += get_dE_fold(fold_norm, ms[:N_lef], ns[:N_lef], m_new, n_new, idx)
    else:
        dE += get_dE_fold(fold_norm2, ms[N_lef:N_lef+N_lef2], ns[N_lef:N_lef+N_lef2], m_new, n_new, idx - N_lef)
    dE += get_dE_bind(L, R, bind_norm, ms, ns, m_new, n_new, idx)
    dE += get_dE_cross(ms, ns, m_new, n_new, idx, N_lef, N_lef2, k_norm, cohesin_blocks_condensin)
    
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
    m_new = rd.randint(0, N_beads - 2)
    n_new = m_new
    return m_new, n_new

@njit
def slide(m_old, n_old, N_beads, f=None, t=0, rw=True):
    '''
    Sliding Monte-Carlo step.
    '''
    # Choose random step for sliding
    if n_old>m_old+1:
        r1 = np.random.choice(np.array([-1, 0, 1], dtype=np.int64)) if rw else -1
        r2 = np.random.choice(np.array([-1, 0, 1], dtype=np.int64)) if rw else 1
    else: # In case that it has just binded it has to extrude
        choices = np.array([-1, 1], dtype=np.int64)
        r1 = np.random.choice(np.array([-1, 0], dtype=np.int64)) if rw else -1
        r2 = np.random.choice(np.array([0, 1], dtype=np.int64)) if rw else 1
    m_new = m_old + r1 if m_old + r1 >= 0 else 0
    n_new = n_old + r2 if n_old + r2 < N_beads else N_beads - 1
    
    # Handle replication forks only if f is provided
    if f is not None:
        if f[m_new, t] != f[m_old, max(t - 1, 0)] and n_old>m_old:
            m_new = m_old+1
        if f[n_new, t] != f[n_old, max(t - 1, 0)] and n_old>m_old:
            n_new = n_old-1

    if n_new<m_new: n_new=m_new
    
    # They cannot go further than chromosome boundaries
    if n_new >= N_beads: n_new = N_beads - 1
    if m_new >= N_beads: m_new = N_beads - 1
    if m_new < 0: m_new = 0
    if n_new < 0: n_new = 0
    
    return int(m_new), int(n_new)

@njit
def initialize(N_lef, N_lef2, N_beads, random_init_spins=True):
    '''
    Random initial condition of the simulation.
    '''
    ms = np.full(N_lef + N_lef2, -5, dtype=np.int64)
    ns = np.full(N_lef + N_lef2, -5, dtype=np.int64)
    for j in range(N_lef):
        ms[j], ns[j] = unbind_bind(N_beads)
    if random_init_spins:
        state = np.random.randint(0, 2, size=N_beads) * 4 - 2
    else:
        state = np.zeros(N_beads, dtype=np.int64)
    return ms, ns, state

@njit
def initialize_J(N_beads, ms, ns):
    J = np.zeros((N_beads, N_beads), dtype=np.int64)
    for i in range(N_beads - 1):
        J[i, i + 1] = 1
        J[i + 1, i] = 1
    for idx in range(len(ms)):
        m, n = ms[idx], ns[idx]
        if m >= 0 and n >= 0 and n>m:  # Ensure valid indices
            J[m, n] += 1
            J[n, m] += 1
    return J

@njit
def run_energy_minimization(
    N_steps, N_sweep, N_lef, N_lef2, N_beads, MC_step, T,
    L, R, k_norm, fold_norm, fold_norm2, bind_norm,
    rep_norm=0.0, t_rep=np.inf, rep_duration=np.inf, f_rep=None,
    potts_norm1=0.0, potts_norm2=0.0, h=None, rw=True, spins=None,
    p_rew=0.5, rep_fork_organizers=True, cohesin_blocks_condensin=False, random_spins=True
):
    '''
    Runs a Monte Carlo or simulated annealing energy minimization for a chromatin simulation.
    [docstring omitted for brevity]
    '''
    ht = np.zeros(N_beads, dtype=np.float64)      # Time-dependent field (for Potts)
    ht_old = np.zeros(N_beads, dtype=np.float64)  # Previous time-dependent field
    mask = (ht_old == 0)  # Mask for updating ht
    n_accepted = 0

    # Possible spin values and indices
    spin_choices = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    lef_idx_choices = np.arange(N_lef, dtype=np.int64)

    # Initialize LEF positions and Potts spins
    ms, ns, spins = initialize(N_lef, N_lef2, N_beads, random_spins)
    spin_traj = np.zeros((N_beads, N_steps // MC_step), dtype=np.int32)

    # Initialize coupling matrix J with current LEF positions
    J = initialize_J(N_beads, ms, ns)

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
    inv_rep_duration = 1.0 / rep_duration if rep_duration != np.inf or rep_duration!=0 else 0.0

    # Progress bar setup
    last_percent = -1

    for i in range(N_steps):
        # Print progress every 5%
        percent = int(100 * i / N_steps)
        if percent % 5 == 0 and percent != last_percent:
            # Numba can't use print with flush, so just print
            print(f"Progress: {percent} % completed.")
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
                if dE <= 0 or np.exp(-dE / T) > np.random.rand():
                    E += dE
                    # Update J matrix for LEF move
                    if m_old >= 0 and n_old>m_old:
                        J[m_old, n_old] -= 1
                        J[n_old, m_old] -= 1
                    if m_new >= 0 and n_new>m_new:
                        J[m_new, n_new] += 1
                        J[n_new, m_new] += 1
                    ms[lef_idx], ns[lef_idx] = m_new, n_new
                    n_accepted += 1
            else:
                # Propose a Potts spin flip
                spin_idx = np.random.randint(N_beads)
                s_choices = spin_choices[spin_choices != spins[spin_idx]]
                s = s_choices[np.random.randint(s_choices.shape[0])]
                dE = get_dE_node(spins, spin_idx, s, J, h, ht, ht_old, potts_norm1, potts_norm2, rt, rep_fork_organizers)
                # Metropolis criterion
                if dE <= 0 or np.exp(-dE / T) > np.random.rand():
                    E += dE
                    spins[spin_idx] = s
                    n_accepted += 1
        
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

    acceptance_rate = n_accepted / (N_steps * N_sweep)
    return Ms, Ns, Es, Es_potts, Fs, Bs, spin_traj, mags, acceptance_rate