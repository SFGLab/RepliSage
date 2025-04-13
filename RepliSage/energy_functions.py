from numba import njit, prange
import numpy as np
import random as rd
from .preproc import *
from tqdm import tqdm

def preprocessing(bedpe_file:str, region:list, chrom:str, N_beads:int):
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
    N_CTCF = np.max([np.count_nonzero(L), np.count_nonzero(R)])
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
def Kappa(mi,ni,mj,nj):
    '''
    Computes the crossing function of LoopSage.
    '''
    k=0.0
    if mi<mj and mj<ni and ni<nj and mi>=0: k+=1 # np.abs(ni-mj)+1
    if mj<mi and mi<nj and nj<ni and mj>=0: k+=1 # np.abs(nj-mi)+1
    # if mi==mj and ni==nj and mi>=0: k+=1
    if mj==ni or mi==nj or ni==nj or mi==mj: k+=1
    return k

@njit
def Rep_Penalty(m, n, f):
    r = 0.0
    
    # The case that cohesin crosses a replication fork: for sure penalized
    if f[m] != f[n] and m>=0: 
        r += 1.0

    if (f[m] == 1 and f[n]==1) and np.any(f[m:n]==0) and m>=0:
        r += 1.0
    
    # if (f[m] == 0 and f[n]==0):
    #     r += 1/max(n-m,1)*np.sum(f[m+1:n]!=f[m])
    return r

@njit
def E_bind(L, R, ms, ns, bind_norm):
    '''
    The binding energy.
    '''
    binding = np.sum(L[ms[ms>=0]] + R[ns[ns>=0]])
    E_b = bind_norm * binding
    return E_b

@njit
def E_rep(f_rep,ms,ns,t,rep_norm):
    '''
    Penalty of the replication energy.
    '''
    E_penalty = 0
    for i in range(len(ms)):
        E_penalty += Rep_Penalty(ms[i], ns[i], f_rep[:,t])
    return rep_norm * E_penalty

@njit(parallel=True)
def E_cross(ms, ns, k_norm):
    '''
    The crossing energy.
    '''
    crossing = 0.0
    N_lef = len(ms)
    for i in range(N_lef):
        for j in range(i + 1, N_lef):
            crossing += Kappa(ms[i], ns[i], ms[j], ns[j])
    return k_norm * crossing

@njit
def E_fold(ms, ns, fold_norm):
    ''''
    The folding energy.
    '''
    folding = np.sum(np.log(np.maximum(ns - ms, 1)))
    return fold_norm * folding

@njit(parallel=True)
def E_potts(spins, J, h, ht, potts_norm1, potts_norm2, t, rep_fork_organizers):
    N_beads = len(J)
    E1 = np.sum(h*spins)/2 + np.sum(h*spins)/2*(1-int(rep_fork_organizers))
    if t>0: E1 += np.sum(ht*spins)/2*int(rep_fork_organizers)
    
    E2 = 0.0
    for i in range(N_beads):
        E2 += np.sum(J[i, i + 1:] * np.abs(spins[i] - spins[i + 1:]))

    return potts_norm1 * E1 + potts_norm2 * E2

@njit
def get_E(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns, t, f_rep, spins, J, h, ht, potts_norm1=0.0, potts_norm2=0.0, rep_fork_organizers=True):
    '''
    The total energy.
    '''
    energy = E_bind(L, R, ms, ns, bind_norm) + E_cross(ms, ns, k_norm) + E_fold(ms, ns, fold_norm)
    if fold_norm2!=0: energy += E_fold(ms[N_lef:N_lef+N_lef2],ns[N_lef:N_lef+N_lef2],fold_norm2)
    if rep_norm!=0.0:
        energy += E_rep(f_rep, ms, ns, t, rep_norm)
    if (potts_norm1!=0.0 or potts_norm2!=0.0): energy += E_potts(spins, J, h, ht, potts_norm1, potts_norm2, t, rep_fork_organizers)
    return energy

@njit
def get_dE_bind(L,R,bind_norm,ms,ns,m_new,n_new,idx):
    '''
    Energy difference for binding energy.
    '''
    B_new = L[m_new]+R[n_new] if m_new>=0 else 0
    B_old = L[ms[idx]]+R[ns[idx]] if ms[idx]>=0 else 0
    return bind_norm*(B_new-B_old)

@njit
def get_dE_fold(fold_norm,ms,ns,m_new,n_new,idx):
    '''
    Energy difference for folding energy.
    '''
    return fold_norm*(np.log(np.maximum(n_new-m_new,1))-np.log(np.maximum(ns[idx]-ms[idx],1)))

@njit
def get_dE_rep(f_rep,rep_norm, ms,ns,m_new,n_new,t,idx):
    '''
    Energy difference for replication energy.
    '''
    dE_rep = Rep_Penalty(m_new, n_new, f_rep[:,t]) - Rep_Penalty(ms[idx], ns[idx], f_rep[:,t-1])
    return rep_norm*dE_rep

@njit(parallel=True)
def get_dE_cross(ms, ns, m_new, n_new, idx, k_norm):
    '''
    Energy difference for crossing energy.
    '''
    K1, K2 = 0, 0
    N_lef = len(ms)
    
    for i in range(N_lef):
        if i != idx:
            K1 += Kappa(ms[idx], ns[idx], ms[i], ns[i])
            K2 += Kappa(m_new, n_new, ms[i], ns[i])
    return k_norm * (K2 - K1)

@njit
def get_dE_node(spins,spin_idx,spin_val,J,h,ht_new,ht_old,ms,ns,potts_norm1,potts_norm2,t,rep_fork_organizers=True):
    # In case that we change node state
    dE1 = h[spin_idx]*(spin_val-spins[spin_idx])/2+h[spin_idx]*(spin_val-spins[spin_idx])/2*(1-int(rep_fork_organizers))
    if t>0: dE1 += ((np.sum(ht_new*spins) - ht_new[spin_idx]*(spins[spin_idx]-spin_val) - np.sum(ht_old*spins))/2)*int(rep_fork_organizers)
    dE2 = np.sum(J[spin_idx, :] * (np.abs(spin_val - spins) - np.abs(spins[spin_idx] - spins)))
    return potts_norm1 * dE1 + potts_norm2 * dE2

@njit
def get_dE_potts_link(spins,J,m_new,n_new,m_old,n_old,N_lef,potts_norm2=0.0):
    N_beads = len(spins)
    if m_new>=0 and m_old>=0:
        dE = J[m_new,n_new]*(spins[m_new]==spins[n_new])-J[m_old,n_old]*(spins[m_old]==spins[n_old])
    elif m_new<0 and m_old>=0:
        dE = -J[m_old,n_old]*(spins[m_old]==spins[n_old])
    elif m_new>=0 and m_old<0:
        dE = J[m_new,n_new]*(spins[m_new]==spins[n_new])
    else:
        dE = 0
    return potts_norm2*dE

@njit
def get_dE_rewiring(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm ,rep_norm, ms, ns, m_new, n_new, idx,  t, f_rep, spins, J, potts_norm2=0.0):
    '''
    Total energy difference.
    '''
    dE = 0.0
    if idx<N_lef:
        dE += get_dE_fold(fold_norm,ms[:N_lef],ns[:N_lef],m_new,n_new,idx)
    else:
        dE += get_dE_fold(fold_norm2,ms[N_lef:N_lef+N_lef2],ns[N_lef:N_lef+N_lef2],m_new,n_new,idx-N_lef)
    dE += get_dE_bind(L, R, bind_norm, ms, ns, m_new, n_new, idx)
    dE += get_dE_cross(ms, ns, m_new, n_new, idx, k_norm)
    if rep_norm!=0.0:
        dE += get_dE_rep(f_rep, rep_norm, ms, ns, m_new, n_new, t, idx)
    if potts_norm2!=0.0: dE += get_dE_potts_link(spins,J, m_new, n_new, ms[idx], ns[idx], len(ms), potts_norm2)
    return dE

@njit
def unbind_bind(N_beads):
    '''
    Rebinding Monte-Carlo step.
    '''
    m_new = rd.randint(0, N_beads - 4)
    n_new = m_new + 3
    return int(m_new), int(n_new)

@njit
def slide(m_old, n_old, N_beads, f, t, rw=True):
    '''
    Sliding Monte-Carlo step.
    '''
    choices = np.array([-1, 1], dtype=np.int64)
    r1 = np.random.choice(choices) if rw else -1
    r2 = np.random.choice(choices) if rw else 1

    m_new = m_old + r1 if m_old + r1>=0 else 0
    if f[m_new,t]!=f[m_old,max(t-1,0)] and np.any(f[:,t]==0):
        m_new = closest_opposite(f[:,t], m_new)
    
    n_new = n_old + r2 if n_old + r2 < N_beads else N_beads-1
    if f[n_new,t]!=f[n_old,max(t-1,0)] and np.any(f[:,t]==0):
        n_new = closest_opposite(f[:,t], n_new)
    return int(m_new), int(n_new)

@njit(parallel=True)
def initialize(N_lef, N_lef2, N_beads):
    '''
    Random initial condition of the simulation.
    '''
    ms, ns = np.zeros(N_lef+N_lef2, dtype=np.int64), np.zeros(N_lef+N_lef2, dtype=np.int64)
    for i in range(N_lef):
        ms[i], ns[i] = unbind_bind(N_beads)
    for i in range(N_lef,N_lef2):
        ms[i], ns[i] = -2, -1
    state = np.random.randint(0, 2, size=N_beads) * 4 - 2
    return ms, ns, state

@njit
def initialize_J(N_beads,J,ms,ns):
    N_lef = len(ms)
    for i in range(N_beads-1):
        J[i,i+1] += 1
        J[i+1,i] += 1
    for m, n in zip(ms,ns):
        J[m,n] += 1
        J[n,m] += 1
    return J

@njit
def run_energy_minimization(N_steps, N_lef, N_lef2, N_beads, MC_step, T, T_min, mode, L, R, k_norm, fold_norm, fold_norm2, bind_norm, rep_norm=0.0, t_rep=np.inf, rep_duration=np.inf, f_rep=np.array([[1,0],[1,0]],dtype=np.int32), potts_norm1=0.0, potts_norm2=0.0, J=None, h=None, rw=True, spins=None, p_rew=0.5, rep_fork_organizers=True):
    '''
    It performs Monte Carlo or simulated annealing of the simulation.
    '''
    # Initialization of parameters@n
    Ti = T

    # Initialization of the time dependent component of the magnetic field
    ht = ht_old = np.zeros(N_beads,dtype=np.float64)
    mask = (ht_old == 0)

    # Choices for MC
    spin_choices = np.array([-2,-1,0,1,2])
    spin_idx_choices = np.arange(N_beads)
    lef_idx_choices = np.arange(N_lef)
    
    # Initialization of matrices
    ms, ns, spins = initialize(N_lef,N_lef2, N_beads)
    spin_traj = np.zeros((N_beads, N_steps//MC_step),dtype=np.int32)
    J = initialize_J(N_beads,J,ms,ns)
    E = get_E(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns, 0, f_rep, spins, J, h, ht, potts_norm1, potts_norm2, rep_fork_organizers)
    Es = np.zeros(N_steps//MC_step, dtype=np.float64)
    Ks = np.zeros(N_steps//MC_step, dtype=np.float64)
    Es_potts = np.zeros(N_steps//MC_step, dtype=np.float64)
    mags = np.zeros(N_steps//MC_step, dtype=np.float64)
    Fs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Bs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Rs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Ms, Ns = np.zeros((N_lef+N_lef2, N_steps//MC_step), dtype=np.int32), np.zeros((N_lef+N_lef2, N_steps//MC_step), dtype=np.int32)

    for i in range(N_steps):
        # Calculate replication time
        rt = 0 if i < t_rep else int(i - t_rep) if (i >= t_rep and i < t_rep + rep_duration) else int(rep_duration)-1
        if rt==(int(rep_duration)-1): lef_idx_choices = np.arange(N_lef+N_lef2)
        mag_field = 2 * (1 - 2 * rt / rep_duration)
        ht += mask * mag_field * f_rep[:, rt]
        Ti = T - (T - T_min) * i / N_steps if mode == 'Annealing' else T
        
        for j in range(N_beads):
            # Propose a move for cohesins (rewiring)
            do_rewiring = rd.random()<p_rew
            if do_rewiring:
                lef_idx = np.random.choice(lef_idx_choices)
                m_old, n_old = ms[lef_idx], ns[lef_idx]
                r = np.random.choice(np.array([0,1]))
                if m_old<0 or n_old<0: r=0
                if r==0:
                    m_new, n_new = unbind_bind(N_beads)
                elif r==1:
                    m_new, n_new = slide(ms[lef_idx], ns[lef_idx], N_beads, f_rep, rt, rw)
                    
                # Cohesin energy difference for rewiring move
                dE = get_dE_rewiring(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns, m_new, n_new, lef_idx, rt, f_rep, spins, J, potts_norm2)
                if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                    E += dE
                    # Change the inetraction matrix
                    if m_old>=0:
                        J[m_old,n_old] -= 1
                        J[n_old,m_old] -= 1
                    if m_new>=0:
                        J[m_new,n_new] += 1
                        J[n_new,m_new] += 1
                    ms[lef_idx], ns[lef_idx] = m_new, n_new
            else:
                # Propose a node state change
                spin_idx = np.random.choice(spin_idx_choices)
                s = np.random.choice(spin_choices[spin_choices!=spins[spin_idx]])

                # Compute the energy that corresponds only to the node change
                dE = get_dE_node(spins,spin_idx,s,J,h,ht,ht_old,ms,ns,potts_norm1,potts_norm2,rt,rep_fork_organizers)
                if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                    E += dE
                    spins[spin_idx] = s
        ht_old = ht
        mask = (ht_old == 0)
        
        # Keep track on energies and trajectories of LEFs and spins
        if i % MC_step == 0:
            Es[i//MC_step] = get_E(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns, i//MC_step, f_rep, spins, J, h, ht, potts_norm1, potts_norm2, rep_fork_organizers)
            Ks[i//MC_step] = E_cross(ms, ns, k_norm)
            mags[i//MC_step] = np.average(spins)
            Ms[:, i//MC_step], Ns[:, i//MC_step] = ms, ns
            spin_traj[:,i//MC_step] = spins
            Es_potts[i//MC_step] = E_potts(spins, J, h, ht, potts_norm1, potts_norm2, rt, rep_fork_organizers)
            Fs[i//MC_step] = E_fold(ms, ns, fold_norm)
            Bs[i//MC_step] = E_bind(L,R,ms,ns,bind_norm)
            if rep_norm!=0.0: Rs[i//MC_step] = E_rep(f_rep,ms,ns,rt,rep_norm)

    return Ms, Ns, Es, Es_potts, Fs, Bs, Rs, Ks, spin_traj, mags