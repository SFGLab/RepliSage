from numba import njit, prange
import numpy as np
import random as rd
from preproc import *

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
def Kappa(mi,ni,mj,nj):
    '''
    Computes the crossing function of LoopSage.
    '''
    k=0.0
    if mi<mj and mj<ni and ni<nj: k+=1 # np.abs(ni-mj)+1
    if mj<mi and mi<nj and nj<ni: k+=1 # np.abs(nj-mi)+1
    if mj==ni or mi==nj or ni==nj or mi==mj: k+=1
    return k

@njit
def E_bind(L, R, ms, ns, bind_norm):
    '''
    The binding energy.
    '''
    binding = np.sum(L[ms] + R[ns])
    E_b = bind_norm * binding
    return E_b

@njit
def E_repli(l_forks, r_forks, ms, ns, t, rep_norm):
    '''
    The replication energy.
    '''
    replication = np.sum(l_forks[ms, t] + l_forks[ns, t] + r_forks[ms, t] + r_forks[ns, t])
    return rep_norm * replication

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
    folding = np.sum(np.log(ns - ms))
    return fold_norm * folding

@njit(parallel=True)
def E_ising(spins, J, h, ising_norm1, ising_norm2):
    N_beads = len(J)
    E1 = np.sum(h*spins)
    
    E2 = 0.0
    for i in range(N_beads):
        E2 += np.sum(J[i, i + 1:] * spins[i] * spins[i + 1:])
    
    return ising_norm1 * E1 + ising_norm2 * E2

# @njit
# def E_comp(spins,ms,ns,comp_norm):
#     Ec = 0
#     for i in range(len(ms)):
#         Ec += np.sum(0.25 * spins[ms[i]:ns[i]]) + 0.75*(ns[i]-ms[i])
#     return comp_norm * Ec

@njit
def get_E(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, t, l_forks, r_forks, spins, J, h, ising_norm1=0.0, ising_norm2=0.0, comp_norm=0.0):
    '''
    The totdal energy.
    '''
    energy = E_bind(L, R, ms, ns, bind_norm) + E_cross(ms, ns, k_norm) + E_fold(ms, ns, fold_norm)
    if rep_norm!=0.0: energy += E_repli(l_forks, r_forks, ms, ns, t, rep_norm)
    if (ising_norm1!=0.0 or ising_norm2!=0): energy += E_ising(spins, J, h, ising_norm1, ising_norm2)
    # if comp_norm!=0.0: energy += E_comp(spins,ms,ns,comp_norm)
    return energy

@njit
def get_dE_bind(L,R,bind_norm,ms,ns,m_new,n_new,idx):
    '''
    Energy difference for binding energy.
    '''
    return bind_norm*(L[m_new]+R[n_new]-L[ms[idx]]-R[ns[idx]])
    
@njit
def get_dE_fold(fold_norm,ms,ns,m_new,n_new,idx):
    '''
    Energy difference for folding energy.
    '''
    return fold_norm*(np.log(n_new-m_new)-np.log(ns[idx]-ms[idx]))

@njit
def get_dE_rep(l_forks,r_forks, rep_norm, ms,ns,m_new,n_new,t,idx):
    '''
    Energy difference for replication energy.
    '''
    E_rep_new = l_forks[m_new,t]+l_forks[n_new,t]+r_forks[m_new,t]+r_forks[n_new,t]
    E_rep_old = l_forks[ms[idx],t-1]+l_forks[ns[idx],t-1]+r_forks[ms[idx],t-1]+r_forks[ns[idx],t-1]
    return rep_norm*(E_rep_new-E_rep_old)

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
def get_dE_ising_node(spins, J, h, spin_idx, ising_norm1, ising_norm2):
    # In case that we change node state
    dE1 = -2*h[spin_idx]*spins[spin_idx]
    dE2 = -2*spins[spin_idx]* np.sum(J[spin_idx,:]*spins)
    return ising_norm1 * dE1 + ising_norm2 * dE2

@njit
def get_dE_ising_link(spins,m_new,n_new,m_old,n_old, ising_norm2=0.0):
    dE = spins[m_new]*spins[n_new]-spins[m_old]*spins[n_old]
    return ising_norm2*dE

# @njit
# def get_dE_comp(spins, comp_norm, m_old, n_old, m_new, n_new):
#     dE = (0.25*np.sum(spins[m_new:n_new])+0.75*(n_new-m_new))-(0.25*np.sum(spins[m_old:n_old])+0.75*(n_old-m_old))
#     return comp_norm * dE

@njit
def get_dE_rewiring(L, R, bind_norm, fold_norm, k_norm ,rep_norm, ms, ns, m_new, n_new, idx,  t, l_forks, r_forks, spins, ising_norm2=0.0, comp_norm=0.0):
    '''
    Total energy difference.
    '''
    dE = 0.0
    dE += get_dE_fold(fold_norm,ms,ns,m_new,n_new,idx)
    dE += get_dE_bind(L, R, bind_norm, ms, ns, m_new, n_new, idx)
    dE += get_dE_cross(ms, ns, m_new, n_new, idx, k_norm)
    if rep_norm!=0.0:
        dE += get_dE_rep(l_forks, r_forks, rep_norm, ms, ns, m_new, n_new, t, idx)
    else:
        dE+= 0
    if ising_norm2!=0.0: dE += get_dE_ising_link(spins, m_new, n_new, ms[idx], ns[idx], ising_norm2)
    # if comp_norm!=0.0: dE += get_dE_comp(spins, comp_norm, ms[idx], ns[idx], m_new, n_new)
    return dE

@njit
def unbind_bind(N_beads):
    '''
    Rebinding Monte-Carlo step.
    '''
    m_new = rd.randint(0, N_beads - 3)
    n_new = m_new + 2
    return int(m_new), int(n_new)

@njit
def slide(m_old, n_old, N_beads, rw=True):
    '''
    Sliding Monte-Carlo step.
    '''
    choices = np.array([-1, 1], dtype=np.int64)
    r1 = np.random.choice(choices) if rw else -1
    r2 = np.random.choice(choices) if rw else 1
    m_new = m_old + r1 if m_old + r1>=0 else 0
    n_new = n_old + r2 if n_old + r2<N_beads else N_beads-1
    return int(m_new), int(n_new)

@njit(parallel=True)
def initialize(N_lef, N_beads):
    '''
    Random initial condition of the simulation.
    '''
    ms, ns = np.zeros(N_lef, dtype=np.int64), np.zeros(N_lef, dtype=np.int64)
    for i in range(N_lef):
        ms[i], ns[i] = unbind_bind(N_beads)
    state = np.random.randint(0, 2, size=N_beads) * 2 - 1
    return ms, ns, state

@njit
def initialize_J(N_beads,ms,ns):
    J = np.zeros((N_beads,N_beads),dtype=np.int64)
    for i in range(N_beads-1):
        J[i,i+1] = 1
        J[i+1,i] = 1
    for m, n in zip(ms,ns):
        J[m,n] = 1
        J[n,m] = 1
    return J

@njit
def run_energy_minimization(N_steps, N_lef, N_CTCF, N_beads, MC_step, T, T_min, mode, L, R, k_norm, fold_norm, bind_norm, rep_norm=0.0, t_rep=np.inf, rep_duration=np.inf, l_forks=np.array([[1,0],[1,0]],dtype=np.int32), r_forks=np.array([[1,0],[1,0]],dtype=np.int32), ising_norm1=0.0, ising_norm2=0.0, comp_norm=0.0, J=None, h=None, rw=True, spins=None):
    '''
    It performs Monte Carlo or simulated annealing of the simulation.
    '''
    # Initialization of parameters
    Ti = T
    
    # Initialization of matrices
    ms, ns, spins = initialize(N_lef, N_beads)
    spin_traj = np.zeros((N_beads, N_steps//MC_step),dtype=np.int32)
    J = initialize_J(N_beads,ms,ns)
    E = get_E(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, 0, l_forks, r_forks, spins, J, h, ising_norm1, ising_norm2)
    Es = np.zeros(N_steps//MC_step, dtype=np.float64)
    Es_ising = np.zeros(N_steps//MC_step, dtype=np.float64)
    Fs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Bs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Rs = np.zeros(N_steps//MC_step, dtype=np.float64)
    E_is = E_ising(spins, J, h, ising_norm1, ising_norm2)
    Ms, Ns = np.zeros((N_lef, N_steps//MC_step), dtype=np.int32), np.zeros((N_lef, N_steps//MC_step), dtype=np.int32)
    
    for i in range(N_steps):
        # Calculate replication time
        rt = 0 if i < t_rep else int(i - t_rep) if (i >= t_rep and i < t_rep + rep_duration) else int(rep_duration)-1
        Ti = T - (T - T_min) * i / N_steps if mode == 'Annealing' else T

        for j in prange(N_lef): # In each swift we move one cohesin and one random spin
            # Propose a move for cohesins (rewiring)
            r = np.random.choice(np.array([0, 1]))
            m_old, n_old = ms[j], ns[j]
            if r==0:
                m_new, n_new = unbind_bind(N_beads)
            elif r==1:
                m_new, n_new = slide(ms[j], ns[j], N_beads, rw)
            
            # Cohesin energy difference for rewiring move
            dE = get_dE_rewiring(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, m_new, n_new, j, rt, l_forks, r_forks, spins, ising_norm2, comp_norm)
            if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                E += dE
                E_is += get_dE_ising_link(spins,ms[j],ns[j],m_old,n_old, ising_norm1)
                J[ms[j],ns[j]], J[ns[j],ms[j]] = 0, 0
                J[m_new,n_new], J[n_new,m_new] = 1, 1
                ms[j], ns[j] = m_new, n_new
            
            # Propose a node state change
            spin_idx = np.random.choice(np.arange(N_beads))

            # Compute the energy that corresponds only to the node change
            dE = get_dE_ising_node(spins, J, h, spin_idx, ising_norm1, ising_norm2)
            if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                E_is += dE
                spins[spin_idx] *= -1

        # Keep track on energies and trajectories of LEFs and spins
        if i % MC_step == 0:
            Es[i//MC_step] = E
            Es_ising[i//MC_step] = E_is
            Ms[:, i//MC_step], Ns[:, i//MC_step] = ms, ns
            spin_traj[:,i//MC_step] = spins
            Fs[i//MC_step] = E_fold(ms, ns, fold_norm)
            Bs[i//MC_step] = E_bind(L,R,ms,ns,bind_norm)
            if rep_norm!=0.0: Rs[i//MC_step] = E_repli(l_forks,r_forks,ms,ns,rt,rep_norm)
    return Ms, Ns, Es, Es_ising, Fs, Bs, Rs, spin_traj, J
