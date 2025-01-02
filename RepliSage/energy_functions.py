from numba import njit, prange
import numpy as np
import random as rd
from preproc import *
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
def Kappa(mi,ni,mj,nj):
    '''
    Computes the crossing function of LoopSage.
    '''
    k=0.0
    if mi<mj and mj<ni and ni<nj and mi>=0: k+=1 # np.abs(ni-mj)+1
    if mj<mi and mi<nj and nj<ni and mj>=0: k+=1 # np.abs(nj-mi)+1
    if mi==mj and ni==nj and mi>=0: k+=1
    return k

@njit
def E_bind(L, R, ms, ns, bind_norm):
    '''
    The binding energy.
    '''
    binding = np.sum(L[ms[ms>=0]] + R[ns[ns>=0]])
    E_b = bind_norm * binding
    return E_b

@njit
def E_repli(l_forks, r_forks, ms, ns, t, rep_norm):
    '''
    The replication energy.
    '''
    replication = np.sum(l_forks[ms, t] + l_forks[ns, t] + r_forks[ms, t] + r_forks[ns, t])
    return rep_norm * replication

@njit
def E_rep_penalty(l_forks,r_forks,ms,ns,t,kr_norm):
    '''
    Penalty of the replication energy.
    '''
    mms, nns = ms[ms>=0], ns[ns>=0]
    E_penalty = 0
    for i in range(len(ms)):
        E_penalty += np.sum(l_forks[mms[i]+1:nns[i]]+r_forks[mms[i]+1:nns[i]])
    return kr_norm * E_penalty

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
    folding = np.sum(np.log(ns[ns>=0] - ms[ms>=0]))
    return fold_norm * folding

@njit(parallel=True)
def E_potts(spins, J, h, ht, potts_norm1, potts_norm2, t, T, l_forks, r_forks):
    N_beads = len(J)
    E1 = np.sum(h*spins)/2
    if t>0: E1 += np.sum(ht*spins)/2
    
    E2 = 0.0
    for i in range(N_beads):
        E2 += np.sum(J[i, i + 1:] * (spins[i] == spins[i + 1:]))
    
    return potts_norm1 * E1 + potts_norm2 * E2

@njit
def get_E(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, kr_norm, gamma, ms, ns, t, l_forks, r_forks, spins, J, h, ht, potts_norm1=0.0, potts_norm2=0.0):
    '''
    The total energy.
    '''
    energy = E_bind(L, R, ms, ns, bind_norm) + E_cross(ms, ns, k_norm) + E_fold(ms, ns, fold_norm)
    if fold_norm2!=0: energy += E_fold(ms[N_lef:N_lef+N_lef2],ns[N_lef:N_lef+N_lef2],fold_norm2)
    if rep_norm!=0.0:
        energy += E_repli(l_forks, r_forks, ms, ns, t, rep_norm)
        energy += E_rep_penalty(l_forks,r_forks,ms,ns,t,kr_norm)
    if (potts_norm1!=0.0 or potts_norm2!=0.0): energy += E_potts(spins, J, h, ht, potts_norm1, potts_norm2, t, 1, l_forks, r_forks)
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
    return fold_norm*(np.log(n_new-m_new)-np.log(ns[idx]-ms[idx]))

@njit
def get_dE_rep(l_forks,r_forks,rep_norm, ms,ns,m_new,n_new,t,idx):
    '''
    Energy difference for replication energy.
    '''
    E_rep_new = l_forks[m_new,t]+l_forks[n_new,t]+r_forks[m_new,t]+r_forks[n_new,t] if m_new>=0 else 0
    E_rep_old = l_forks[ms[idx],t-1]+l_forks[ns[idx],t-1]+r_forks[ms[idx],t-1]+r_forks[ns[idx],t-1] if ms[idx]>=0 else 0
    return rep_norm*(E_rep_new-E_rep_old)

@njit
def get_dE_rep_penalty(l_forks,r_forks, kr_norm, ms,ns,m_new,n_new,t,idx):
    E_kr_new = np.sum(l_forks[(m_new+1):n_new,t]+r_forks[(m_new+1):n_new,t]) if m_new>=0 else 0
    E_kr_old = np.sum(l_forks[(ms[idx]+1):ns[idx],t-1]+r_forks[(ms[idx]+1):ns[idx],t-1]) if ms[idx]>=0 else 0
    return kr_norm*(E_kr_new-E_kr_old)

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
def get_dE_node(spins,spin_idx,spin_val,J,h,ht_new,ht_old,ms,ns,potts_norm1,potts_norm2, t, T, l_forks, r_forks):
    # In case that we change node state
    dE1 = h[spin_idx]*(spin_val-spins[spin_idx])/2
    if t>0: dE1 += (np.sum(ht_new*spins) - ht_new[spin_idx]*(spins[spin_idx]-spin_val) - np.sum(ht_old*spins))/2
    dE2 = np.sum(J[spin_idx, :] * ((spin_val == spins) - (spins[spin_idx] == spins)))
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
def get_dE_rewiring(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm ,rep_norm, kr_norm, gamma, ms, ns, m_new, n_new, idx,  t, l_forks, r_forks, spins, J, potts_norm2=0.0):
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
        dE += get_dE_rep(l_forks, r_forks, rep_norm, ms, ns, m_new, n_new, t, idx)
        dE += get_dE_rep_penalty(l_forks, r_forks, kr_norm, ms, ns, m_new, n_new, t, idx)
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
def slide(m_old, n_old, N_beads, rw=True):
    '''
    Sliding Monte-Carlo step.
    '''
    dist = 0
    while dist<3:
        choices = np.array([-1, 1], dtype=np.int64)
        r1 = np.random.choice(choices) if rw else -1
        r2 = np.random.choice(choices) if rw else 1
        m_new = m_old + r1 if m_old + r1>=0 else 0
        n_new = n_old + r2 if n_old + r2<N_beads else N_beads-1
        dist = n_new-m_new
    return int(m_new), int(n_new)

@njit(parallel=True)
def initialize(N_lef, N_beads):
    '''
    Random initial condition of the simulation.
    '''
    ms, ns = np.zeros(N_lef, dtype=np.int64), np.zeros(N_lef, dtype=np.int64)
    for i in range(N_lef):
        ms[i], ns[i] = unbind_bind(N_beads)
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
def run_energy_minimization(N_steps, N_lef, N_lef2, N_CTCF, N_beads, MC_step, T, T_min, mode, L, R, k_norm, fold_norm, fold_norm2, bind_norm, gamma, kr_norm=0.0, rep_norm=0.0, t_rep=np.inf, rep_duration=np.inf, l_forks=np.array([[1,0],[1,0]],dtype=np.int32), r_forks=np.array([[1,0],[1,0]],dtype=np.int32), potts_norm1=0.0, potts_norm2=0.0, J=None, h=None, rw=True, spins=None, p1=0.5, p2=0.5):
    '''
    It performs Monte Carlo or simulated annealing of the simulation.
    '''
    # Initialization of parameters@n
    Ti = T

    # Initialization of the time dependent component of the magnetic field
    ht = ht_old = np.zeros(N_beads,dtype=np.float64)

    # Choices for MC
    N_rep = np.max(np.sum(l_forks,axis=0))
    spin_choices = np.array([-2,-1,0,1,2])
    spin_idx_choices = np.arange(N_beads)
    lef_idx_choices = np.arange(N_lef+N_lef2)
    
    # Initialization of matrices
    ms, ns, spins = initialize(N_lef+N_lef2, N_beads)
    spin_traj = np.zeros((N_beads, N_steps//MC_step),dtype=np.int32)
    J = initialize_J(N_beads,J,ms,ns)
    E = get_E(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, kr_norm, gamma, ms, ns, 0, l_forks, r_forks, spins, J, h, ht, potts_norm1, potts_norm2)
    Es = np.zeros(N_steps//MC_step, dtype=np.float64)
    Es_potts = np.zeros(N_steps//MC_step, dtype=np.float64)
    mags = np.zeros(N_steps//MC_step, dtype=np.float64)
    N_lefs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Fs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Bs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Rs = np.zeros(N_steps//MC_step, dtype=np.float64)
    E_is = E_potts(spins, J, h, ht, potts_norm1, potts_norm2, 0, 1, l_forks, r_forks)
    Ms, Ns = np.zeros((N_lef+N_lef2, N_steps//MC_step), dtype=np.int32), np.zeros((N_lef+N_lef2, N_steps//MC_step), dtype=np.int32)

    for i in range(N_steps):
        # Calculate replication time
        rt = 0 if i < t_rep else int(i - t_rep) if (i >= t_rep and i < t_rep + rep_duration) else int(rep_duration)-1
        mag_field = - 2 * (1 - 2 * rt / rep_duration)
        ht += mag_field * (l_forks[:, rt] + r_forks[:, rt])/N_lef
        Ti = T - (T - T_min) * i / N_steps if mode == 'Annealing' else T
        
        for j in range(N_lef):
            # Propose a move for cohesins (rewiring)
            do_rewiring = rd.random()<p1
            if do_rewiring:
                lef_idx = np.random.choice(lef_idx_choices)
                m_old, n_old = ms[lef_idx], ns[lef_idx]
                r = np.random.choice(np.array([0,1]))
                if r==0:
                    m_new, n_new = unbind_bind(N_beads)
                else:
                    m_new, n_new = slide(ms[lef_idx], ns[lef_idx], N_beads, rw)
                
                # Cohesin energy difference for rewiring move
                dE = get_dE_rewiring(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, kr_norm, gamma, ms, ns, m_new, n_new, lef_idx, rt, l_forks, r_forks, spins, J, potts_norm2)
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
                dE = get_dE_node(spins,spin_idx,s,J,h,ht,ht_old,ms,ns,potts_norm1,potts_norm2, rt, rep_duration, l_forks, r_forks)
                if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                    E_is += dE
                    spins[spin_idx] = s
        ht_old = ht

        # Keep track on energies and trajectories of LEFs and spins
        if i % MC_step == 0:
            Es[i//MC_step] = E
            N_lefs[i//MC_step] = len(ms[ms>=0])
            mags[i//MC_step] = np.average(spins)
            Ms[:, i//MC_step], Ns[:, i//MC_step] = ms, ns
            spin_traj[:,i//MC_step] = spins
            Es_potts[i//MC_step] = E_is
            Fs[i//MC_step] = E_fold(ms, ns, fold_norm)
            Bs[i//MC_step] = E_bind(L,R,ms,ns,bind_norm)
            if rep_norm!=0.0: Rs[i//MC_step] = E_repli(l_forks,r_forks,ms,ns,rt,rep_norm)
    return Ms, Ns, Es, Es_potts, Fs, Bs, Rs, spin_traj, J, mags, N_lefs