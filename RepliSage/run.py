##############################################################################
################### SEBASTIAN KORSAK 2024, WARSAW, POLAND ####################
##############################################################################
## This script runs a stochastic simulation, similar like LoopSage. ##########
## The given script is parallelized across CPU cores and has been modified ###
## to simulate the propagation of replication forks, which act as barriers. ##
##############################################################################
##############################################################################
# 
# Basic Libraries
import numpy as np
import random as rd

# Hide warnings
import warnings
import time
from numba import njit, prange
warnings.filterwarnings('ignore')

# My own libraries
from Replikator import *
from preproc import *
from plots import *
from em import *

def preprocessing(bedpe_file:str, region:list, chrom:str, N_beads:int):
    '''
    It computes the binding potential and the number of CTCF motifs.
    ---------------------------------------------------------------
    Input:
    bedpe_file: the path of the bedpe file.
    region: the coordinates of region in genomic distance units, in format of list [start,end].
    chrom: the chromosome of interest.
    N_beads: the number of simulation beads.
    '''
    L, R, J, dists = binding_vectors_from_bedpe(bedpe_file, N_beads, region, chrom, False, False)
    N_CTCF = np.max([np.count_nonzero(L), np.count_nonzero(R)])
    return L, R, J, dists, N_CTCF

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

@njit(parallel=True)
def compute_J(ms,ns,N_beads):
    '''
    Compute interaction coefficient of the Ising model.
    We assume that there are interactions where are cohesins.
    '''
    J = np.zeros((N_beads,N_beads),dtype=np.int32)
    N_lef = len(ms)
    
    for i in range(N_beads):
        J[i,(i+1)%N_beads], J[i,(i-1)%N_beads] = 1, 1
        J[(i+1)%N_beads,i], J[(i-1)%N_beads,i] = 1, 1
    return J

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
    E1 = np.sum(h * spins)
    
    E2 = 0.0
    for i in range(N_beads):
        E2 += np.sum(J[i, i + 1:] * spins[i] * spins[i + 1:])
    
    return ising_norm1 * E1 + ising_norm2 * E2

@njit
def get_E(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, t, l_forks, r_forks, spins, J, h, ising_norm1, ising_norm2):
    ''''
    The totdal energy.
    '''
    energy = E_bind(L, R, ms, ns, bind_norm) + E_cross(ms, ns, k_norm) + E_fold(ms, ns, fold_norm)
    if rep_norm!=0.0: energy += E_repli(l_forks, r_forks, ms, ns, t, rep_norm)
    if ising_norm1!=0.0 or ising_norm2!=0.0: energy += E_ising(spins, J, h, ising_norm1, ising_norm2)
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
def get_dE_ising(spins, J, h, spin_idx, ising_norm1, ising_norm2, m_old, n_old, m_new, n_new):
    dE1 = -2 * h[spin_idx] * spins[spin_idx]
    dE2 = -2*spins[spin_idx]* np.sum(J[spin_idx,:]*spins)
    dE2 += np.sum(np.outer(spins[m_new:n_new],spins[m_new:n_new]))-np.sum(np.outer(spins[m_old:n_old],spins[m_old:n_old]))
    return ising_norm1 * dE1 + ising_norm2 * dE2

@njit
def get_dE(L, R, bind_norm, fold_norm, k_norm ,rep_norm, ms, ns, m_new, n_new, idx,  t, l_forks, r_forks, spins, J, h, spin_idx, ising_norm1=0, ising_norm2=0.0):
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
    if ising_norm1!=0 or ising_norm2!=0: dE += get_dE_ising(spins, J, h, spin_idx, ising_norm1, ising_norm2, ms[idx], ns[idx], m_new, n_new)
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
    return ms, ns

@njit
def run_energy_minimization(N_steps, N_lef, N_CTCF, N_beads, MC_step, T, T_min, mode, L, R, kappa, f, b, c_rep=0.0, t_rep=np.inf, rep_duration=np.inf, l_forks=np.array([[1,0],[1,0]],dtype=np.int32), r_forks=np.array([[1,0],[1,0]],dtype=np.int32), c_ising1=0.0, c_ising2=0.0, ising_field=None, spin_state=None, rw=True):
    '''
    It performs Monte Carlo or simulated annealing of the simulation.
    '''
    # Initialization of parameters
    N_rep = np.max(np.sum(l_forks,axis=0))
    fold_norm, bind_norm, k_norm, rep_norm = -N_beads*f/(N_lef*np.log(N_beads/N_CTCF)), -N_beads*b/(np.sum(L)+np.sum(R)), kappa*1e4, -N_beads*c_rep/N_rep
    ising_norm1, ising_norm2 = -c_ising1, -c_ising2/(2*N_lef)
    Ti = T
    
    # Initialization of matrices
    ms, ns = initialize(N_lef, N_beads)
    spin_traj = np.zeros((N_beads, N_steps//MC_step),dtype=np.int32)
    J = compute_J(ms,ns,N_beads)
    E = get_E(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, 0, l_forks, r_forks, spin_state, J, ising_field, ising_norm1, ising_norm2)
    Es = np.zeros(N_steps//MC_step, dtype=np.float64)
    Es_ising = np.zeros(N_steps//MC_step, dtype=np.float64)
    Fs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Bs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Rs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Ms, Ns = np.zeros((N_lef, N_steps//MC_step), dtype=np.int32), np.zeros((N_lef, N_steps//MC_step), dtype=np.int32)

    for i in range(N_steps):
        # Calculate replication time
        rt = 0 if i < t_rep else int(i - t_rep) if (i >= t_rep and i < t_rep + rep_duration) else int(rep_duration)-1
        Ti = T - (T - T_min) * i / N_steps if mode == 'Annealing' else T

        for j in prange(N_lef): # In each swift we move one cohesin and one random spin
            # Propose a move for cohesins
            r = np.random.choice(np.array([0, 1]))
            m_old, n_old = ms[j], ns[j]
            if r == 0:
                m_new, n_new = unbind_bind(N_beads)
            else:
                m_new, n_new = slide(ms[j], ns[j], N_beads, rw)
            
            # Propose a move for spin
            spin_idx = np.random.choice(np.arange(N_beads))
            
            # Cohesin energy difference
            dE = get_dE(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, m_new, n_new, j, rt, l_forks, r_forks, spin_state, J, ising_field, spin_idx, ising_norm1, ising_norm2) 

            # Check if the move would be accepted
            if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                E += dE
                ms[j], ns[j] = m_new, n_new
                if c_ising1!=0.0 or c_ising2!=0.0:
                    J[m_old:n_old,m_old:n_old] = 0
                    J[ms[j]:ns[j],ms[j]:ns[j]] = 1
                    spin_state[spin_idx] *= -1

        # Keep track on energies and trajectories of LEFs and spins
        if i % MC_step == 0:
            Es[i//MC_step] = E
            Ms[:, i//MC_step], Ns[:, i//MC_step] = ms, ns
            spin_traj[:,i//MC_step] = spin_state
            Fs[i//MC_step] = E_fold(ms, ns, fold_norm)
            Bs[i//MC_step] = E_bind(L,R,ms,ns,bind_norm)
            if rep_norm!=0.0: Rs[i//MC_step] = E_repli(l_forks,r_forks,ms,ns,rt,rep_norm)
    return Ms, Ns, Es, Es_ising, Fs, Bs, Rs, spin_traj

class StochasticSimulation:
    def __init__(self,N_beads,chrom,region, bedpe_file, out_path, N_lef=None, rept_path=None, t_rep=None, rep_duration=None):
        '''
        Import simulation parameters and data.
        '''
        # Import parameters
        self.N_beads = N_beads if N_beads!=None else int(np.round((region[1]-region[0])/2000))
        self.chrom, self.region = chrom, region
        self.t_rep, self.rep_duration = t_rep, rep_duration
        self.out_path = out_path
        make_folder(self.out_path)

        # Import replication data
        self.run_replication = rept_path!=None
        if self.run_replication:
            rep = Replikator(rept_path,self.N_beads,rep_duration,chrom,region)
            self.rep_frac, self.l_forks, self.r_forks = rep.run()
            self.epigenetic_field, self.state = rep.calculate_ising_parameters()
        else:
            self.l_forks, self.r_forks = np.array([[1,0],[1,0]],dtype=np.int32),  np.array([[1,0],[1,0]],dtype=np.int32)
            self.epigenetic_field, self.state = np.array(list(),dtype=np.float64), np.array(list(),dtype=np.int32)

        # Import loop data
        self.L, self.R, J, dists, self.N_CTCF = preprocessing(bedpe_file=bedpe_file, region=region, chrom=chrom, N_beads=self.N_beads)
        self.N_lef= self.N_CTCF if N_lef==None else N_lef
        print(f'Simulation starts with number of beads: {self.N_beads}')
        print(f'Number of CTCFs is N_CTCF={self.N_CTCF}, and number of LEFs is N_lef={self.N_lef}.')

    def run_stochastic_simulation(self, N_steps, MC_step, burnin, T, T_min, f=1.0, b=1.0, kappa=10.0, c_rep=None, c_ising1=0.0, c_ising2=0.0, mode='Metropolis',rw=True, random_spin_state=True):
        '''
        Energy minimization script.
        '''
        if not self.run_replication: c_rep, c_ising1, c_ising2 = 0.0, 0.0, 0.0
        self.is_ising = c_ising1!=0.0 or c_ising2!=0.0

        # Running the simulation
        start = time.time()
        print('\nRunning RepliSage...')
        self.state = np.random.choice([-1, 1], size=self.N_beads).astype(np.int32) if random_spin_state else self.state
        self.N_steps,self.MC_step, self.burnin, self.T, self.T_in = N_steps,MC_step, burnin, T, T_min
        self.Ms, self.Ns, self.Es, self.Es_ising, self.Fs, self.Bs, self.Rs, self.spin_traj = run_energy_minimization(
        N_steps=N_steps, MC_step=MC_step, T=T, T_min=T_min, t_rep=self.t_rep, rep_duration=self.rep_duration,
        mode=mode, N_lef=self.N_lef, N_CTCF=self.N_CTCF, N_beads=self.N_beads,
        L=self.L, R=self.R, kappa=kappa, f=f, b=b, c_rep=c_rep,
        l_forks=self.l_forks, r_forks=self.r_forks,
        c_ising1=c_ising1, c_ising2=c_ising2, ising_field=self.epigenetic_field, spin_state=self.state,
        rw=rw)
        end = time.time()
        elapsed = end - start
        print(f'Computation finished succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')

        np.save(f'{self.out_path}/other/Ms.npy', self.Ms)
        np.save(f'{self.out_path}/other/Ns.npy', self.Ns)
        np.save(f'{self.out_path}/other/Es.npy', self.Es)
        return self.Ms, self.Ns, self.Es, self.Es_ising, self.Fs, self.Bs, self.Rs, self.spin_traj
    
    def show_plots(self):
        '''
        Draw plots.
        '''
        make_timeplots(self.Es, self.Fs, self.Bs, self.Rs, self.burnin//self.MC_step, self.out_path)
        coh_traj_plot(self.Ms, self.Ns, self.N_beads, self.out_path)
        if self.is_ising: ising_traj_plot(self.spin_traj,self.out_path)

    def run_openmm(self,platform='CPU'):
        ''' 
        Run OpenMM energy minimization.
        '''
        md = EM_LE(self.Ms,self.Ns,self.N_beads,self.burnin,self.MC_step,self.out_path,platform,self.rep_frac,self.t_rep,self.spin_traj)
        md.run_pipeline()

def main():
    # Set parameters
    N_beads = None
    N_steps, MC_step, burnin, T, T_min, t_rep, rep_duration = int(4e4), int(1e2), int(1e3), 2.0, 0.0, int(1e4), int(2e4)
    f, b, kappa, c_rep = 1.0, 0.2, 1.0, 1.0
    c_ising1, c_ising2 = 1.0, 1.0
    mode, rw, random_spin_state = 'Metropolis', True, True

    # Define data and coordinates
    region, chrom =  [178421513, 186421513], 'chr1'
    bedpe_file = '/home/skorsak/Data/method_paper_data/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe'
    rept_path = '/home/skorsak/Data/Replication/sc_timing/GM12878_single_cell_data_hg37.mat'
    out_path = '../output'
    
    # Run simulation
    sim = StochasticSimulation(N_beads,chrom,region, bedpe_file, out_path, None, rept_path, t_rep, rep_duration)
    sim.run_stochastic_simulation(N_steps, MC_step, burnin, T, T_min, f, b, kappa, c_rep, c_ising1, c_ising2, mode, rw, random_spin_state)
    sim.show_plots()
    sim.run_openmm('CPU')

if __name__=='__main__':
    main()