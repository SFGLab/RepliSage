#Basic Libraries
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import scipy.stats as stats
from tqdm import tqdm

# Hide warnings
import warnings
from numba import njit
warnings.filterwarnings('ignore')

# scipy
from scipy.stats import norm
from scipy.stats import poisson

# My own libraries
from Replikator import *
from RepliSage_preproc import *
from RepliSage_plots import *
from RepliSage_md import *
from RepliSage_em import *

def preprocessing(bedpe_file, region, chrom, N_beads):
    L, R, dists = binding_vectors_from_bedpe(bedpe_file, N_beads, region, chrom, False, False)
    N_CTCF = np.max([np.count_nonzero(L), np.count_nonzero(R)])
    return L, R, dists, N_CTCF

@njit
def Kappa(mi,ni,mj,nj):
    k=0
    if mi<mj and mj<ni and ni<nj: k+=1 # np.abs(ni-mj)+1
    if mj<mi and mi<nj and nj<ni: k+=1 # np.abs(nj-mi)+1
    if mj==ni or mi==nj or ni==nj or mi==mj: k+=1
    return k

@njit(parallel=True) 
def E_bind(L, R, ms, ns, bind_norm):
    binding = np.sum(L[ms] + R[ns])
    E_b = bind_norm * binding
    return E_b

@njit
def E_repli(l_forks, r_forks, ms, ns, t, rep_norm):
    replication = np.sum(l_forks[ms, t] + l_forks[ns, t] + r_forks[ms, t] + r_forks[ns, t])
    E_rep = rep_norm * replication
    return E_rep

@njit(parallel=True) 
def E_cross(ms, ns, k_norm):
    crossing = 0
    N_lef = len(ms)
    for i in range(N_lef):
        for j in range(i + 1, N_lef):
            crossing += Kappa(ms[i], ns[i], ms[j], ns[j])
    return k_norm * crossing

@njit
def E_fold(ms, ns, fold_norm):
    N_lef = len(ms)
    folding = np.sum(np.log(ns - ms))
    return fold_norm * folding

@njit
def get_E(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, t, l_forks, r_forks):
    energy = E_bind(L, R, ms, ns, bind_norm) + E_cross(ms, ns, k_norm) + E_fold(ms, ns, fold_norm)
    if rep_norm is not None:
        energy += E_repli(l_forks, r_forks, ms, ns, t, rep_norm)
    return energy

@njit
def get_dE_bind(L,R,bind_norm,ms,ns,m_new,n_new,idx):
    return bind_norm*(L[m_new]+R[n_new]-L[ms[idx]]-R[ns[idx]])
    
@njit
def get_dE_fold(fold_norm,ms,ns,m_new,n_new,idx):
    return fold_norm*(np.log(n_new-m_new)-np.log(ns[idx]-ms[idx]))

@njit
def get_dE_rep(l_forks,r_forks, rep_norm, ms,ns,m_new,n_new,t,idx):
    E_rep_new = l_forks[m_new,t]+l_forks[n_new,t]+r_forks[m_new,t]+r_forks[n_new,t]
    E_rep_old = l_forks[ms[idx],t-1]+l_forks[ns[idx],t-1]+r_forks[ms[idx],t-1]+r_forks[ns[idx],t-1]
    dE_rep = rep_norm*(E_rep_new-E_rep_old)
    return dE_rep

@njit(parallel=True)
def get_dE_cross(ms, ns, m_new, n_new, idx, k_norm):
    K1, K2 = 0, 0
    for i in range(N_lef):
        if i != idx:
            K1 += Kappa(ms[idx], ns[idx], ms[i], ns[i])
            K2 += Kappa(m_new, n_new, ms[i], ns[i])
    return k_norm * (K2 - K1)

@njit
def get_dE(L, R, bind_norm, fold_norm, k_norm ,rep_norm, ms, ns, m_new, n_new, idx,  t, l_forks, r_forks):
    dE = 0
    dE += get_dE_fold(fold_norm,ms,ns,m_new,n_new,idx)
    dE += get_dE_bind(L, R, bind_norm, ms, ns, m_new, n_new, idx)
    dE += get_dE_cross(ms, ns, m_new, n_new, idx, k_norm)
    if rep_norm is not None:
        dE += get_dE_rep(l_forks, r_forks, rep_norm, ms, ns, m_new, n_new, t, idx)
    return dE

@njit
def unbind_bind(N_beads, avg_loop):
    m_new = rd.randint(0, N_beads - 2)
    n_new = m_new + 5
    if n_new >= N_beads:
        n_new = rd.randint(m_new + 1, N_beads - 1)
    return int(m_new), int(n_new)

@njit
def slide(m_old, n_old, N_beads):
    choices = np.array([-1, 1], dtype=np.int64)
    r1 = np.random.choice(choices)
    r2 = np.random.choice(choices)
    m_new = (m_old + r1) % N_beads
    n_new = (n_old + r2) % N_beads
    return m_new, n_new

@njit(parallel=True) 
def initialize(N_lef, N_beads, avg_loop):
    ms, ns = np.zeros(N_lef, dtype=np.int64), np.zeros(N_lef, dtype=np.int64)
    for i in range(N_lef):
        ms[i], ns[i] = unbind_bind(N_beads, avg_loop)
    return ms, ns

def run_energy_minimization(N_steps, MC_step, burnin, T, T_min, t_rep, rep_duration, mode, avg_loop, L, R, kappa, f, b, c_rep, N_lef, N_beads, N_CTCF, l_forks, r_forks, viz, save, path):
    N_rep = np.max(np.sum(l_forks+r_forks,axis=0))
    fold_norm, bind_norm, k_norm, rep_norm = f/(N_lef*np.log(avg_loop)), b/(np.sum(L)+np.sum(R)), kappa/N_lef, c_rep/N_rep
    
    Ti = T
    Ts = []
    bi = burnin // MC_step
    ms, ns = initialize(N_lef, N_beads, avg_loop)
    E = get_E(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, 0, l_forks, r_forks)
    Es = []
    Ms, Ns = np.zeros((N_lef, N_steps), dtype=np.int32), np.zeros((N_lef, N_steps), dtype=np.int32)
    
    if viz: print('Running simulation...')
    for i in tqdm(range(N_steps)):
        Ti = T - (T - T_min) * (i + 1) / N_steps if mode == 'Annealing' else T
        Ts.append(Ti)
        for j in range(N_lef):
            r = rd.choice([0, 1, 2])
            if r == 0:
                m_new, n_new = unbind_bind(N_beads, avg_loop)
            else:
                m_new, n_new = slide(ms[j], ns[j], N_beads)
            if i < t_rep:
                dE = get_dE(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, m_new, n_new, j,  0, l_forks, r_forks)
            elif i >= t_rep and i < t_rep + rep_duration:
                dE = get_dE(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, m_new, n_new, j, i - t_rep,  l_forks, r_forks)
            else:
                dE = get_dE(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, m_new, n_new, j,  rep_duration - 1, l_forks, r_forks)
            if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                ms[j], ns[j] = m_new, n_new
                E += dE
            Ms[j, i], Ns[j, i] = ms[j], ns[j]
        if i % MC_step == 0:
            Es.append(E)
    if viz: 
        print('Done! ;D')
        coh_traj_plot(Ms, Ns, N_beads, path)
        make_timeplots(Es, bi, mode, path)
    if save:
        np.save(f'{path}/other/Ms.npy', Ms)
        np.save(f'{path}/other/Ns.npy', Ns)
        np.save(f'{path}/other/Ts.npy', Ts)
        np.save(f'{path}/other/Es.npy', Es)

    return Ms, Ns, Es

# Set MC parameters
N_beads, N_lef = 10000, 1000
N_steps, MC_step, burnin, T, T_min, t_rep, rep_duration = int(2e4), int(5e2), 1000, 4, 1, int(5e3), int(1e4)

# For method paper
region, chrom =  [0, 150617247], 'chr9'

out_path=f'with_md'
bedpe_file = '/home/skorsak/Data/method_paper_data/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe'
rept_path = '/home/skorsak/Data/Replication/sc_timing/Chr9_replication_state_filtered.mat'
ori_path = '/home/skorsak/Data/Replication/origins/LCL_MCM_replication_origins.bed'
out_path = 'output'
make_folder(out_path)

rep = Replikator(rept_path,ori_path,N_beads,rep_duration)
rep_frac, l_forks, r_forks = rep.run()
N_rep = np.max(np.sum(l_forks+r_forks,axis=0))

# Preprocessing
L, R, dists, N_CTCF = preprocessing(bedpe_file=bedpe_file, region=region, chrom=chrom, N_beads=N_beads)
avg_loop = int(np.average(dists))+1

# Running the simulation
Ms, Ns, Es = run_energy_minimization(
    N_steps=N_steps, MC_step=MC_step, burnin=burnin, T=T, T_min=T_min, t_rep=1e5, rep_duration=rep_duration,
    mode='Annealing', viz=True, save=False, N_lef=N_lef, N_beads=N_beads,
    avg_loop=avg_loop, L=L, R=R, kappa=1e5, f=-2000, b=-1000, c_rep=-4000, N_CTCF=N_CTCF, 
    l_forks=l_forks, r_forks=r_forks, path='./output'
)