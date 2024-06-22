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
def E_bind(L, R, ms, ns, b):
    binding = np.sum(L[ms] + R[ns])
    E_b = b * binding / (np.sum(L) + np.sum(R))
    return E_b

@njit
def E_repli(l_forks, r_forks, ms, ns, t, c_rep, N_rep):
    replication = np.sum(l_forks[ms, t] + l_forks[ns, t] + r_forks[ms, t] + r_forks[ns, t])
    E_rep = c_rep * replication / N_rep
    return E_rep

@njit(parallel=True) 
def E_cross(ms, ns, N_lef, kappa):
    crossing = 0
    for i in range(N_lef):
        for j in range(i + 1, N_lef):
            crossing += Kappa(ms[i], ns[i], ms[j], ns[j])
    return kappa * crossing / N_lef

@njit
def E_fold(ms, ns, N_lef, f, log_avg_loop):
    folding = np.sum(np.log(ns - ms))
    return f * folding / (N_lef * log_avg_loop)

@njit
def get_E(L, R, ms, ns, t, b, kappa, f, avg_loop, N_lef, log_avg_loop, c_rep, N_rep, l_forks, r_forks):
    energy = E_bind(L, R, ms, ns, b) + E_cross(ms, ns, N_lef, kappa) + E_fold(ms, ns, N_lef, f, avg_loop)
    if c_rep is not None:
        energy += E_repli(l_forks, r_forks, ms, ns, t, c_rep, N_rep)
    return energy

@njit
def get_dE_bind(L,R,bind_norm,ms,ns,m_new,n_new,idx):
    return bind_norm*(L[m_new]+R[n_new]-L[ms[idx]]-R[ns[idx]])
    
@njit
def get_dE_fold(fold_norm,ms,ns,m_new,n_new,idx):
    return fold_norm*(np.log(n_new-m_new)-np.log(ns[idx]-ms[idx]))

@njit
def get_dE_rep(l_forks,r_forks, c_rep, N_rep, ms,ns,m_new,n_new,t,idx):
    E_rep_new = l_forks[m_new,t]+l_forks[n_new,t]+r_forks[m_new,t]+r_forks[n_new,t]
    E_rep_old = l_forks[ms[idx],t-1]+l_forks[ns[idx],t-1]+r_forks[ms[idx],t-1]+r_forks[ns[idx],t-1]
    dE_rep = c_rep*(E_rep_new-E_rep_old)/N_rep
    return dE_rep

@njit(parallel=True)
def get_dE_cross(ms, ns, m_new, n_new, idx, N_lef, kappa):
    K1, K2 = 0, 0
    for i in range(N_lef):
        if i != idx:
            K1 += Kappa(ms[idx], ns[idx], ms[i], ns[i])
            K2 += Kappa(m_new, n_new, ms[i], ns[i])
    return kappa * (K2 - K1) / N_lef

@njit
def get_dE(L, R, bind_norm, fold_norm, ms, ns, m_new, n_new, idx, N_lef, kappa, t, c_rep, N_rep, l_forks, r_forks):
    dE = 0
    dE += get_dE_bind(L, R, bind_norm, ms, ns, m_new, n_new, idx)
    dE += get_dE_cross(ms, ns, m_new, n_new, idx, N_lef, kappa)
    if c_rep is not None:
        dE += get_dE_rep(l_forks, r_forks, c_rep, N_rep, ms, ns, m_new, n_new, t, idx)
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

def run_energy_minimization(N_steps, MC_step, burnin, T, T_min, t_rep, mode, viz, save, N_lef, N_beads, avg_loop, L, R, kappa, f, b, c_rep, N_rep, N_CTCF, l_forks, r_forks, path):
    Ti = T
    Ts = []
    bi = burnin // MC_step
    ms, ns = initialize(N_lef, N_beads, avg_loop)
    E = get_E(L, R, ms, ns, 0, b, kappa, f, avg_loop, N_lef, np.log(avg_loop + 1), c_rep, N_rep, l_forks, r_forks)
    Es, Ks, Fs, Bs, ufs = [], [], [], [], []
    Ms, Ns = np.zeros((N_lef, N_steps), dtype=np.int32), np.zeros((N_lef, N_steps), dtype=np.int32)
    
    if viz:
        print('Running simulation...')
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
                dE = get_dE(L, R, b, f, ms, ns, m_new, n_new, j, N_lef, kappa, 0, c_rep, N_rep, l_forks, r_forks)
            elif i >= t_rep and i < t_rep + rep_duration:
                dE = get_dE(L, R, b, f, ms, ns, m_new, n_new, j, N_lef, kappa, i - t_rep, c_rep, N_rep, l_forks, r_forks)
            else:
                dE = get_dE(L, R, b, f, ms, ns, m_new, n_new, j, N_lef, kappa, rep_duration - 1, c_rep, N_rep, l_forks, r_forks)
            if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                ms[j], ns[j] = m_new, n_new
                E += dE
            Ms[j, i], Ns[j, i] = ms[j], ns[j]
        if i % MC_step == 0:
            Es.append(E)
    if viz:
        print('Done! ;D')
    if save:
        np.save(f'{path}/other/Ms.npy', Ms)
        np.save(f'{path}/other/Ns.npy', Ns)
        np.save(f'{path}/other/Ts.npy', Ts)
        np.save(f'{path}/other/Es.npy', Es)
    if viz:
        coh_traj_plot(Ms, Ns, N_beads, path)
        make_timeplots(Es, burnin, mode, path)
    return Ms, Ns, Es

# Set MC parameters
N_beads=10000
N_steps, MC_step, burnin, T, T_min, rep_duration = int(2e4), int(5e2), 1000, 4, 1, 10000

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

# Running the simulation
Ms, Ns, Es = run_energy_minimization(
    N_steps=N_steps, MC_step=MC_step, burnin=burnin, T=T, T_min=T_min, t_rep=2000,
    mode='Annealing', viz=True, save=False, N_lef=50, N_beads=N_beads,
    avg_loop=5000, L=L, R=R, kappa=0.1, f=-1000, b=-1000, c_rep=-5000, N_rep=N_rep, N_CTCF=N_CTCF, 
    l_forks=l_forks, r_forks=r_forks, path='./output'
)