#Basic Libraries
import numpy as np
import random as rd

# Hide warnings
import warnings
import time
from numba import njit
warnings.filterwarnings('ignore')

# My own libraries
from Replikator import *
from RepliSage_preproc import *
from RepliSage_plots import *
from RepliSage_md import *

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
def compute_J(ms,ns,N_beads):
    J = np.zeros((N_beads,N_beads),dtype=np.int32)
    N_lef = len(ms)
    for i in range(N_lef):
        m, n = ms[i], ns[i]
        J[m,n] = 1
        J[n,m] = 1
    return J

@njit(parallel=True)
def E_ising(spins, J, h, ising_norm1, ising_norm2):
    N_beads = len(J)
    E1 = np.sum(h * spins)
    
    E2 = 0.0
    for i in range(N_beads):
        E2 += np.sum(J[i, i + 1:] * spins[i] * spins[i + 1:])
    
    return ising_norm1 * E1 + ising_norm2 * E2

@njit
def E_bind(L, R, ms, ns, bind_norm):
    binding = np.sum(L[ms] + R[ns])
    E_b = bind_norm * binding
    return E_b

@njit
def E_repli(l_forks, r_forks, ms, ns, t, rep_norm):
    replication = np.sum(l_forks[ms, t] + l_forks[ns, t] + r_forks[ms, t] + r_forks[ns, t])
    return rep_norm * replication

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
    folding = np.sum(np.log(ns - ms))
    return fold_norm * folding

@njit
def get_E(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, t, l_forks, r_forks,spins=None,J=None,h=None,ising_norm1=None,ising_norm2=None):
    energy = E_bind(L, R, ms, ns, bind_norm) + E_cross(ms, ns, k_norm) + E_fold(ms, ns, fold_norm)
    if rep_norm!=0: energy += E_repli(l_forks, r_forks, ms, ns, t, rep_norm)
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
    return rep_norm*(E_rep_new-E_rep_old)

@njit(parallel=True)
def get_dE_cross(ms, ns, m_new, n_new, idx, k_norm):
    K1, K2 = 0, 0
    for i in range(N_lef):
        if i != idx:
            K1 += Kappa(ms[idx], ns[idx], ms[i], ns[i])
            K2 += Kappa(m_new, n_new, ms[i], ns[i])
    return k_norm * (K2 - K1)

@njit
def get_dE_ising(spins, J, h, spin_idx, ising_norm1, ising_norm2):
    dE1 = 2 * h[spin_idx] * spins[spin_idx]
    dE2 = 2 * (np.sum(J[spin_idx, :] * spins)-J[spin_idx, spin_idx] * spins[spin_idx])
    return ising_norm1 * dE1 + ising_norm2 * dE2

@njit
def get_dE(L, R, bind_norm, fold_norm, k_norm ,rep_norm, ms, ns, m_new, n_new, idx,  t, l_forks, r_forks):
    dE = 0
    dE += get_dE_fold(fold_norm,ms,ns,m_new,n_new,idx)
    dE += get_dE_bind(L, R, bind_norm, ms, ns, m_new, n_new, idx)
    dE += get_dE_cross(ms, ns, m_new, n_new, idx, k_norm)
    if rep_norm!=0: dE += get_dE_rep(l_forks, r_forks, rep_norm, ms, ns, m_new, n_new, t, idx)
    return dE

@njit
def unbind_bind(N_beads, avg_loop):
    m_new = rd.randint(0, N_beads - 2)
    n_new = m_new + 1
    return int(m_new), int(n_new)

@njit
def slide(m_old, n_old, N_beads):
    choices = np.array([-1, 1], dtype=np.int64)
    r1 = np.random.choice(choices)
    r2 = np.random.choice(choices)
    m_new = (m_old + r1) % N_beads
    n_new = (n_old + r2) % N_beads
    return int(m_new), int(n_new)

@njit(parallel=True)
def initialize(N_lef, N_beads, avg_loop):
    ms, ns = np.zeros(N_lef, dtype=np.int64), np.zeros(N_lef, dtype=np.int64)
    for i in range(N_lef):
        ms[i], ns[i] = unbind_bind(N_beads, avg_loop)
    return ms, ns

@njit
def run_energy_minimization(N_steps, MC_step, T, T_min, t_rep, rep_duration, mode, avg_loop, L, R, kappa, f, b, c_rep, N_lef, N_beads, l_forks, r_forks, c_ising1=None, c_ising2=None, ising_field=None, spin_state=None):
    N_rep = np.max(np.sum(l_forks+r_forks,axis=0))
    fold_norm, bind_norm, k_norm, rep_norm = -N_beads*f/(N_lef*np.log(avg_loop)), -N_beads*b/(np.sum(L)+np.sum(R)), N_beads*kappa/N_lef, -N_beads*c_rep/N_rep
    ising_norm1, ising_norm2 = -c_ising1, -N_beads*c_ising2/N_lef
    Ti = T
    ms, ns = initialize(N_lef, N_beads, avg_loop)
    spin_traj = np.zeros((N_beads, N_steps),dtype=np.int32)
    if ising_norm1!=0 or ising_norm2!=0: J = compute_J(ms,ns,N_beads)
    E = get_E(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, 0, l_forks, r_forks)
    if ising_norm1!=0 or ising_norm2!=0: E_is = E_ising(spin_state,J,ising_field,ising_norm1,ising_norm2)
    Es = np.zeros(N_steps//MC_step, dtype=np.float64)
    Es_ising = np.zeros(N_steps//MC_step, dtype=np.float64)
    Fs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Bs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Rs = np.zeros(N_steps//MC_step, dtype=np.float64)
    Ms, Ns = np.zeros((N_lef, N_steps), dtype=np.int32), np.zeros((N_lef, N_steps), dtype=np.int32)

    for i in range(N_steps):
        rt = 0 if i < t_rep else int(i - t_rep) if (i >= t_rep and i < t_rep + rep_duration) else int(rep_duration - 1)
        Ti = T - (T - T_min) * i / N_steps if mode == 'Annealing' else T
        for j in range(N_lef):
            r = np.random.choice(np.array([0, 1, 2]))
            if r == 0:
                m_new, n_new = unbind_bind(N_beads, avg_loop)
            else:
                m_new, n_new = slide(ms[j], ns[j], N_beads)
            
            dE = get_dE(L, R, bind_norm, fold_norm, k_norm, rep_norm, ms, ns, m_new, n_new, j, rt, l_forks, r_forks)
            if dE <= 0 or np.exp(-dE / Ti) > np.random.rand():
                ms[j], ns[j] = m_new, n_new
                E += dE

                if ising_norm1 != 0 or ising_norm2 != 0:
                    J[m_new,n_new],J[n_new,m_new]=1,1
                    J[ms[j],ns[j]],J[ns[j],ms[j]]=0,0
                    spin_idx = np.random.randint(N_beads)
                    dE_ising = get_dE_ising(spin_state, J, ising_field, spin_idx, ising_norm1, ising_norm2)
                    if dE_ising <= 0 or np.exp(-dE_ising / Ti) > np.random.rand():
                        E_is += dE_ising
                        spin_state[spin_idx] *= -1
                
            Ms[j, i], Ns[j, i] = ms[j], ns[j]
            spin_traj[:,i] = spin_state

        if i % MC_step == 0:
            Es[i//MC_step] = E
            if ising_norm1!=0 or ising_norm2!=0: Es_ising[i//MC_step] = E_is
            Fs[i//MC_step] = E_fold(ms, ns, fold_norm)
            Bs[i//MC_step] = E_bind(L,R,ms,ns,bind_norm)
            Rs[i//MC_step] = E_repli(l_forks,r_forks,ms,ns,rt,rep_norm)
    return Ms, Ns, Es, Es_ising, Fs, Bs, Rs, spin_traj

# Set parameters
N_beads = int(1e4)
N_steps, MC_step, burnin, T, T_min, t_rep, rep_duration = int(1e4), int(5e2), int(1e3), 1.5, 0, int(2e3), int(5e3)

# For method paper
region, chrom =  [0, 150617247], 'chr9'
out_path = f'with_md'
bedpe_file = '/home/skorsak/Data/method_paper_data/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe'
rept_path = '/home/skorsak/Data/Replication/sc_timing/GM12878_single_cell_data_hg37.mat'
out_path = 'output'
make_folder(out_path)

rep = Replikator(rept_path,N_beads,rep_duration,chrom,region)
rep_frac, l_forks, r_forks = rep.run()
epigenetic_field, state = rep.calculate_ising_parameters()

plt.plot(epigenetic_field)
plt.ylabel('Epigenetic Field')
plt.grid()
plt.show()

# Preprocessing
L, R, dists, N_CTCF = preprocessing(bedpe_file=bedpe_file, region=region, chrom=chrom, N_beads=N_beads)
N_lef= 1000
avg_loop = int(np.average(dists))+1
print('N_CTCF=',N_CTCF)

# Running the simulation
start = time.time()
print('\nRunning RepliSage...')
Ms, Ns, Es, Es_ising, Fs, Bs, Rs, spin_traj = run_energy_minimization(
    N_steps=N_steps, MC_step=MC_step, T=T, T_min=T_min, t_rep=t_rep, rep_duration=rep_duration,
    mode='Metropolis', N_lef=N_lef, N_beads=N_beads,
    avg_loop=avg_loop, L=L, R=R, kappa=10, f=1, b=0.2, c_rep=2.0,
    l_forks=l_forks, r_forks=r_forks,
    c_ising1=1.0, c_ising2=0.5, spin_state=state, ising_field=epigenetic_field
)
end = time.time()
elapsed = end - start
print(f'Computation finished succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')

make_timeplots(Es, Fs, Bs, Rs, burnin//MC_step, out_path)
coh_traj_plot(Ms, Ns, N_beads, out_path)
ising_traj_plot(spin_traj,out_path)

np.save(f'{out_path}/other/Ms.npy', Ms)
np.save(f'{out_path}/other/Ns.npy', Ns)
np.save(f'{out_path}/other/Es.npy', Es)
np.save(f'{out_path}/other/spin_traj.npy', spin_traj)

# platform='OpenCL'
# md = MD_LE(Ms,Ns,l_forks,r_forks,t_rep,N_beads,burnin,MC_step,out_path,platform,spin_traj)
# md.run_pipeline(write_files=True)