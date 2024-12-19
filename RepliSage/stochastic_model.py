##############################################################################
################### SEBASTIAN KORSAK 2024, WARSAW, POLAND ####################
##############################################################################
## This script runs a stochastic simulation, similar like LoopSage. ##########
## The given script is parallelized across CPU cores and has been modified ###
## to simulate the propagation of replication forks, which act as barriers. ##
##############################################################################
##############################################################################

# Hide warnings
import warnings
import time
warnings.filterwarnings('ignore')

# My own libraries
from Replikator import *
from common import *
from plots import *
from energy_functions import *
from md_model import *

class StochasticSimulation:
    def __init__(self,N_beads,chrom,region, bedpe_file, out_path, N_lef=None, N_lef2=0, rept_path=None, t_rep=None, rep_duration=None,scale=10):
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
            self.rep_frac, self.l_forks, self.r_forks = rep.run(scale=scale)
            self.h, _ = rep.calculate_ising_parameters()
        else:
            self.l_forks, self.r_forks = np.array([[1,0],[1,0]],dtype=np.int32),  np.array([[1,0],[1,0]],dtype=np.int32)

        # Import loop data
        self.L, self.R, self.J, self.N_CTCF = preprocessing(bedpe_file=bedpe_file, region=region, chrom=chrom, N_beads=self.N_beads)
        self.N_lef= 2*self.N_CTCF if N_lef==None else N_lef
        self.N_lef2 = N_lef2
        print(f'Simulation starts with number of beads: {self.N_beads}')
        print(f'Number of CTCFs is N_CTCF={self.N_CTCF}, and number of LEFs is N_lef={self.N_lef}.\nNumber of LEFs in the second family N_lef2={self.N_lef2}.')

    def run_stochastic_simulation(self, N_steps, MC_step, burnin, T, T_min, f=1.0, f2=0, b=1.0, kr=1.0, kappa=1.0, c_rep=None, c_ising1=0.0, c_ising2=0.0, mode='Metropolis',rw=True):
        '''
        Energy minimization script.
        '''
        # Normalize strengths
        if not self.run_replication: c_rep, c_ising1, c_ising2 = 0.0, 0.0, 0.0
        N_rep = np.max(np.sum(self.l_forks,axis=0))
        fold_norm, fold_norm2 = -self.N_beads*f/(self.N_lef*np.log(self.N_beads/self.N_lef)), -self.N_beads*f2/(self.N_lef*np.log(self.N_beads/self.N_lef))
        bind_norm, k_norm = -self.N_beads*b/(2*(np.sum(self.L)+np.sum(self.R))), kappa*1e5
        rep_norm, kr_norm = -self.N_beads*c_rep/N_rep, kr*1e5
        ising_norm1, ising_norm2 = -c_ising1, -c_ising2

        self.is_ising = (c_ising1!=0.0 or c_ising2!=0.0) and np.all(self.J!=None)

        # Running the simulation
        print('\nRunning RepliSage...')
        start = time.time()
        self.N_steps,self.MC_step, self.burnin, self.T, self.T_in = N_steps,MC_step, burnin, T, T_min
        self.Ms, self.Ns, self.Es, self.Es_ising, self.Fs, self.Bs, self.Rs, self.spin_traj, self.J, self.mags = run_energy_minimization(
        N_steps=N_steps, MC_step=MC_step, T=T, T_min=T_min, t_rep=self.t_rep, rep_duration=self.rep_duration,
        mode=mode, N_lef=self.N_lef, N_lef2=self.N_lef2, N_CTCF=self.N_CTCF, N_beads=self.N_beads,
        L=self.L, R=self.R, k_norm=k_norm, fold_norm=fold_norm, fold_norm2=fold_norm2,
        bind_norm=bind_norm, rep_norm=rep_norm, kr_norm=kr_norm, 
        l_forks=self.l_forks, r_forks=self.r_forks,
        ising_norm1=ising_norm1, ising_norm2=ising_norm2,
        h=self.h, rw=rw)
        end = time.time()
        elapsed = end - start
        print(f'Computation finished succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')

        np.save(f'{self.out_path}/other/Ms.npy', self.Ms)
        np.save(f'{self.out_path}/other/Ns.npy', self.Ns)
        np.save(f'{self.out_path}/other/Es.npy', self.Es)
        np.save(f'{self.out_path}/other/Fs.npy', self.Fs)
        np.save(f'{self.out_path}/other/Bs.npy', self.Bs)
        np.save(f'{self.out_path}/other/Rs.npy', self.Rs)
        np.save(f'{self.out_path}/other/J.npy', self.J)
        np.save(f'{self.out_path}/other/spin_traj.npy', self.spin_traj)
    
    def show_plots(self):
        '''
        Draw plots.
        '''
        make_timeplots(self.Es, self.Es_ising, self.Fs, self.Bs, self.Rs, self.mags, self.burnin//self.MC_step, self.out_path)
        coh_traj_plot(self.Ms, self.Ns, self.N_beads, self.out_path)
        if self.is_ising: ising_traj_plot(self.spin_traj,self.out_path)

    def run_openmm(self,platform='CPU',init_struct='hilbert',mode='MD'):
        ''' 
        Run OpenMM energy minimization.
        '''
        md = MD_MODEL(self.Ms,self.Ns,self.N_beads,self.burnin,self.MC_step,self.out_path,platform,self.rep_frac,self.t_rep,self.spin_traj)
        md.run_pipeline(init_struct,mode='EM')

def main():
    # Set parameters
    N_beads, N_lef, N_lef2 = 2000, 200, 10
    N_steps, MC_step, burnin, T, T_min, t_rep, rep_duration = int(2e5), int(5e2), int(1e3), 1.8, 1.0, int(6e4), int(8e4)
    f, f2, b, kappa  = 1.0, 10.0, 1.0, 1.0
    c_rep, kr = 1.0, 1.0
    c_state_field, c_state_interact = 0.01, 3.0
    mode, rw, random_spins = 'Metropolis', True, True
    scale = 200
    
    # Define data and coordinates
    region, chrom =  [80935000, 89874700], 'chr14'
    bedpe_file = '/home/skorsak/Data/method_paper_data/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe'
    rept_path = '/home/skorsak/Data/Replication/sc_timing/GM12878_single_cell_data_hg37.mat'
    out_path = '../output'
    
    # Run simulation
    sim = StochasticSimulation(N_beads, chrom, region, bedpe_file, out_path, N_lef, N_lef2, rept_path, t_rep, rep_duration, scale)
    sim.run_stochastic_simulation(N_steps, MC_step, burnin, T, T_min, f, f2, b, kappa, kr, c_rep, c_state_field, c_state_interact, mode, rw)
    sim.show_plots()
    sim.run_openmm('OpenCL')

if __name__=='__main__':
    main()