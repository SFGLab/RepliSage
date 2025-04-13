##############################################################################
################### SEBASTIAN KORSAK 2024, WARSAW, POLAND ####################
##############################################################################
##### This script runs a stochastic simulation, similar like LoopSage. #######
## The given script is parallelized across CPU cores and has been modified ###
## to simulate the propagation of replication forks, which act as barriers. ##
##############################################################################
##############################################################################

# Hide warnings
import warnings
import time
warnings.filterwarnings('ignore')

# My own libraries
from .Replikator import *
from .common import *
from .plots import *
from .energy_functions import *
from .structure_metrics import *
from .network_analysis import *
from .md_model import *

class StochasticSimulation:
    def __init__(self, N_beads, chrom, region, bedpe_file, out_path, N_lef=None, N_lef2=0, rept_path=None, t_rep=None, rep_duration=None, Tstd_factor=0.1, speed_scale=20, scale=1):
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
            rep = Replikator(rept_path,self.N_beads,1000,chrom,region,Tstd_factor=Tstd_factor,speed_factor=speed_scale)
            rep_frac, _, _ = rep.run(scale=scale,out_path=self.out_path+'/plots')
            self.rep_frac = expand_columns(rep_frac, rep_duration)
            self.h, _ = rep.calculate_ising_parameters()
        else:
            self.l_forks, self.r_forks = np.array([[1,0],[1,0]],dtype=np.int32),  np.array([[1,0],[1,0]],dtype=np.int32)

        # Import loop data
        self.L, self.R, self.J, self.N_CTCF = preprocessing(bedpe_file=bedpe_file, region=region, chrom=chrom, N_beads=self.N_beads)
        self.N_lef= 2*self.N_CTCF if N_lef==None else N_lef
        self.N_lef2 = N_lef2
        print(f'Simulation starts with number of beads: {self.N_beads}')
        print(f'Number of CTCFs is N_CTCF={self.N_CTCF}, and number of LEFs is N_lef={self.N_lef}.\nNumber of LEFs in the second family N_lef2={self.N_lef2}.')

    def run_stochastic_simulation(self, N_steps, MC_step, burnin, T, T_min, f=1.0, f2=0.0, b=1.0, kappa=1.0, c_rep=None, c_potts1=0.0, c_potts2=0.0, mode='Metropolis',rw=True, p_rew=0.5, rep_fork_organizers=True, save_MDT=True):
        '''
        Energy minimization script.
        '''
        # Normalize strengths
        if not self.run_replication: c_rep, c_potts1, c_potts2 = 0.0, 0.0, 0.0
        fold_norm, fold_norm2 = -self.N_beads*f/(self.N_lef*np.log(self.N_beads/self.N_lef)), -self.N_beads*f2/(self.N_lef*np.log(self.N_beads/self.N_lef))
        bind_norm, k_norm = -self.N_beads*b/(np.sum(self.L)+np.sum(self.R)), kappa*1e5
        rep_norm = c_rep*1e5
        potts_norm1, potts_norm2 = -c_potts1, -c_potts2
        self.is_potts = (c_potts1!=0.0 or c_potts2!=0.0) and np.all(self.J!=None)
        
        # Running the simulation
        print('\nRunning RepliSage...')
        start = time.time()
        self.N_steps,self.MC_step, self.burnin, self.T, self.T_in = N_steps,MC_step, burnin, T, T_min
        self.Ms, self.Ns, self.Es, self.Es_potts, self.Fs, self.Bs, self.Rs, self.Ks, self.spin_traj, self.mags = run_energy_minimization(
        N_steps=N_steps, MC_step=MC_step, T=T, T_min=T_min, t_rep=self.t_rep, rep_duration=self.rep_duration, p_rew=p_rew,
        mode=mode, N_lef=self.N_lef, N_lef2=self.N_lef2, N_beads=self.N_beads,
        L=self.L, R=self.R, k_norm=k_norm, fold_norm=fold_norm, fold_norm2=fold_norm2,
        bind_norm=bind_norm, rep_norm=rep_norm,
        f_rep=self.rep_frac, potts_norm1=potts_norm1, potts_norm2=potts_norm2,
        h=self.h, J=self.J, rw=rw, rep_fork_organizers=rep_fork_organizers)
        end = time.time()
        elapsed = end - start
        print(f'Computation finished succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')

        if save_MDT:
            np.save(f'{self.out_path}/other/Ms.npy', self.Ms)
            np.save(f'{self.out_path}/other/Ns.npy', self.Ns)
            np.save(f'{self.out_path}/other/Es.npy', self.Es)
            np.save(f'{self.out_path}/other/Fs.npy', self.Fs)
            np.save(f'{self.out_path}/other/Bs.npy', self.Bs)
            np.save(f'{self.out_path}/other/Rs.npy', self.Rs)
            np.save(f'{self.out_path}/other/Ks.npy', self.Ks)
            np.save(f'{self.out_path}/other/loop_lengths.npy', self.Ns-self.Ms)
            np.save(f'{self.out_path}/other/Es_potts.npy', self.Es_potts)
            np.save(f'{self.out_path}/other/mags.npy',self.mags)
            np.save(f'{self.out_path}/other/spins.npy', self.spin_traj)
    
    def show_plots(self):
        '''
        Draw plots.
        '''
        make_timeplots(self.Es, self.Es_potts, self.Fs, self.Bs, self.Rs, self.mags, self.burnin//self.MC_step, self.out_path)
        coh_traj_plot(self.Ms, self.Ns, self.N_beads, self.out_path)
        compute_potts_metrics(self.Ms, self.Ns, self.spin_traj, self.N_beads,self.out_path)
        if self.is_potts: ising_traj_plot(self.spin_traj,self.out_path)
        plot_loop_length(self.Ns-self.Ms, self.t_rep//self.MC_step,  (self.t_rep+self.rep_duration)//self.MC_step, self.out_path)
        compute_state_proportions_sign_based(self.Ms, self.Ns, self.spin_traj, self.t_rep//self.MC_step,  (self.t_rep+self.rep_duration)//self.MC_step, self.out_path)

    def compute_structure_metrics(self):
        '''
        It computes plots with metrics for analysis after simulation.
        '''      
        compute_metrics_for_ensemble(self.out_path+'/ensemble',duplicated_chain=True,path=self.out_path)

    def run_openmm(self,platform='CPU',init_struct='rw',mode='EM',integrator_mode='langevin',integrator_step=10.0 * mm.unit.femtosecond,tol=1.0,sim_step=10000,p_ev=0.01,reporters=False, md_temperature=310*mm.unit.kelvin, ff_path='RepliSage/forcefields/classic_sm_ff.xml'):
        '''
        Run OpenMM energy minimization.
        '''
        md = MD_MODEL(self.Ms,self.Ns,self.N_beads,self.burnin,self.MC_step,self.out_path,platform,self.rep_frac,self.t_rep,self.spin_traj)
        md.run_pipeline(init_struct,mode=mode,integrator_mode=integrator_mode,p_ev=p_ev,md_temperature=md_temperature,ff_path=ff_path,integrator_step=integrator_step,tol=tol,reporters=reporters,sim_step=sim_step)

def main():
    # Set parameters
    N_beads, N_lef, N_lef2 = 1000, 100, 20
    N_steps, MC_step, burnin, T, T_min, t_rep, rep_duration = int(8e4), int(4e2), int(1e3), 1.6, 1.0, int(1e4), int(2e4)

    f, f2, b, kappa= 1.0, 5.0, 1.0, 1.0
    c_state_field, c_state_interact, c_rep = 2.0, 1.0, 1.0
    mode, rw, random_spins, rep_fork_organizers = 'Metropolis', True, True, True
    Tstd_factor, speed_scale, init_rate_scale, p_rew = 0.1, 20, 1.0, 0.5
    save_MDT, save_plots = True, True

    # for stress scale=5.0, sigma_t = T*0.2, speed=5*
    # for normal replication scale=1.0, sigma_t = T*0.1, speed=20*
    
    # Define data and coordinates
    region, chrom =  [80835000, 98674700], 'chr14'
    # region, chrom =  [10835000, 97674700], 'chr14'
    
    # Data
    bedpe_file = '/home/blackpianocat/Data/method_paper_data/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe'
    rept_path = '/home/blackpianocat/Data/Replication/sc_timing/GM12878_single_cell_data_hg37.mat'
    out_path = '/home/blackpianocat/Data/Simulations/RepliSage_test'
    
    # Run simulation
    sim = StochasticSimulation(N_beads, chrom, region, bedpe_file, out_path, N_lef, N_lef2, rept_path, t_rep, rep_duration, Tstd_factor, speed_scale, init_rate_scale)
    sim.run_stochastic_simulation(N_steps, MC_step, burnin, T, T_min, f, f2, b, kappa, c_rep, c_state_field, c_state_interact, mode, rw, p_rew, rep_fork_organizers, save_MDT)
    if save_plots: sim.show_plots()
    sim.run_openmm('OpenCL',mode='MD')
    if save_plots: sim.compute_structure_metrics()

    # Save Parameters
    if save_MDT:
        params = {k: v for k, v in locals().items() if k not in ['args','sim']}
        save_parameters(out_path+'/other/params.txt',**params)
    
if __name__=='__main__':
    main()