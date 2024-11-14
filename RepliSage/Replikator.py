import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
from sklearn import preprocessing
from common import *
from replication_simulator import *
import mat73
from minepy import MINE
import time
from tqdm import tqdm

class Replikator:
    def __init__(self,rept_data_path:str,sim_L:int,sim_T:int,chrom:str,coords=None):
        '''
        Initialization of the data preprocessing.
        ------------------------------------------
        Input data needed:
        rept_data_path: the path with single cell replication timing data.
        sim_L: the simulation length of the replication simulation
        sim_T: the replication time duration of replication simulation.
        chrom: the chromosome of interest
        coords: the region of interest as list [start,end]. It should be a list.
        '''
        self.chrom, self.coords, self.is_region = chrom, np.array(coords), np.all(coords!=None)
        self.data = mat73.loadmat(rept_data_path)
        self.gen_windows = self.data['genome_windows'][eval(chrom[-1])-1][0]
        self.mat = self.data['replication_state_filtered'][eval(chrom[-1])-1][0].T
        self.L, self.T = sim_L, sim_T
    
    def process_matrix(self):
        '''
        Import and rescale the matrices of single cell replication timing.
        '''
        min_value = np.min(np.nan_to_num(self.mat[self.mat>0]))
        self.mat = np.nan_to_num(self.mat,nan=min_value)
        min_max_scaler = preprocessing.MinMaxScaler()
        self.mat = min_max_scaler.fit_transform(self.mat)

    def compute_f(self):
        '''
        Compute the averages and standard deviations across the single cell replication matrix.
        Here we compute both averages over cell circle time and over spartial dimensions.
        '''
        afx, sfx, aft, sft = np.average(self.mat,axis=0), np.std(self.mat,axis=0), np.average(self.mat,axis=1), np.std(self.mat,axis=1)
        min_avg, avg_std = np.min(afx[afx>0]), np.average(sfx)
        afx[afx<=0], sfx[sfx<=0] = min_avg, avg_std
        self.avg_fx = min_max_normalize(afx)
        self.std_fx = min_max_normalize(sfx)
        self.avg_ft = min_max_normalize(aft)
        self.std_ft = min_max_normalize(sft)
        if self.is_region:
            N = len(self.avg_fx)
            self.avg_fx = self.avg_fx[(self.coords[0]*N)//chrom_sizes[self.chrom]:self.coords[1]*N//chrom_sizes[self.chrom]]
            self.std_fx = self.std_fx[(self.coords[0]*N)//chrom_sizes[self.chrom]:self.coords[1]*N//chrom_sizes[self.chrom]]
        self.avg_fx = reshape_array(self.avg_fx,self.L)
        self.std_fx = reshape_array(self.std_fx,self.L)
        self.avg_ft = reshape_array(self.avg_ft,self.T)
        self.std_ft = reshape_array(self.std_ft,self.T)

    def compute_peaks(self,prominence=0.01):
        '''
        Here we compute peaks and dips of the replication timing curves.
        ----------------------------------------------
        Input:
        prominence: it is the prominence parameter from the scipy function: find_peaks().
        '''
        self.peaks, _ = find_peaks(self.avg_fx,prominence=prominence)
        self.dips, _ = find_peaks(-self.avg_fx,prominence=prominence)

    def compute_slopes(self):
        '''
        Here the slopes between successive maxima of the replication curves are estimated.
        Slopes of replication timing curve should correlate with the speed of replication forks.
        '''
        extrema_indices = np.sort(np.concatenate((self.peaks, self.dips)))
        extrema_indices_sorted = np.sort(extrema_indices)
        print('Computing slopes of replication curves...')

        avg_slopes, std_slopes = np.zeros(self.L), np.zeros(self.L)
        for i, extr in enumerate(extrema_indices_sorted[:-1]):
            start_idx = extrema_indices_sorted[i]
            end_idx = extrema_indices_sorted[i + 1]
            delta_x = (end_idx - start_idx)
            segment_slope = (self.avg_fx[end_idx] - self.avg_fx[start_idx]) / delta_x
            sigma_slope = np.sqrt((self.std_fx[start_idx] / delta_x) ** 2 + (self.std_fx[end_idx] / delta_x) ** 2)
            avg_slopes[extr] = np.abs(segment_slope)
            std_slopes[extr] = sigma_slope
        self.speed_avg = np.average(avg_slopes)
        self.speed_std = np.average(std_slopes)
        self.speed_ratio = self.speed_std/self.speed_avg
        print('Done!\n')

    def compute_init_rate(self):
        '''
        Estimation of the initiation rate function I(x,t).
        '''
        self.initiation_rate = np.zeros((self.L,self.T))
        print('Computing initiation rate...')
        mus = self.T*(1-self.avg_fx)
        stds = self.T*self.std_fx
        for ori in tqdm(range(len(mus))):
            s = np.round(np.random.normal(mus[ori], stds[ori], 20000)).astype(int)
            s[s<0] = 0
            s[s>=self.T] = self.T-1
            unique_locations, counts = np.unique(s, return_counts=True)
            self.initiation_rate[ori,unique_locations] = counts
            self.initiation_rate[ori,:] /= np.sum(self.initiation_rate[ori,:])
        print('Computation Done! <3\n')

    def prepare_data(self):
        '''
        This function prepares the data and computes the initiation rate.
        The functions are called in the correct sequence here.
        '''
        self.process_matrix()
        self.compute_f()
        self.compute_peaks()
        self.compute_slopes()
        self.compute_init_rate()

    def run(self,scale=10):
        '''
        This function calls replication simulation.
        '''
        self.prepare_data()
        repsim = ReplicationSimulator(self.L, self.T, scale*self.initiation_rate, self.speed_ratio)
        self.sim_f, l_forks, r_forks, T_final, rep_fract = repsim.run_simulator()
        repsim.visualize_simulation()
        return self.sim_f, l_forks, r_forks
    
    def calculate_ising_parameters(self):
        '''
        Calculate compartmentalization related data.
        We connect compartmentalization with early and late replication timing sites.
        '''
        magnetic_field = -2*self.avg_fx+1
        state =  np.where(min_max_normalize(np.average(self.sim_f,axis=1),-1,1) > 0, 1, -1)
        return np.array(magnetic_field,dtype=np.float64), np.array(state,dtype=np.int32)

def run_loop(N_trials:int,scale=1,N_beads=10000,rep_duration=1000):
    '''
    For validation purposes, we can run a number of independent replication timing experiments.
    When we run these experiments, we can average the replication fraction of each one of them.
    The result should correlate highly with the experimental replication timing.
    Otherwise, the hyperparameters needs to be reconfigured.
    '''
    sf = np.zeros((N_beads,rep_duration))
    chrom = 'chr9'
    rept_path = '/home/skorsak/Data/Replication/sc_timing/GM12878_single_cell_data_hg37.mat'
    rep = Replikator(rept_path,N_beads,rep_duration,chrom)
    rep.prepare_data()

    print('Running Replikators...')
    start = time.time()
    sf = run_Ntrials(N_trials,rep.L,rep.T,scale*rep.initiation_rate,rep.speed_ratio,rep.speed_avg)
    end = time.time()
    elapsed = end - start
    print(f'Computation finished succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')

    # Replication fraction plots
    plt.figure(figsize=(20, 5),dpi=200)
    plt.plot(min_max_normalize(np.average(sf,axis=1)),'b-',label='Simulated')
    plt.plot(rep.avg_fx,'r-',label='Experimental')
    plt.xlabel('DNA position',fontsize=18)
    plt.ylabel('Replication Fraction',fontsize=18)
    # plt.ylim((0.5,1))
    plt.legend()
    # plt.savefig(f'repfrac_Ntrials{N_trials}_scale_{scale}.png',format='png',dpi=200)
    plt.show()

    # Correlations computations
    corr, pval = stats.pearsonr(np.average(sf,axis=1), rep.avg_fx)
    print(f'Pearson correlation: {corr}, with p-value {pval}.')
    corr, pval = stats.spearmanr(np.average(sf,axis=1), rep.avg_fx)
    print(f'Spearman correlation: {corr}, with p-value {pval}.')
    corr, pval = stats.kendalltau(np.average(sf,axis=1), rep.avg_fx)
    print(f'Kendall tau correlation: {corr}, with p-value {pval}.')
    mine = MINE()
    mine.compute_score(np.average(sf,axis=1), rep.avg_fx)
    mic = mine.mic()
    print(f"Maximal Information Coefficient: {mic}")
    return sf

def main():
    # Parameters
    region, chrom =  [170421513, 185491193], 'chr1'
    N_beads,rep_duration = 5000,5000
    
    # Paths
    rept_path = '/home/skorsak/Data/Replication/sc_timing/GM12878_single_cell_data_hg37.mat'

    # Run simulation
    rep = Replikator(rept_path,N_beads,rep_duration,chrom,region)
    f, l_forks, r_forks = rep.run()
    magnetic_field, state = rep.calculate_ising_parameters()