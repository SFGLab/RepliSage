import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
from sklearn import preprocessing
from sklearn.metrics import mutual_info_score
from common import *
from numsimulator import *
import mat73
from minepy import MINE
import time
from tqdm import tqdm

class Replikator:
    def __init__(self,rept_data_path,sim_L,sim_T,chrom,coords=None):
        self.chrom, self.coords, self.is_region = chrom, np.array(coords), np.all(coords!=None)
        self.data = mat73.loadmat(rept_data_path)
        self.gen_windows = self.data['genome_windows'][eval(chrom[-1])-1][0]
        self.mat = self.data['replication_state_filtered'][eval(chrom[-1])-1][0].T
        self.L, self.T = sim_L, sim_T
    
    def process_matrix(self):
        min_value = np.min(np.nan_to_num(self.mat[self.mat>0]))
        self.mat = np.nan_to_num(self.mat,nan=min_value)
        min_max_scaler = preprocessing.MinMaxScaler()
        self.mat = min_max_scaler.fit_transform(self.mat)
        if self.is_region:
            winds_str, winds_end = self.gen_windows[:,0], self.gen_windows[:,2]
            self.mat = self.mat[:,(winds_end > self.coords[0]) & (winds_str < self.coords[1])]

    def compute_f(self):
        afx, sfx, aft, sft = np.average(self.mat,axis=0), np.std(self.mat,axis=0), np.average(self.mat,axis=1), np.std(self.mat,axis=1)
        min_avg, max_std = np.min(afx[afx>0]), np.min(sfx)
        afx[afx<=0], sfx[sfx<=0] = min_avg, max_std
        self.avg_fx = min_max_normalize(reshape_array(afx,self.L))
        self.std_fx = min_max_normalize(reshape_array(sfx,self.L))
        self.avg_ft = min_max_normalize(reshape_array(aft,self.T))
        self.std_ft = min_max_normalize(reshape_array(sft,self.T))        

    def compute_peaks(self,prominence=0.01):
        self.peaks, _ = find_peaks(self.avg_fx,prominence=prominence)
        self.dips, _ = find_peaks(-self.avg_fx,prominence=prominence)

    def compute_slopes(self):
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
        # print(f'Average speed {self.speed_avg}, std speed {self.speed_std}.')
        # print(f'Std - Average ratio is {self.speed_std/self.speed_avg}.')
        print('Done!\n')

    def compute_init_rate(self):
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
        self.process_matrix()
        self.compute_f()
        self.compute_peaks()
        self.compute_slopes()
        self.compute_init_rate()

    def run(self,scale=10,viz=True):
        self.prepare_data()
        self.sim_f, l_forks, r_forks = numerical_simulator(self.L,self.T,scale*self.initiation_rate,self.speed_ratio,self.speed_avg,viz=viz)
        return self.sim_f, l_forks, r_forks
    
    def calculate_ising_parameters(self):
        magnetic_field = -2*(self.avg_fx-np.mean(self.avg_fx))/(np.max(self.avg_fx)-np.min(self.avg_fx))
        state =  np.where(min_max_normalize(np.average(self.sim_f,axis=1),-1,1) > 0, 1, -1)
        return np.array(magnetic_field,dtype=np.float64), np.array(state,dtype=np.int32)

def run_loop(N_trials,scale=10,N_beads=10000,rep_duration=5000):
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
    plt.plot(np.average(sf,axis=1),'b-',label='Simulated')
    plt.xlabel('DNA position',fontsize=18)
    plt.ylabel('Simulated Replication Fraction',fontsize=18)
    plt.ylim((0.5,1))
    plt.savefig(f'sim_repfrac_Ntrials{N_trials}_scale_{scale}.png',format='png',dpi=200)
    plt.show()

    # Replication fraction plots
    plt.figure(figsize=(20, 5),dpi=200)
    plt.plot(rep.avg_fx,'r-',label='Experimental')
    plt.xlabel('DNA position',fontsize=18)
    plt.ylabel('Experimental Replication Fraction',fontsize=18)
    plt.savefig(f'exp_repfrac_Ntrials{N_trials}_scale_{scale}.png',format='png',dpi=200)
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
    region, chrom =  [0, 248387328], 'chr1'
    N_beads,rep_duration = 50000,5000

    # Paths
    rept_path = '/home/skorsak/Data/Replication/sc_timing/GM12878_single_cell_data_hg37.mat'

    # Run simulation
    rep = Replikator(rept_path,N_beads,rep_duration,chrom,region)
    f, l_forks, r_forks = rep.run()
    magnetic_field, state = rep.calculate_ising_parameters()