import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.signal import find_peaks
from sklearn import preprocessing
from common import *
from Replikator_numsimulator import *
import scipy.io
import time
from tqdm import tqdm

class Replikator:
    def __init__(self,rept_data_path,sim_L,sim_T,chrom='chr9',coords=None):
        self.chrom, self.coords, self.is_region = chrom, np.array(coords), np.all(coords!=None)
        self.mat = scipy.io.loadmat(rept_data_path)['Chr9_replication_state_filtered']
        self.L, self.T = sim_L, sim_T
        self.chrom_length = chrom_sizes[self.chrom]

    def process_matrix(self):
        min_value = np.min(np.nan_to_num(self.mat[self.mat>0]))
        self.mat = np.nan_to_num(self.mat,nan=min_value)
        min_max_scaler = preprocessing.MinMaxScaler()
        self.mat = min_max_scaler.fit_transform(self.mat)
        if self.is_region:
            resolution = self.chrom_length//len(self.mat)
            idxs = self.coords//resolution
            self.mat = self.mat[:,idxs[0]:idxs[1]]

    def compute_f(self):
        afx, sfx, aft, sft = np.average(self.mat,axis=0), np.std(self.mat,axis=0), np.average(self.mat,axis=1), np.std(self.mat,axis=1)
        min_avg, max_std = np.min(afx[afx>0]), np.max(sfx)
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
        speed_avg = np.average(avg_slopes)
        speed_std = np.average(std_slopes)
        self.speed_ratio = speed_std/speed_avg
        print(f'Std - Average ratio is {speed_std/speed_avg}.')
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
        f, l_forks, r_forks = numerical_simulator(self.L,self.T,scale*self.initiation_rate,self.speed_ratio,viz=viz)
        return f, l_forks, r_forks

def run_loop(N_trials,scale=10,N_beads=10000,rep_duration=5000):
    sf = np.zeros((N_beads,rep_duration))
    rept_path = '/home/skorsak/Data/Replication/sc_timing/Chr9_replication_state_filtered.mat'
    rep = Replikator(rept_path,N_beads,rep_duration)
    rep.prepare_data()

    print('Running Replikators...')
    start = time.time()
    sf = run_Ntrials(N_trials,rep.L,rep.T,scale*rep.initiation_rate,rep.speed_ratio)
    end = time.time()
    elapsed = end - start
    print(f'Computation finished succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')
    
    # Replication fraction
    plt.figure(figsize=(17, 4))
    plt.imshow(1-sf.T, cmap='bwr', aspect='auto', origin='lower')
    plt.colorbar(label='Replication Fraction')
    plt.title('DNA Replication Simulation')
    plt.xlabel('DNA position',fontsize=16)
    plt.ylabel('Computational Time',fontsize=16)
    plt.savefig(f'averafe_rep_frac_Ntrials{N_trials}_scale_{scale}.png',format='png',dpi=200)
    plt.show()

    # Replication fraction
    plt.figure(figsize=(17, 4))
    plt.plot(np.average(sf,axis=1))
    plt.xlabel('DNA position',fontsize=16)
    plt.ylabel('Average Replication Fraction',fontsize=16)
    plt.savefig(f'averafe_rep_frac_x_Ntrials{N_trials}_scale_{scale}.png',format='png',dpi=200)
    plt.show()
    return sf

def main():
    # Parameters
    region, chrom =  [0, 150617247], 'chr9'
    N_beads,rep_duration = 10000,5000

    # Paths
    rept_path = '/home/skorsak/Data/Replication/sc_timing/Chr9_replication_state_filtered.mat'

    # Run simulation
    rep = Replikator(rept_path,N_beads,rep_duration)
    f, l_forks, r_forks = rep.run()