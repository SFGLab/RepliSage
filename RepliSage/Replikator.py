import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
from .common import *
from .replication_simulator import *
import pandas as pd
import time
from tqdm import tqdm
from numba import njit

@njit
def gaussian(x, mu, sig):
    return np.exp(-(x - mu)*(x - mu) / (sig*sig) / 2)/np.sqrt(2*np.pi*sig)

@njit
def get_p_vector(T, mu, sig):
    """Creates a vector of probabilities for a given locus with desired initiation mu and standard deviation sig"""
    ps = [gaussian(0, mu, sig)]
    for i in range(1, T):
        ps.append(gaussian(i, mu, sig)/(1-ps[-1]))
    return ps

def sanitize_chr_dataframe(df):
    """
    Takes a pandas DataFrame and checks whether the first column header looks like 'Chr', 'chrom', etc.
    If so, the header row is discarded and column names are reset to integers: 0, 1, 2, ...
    """
    chr_like = ['chr', 'chrom', 'chromosome']
    
    # Get the first column header, normalized
    first_col = str(df.columns[0]).strip().lower()
    
    if any(first_col.startswith(prefix) for prefix in chr_like):
        # Header is present, so remove the first row and reset index
        df = df.drop(index=0).reset_index(drop=True)

    # Rename all columns to integers
    df.columns = range(df.shape[1])
    
    return df

class Replikator:
    def __init__(self,rept_data_path:str,sim_L:int,sim_T:int,chrom:str,coords=None,Tstd_factor=0.1,speed_factor=20,sc=True):
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
        self.chrom, self.coords, self.is_region, self.sc = chrom, np.array(coords), np.all(coords!=None), sc
        chrom_nr = int(re.sub(r'\D', '', self.chrom))
        if sc:
            self.data = pd.read_parquet(rept_data_path)
            self.gen_windows = self.data[self.data['chromosome'] == chrom_nr][['start', 'end', 'center']].values
            self.chrom_size = int(np.max(self.gen_windows))
            single_cell_cols = [col for col in self.data.columns if col.startswith('SC_')]
            self.mat = self.data[self.data['chromosome'] == chrom_nr][single_cell_cols].T.values
        else:
            self.data = pd.read_csv(rept_data_path, sep='\t', header=None)
            self.data = self.data.fillna(0)
            self.data = sanitize_chr_dataframe(self.data)
            if not self.data[0].str.startswith('chr').all():
                self.data[0] = 'chr' + self.data[0].astype(str)

        print('\nStep #0: Preprocessing replication timing data...')
        self.L, self.T = sim_L, sim_T
        self.sigma_t = self.T*Tstd_factor
        self.speed_factor = speed_factor

    def process_df(self):
        self.data = self.data[self.data[0] == self.chrom].reset_index(drop=True)
        self.gen_windows = self.data[1].values
        self.avg_fx = self.data[2].values
        self.avg_fx = (self.avg_fx - np.min(self.avg_fx)) / (np.max(self.avg_fx) - np.min(self.avg_fx))
        if self.is_region:
            start, end = self.coords[0], self.coords[1]
            self.avg_fx = self.avg_fx[(self.gen_windows >= start) & (self.gen_windows <= end)]
        self.avg_fx = reshape_array(self.avg_fx, self.L)
    
    def process_matrix(self):
        '''
        Import and rescale the matrices of single cell replication timing.
        '''
        min_value = np.min(np.nan_to_num(self.mat[self.mat>0]))
        self.mat = np.nan_to_num(self.mat,nan=min_value)
        self.mat = (self.mat-np.min(self.mat))/(np.max(self.mat)-np.min(self.mat))

    def compute_f(self):
        '''
        Compute the averages and standard deviations across the single cell replication matrix.
        Here we compute both averages over cell circle time and over spartial dimensions.
        '''
        afx, sfx, aft, sft = np.average(self.mat,axis=0), np.std(self.mat,axis=0), np.average(self.mat,axis=1), np.std(self.mat,axis=1)
        min_avg, avg_std = np.min(afx[afx>0]), np.average(sfx)
        afx[afx<=0], sfx[sfx<=0] = min_avg, avg_std
        self.avg_fx = afx
        self.std_fx = sfx
        self.avg_ft = aft
        self.std_ft = sft
        if self.is_region:
            N = len(self.avg_fx)
            self.avg_fx = self.avg_fx[(self.coords[0]*N)//self.chrom_size:self.coords[1]*N//self.chrom_size]
            self.std_fx = self.std_fx[(self.coords[0]*N)//self.chrom_size:self.coords[1]*N//self.chrom_size]
        self.avg_fx = reshape_array(self.avg_fx, self.L)
        self.std_fx = reshape_array(self.std_fx, self.L)
        self.avg_ft = reshape_array(self.avg_ft, self.T)
        self.std_ft = reshape_array(self.std_ft, self.T)

    def compute_peaks(self,prominence=0.01):
        '''
        Here we compute peaks and dips of the replication timing curves.
        ----------------------------------------------------------------
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
            delta_fx = self.avg_fx[end_idx] - self.avg_fx[start_idx]
            delta_x = (end_idx - start_idx)
            # Avoid division by zero or near-zero to prevent infinite slopes
            if np.isclose(delta_fx, 0):
                segment_slope = 0.0
                sigma_slope = 0.0
            else:
                segment_slope = delta_x / delta_fx
                sigma_slope = delta_x * np.sqrt(2 * (self.sigma_t / self.T) ** 2) / (delta_fx ** 2)
            avg_slopes[extr] = np.abs(segment_slope)
            std_slopes[extr] = sigma_slope
        self.speed_avg = self.speed_factor * np.average(avg_slopes)
        self.speed_std = self.speed_factor * np.average(std_slopes)
        self.speed_ratio = self.speed_std / self.speed_avg if self.speed_avg != 0 else 0
        print('Done!')

    def compute_init_rate(self,viz=False):
        '''
        Estimation of the initiation rate function I(x,t).
        '''
        self.initiation_rate = np.zeros((self.L, self.T))
        print('Computing initiation rate...')
        mus = self.T*(1-self.avg_fx)

        for ori in tqdm(range(len(mus))):
            m = int(mus[ori])
            p_i = get_p_vector(self.T, m, sig=self.sigma_t)
            self.initiation_rate[ori, :] = p_i
        
        if viz:
            plt.figure(figsize=(15, 8),dpi=200)
            plt.imshow(self.initiation_rate.T,cmap='rainbow',aspect='auto',vmax=np.mean(self.initiation_rate)+np.std(self.initiation_rate))
            plt.title('Initiation Rate Function',fontsize=24,pad=20)
            plt.xlabel('Genomic Distance',fontsize=20)
            plt.ylabel('Pseudo-Time',fontsize=20)
            plt.xticks([])
            plt.yticks([])
            cbar = plt.colorbar()
            cbar.set_ticks([])  # Removes the tick marks
            cbar.ax.tick_params(size=0)  # Removes the tick lines
            plt.show()
        print('Computation Done! <3\n')

    def prepare_data(self):
        '''
        This function prepares the data and computes the initiation rate.
        The functions are called in the correct sequence here.
        '''
        if self.sc:
            self.process_matrix()
            self.compute_f()
        else:
            self.process_df()
        self.compute_peaks()
        self.compute_slopes()
        self.compute_init_rate()
    
    def run(self, scale=1, out_path=None):
        '''
        This function calls replication simulation.
        '''
        self.prepare_data()
        repsim = ReplicationSimulator(self.L, self.T, scale*self.initiation_rate, self.speed_ratio, self.speed_avg)
        self.sim_f, l_forks, r_forks, T_final, rep_fract = repsim.run_simulator()
        repsim.visualize_simulation(out_path)
        return self.sim_f, l_forks, r_forks
    
    def calculate_ising_parameters(self):
        '''
        Calculate compartmentalization related data.
        We connect compartmentalization with early and late replication timing sites.
        '''
        magnetic_field = 2*self.avg_fx-1
        state =  np.where(min_max_normalize(np.average(self.sim_f,axis=1),-1,1) > 0, 1, -1)
        return np.array(magnetic_field,dtype=np.float64), np.array(state,dtype=np.int32)

def run_loop(N_trials:int, scale=1.0, N_beads=5000, rep_duration=1000):
    '''
    For validation purposes, we can run a number of independent replication timing experiments.
    When we run these experiments, we can average the replication fraction of each one of them.
    The result should correlate highly with the experimental replication timing.
    Otherwise, the hyperparameters needs to be reconfigured.
    '''
    sf = np.zeros((N_beads,rep_duration))
    chrom = 'chr14'
    rept_path = '/home/skorsak/Data/Replication/sc_timing/GM12878_single_cell_data_hg37.mat'
    rep = Replikator(rept_path,N_beads,rep_duration,chrom)
    rep.prepare_data()

    print('Running Replikators...')
    start = time.time()
    sf = run_Ntrials(N_trials,rep.L,rep.T,scale*rep.initiation_rate,rep.speed_ratio,rep.speed_avg)
    end = time.time()
    elapsed = end - start
    print(f'Computation finished succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')

    # Correlations computations
    pears, pval = stats.pearsonr(np.average(sf,axis=1), rep.avg_fx)
    print(f'Pearson correlation: {pears:.3f} %, with p-value {pval}.')
    spear, pval = stats.spearmanr(np.average(sf,axis=1), rep.avg_fx)
    print(f'Spearman correlation: {spear:.3f} %, with p-value {pval}.')
    kend, pval = stats.kendalltau(np.average(sf,axis=1), rep.avg_fx)
    print(f'Kendall tau correlation: {kend:.3f} %, with p-value {pval}.')

    # Improved plot lines with thicker width and transparency
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(15, 5),dpi=200)
    plt.plot(
        min_max_normalize(np.average(sf, axis=1)), 
        'b-', 
        label='Simulated', 
        linewidth=2.5, 
        alpha=0.8
    )
    plt.plot(
        rep.avg_fx, 
        'r-', 
        label='Experimental', 
        linewidth=2.5, 
        alpha=0.8
    )

    # Enhanced axis labels
    plt.xlabel('Genomic Distance', fontsize=18, labelpad=10)
    plt.ylabel('Replication Fraction', fontsize=18, labelpad=10)

    # Custom ticks with lighter grid
    plt.xticks([], fontsize=12, color='grey')
    plt.yticks([], fontsize=12, color='grey')
    plt.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)

    # Adding a styled text box
    text_x = 2700  # X-coordinate for the text box (adjust as needed)
    text_y = 0.05  # Y-coordinate for the text box (adjust as needed)
    plt.text(
        text_x, text_y, 
        f'Pearson Correlation: {100 * pears:.2f}%\nSpearman Correlation: {100 * spear:.2f}%', 
        fontsize=12, 
        color='black', 
        bbox=dict(facecolor='lightblue', alpha=0.7, edgecolor='navy', boxstyle='round,pad=0.5')
    )

    # Stylish legend with custom frame
    plt.legend(
        fontsize=14, 
        loc='upper left', 
        frameon=True, 
        framealpha=0.8, 
        edgecolor='black', 
        facecolor='white'
    )

    # Save as high-resolution images with consistent naming
    plt.savefig(f'ntrial_{N_trials}_scale_{scale}_rep_frac.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(f'ntrial_{N_trials}_scale_{scale}_rep_frac.pdf', format='pdf', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
    
    return sf

def main():
    # Parameters
    chrom =  'chr14'
    coords = [10835000, 98674700]
    N_beads,rep_duration = 20000,1000
    
    # Paths
    rept_path = '/home/skorsak/Data/Replication/sc_timing/GM12878_single_cell_data_hg37.mat'

    # Run simulation
    rep = Replikator(rept_path,N_beads,rep_duration,chrom,coords)
    f, l_forks, r_forks = rep.run(scale=1)
    magnetic_field, state = rep.calculate_ising_parameters()