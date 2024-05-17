import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.signal import find_peaks
from tqdm import tqdm

def expand_columns(array, new_columns):
    """
    Expand each column of a given array by repeating its elements to fit the desired number of columns.
    
    Parameters:
        array (numpy.ndarray): The input array of shape (N, T1).
        new_columns (int): The desired number of columns (T2 > T1).
    
    Returns:
        numpy.ndarray: The expanded array of shape (N, new_columns).
    """
    N, T1 = array.shape
    
    if new_columns <= T1:
        raise ValueError("Number of new columns (T2) must be greater than the original number of columns (T1).")
    
    # Compute the number of times to repeat each element within a column
    repeat_factor = new_columns // T1
    
    # Reshape the array to repeat each element in the column
    expanded_array = np.repeat(array, repeat_factor, axis=1)
    
    # Trim or slice the expanded array to match the desired number of columns
    expanded_array = expanded_array[:, :new_columns]

    # Calculate the remaining columns needed
    remaining_columns = new_columns - expanded_array.shape[1]
    
    if remaining_columns > 0:
        # Pad the array with zeros to reach the desired number of columns
        zero_padding = np.zeros((N, remaining_columns))
        expanded_array = np.hstack((expanded_array, zero_padding))
    
    return expanded_array

def replikator_num_simulator(L,time_steps,initiation_rate,mu_v,std_v,viz=False):
    # Initialize arrays
    vs = np.zeros(L)  # Fork propagation speed
    f = np.zeros((L, time_steps))  # Replication fraction
    rep_fract = list()
    replicated_dna = np.zeros((L, time_steps), dtype=int)  # Forks position
    r_forks, l_forks = np.zeros((L, time_steps), dtype=int), np.zeros((L, time_steps), dtype=int)
    init_t, init_x, coal_t, coal_x = list(), list(), list(), list()

    # Monte Carlo simulation
    print('Running replikator....')
    dna_is_replicated = False
    t, T = 1, time_steps
    while not dna_is_replicated:
        # Fork initiation
        initiate_forks = np.random.rand(L) < initiation_rate[:,t]
        init_locs = np.nonzero(initiate_forks)[0]
        for init in init_locs:
            if replicated_dna[init, t-1]==0:
                vel = np.random.normal(mu_v[init], std_v[init], 1)[0]
                vs[init] = vel if vel>=1 else 1
        replicated_dna[initiate_forks, t] = 1
        
        previously_initiated = replicated_dna[:, t-1] == 1
        xs = np.nonzero(initiate_forks*(~previously_initiated))[0]
        for x in xs:
            init_x.append(x)
            init_t.append(t)
        
        # Fork propagation
        for i in range(L):
            if replicated_dna[i, t - 1] == 1:
                # Move the fork according to the propagation speed
                v = vs[i]
                distance = int(round(np.random.uniform(0,v+1,1)[0]))
                if replicated_dna[(i + distance + 1) % L, t-1]==1 and replicated_dna[(i+1)%L, t - 1] == 0: 
                    coal_t.append(t)
                    coal_x.append((i + distance) % L)
                if replicated_dna[(i - distance -1) % L, t-1]==1 and replicated_dna[(i-1)%L, t - 1] == 0: 
                    coal_t.append(t)
                    coal_x.append((i - distance) % L)
                
                if (i-distance)%L<(i+distance)%L: # they have not met the boundaries
                    replicated_dna[(i-distance)%L:(i+distance)%L,t] = 1
                    vs[(i-distance)%L:(i+distance)%L] = v
                else: # they meet boundaries
                    if (i+distance)>L:
                        replicated_dna[i:L,t], replicated_dna[0:(i+distance)%L,t] = 1, 1
                        vs[i:L], vs[0:(i+distance)%L] = v, v
                    if (i-distance)<0:
                        replicated_dna[0:i,t], replicated_dna[(i-distance)%L:L,t] = 1, 1
                        vs[0:i], vs[(i-distance)%L:L] = v, v
                replicated_dna[(i-distance)%L, t], replicated_dna[(i+distance)%L, t] = 1, 1
                r_forks[(i + distance) % L,t] = 1 if replicated_dna[(i+distance)%L, t - 1] == 0 else 0
                l_forks[(i - distance) % L,t] = 1 if replicated_dna[(i-distance)%L, t - 1] == 0 else 0

        # Check if termination condition is satisfied
        rep_fract.append(np.count_nonzero(replicated_dna[:,t-1])/L)
        if np.all(replicated_dna[:,t-1]==1): 
            dna_is_replicated = True
            T = t

        # Calculate replication fraction
        f[:, t] = replicated_dna[:, t]
        t+=1
    print('Done! ;)')

    replicated_dna, f = replicated_dna[:,:T], f[:,:T]
    r_forks, l_forks = r_forks[:,:T], l_forks[:,:T]

    if T<time_steps:
        replicated_dna, f = expand_columns(replicated_dna,time_steps), expand_columns(f,time_steps)
        r_forks, l_forks = expand_columns(r_forks,time_steps), expand_columns(l_forks,time_steps)
        zero_columns = np.all(f == 0, axis=0) & (np.arange(time_steps)>time_steps/2)
        f[:, zero_columns] = 1
    print('Dimension of r_forks:', r_forks.shape)

    if viz:
        # Replication fraction
        plt.figure(figsize=(12.6, 6))
        plt.imshow(f.T, cmap='bwr', aspect='auto', origin='lower')
        plt.colorbar(label='Replication Fraction')
        plt.title('DNA Replication Simulation')
        plt.xlabel('DNA position')
        plt.ylabel('Computational Time')
        plt.show()

        # Plot Replication Fraction
        plt.figure(figsize=(8, 6))
        plt.plot(rep_fract)
        plt.xlabel('Time',fontsize=18)
        plt.ylabel(r'Replication Fraction',fontsize=18)
        plt.show()

        # Fork Locations
        plt.figure(figsize=(10, 6))
        plt.imshow(r_forks.T, cmap='Reds', aspect='auto', origin='lower',vmax=0.1)
        plt.title('Right Fork Locations',fontsize=28)
        plt.xlabel('DNA position',fontsize=18)
        plt.ylabel('Computational Time',fontsize=18)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.imshow(l_forks.T, cmap='Blues', aspect='auto', origin='lower',vmax=0.1)
        plt.title('Left Fork Locations',fontsize=28)
        plt.xlabel('DNA position',fontsize=18)
        plt.ylabel('Computational Time',fontsize=18)
        plt.show()

    return l_forks, r_forks

def min_max_normalize(matrix, Min, Max):
    # Calculate the minimum and maximum values of the matrix
    matrix = np.nan_to_num(matrix)
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    # Normalize the matrix using the min-max formula
    normalized_matrix = Min + (Max - Min) * ((matrix - min_val) / (max_val - min_val))

    return normalized_matrix

def reshape_array_by_averaging(input_array, new_dimension):
    """
    Reshape the input numpy array by computing averages across windows.

    Parameters:
    input_array (numpy.ndarray): Input array of dimension M.
    new_dimension (int): Desired new dimension N for the reshaped array.

    Returns:
    numpy.ndarray: Reshaped array of dimension N.
    """

    # Calculate the size of each window
    original_length = len(input_array)
    window_size = original_length // new_dimension  # Calculate the size of each window

    # Initialize the reshaped array
    reshaped_array = np.zeros(new_dimension)

    # Iterate over each segment/window and compute the average
    for i in range(new_dimension):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size

        # Compute the average of values in the current window
        if i == new_dimension - 1:  # Handle the last segment
            reshaped_array[i] = np.mean(input_array[start_idx:])
        else:
            reshaped_array[i] = np.mean(input_array[start_idx:end_idx])

    return reshaped_array

class RepTimingPreproc:
    '''
    This is a class that imports single cell replication timing data and performs some operations,
    to convert them to useful data that can be used as input for the simulation.
    '''
    def __init__(self,L,screp_time_path,chrom,oris_path=None,region=None):
        '''
        Importing data.

        - L (int): simulation length
        - screp_time_path (str): the path with single cell replication timing data.
        - chrom (int): the number of the chromosome of interest.
        - oris_path (str): the path with the replication origin data.
        - region (list): list with start and end coordinate of the region of interest.
        '''
        self.L = L
        self.chrom, self.region = chrom, region
        self.rept_preproc(screp_time_path)
        self.oris_reproc(oris_path)

    def rept_preproc(self,screp_time_path):
        '''
        This function performs a preprocessing of single cell replication curves.
        '''
        rep_timing_df = pd.read_csv(screp_time_path,sep='\t')
        self.experiment_columns = [col for col in rep_timing_df if col.startswith('HPSI')]
        self.num_experiments = len(rep_timing_df)
        self.rept_chrom_df = rep_timing_df[rep_timing_df['chr'] == self.chrom].reset_index(drop=True)
        self.rept_chrom_df[self.experiment_columns] = self.rept_chrom_df[self.experiment_columns].fillna(self.rept_chrom_df[self.experiment_columns].mean())
        if np.all(self.region!=None):
            self.rept_chrom_df = self.rept_chrom_df[(self.rept_chrom_df['start']>self.region[0])&(self.rept_chrom_df['end']<self.region[1])].reset_index(drop=True)       
        self.original_region_length = self.rept_chrom_df['end'].values[-1] if self.region==None else self.region[1]-self.region[0]
        
    def oris_reproc(self,oris_path):
        '''
        Replication origins preprocessing.
        '''
        if oris_path!=None:
            ori_df = pd.read_csv(oris_path,sep='\t',header=None)
            chrom_ori_df = ori_df[ori_df[0]=='chr'+str(self.chrom)].reset_index(drop=True)
            chrom_ori_df[3] = chrom_ori_df[3].fillna(chrom_ori_df[3].mean())
            if np.all(self.region!=None):
                chrom_ori_df = chrom_ori_df[(chrom_ori_df[1]>self.region[0])&(chrom_ori_df[2]<self.region[1])].reset_index(drop=True)
                chrom_ori_df[[1,2]] = chrom_ori_df[[1,2]]-self.region[0]
            chrom_ori_df[[1,2]] = self.L*(chrom_ori_df[[1,2]]/self.original_region_length)
            chrom_ori_df[[1,2]] = chrom_ori_df[[1,2]].round(0).astype(int)
            
            chrom_ori_df[3] = chrom_ori_df.groupby([1, 2])[3].transform('mean')
            self.oris = np.unique(chrom_ori_df[1].values)
        else:
            self.oris = np.arange(L)
        print(f'The number of replication origins is {len(self.oris)}')

    def vizualize_repcurves(self):
        '''
        Plotting single cell replication curves.
        '''
        fig, axs = plt.subplots(self.num_experiments, 1, figsize=(12, 400))
        for i, experiment_col in enumerate(self.experiment_columns):
            axs[i].plot(chromosome_df['start'], chromosome_df[experiment_col], label=experiment_col)
        plt.show()

    def aggregate_signals(self):
        '''
        Compute the average and the standard deviation of the single cell replication curves.
        '''
        # Compute average and standard deviation across experiments for each location
        self.rept_chrom_df['average_signal'] = self.rept_chrom_df[self.experiment_columns].mean(axis=1)
        self.rept_chrom_df['std_dev_signal'] = self.rept_chrom_df[self.experiment_columns].std(axis=1)
        self.avg_signal, self.std_signal = np.zeros(self.original_region_length), np.zeros(self.original_region_length)
        print('Aggregating signals...')
        starts, ends = self.rept_chrom_df['start'].values-self.region[0], self.rept_chrom_df['end'].values-self.region[0]
        mean_values, std_values = self.rept_chrom_df['average_signal'].values, self.rept_chrom_df['std_dev_signal'].values
        for i in tqdm(range(len(self.rept_chrom_df))):
            self.avg_signal[starts[i]:ends[i]], self.std_signal[starts[i]:ends[i]] = mean_values[i], std_values[i]
        self.avg_signal = reshape_array_by_averaging(self.avg_signal,self.L)
        self.std_signal = reshape_array_by_averaging(self.std_signal,self.L)
        print('Done! :D')

    def find_peaks_dips(self,viz=False,prominence=0.1):
        '''
        It computes the peaks of the averaged replication curve.
        '''
        self.peaks, _ = find_peaks(self.avg_signal,prominence=prominence)
        self.dips, _ = find_peaks(-self.avg_signal,prominence=prominence)

    def compute_slopes(self):
        '''
        It computes the slopes between the maxima of the average replicaton curve.
        These slopes are linked with the replication fork velocities.
        '''
        extrema_indices = np.sort(np.concatenate((self.peaks, self.dips)))
        extrema_indices_sorted = np.sort(extrema_indices)
        exp_slopes = list()
        print('Computing slopes of replication curves...')
        for exp_col in tqdm(self.experiment_columns):
            exp_array = np.zeros(self.original_region_length)
            starts = self.rept_chrom_df['start'].values - self.region[0]
            ends = self.rept_chrom_df['end'].values - self.region[0]
            values = self.rept_chrom_df[exp_col].values
            for start, end, value in zip(starts, ends, values):
                exp_array[start:end] = value
            exp_array = reshape_array_by_averaging(exp_array,self.L)
            slopes = np.zeros(self.L)
            for i, extr in enumerate(extrema_indices_sorted[:-1]):
                start_idx = extrema_indices_sorted[i]
                end_idx = extrema_indices_sorted[i + 1]
                segment_slope = (exp_array[end_idx] - exp_array[start_idx]) / (end_idx - start_idx)
                slopes[extr] = segment_slope
            exp_slopes.append(slopes)
        print('Done!\n')
        exp_slopes = np.nan_to_num(np.array(exp_slopes))

        # Aggregate across experiments
        self.average_slopes = np.average(exp_slopes,axis=0)
        self.std_slopes = np.std(exp_slopes,axis=0)

    def compute_initiation_rate(self,time_steps,viz=False):
        '''
        Computes the initiation rates of each origin.

        time_steps (int): the simulation time steps.
        '''
        self.initiation_rate = np.zeros((time_steps,self.L))
        mus = min_max_normalize(self.avg_signal, 0, time_steps)
        stds = self.std_signal*time_steps/(np.max(self.avg_signal)-np.min(self.avg_signal))

        print('Computing initiation rate...')
        for i in tqdm(range(self.L)):
            if i in self.oris:
                mu, std = mus[i], stds[i]
                s = np.round(np.random.normal(mu, std, 10000)).astype(int)
                s[s>=time_steps] = time_steps-1
                unique_locations, counts = np.unique(s, return_counts=True)
                self.initiation_rate[unique_locations,i] = counts
                self.initiation_rate[:,i] /= np.sum(self.initiation_rate[:,i])
        print('Computation Done! <3')

        if viz:
            plt.figure(figsize=(10.6, 5))
            plt.imshow(self.initiation_rate, cmap='jet', aspect='auto', origin='lower',vmax=5/time_steps)
            plt.colorbar(label='Initiation Rate')
            plt.xlabel('DNA position')
            plt.ylabel('Computational Time')
            plt.show()

def replikator(ori_path,rept_path, L, chrom, region, time_steps):

    prproc = RepTimingPreproc(L,rept_path,chrom,ori_path,region)
    prproc.aggregate_signals()
    prproc.find_peaks_dips()
    prproc.compute_slopes()
    prproc.compute_initiation_rate(time_steps,viz=True)
    lfs, rfs = replikator_num_simulator(L=L,time_steps=time_steps,initiation_rate=prproc.initiation_rate.T,\
        mu_v=5*prproc.average_slopes,std_v=5*prproc.std_slopes,viz=True)
    return lfs, rfs

# L, chrom,time_steps = 1000, 1, int(1e3)
# region = [178421513, 179491193]
# rept_path = '/home/skorsak/Documents/data/Replication/timing/iPSC_individual_level_data.txt'
# ori_path = '/home/skorsak/Documents/data/Replication/LCL_MCM_replication_origins.bed'