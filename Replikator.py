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

def run_replikator(L,time_steps,initiation_rate,mu_v,std_v,viz=False):
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
        initiate_forks = np.random.rand(L) < initiation_rate
        init_locs = np.nonzero(initiate_forks)[0]
        for init in init_locs:
            if replicated_dna[init, t-1]==0:
                vel = np.random.normal(mu_v, std_v, 1)[0]
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
        f[:, t] = np.sum(replicated_dna[:, :t + 1], axis=1) / (t + 1)
        t+=1
    print('Done! ;)')

    replicated_dna, f = replicated_dna[:,:T], f[:,:T]
    r_forks, l_forks = r_forks[:,:T], l_forks[:,:T]
    

    if T<time_steps:
        replicated_dna, f = expand_columns(replicated_dna,time_steps), expand_columns(f,time_steps)
        r_forks, l_forks = expand_columns(r_forks,time_steps), expand_columns(l_forks,time_steps)

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

# def sigmoid_f(x,x0):
#     return 1/(1+np.exp(-x+5+x0))

# def generate_replifrac(rep_t,time_steps,viz=False):
#     f_x = min_max_normalize(np.array(rep_t), 0, 5)
#     f = np.zeros((time_steps,len(f_x)))
    
#     for i in range(len(f_x)):
#         s = sigmoid_f(np.linspace(0,15,time_steps),f_x[i])
#         f[:,i] = s

#     if viz:
#         plt.figure(figsize=(12.6, 6))
#         plt.imshow(f, cmap='bwr', aspect='auto', origin='lower')
#         plt.colorbar(label='Experimental Replication Fraction')
#         plt.xlabel('DNA position')
#         plt.ylabel('Computational Time')
#         plt.show()
    
#     return f

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
    def __init__(self,L,screp_time_path,chrom,region=None):
        self.L = L
        self.chrom = chrom
        self.df = df = pd.read_csv(screp_time_path,sep='\t')
        self.chromosome_df = self.df[self.df['chr'] == self.chrom]
    
    def vizualize_repcurves(self):
        # Extract experiment columns (assuming they start with 'HPSI')
        self.experiment_columns = [col for col in self.chromosome_df.columns if col.startswith('HPSI')]

        # Determine the number of experiments
        num_experiments = len(experiment_columns)

        # Create subplots
        fig, axs = plt.subplots(num_experiments, 1, figsize=(12, 400))

        # Plot each experiment in a separate subplot
        for i, experiment_col in enumerate(experiment_columns):
            axs[i].plot(chromosome_df['start'], chromosome_df[experiment_col], label=experiment_col)

        # Adjust layout and show plots
        plt.show()

    def aggregate_signals(self,viz=False):
        # Compute average and standard deviation across experiments for each location
        self.chromosome_df['average_signal'] = self.chromosome_df[experiment_columns].mean(axis=1)
        self.chromosome_df['std_dev_signal'] = self.chromosome_df[experiment_columns].std(axis=1)
        self.avg_signal = reshape_array_by_averaging(self.chromosome_df['average_signal'].values,self.L)
        self.std_signal = reshape_array_by_averaging(self.chromosome_df['std_dev_signal'].values,self.L)
        if viz:
            # Plot average signal
            x = np.arange(self.L)
            plt.figure(figsize=(20, 5))
            plt.plot(x, self.avg_signal, label='Average Signal')
            plt.xlabel('Genomic Coordinate')
            plt.ylabel('Average Signal')
            plt.title(f'Average Signal Across Experiments for Chromosome {chromosome_of_interest}')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot standard deviation signal
            plt.figure(figsize=(20, 5))
            plt.plot(x, self.std_signal, label='Standard Deviation of Signal', color='orange')
            plt.xlabel('Genomic Coordinate')
            plt.ylabel('Standard Deviation')
            plt.title(f'Standard Deviation of Signal Across Experiments for Chromosome {chromosome_of_interest}')
            plt.legend()
            plt.grid(True)
            plt.show()

    def find_peaks_dips(self,viz=False,prominence=0.1):
        # Find peaks in the average signal
        self.peaks, _ = find_peaks(self.avg_signal,prominence=prominence)

        # Find dips (valleys) in the average signal
        self.dips, _ = find_peaks(-self.avg_signal,prominence=prominence)  # Using negative signal for dips

        print('Number of peaks',len(self.peaks))
        print('Number of dips',len(self.dips))

        # Plot average signal with identified peaks and dips
        if viz:
            plt.figure(figsize=(20, 6))
            x = np.arange(self.L)
            plt.plot(x, self.avg_signal, label='Average Signal')
            plt.plot(x[peaks], self.avg_signal[peaks], 'rx', markersize=8, label='Peaks')
            plt.plot(x[dips], self.avg_signal[dips], 'gx', markersize=8, label='Dips')
            plt.xlabel('Genomic Coordinate')
            plt.ylabel('Average Signal')
            plt.title(f'Average Signal with Peaks and Dips for Chromosome {chromosome_of_interest}')
            plt.legend()
            plt.grid(True)
            plt.show()

    def compute_slopes(self):
        # Combine indices of peaks and dips to create a union of extrema
        extrema_indices = np.sort(np.concatenate((self.peaks, self.dips)))

        # Sort the union of extrema based on genomic coordinates
        extrema_indices_sorted = np.sort(extrema_indices)

        # Compute average slopes between consecutive maxima (peaks) for each experiment
        exp_slopes = list()

        for exp_col in self.experiment_columns:
            exp_array = chromosome_df[exp_col].values
            slopes = list()
            for i in range(len(extrema_indices_sorted) - 1):
                start_idx = extrema_indices_sorted[i]
                end_idx = extrema_indices_sorted[i + 1]
                segment_slope = (exp_array[end_idx] - exp_array[start_idx]) / (end_idx - start_idx)
                slopes.append(segment_slope)
            exp_slopes.append(slopes)
        exp_slopes = np.nan_to_num(np.array(exp_slopes))

        # Aggregate across experiments
        self.average_slopes = np.average(exp_slopes,axis=0)
        self.std_slopes = np.std(exp_slopes,axis=0)

    def compute_initiation_rate(self,time_steps),viz=False:
        self.initiation_rate = np.zeros((time_steps,self.L))
        mus = min_max_normalize(self.avg_signal, 0, time_steps)
        stds = self.std_signal*time_steps/(np.max(self.avg_signal)-np.min(self.avg_signal))

        print('Computing initiation rate...')
        for i in tqdm(range(self.L)):
            mu, std = mus[i], stds[i]
            s = np.round(np.random.normal(mu, std, 10000)).astype(int)
            s[s>=time_steps] = time_steps-1
            unique_locations, counts = np.unique(s, return_counts=True)
            self.initiation_rate[unique_locations,i] = counts
            self.initiation_rate[:,i] /= np.sum(self.initiation_rate[:,i])
        print('Computation Done! <3')

        if viz:
            plt.figure(figsize=(10.6, 5))
            plt.imshow(self.initiation_rate, cmap='jet', aspect='auto', origin='lower',vmax=10/time_steps)
            plt.colorbar(label='Initiation Rate')
            plt.xlabel('DNA position')
            plt.ylabel('Computational Time')
            plt.show()

def main():
    run_replikator(L=1000,time_steps=int(1e3),initiation_rate=0.001,mu_v=3,std_v=2,viz=True)