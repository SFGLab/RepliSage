import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.signal import find_peaks
from sklearn import preprocessing
import scipy.io
from scipy.interpolate import interp1d
from tqdm import tqdm

chrom_sizes = {'chr1':248387328,'chr2':242696752,'chr3':201105948,'chr4':193574945,
               'chr5':182045439,'chr6':172126628,'chr7':160567428,'chr8':146259331,
               'chr9':150617247,'chr10':134758134,'chr11':135127769,'chr12':133324548,
               'chr13':113566686,'chr14':101161492,'chr15':99753195,'chr16':96330374,
               'chr17':84276897,'chr18':80542538,'chr19':61707364,'chr20':66210255,
               'chr21':45090682,'chr22':51324926,'chrX':154259566,'chrY':62460029}

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

def min_max_normalize(matrix, Min, Max):
    # Calculate the minimum and maximum values of the matrix
    matrix = np.nan_to_num(matrix)
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    # Normalize the matrix using the min-max formula
    normalized_matrix = Min + (Max - Min) * ((matrix - min_val) / (max_val - min_val))

    return normalized_matrix

def reshape_array(input_array, new_dimension):
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
    reshaped_array = np.zeros(new_dimension)

    # Iterate over each segment/window and compute the average
    if original_length>new_dimension:
        # In case that we want to downgrade the dimension we can compute averages
        for i in range(new_dimension):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size

            # Compute the average of values in the current window
            if i == new_dimension - 1:  # Handle the last segment
                reshaped_array[i] = np.average(input_array[start_idx:])
            else:
                reshaped_array[i] = np.average(input_array[start_idx:end_idx])
    else:
        # In case that we need higher dimension we perform spline interpolation
        original_indices = np.linspace(0, original_length - 1, original_length)
        new_indices = np.linspace(0, original_length - 1, new_dimension)
        spline_interpolation = interp1d(original_indices, input_array, kind='cubic')
        reshaped_array = spline_interpolation(new_indices)

    return reshaped_array

class Replikator:
    def __init__(self,rept_data_path,oris_path,sim_L,sim_T,chrom='chr9',coords=None):
        self.chrom, self.coords, self.is_region = chrom, np.array(coords), np.all(coords!=None)
        self.mat = scipy.io.loadmat(rept_data_path)['Chr9_replication_state_filtered']
        self.oris_df = pd.read_csv(oris_path,sep='\t',header=None)
        self.L, self.T = sim_L, sim_T

    def proces_oris(self,chrom_length=150617247):
        chrom_length = chrom_sizes[self.chrom]
        self.oris_df = self.oris_df[self.oris_df[0]=='chr9'].reset_index(drop=True)
        if self.is_region:
            self.oris_df = self.oris_df[(self.oris_df[1]>coords[0])&(self.oris_df[2]<coords[1])].reset_index(drop=True)
            self.oris_df[1],  self.oris_df[2] = self.oris_df[1]-coords[0],  self.oris_df[2]-coords[1]
            chrom_length = coords[1]-coords[0]
        self.oris, self.oris_log_pvals = self.L*self.oris_df[1].values//chrom_length, self.oris_df[3].values

    def process_matrix(self):
        min_value = np.min(np.nan_to_num(self.mat[self.mat>0]))
        self.mat = np.nan_to_num(self.mat,nan=min_value)
        min_max_scaler = preprocessing.MinMaxScaler()
        self.mat = min_max_scaler.fit_transform(self.mat)
        if self.is_region:
            resolution = chrom_length//len(self.mat)
            idxs = self.coords//resolution
            self.mat = self.mat[:,idxs[0]:idxs[1]]

    def compute_f(self):
        self.avg_fx = reshape_array(np.average(self.mat,axis=0),self.L)
        self.std_fx = reshape_array(np.std(self.mat,axis=0),self.L)
        self.avg_ft = reshape_array(np.average(self.mat,axis=1),self.T)
        self.std_ft = reshape_array(np.std(self.mat,axis=1),self.T)
        min_avg, min_std = np.min(self.avg_fx[self.avg_fx>0]), np.min(self.std_fx[self.std_fx>0])
        self.avg_fx[self.avg_fx<=0], self.std_fx[self.std_fx<=0] = min_avg, min_std

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
        self.speed_avg = 10000*np.average(avg_slopes)
        self.speed_std = 10000*np.average(std_slopes)
        print(f'Speed = {self.speed_avg}+/-{self.speed_std}')
        print('Done!\n')

    def compute_init_rate(self):
        ms = self.avg_fx[self.oris]
        ss = self.std_fx[self.oris]

        self.initiation_rate = np.zeros((self.L,self.T))
        print('Computing initiation rate...')
        mus = self.T*(1-ms)
        stds = self.T*ss
        for i, ori in enumerate(self.oris):
            s = np.round(np.random.normal(mus[i], stds[i], 20000)).astype(int)
            s[s<0] = 0
            s[s>=self.T] = self.T-1
            unique_locations, counts = np.unique(s, return_counts=True)
            self.initiation_rate[ori,unique_locations] = counts
            self.initiation_rate[ori,:] /= np.sum(self.initiation_rate[ori,:])
        print('Computation Done! <3\n')

    def prepare_data(self):
        self.process_matrix()
        self.proces_oris()
        self.compute_f()
        self.compute_peaks()
        self.compute_slopes()
        self.compute_init_rate()
    
    def numerical_simulator(self,scale,viz=False):
        # Initialize arrays
        vs = np.zeros(self.L)  # Fork propagation speed
        f = np.zeros((self.L, self.T))  # Replication fraction
        rep_fract = list()
        replicated_dna = np.zeros((self.L, self.T), dtype=int)  # Forks position
        r_forks, l_forks = np.zeros((self.L, self.T), dtype=int), np.zeros((self.L, self.T), dtype=int)
        init_t, init_x, coal_t, coal_x = list(), list(), list(), list()

        # Monte Carlo simulation
        print('Running replikator....')
        dna_is_replicated = False
        t, T = 1, self.T
        while not dna_is_replicated:
            # Fork initiation
            initiate_forks = np.random.rand(self.L) < scale*self.initiation_rate[:,t]
            init_locs = np.nonzero(initiate_forks)[0]
            for init in init_locs:
                if replicated_dna[init, t-1]==0:
                    vel = np.random.normal(self.speed_avg, self.speed_std, 1)[0]
                    vs[init] = vel if vel>=1 else 1
            replicated_dna[initiate_forks, t] = 1
            
            previously_initiated = replicated_dna[:, t-1] == 1
            xs = np.nonzero(initiate_forks*(~previously_initiated))[0]
            for x in xs:
                init_x.append(x)
                init_t.append(t)
            
            # Fork propagation
            for i in range(self.L):
                if replicated_dna[i, t - 1] == 1:
                    # Move the fork according to the propagation speed
                    v = vs[i]
                    distance = int(round(np.random.uniform(0,v+1,1)[0]))
                    if replicated_dna[(i + distance + 1) % self.L, t-1]==1 and replicated_dna[(i+1)%self.L, t - 1] == 0:
                        coal_t.append(t)
                        coal_x.append((i + distance) % self.L)
                    if replicated_dna[(i - distance -1) % self.L, t-1]==1 and replicated_dna[(i-1)%self.L, t - 1] == 0:
                        coal_t.append(t)
                        coal_x.append((i - distance) % self.L)
                    
                    if (i-distance)%self.L<(i+distance)%self.L: # they have not met the boundaries
                        replicated_dna[(i-distance)%self.L:(i+distance)%self.L,t] = 1
                        vs[(i-distance)%self.L:(i+distance)%self.L] = v
                    else: # they meet boundaries
                        if (i+distance)>self.L:
                            replicated_dna[i:self.L,t], replicated_dna[0:(i+distance)%self.L,t] = 1, 1
                            vs[i:self.L], vs[0:(i+distance)%self.L] = v, v
                        if (i-distance)<0:
                            replicated_dna[0:i,t], replicated_dna[(i-distance)%self.L:self.L,t] = 1, 1
                            vs[0:i], vs[(i-distance)%self.L:self.L] = v, v
                    replicated_dna[(i-distance)%self.L, t], replicated_dna[(i+distance)%self.L, t] = 1, 1
                    r_forks[(i + distance) % self.L,t] = 1 if replicated_dna[(i+distance)%self.L, t - 1] == 0 else 0
                    l_forks[(i - distance) % self.L,t] = 1 if replicated_dna[(i-distance)%self.L, t - 1] == 0 else 0

            # Check if termination condition is satisfied
            rep_fract.append(np.count_nonzero(replicated_dna[:,t-1])/self.L)
            if np.all(replicated_dna[:,t-1]==1): 
                dna_is_replicated = True
                T = t

            # Calculate replication fraction
            f[:, t] = replicated_dna[:, t]
            t+=1
        print('Done! ;)')

        replicated_dna, f = replicated_dna[:,:T], f[:,:T]
        r_forks, l_forks = r_forks[:,:T], l_forks[:,:T]

        if T<self.T:
            replicated_dna, f = expand_columns(replicated_dna,self.T), expand_columns(f,self.T)
            r_forks, l_forks = expand_columns(r_forks,self.T), expand_columns(l_forks,self.T)
            zero_columns = np.all(f == 0, axis=0) & (np.arange(self.T)>self.T/2)
            f[:, zero_columns] = 1

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

        return f, l_forks, r_forks

    def run(self,scale=100,viz=True):
        self.prepare_data()
        f, l_forks, r_forks = self.numerical_simulator(scale,viz=viz)
        return f, l_forks, r_forks

def run_loop(N_trials):
    N_beads,rep_duration = 10000,5000
    sf = np.zeros((N_beads,rep_duration))
    rept_path = '/mnt/raid/data/replication/single_cell/Chr9_replication_state_filtered.mat'
    ori_path = '/mnt/raid/data/replication/LCL_MCM_replication_origins.bed'
    rep = Replikator(rept_path,ori_path,N_beads,rep_duration)
    rep.prepare_data()
    for i in range(100):
        f, l_forks, r_forks = rep.numerical_simulator(100,viz=False)
        sf += f
    sf /= 10
    
    # Replication fraction
    plt.figure(figsize=(12.6, 6))
    plt.imshow(1-sf.T, cmap='bwr', aspect='auto', origin='lower')
    plt.colorbar(label='Replication Fraction')
    plt.title('DNA Replication Simulation')
    plt.xlabel('DNA position',fontsize=16)
    plt.ylabel('Computational Time',fontsize=16)
    plt.show()

    # Replication fraction
    plt.figure(figsize=(15, 6))
    plt.plot(np.average(sf,axis=1))
    plt.xlabel('DNA position',fontsize=16)
    plt.ylabel('Average Replication Fraction',fontsize=16)
    plt.show()

def main():
    N_beads,rep_duration = 10000,5000
    rept_path = '/mnt/raid/data/replication/single_cell/Chr9_replication_state_filtered.mat'
    ori_path = '/mnt/raid/data/replication/LCL_MCM_replication_origins.bed'
    rep = Replikator(rept_path,ori_path,N_beads,rep_duration)
    l_forks, r_forks = rep.run()
