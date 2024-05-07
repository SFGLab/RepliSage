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

    if viz:
        # Replication fraction
        plt.figure(figsize=(12.6, 6))
        plt.imshow(f.T, cmap='bwr', aspect='auto', origin='lower')
        plt.colorbar(label='Replication Fraction')
        plt.title('DNA Replication Simulation')
        plt.xlabel('DNA position')
        plt.ylabel('Computational Time')
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(np.average(f,axis=0))
        plt.xlabel('Time',fontsize=18)
        plt.ylabel(r'$f(t)$',fontsize=18)
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(np.average(f,axis=1))
        plt.xlabel('Location',fontsize=18)
        plt.ylabel(r'$f(x)$',fontsize=18)
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(rep_fract)
        plt.xlabel('Time',fontsize=18)
        plt.ylabel(r'Replication Fraction',fontsize=18)
        plt.show()

        # # Fork Locations
        # plt.figure(figsize=(10, 6))
        # plt.imshow(r_forks.T, cmap='Reds', aspect='auto', origin='lower',vmax=0.1)
        # plt.title('Right Fork Locations',fontsize=28)
        # plt.xlabel('DNA position',fontsize=18)
        # plt.ylabel('Computational Time',fontsize=18)
        # plt.show()

        # plt.figure(figsize=(10, 6))
        # plt.imshow(l_forks.T, cmap='Blues', aspect='auto', origin='lower',vmax=0.1)
        # plt.title('Left Fork Locations',fontsize=28)
        # plt.xlabel('DNA position',fontsize=18)
        # plt.ylabel('Computational Time',fontsize=18)
        # plt.show()

        # # Initiations and coalesences
        # plt.figure(figsize=(10, 6))
        # plt.scatter(init_x, init_t,marker='d',color='green')
        # plt.title('Initiations',fontsize=28)
        # plt.xlabel('DNA position',fontsize=18)
        # plt.ylabel('Computational Time',fontsize=18)
        # plt.xlim((0,L))
        # plt.ylim((0,time_steps))
        # plt.show()

        # plt.figure(figsize=(10, 6))
        # plt.scatter(coal_x, coal_t,marker='d',color='purple')
        # plt.title('Coalescences',fontsize=28)
        # plt.xlabel('DNA position',fontsize=18)
        # plt.ylabel('Computational Time',fontsize=18)
        # plt.xlim((0,L))
        # plt.ylim((0,time_steps))
        # plt.show()
    return l_forks, r_forks

def min_max_normalize(matrix, Min, Max):
    # Calculate the minimum and maximum values of the matrix
    matrix = np.nan_to_num(matrix)
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    # Normalize the matrix using the min-max formula
    normalized_matrix = Min + (Max - Min) * ((matrix - min_val) / (max_val - min_val))

    return normalized_matrix

# '/home/skorsak/Documents/data/Replication/Timing/GM12878_hg38_smoothed.txt'
def analyze_repcurve(file,chrom,viz=False):
    # Read replication timing data
    rep_timimg_df = pd.read_csv(file,sep='\t')
    RT_chr6_df = rep_timimg_df[rep_timimg_df['Chr ']==chrom].reset_index(drop=True)
    pos, rep_t = RT_chr6_df[' Coordinate '].values, RT_chr6_df[' Replication Timing '].values
    rep_peaks, _ = find_peaks(rep_t)
    rep_deaps, _ = find_peaks(-rep_t)

    if viz:
        figure(figsize=(25, 4), dpi=200)
        plt.plot(rep_t, 'ko',markersize=0.5,label='Replication Timing Curve')
        plt.plot(rep_peaks, rep_t[rep_peaks], "bx",label='Initiations')
        plt.plot(rep_deaps, rep_t[rep_deaps], "rx",label='Coalesences')
        plt.xlabel('Genomic Distance',fontsize=16)
        plt.ylabel('Replication Timing',fontsize=16)
        plt.legend()
        plt.show()
    return rep_t, rep_peaks, rep_deaps

def sigmoid_f(x,x0):
    return 1/(1+np.exp(-x+5+x0))

def generate_replifrac(rep_t,time_steps,viz=False):
    f_x = min_max_normalize(np.array(rep_t), 0, 5)
    f = np.zeros((time_steps,len(f_x)))
    
    for i in range(len(f_x)):
        s = sigmoid_f(np.linspace(0,15,time_steps),f_x[i])
        f[:,i] = s

    if viz:
        plt.figure(figsize=(12.6, 6))
        plt.imshow(f, cmap='bwr', aspect='auto', origin='lower')
        plt.colorbar(label='Experimental Replication Fraction')
        plt.xlabel('DNA position')
        plt.ylabel('Computational Time')
        plt.show()
    
    return f

def main():
    run_replikator(L=1000,time_steps=int(1e3),initiation_rate=0.001,mu_v=3,std_v=2,viz=True)