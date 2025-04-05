import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from .common import *
from numba import njit, prange


def simulator(initiation_rate, speed_mean, speed_std):
    '''
    Run replication simulation.
    '''
    L, T_orig = initiation_rate.shape
    t, T_final = 1, 1
    dna_is_replicated = False

    # Initialize arrays
    replicated_dna = rep_dna_col = np.zeros((L, 1), dtype=np.int64)
    l_forks_mat = l_forks_mat_col = np.zeros((L, 1), dtype=np.int64)
    r_forks_mat = r_forks_mat_col = np.zeros((L, 1), dtype=np.int64)
    l_forks, r_forks = np.ones((1, 6)), np.ones((1,6))

    # Fire randomly origins and propagate forks till dna will be fully replicated
    while not dna_is_replicated:
        # propagate the forks from the previous column:
        rep_dna_col, l_forks, r_forks, l_forks_mat_col, r_forks_mat_col = propagate_forks(t, replicated_dna[:,-1], l_forks, r_forks)

        # fire new origins and forks:
        initiate_forks = np.random.rand(L) < initiation_rate[:, t] 
        init_locs = np.nonzero(initiate_forks)[0]
        for init in init_locs: 
            if replicated_dna[init, -1] == 0:
                v = np.abs(np.random.normal(speed_mean, speed_std, 1)[0]/T_orig)
                l_forks = np.vstack((l_forks, [init, v, t, init, init, 0])) # start, v, t0, pos, new_pos, to_del
                r_forks = np.vstack((l_forks, [init, v, t, init, init, 0]))
        rep_dna_col[initiate_forks] = 1

        # Update replicated_dna and forks matrices:
        replicated_dna = np.hstack((replicated_dna, rep_dna_col.reshape((L, 1))))
        l_forks_mat = np.hstack((l_forks_mat, l_forks_mat_col.reshape((L, 1))))
        r_forks_mat = np.hstack((r_forks_mat, r_forks_mat_col.reshape((L, 1))))
        
        # check whether already replicated:
        if np.all(replicated_dna[:, -1] == 1):
            dna_is_replicated = True
            T_final = t
        t += 1
        
        if t>=1_000_000:
            raise(Exception(f"The replication simulation failed to finish before time T={1000000}. \
                \nPlease increase the initiation rate or velocities parameters and rerun the simulation."))

    replicated_dna = rescale_matrix(replicated_dna, T_orig, fork_mat=False)
    l_forks_mat = rescale_matrix(l_forks_mat, T_orig, fork_mat=True)
    r_forks_mat = rescale_matrix(r_forks_mat, T_orig, fork_mat=True)
    rep_fract = [np.mean(replicated_dna[:, i]) for i in range(replicated_dna.shape[1])]

    return replicated_dna, l_forks_mat, r_forks_mat, T_final, rep_fract


def propagate_forks(t, rep_dna_col, l_forks, r_forks):
    '''
    Propagation of replication forks.
    ------------------------------------------------
    Input:
    t: it is the given time point of the simulation.
    rep_dna_col: is the previous column of replicated dna
    l_forks and r_forks: are the objects representing active (moving) left and right forks respectively.
    Returns new columns of replicated dna, updated l_forks and r_forks and new columns for the fork matrices.
    '''
    L = len(rep_dna_col)
    rep_dna_col_new = rep_dna_col.copy()
    l_forks_mat_col = np.zeros((L, 1), dtype=np.int64)
    r_forks_mat_col = np.zeros((L, 1), dtype=np.int64)
    
    if any([f[5] == 0 for f in l_forks]) > 0 or any([f[5] == 0 for f in r_forks]):

        for lf in l_forks:
            if lf[5] != 1:
                distance = np.abs(int(round(lf[1]*(t-lf[2]))))
                lf[4] = lf[0]-distance
        for rf in r_forks:
            if rf[5] != 1:
                distance = np.abs(int(round(rf[1]*(t-rf[2]))))
                rf[4] = rf[0]+distance

        # if forks meet each other (merge):
        for lf in l_forks:
            for rf in r_forks:
                if lf[3]>rf[3] and lf[4]<=rf[4]:
                    rep_dna_col_new[int(rf[3]+1):int(lf[3])] = 1
                    l_forks_mat_col[int(rf[3]+1):int(lf[3])] = 1 
                    r_forks_mat_col[int(rf[3]+1):int(lf[3])] = 1 
                    lf[5] = 1
                    rf[5] = 1

        # deleting merged forks:
        l_forks = [lf for lf in l_forks if lf[5] != 1]
        r_forks = [rf for rf in r_forks if lf[5] != 1]
        
        # if forks did not meet any opposite fork:
        for lf in l_forks:
            rep_dna_col_new[max(0, int(lf[4])):int(lf[3])] = 1
            l_forks_mat_col[max(0, int(lf[4]))] = 1
            if lf[4] <= 0:
                lf[5] = 1

        for rf in r_forks:
            rep_dna_col_new[int(rf[3]+1):int(rf[4]+1)] = 1
            r_forks_mat_col[min(L-1, int(rf[4]))] = 1
            if rf[4] >= L-1:
                rf[5] = 1

        # deleting out-of-range forks:
        l_forks = [lf for lf in l_forks if lf[5] != 1]
        r_forks = [rf for rf in r_forks if lf[5] != 1]

        # updating forks positions:
        for lf in l_forks:
            lf[3] = lf[4]
        for rf in r_forks:
            rf[3] = rf[4]
    
    return rep_dna_col_new, l_forks, r_forks, l_forks_mat_col, r_forks_mat_col

class ReplicationSimulator:
    def __init__(self,L:int, T:int, initiation_rate:np.ndarray, speed_ratio:float, speed_mean:float):
        '''
        Set parameters for replication simulation.
        ------------------------------------------
        Input:
        L: simlation length
        T: simulation time steps
        initiation_rate: the initiation rate function that gives the probability that a specific origin 
                         would fire replication in space-time point (x,t).
        speed_ratio: the is the redio of standard deviation over the average of the slopes of replication curve.
        speed_mean: the average speed of replication fork, set by user, it should be in simulation units.
        '''
        self.L, self.T = L, T
        self.initiation_rate, self.speed_ratio = initiation_rate, speed_ratio
        self.speed_mean = speed_mean
        self.speed_std = speed_ratio*speed_mean

    def run_simulator(self):
        '''
        Run replication simulation.
        '''
        print("Running replication simulation...")
        self.f, self.l_forks, self.r_forks, T_final, self.rep_fract = simulator(self.initiation_rate, self.speed_mean, self.speed_std)
        print(f"Replication simulation finished in time {str(T_final)}.")

        return self.f, self.l_forks, self.r_forks, T_final, self.rep_fract

    def visualize_simulation(self,path=None):
        '''
        Vizualize the results.
        '''
        plt.figure(figsize=(12.6, 6))
        plt.imshow(self.f.T, cmap='bwr', aspect='auto', origin='lower')
        plt.colorbar(label='Replication Fraction')
        plt.title('DNA Replication Simulation')
        plt.xlabel('DNA position')
        plt.ylabel('Computational Time')
        if path!=None:
            plt.savefig(path+'/rep_simulation.png', dpi=200)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(self.rep_fract)
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Replication Fraction', fontsize=18)
        if path!=None:
            plt.savefig(path+'/rep_frac.png', dpi=200)
        plt.close()

def run_Ntrials(N_trials, L, T, initiation_rate, speed_ratio, speed_mean=3):
    '''
    A function that runs N_trials of the simulation.
    '''
    sf = np.zeros((L,T), dtype=np.float64)
    for i in tqdm(range(N_trials)):
        # Run the simulation
        repsim = ReplicationSimulator(L, T, initiation_rate, speed_ratio, speed_mean)
        f, l_forks, r_forks, T_final, rep_fract = repsim.run_simulator()
        sf += f
    sf /= N_trials
    return sf