import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from common import *
from numba import njit, prange

@njit
def simulator(L,T,initiation_rate,speed_ratio,speed_mean):
    '''
    Run replication simulation.
    '''
    t, T_final = 1, T
    dna_is_replicated = False

    # Initialize typed lists
    rep_fract = list()

    # Initialize arrays
    vs, t0s = np.zeros(L, dtype=np.float64), np.zeros(L, dtype=np.float64) # Fork propagation speed
    f = np.zeros((L, T), dtype=np.float64)  # Replication fraction
    replicated_dna = np.zeros((L, T), dtype=np.int64)  # Forks position
    r_forks = np.zeros((L, T), dtype=np.int64)
    l_forks = np.zeros((L, T), dtype=np.int64)

    # Fire randomly origins and propagate forks till dna will be fully replicated
    while not dna_is_replicated:
        initiate_forks = np.random.rand(L) < initiation_rate[:, t] # fire origins
        init_locs = np.nonzero(initiate_forks)[0]
        for init in init_locs: 
            if replicated_dna[init, t-1] == 0:
                vs[init] = np.random.normal(speed_mean, speed_mean * speed_ratio, 1)[0]
                t0s[init] = t
        replicated_dna[initiate_forks, t] = 1
        replicated_dna, vs, t0s, l_forks, r_forks = propagate_forks(L,t,replicated_dna, vs, t0s, l_forks, r_forks)
        rep_fract.append(np.count_nonzero(replicated_dna[:, t-1]) / L)
        if np.all(replicated_dna[:, t-1] == 1):
            dna_is_replicated = True
            T_final = t
        
        f[:, t] = replicated_dna[:, t]
        t += 1

    f, l_forks, r_forks = f[:, :T_final], l_forks[:, :T_final], r_forks[:, :T_final]

    return f, l_forks, r_forks, T_final, rep_fract

@njit
def propagate_forks(L,t,replicated_dna, vs, t0s, l_forks, r_forks):
    '''
    Propagation of replication forks.
    ------------------------------------------------
    Input:
    t: it is the given time point of the simulation.
    '''
    for i in range(L):
        if replicated_dna[i, t - 1] == 1:
            v, t0 = vs[i], t0s[i]
            distance = np.abs(int(round(v*(t-t0))))

            if (i - distance) % L < (i + distance) % L:
                replicated_dna[(i - distance) % L:(i + distance) % L, t] = 1
                vs[(i - distance) % L:(i + distance) % L] = v
                t0s[(i - distance) % L:(i + distance) % L] = t0
            else:
                if (i + distance) > L:
                    replicated_dna[i:L, t], replicated_dna[0:(i + distance) % L, t] = 1, 1
                    vs[i:L], vs[0:(i + distance) % L] = v, v
                    t0s[i:L], t0s[0:(i + distance) % L] = t0, t0
                if (i - distance) < 0:
                    replicated_dna[0:i, t], replicated_dna[(i - distance) % L:L, t] = 1, 1
                    vs[0:i], vs[(i - distance) % L:L] = v, v
                    t0s[0:i], t0s[(i - distance) % L:L] = t0, t0
            replicated_dna[(i - distance) % L, t], replicated_dna[(i + distance) % L, t] = 1, 1
            r_forks[(i + distance) % L, t] = 1 if replicated_dna[(i + distance) % L, t - 1] == 0 else 0
            l_forks[(i - distance) % L, t] = 1 if replicated_dna[(i - distance) % L, t - 1] == 0 else 0
    return replicated_dna, vs, t0s, l_forks, r_forks

class ReplicationSimulator:
    def __init__(self,L:int, T:int, initiation_rate:np.ndarray, speed_ratio:float, speed_mean=0.1):
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

    def run_simulator(self):
        '''
        Run replication simulation.
        '''
        self.f, self.l_forks, self.r_forks, T_final, self.rep_fract = simulator(self.L,self.T,self.initiation_rate,self.speed_ratio,self.speed_mean)

        if T_final < self.T:
            self.f = expand_columns(self.f, self.T)
            self.r_forks = expand_columns(self.r_forks, self.T)
            self.l_forks = expand_columns(self.l_forks, self.T)
            zero_columns = np.all(self.f == 0, axis=0) & (np.arange(self.T) > self.T / 2)
            self.f[:, zero_columns] = 1
        return self.f, self.l_forks, self.r_forks, T_final, self.rep_fract

    def visualize_simulation(self):
        '''
        Vizualize the results.
        '''
        plt.figure(figsize=(12.6, 6))
        plt.imshow(self.f.T, cmap='bwr', aspect='auto', origin='lower')
        plt.colorbar(label='Replication Fraction')
        plt.title('DNA Replication Simulation')
        plt.xlabel('DNA position')
        plt.ylabel('Computational Time')
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(self.rep_fract)
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Replication Fraction', fontsize=18)
        plt.show()

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