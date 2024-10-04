import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from common import *

class ReplicationSimulator:
    def __init__(self,L:int, T:int, initiation_rate:np.ndarray, speed_ratio:float, speed_mean=0.05):
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
        t, T_final = 1, self.T
        dna_is_replicated = False

        # Initialize typed lists
        self.rep_fract = list()

        # Initialize arrays
        self.vs, self.t0s = np.zeros(self.L, dtype=np.float64), np.zeros(self.L, dtype=np.float64) # Fork propagation speed
        self.f = np.zeros((self.L, self.T), dtype=np.float64)  # Replication fraction
        self.replicated_dna = np.zeros((self.L, self.T), dtype=np.int64)  # Forks position
        self.r_forks = np.zeros((self.L, self.T), dtype=np.int64)
        self.l_forks = np.zeros((self.L, self.T), dtype=np.int64)

        while not dna_is_replicated:
            initiate_forks = np.random.rand(self.L) < self.initiation_rate[:, t]
            init_locs = np.nonzero(initiate_forks)[0]
            for init in init_locs:
                if self.replicated_dna[init, t-1] == 0:
                    vel = np.random.normal(2, self.speed_mean * self.speed_ratio, 1)[0]
                    self.vs[init] = max(vel, 1)
                    self.t0s[init] = t
            self.replicated_dna[initiate_forks, t] = 1
            self.propagate_forks(t)
            self.rep_fract.append(np.count_nonzero(self.replicated_dna[:, t-1]) / self.L)
            if np.all(self.replicated_dna[:, t-1] == 1):
                dna_is_replicated = True
                T_final = t
            
            self.f[:, t] = self.replicated_dna[:, t]
            t += 1

        self.f, self.l_forks, self.r_forks = self.f[:, :T_final], self.l_forks[:, :T_final], self.r_forks[:, :T_final]

        if T_final < self.T:
            self.f = expand_columns(self.f, self.T)
            self.r_forks = expand_columns(self.r_forks, self.T)
            self.l_forks = expand_columns(self.l_forks, self.T)
            zero_columns = np.all(self.f == 0, axis=0) & (np.arange(self.T) > self.T / 2)
            self.f[:, zero_columns] = 1
        return self.f, self.l_forks, self.r_forks, T_final, self.rep_fract
    
    def propagate_forks(self,t:int):
        '''
        Propagation of replication forks.
        ---------------------------------
        Input:
        t: it is the given time point of the simulation.
        '''
        for i in range(self.L):
            if self.replicated_dna[i, t - 1] == 1:
                v, t0 = self.vs[i], self.t0s[i]
                distance = int(round(v*(t-t0)))#int(round(np.random.uniform(0, v + 1, 1)[0]))

                if (i - distance) % self.L < (i + distance) % self.L:
                    self.replicated_dna[(i - distance) % self.L:(i + distance) % self.L, t] = 1
                    self.vs[(i - distance) % self.L:(i + distance) % self.L] = v
                    self.t0s[(i - distance) % self.L:(i + distance) % self.L] = t0
                else:
                    if (i + distance) > self.L:
                        self.replicated_dna[i:self.L, t], self.replicated_dna[0:(i + distance) % self.L, t] = 1, 1
                        self.vs[i:self.L], self.vs[0:(i + distance) % self.L] = v, v
                        self.t0s[i:self.L], self.t0s[0:(i + distance) % self.L] = t0, t0
                    if (i - distance) < 0:
                        self.replicated_dna[0:i, t], self.replicated_dna[(i - distance) % self.L:self.L, t] = 1, 1
                        self.vs[0:i], self.vs[(i - distance) % self.L:self.L] = v, v
                        self.t0s[0:i], self.t0s[(i - distance) % self.L:self.L] = t0, t0
                self.replicated_dna[(i - distance) % self.L, t], self.replicated_dna[(i + distance) % self.L, t] = 1, 1
                self.r_forks[(i + distance) % self.L, t] = 1 if self.replicated_dna[(i + distance) % self.L, t - 1] == 0 else 0
                self.l_forks[(i - distance) % self.L, t] = 1 if self.replicated_dna[(i - distance) % self.L, t - 1] == 0 else 0

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