import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from common import *

def numerical_simulator(L, T, initiation_rate, speed_ratio, speed_mean=3, viz=False):
    # Run the simulation
    f, l_forks, r_forks, T_final, rep_fract = run_simulation(L, T, initiation_rate, speed_ratio, speed_mean)

    if T_final < T:
        f = expand_columns(f, T)
        r_forks = expand_columns(r_forks, T)
        l_forks = expand_columns(l_forks, T)
        zero_columns = np.all(f == 0, axis=0) & (np.arange(T) > T / 2)
        f[:, zero_columns] = 1

    if viz:
        visualize_simulation(f, rep_fract)

    return np.array(f, dtype=np.float64), np.array(l_forks, dtype=np.int64), np.array(r_forks, dtype=np.int64)

def run_simulation(L, T, initiation_rate, speed_ratio, speed_mean):
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

    while not dna_is_replicated:
        initiate_forks = np.random.rand(L) < initiation_rate[:, t]
        init_locs = np.nonzero(initiate_forks)[0]
        for init in init_locs:
            if replicated_dna[init, t-1] == 0:
                vel = np.random.normal(2, speed_mean * speed_ratio, 1)[0]
                vs[init] = max(vel, 1)
                t0s[init] = t
        replicated_dna[initiate_forks, t] = 1
        
        vs, t0s, r_forks, l_forks = propagate_forks(L, t, vs, t0s, replicated_dna, r_forks, l_forks)
        
        rep_fract.append(np.count_nonzero(replicated_dna[:, t-1]) / L)
        if np.all(replicated_dna[:, t-1] == 1):
            dna_is_replicated = True
            T_final = t
        
        f[:, t] = replicated_dna[:, t]
        t += 1

    return f[:, :T_final], l_forks[:, :T_final], r_forks[:, :T_final], T_final, rep_fract

def propagate_forks(L, t, vs, t0s, replicated_dna, r_forks, l_forks):
    for i in range(L):
        if replicated_dna[i, t - 1] == 1:
            v, t0 = vs[i], t0s[i]
            distance = int(round(v*(t-t0)))#int(round(np.random.uniform(0, v + 1, 1)[0]))

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
    return vs, t0s, r_forks, l_forks

def run_Ntrials(N_trials, L, T, initiation_rate, speed_ratio, speed_mean=2):
    sf = np.zeros((L,T), dtype=np.float64)
    for i in tqdm(range(N_trials)):
        # Run the simulation
        f, l_forks, r_forks, T_final, rep_fract = run_simulation(L, T, initiation_rate, speed_ratio, speed_mean)
        if T_final < T:
            f = expand_columns(f, T)
            r_forks = expand_columns(r_forks, T)
            l_forks = expand_columns(l_forks, T)
            zero_columns = np.all(f == 0, axis=0) & (np.arange(T,dtype=np.int64) > T / 2)
            f[:, zero_columns] = 1
        sf += f
    sf /= N_trials
    return sf

def visualize_simulation(f, rep_fract):
    plt.figure(figsize=(12.6, 6))
    plt.imshow(f.T, cmap='bwr', aspect='auto', origin='lower')
    plt.colorbar(label='Replication Fraction')
    plt.title('DNA Replication Simulation')
    plt.xlabel('DNA position')
    plt.ylabel('Computational Time')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(rep_fract)
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Replication Fraction', fontsize=18)
    plt.show()