import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

def run_replikator(L,time_steps,initiation_rate,mu_v,std_v,viz=False):
    # Initialize arrays
    vs = np.zeros(L)  # Fork propagation speed
    f = np.zeros((L, time_steps))  # Replication fraction
    replicated_dna = np.zeros((L, time_steps), dtype=int)  # Forks position
    r_forks, l_forks = np.zeros((L, time_steps), dtype=int), np.zeros((L, time_steps), dtype=int)
    init_t, init_x, coal_t, coal_x = list(), list(), list(), list()

    # Monte Carlo simulation
    print('Running replikator....')
    for t in tqdm(range(1, time_steps)):
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

        # if np.count_nonzero(replicated_dna[:,t])==L: break
        
        # Calculate replication fraction
        f[:, t] = np.sum(replicated_dna[:, :t + 1], axis=1) / (t + 1)
    print('Done! ;)')

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

def main():
    run_replikator(L=1000,time_steps=int(1e3),initiation_rate=0.001,mu_v=3,std_v=2,viz=True)