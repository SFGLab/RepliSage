"""Simple folding statistics of loop configurations."""
import numpy as np

def get_stats(ms,ns,N_beads):
    '''
    This is a function that computes maximum compaction score in every step of the simulation.
    '''
    # Computing Folding Metrics
    N_coh = len(ms)
    chromatin = np.zeros(N_beads)
    chromatin2 = np.zeros(N_beads)
    for nn in range(N_coh):
        m,n = int(ms[nn]),int(ns[nn])
        if m<=n:
            chromatin[m:n] = 1
            chromatin2[m:n]+=1
        else:
            chromatin[0:n] = 1
            chromatin[m:] = 1
            chromatin2[0:n]+=1
            chromatin2[m:]+=1
    f = np.mean(chromatin)
    F = np.mean(chromatin2)
    f_std = np.std(chromatin)
    FC = 1/(1-f+0.001)
    
    return f, f_std, F, FC
