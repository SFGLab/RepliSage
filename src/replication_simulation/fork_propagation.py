"""Propagation of replication forks (one time step)."""
import numpy as np

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
        r_forks = [rf for rf in r_forks if rf[5] != 1]

        # updating forks positions:
        for lf in l_forks:
            lf[3] = lf[4]
        for rf in r_forks:
            rf[3] = rf[4]
    
    return rep_dna_col_new, l_forks, r_forks, l_forks_mat_col, r_forks_mat_col
