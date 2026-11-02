"""Stochastic replication simulation: origin firing + fork propagation."""
import numpy as np
from .fork_propagation import propagate_forks
from ..utils.arrays import rescale_matrix
from ..logger import log

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

    activated_origins = set()

    # Identify potential origins: any position with nonzero initiation rate at any time
    potential_origins = set(np.where(np.any(initiation_rate > 0, axis=1))[0])

    bar = log.progress(total=100, desc="Replication", unit="%", bar_format="{l_bar}{bar}| {n:.0f}/{total}% [{elapsed}<{remaining}]")

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
                l_forks = np.vstack((np.asarray(l_forks).reshape(-1, 6), [init, v, t, init, init, 0]))  # reshape: safe when no forks are active # start, v, t0, pos, new_pos, to_del
                r_forks = np.vstack((l_forks, [init, v, t, init, init, 0]))
                activated_origins.add(init)
        rep_dna_col[initiate_forks] = 1

        # Update replicated_dna and forks matrices:
        replicated_dna = np.hstack((replicated_dna, rep_dna_col.reshape((L, 1))))
        l_forks_mat = np.hstack((l_forks_mat, l_forks_mat_col.reshape((L, 1))))
        r_forks_mat = np.hstack((r_forks_mat, r_forks_mat_col.reshape((L, 1))))
        
        done = 100 * replicated_dna[:, -1].mean(); bar.update(max(0, done - bar.n))      # progress = replicated fraction

        # check whether already replicated:
        if np.all(replicated_dna[:, -1] == 1):
            dna_is_replicated = True
            T_final = t
        t += 1
        
        if t>=1_000_000:
            bar.close(); raise(Exception(f"The replication simulation failed to finish before time T={1000000}. \
                \nPlease increase the initiation rate or velocities parameters and rerun the simulation."))
    
    bar.update(100 - bar.n); bar.close()
    replicated_dna = rescale_matrix(replicated_dna, T_orig, fork_mat=False)
    l_forks_mat = rescale_matrix(l_forks_mat, T_orig, fork_mat=True)
    r_forks_mat = rescale_matrix(r_forks_mat, T_orig, fork_mat=True)
    rep_fract = [np.mean(replicated_dna[:, i]) for i in range(replicated_dna.shape[1])]

    num_activated = len(activated_origins)
    num_potential = len(potential_origins)
    if num_potential > 0:
        proportion = num_activated / num_potential # 2 because forks propagate in both directions
        log.info(f"{proportion:.2%} of potential origins were fired ({num_activated} out of {num_potential})")
    else:
        log.error("No potential origins found (initiation_rate is zero everywhere)")

    return replicated_dna, l_forks_mat, r_forks_mat, T_final, rep_fract
