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
    max_t = 20 * T_orig                     # safety cap on the simulated steps (the rate of the last column is reused after T_orig)
    t, T_final, extended = 1, 1, False
    dna_is_replicated = False

    # columns of the three matrices are collected in lists and stacked once (int8: values are 0/1)
    zero = np.zeros((L, 1), dtype=np.int8)
    rep_cols, lf_cols, rf_cols = [zero[:, 0]], [zero[:, 0]], [zero[:, 0]]
    last_rep = np.zeros(L, dtype=np.int64)
    l_forks, r_forks = np.ones((1, 6)), np.ones((1, 6))

    activated_origins = set()

    # Identify potential origins: any position with nonzero initiation rate at any time
    potential_origins = set(np.where(np.any(initiation_rate > 0, axis=1))[0])

    bar = log.progress(total=100, desc="Replication", unit="%", bar_format="{l_bar}{bar}| {n:.0f}/{total}% [{elapsed}<{remaining}]")

    # Fire randomly origins and propagate forks till dna will be fully replicated
    while not dna_is_replicated:
        # propagate the forks from the previous column:
        rep_dna_col, l_forks, r_forks, l_forks_mat_col, r_forks_mat_col = propagate_forks(t, last_rep, l_forks, r_forks)

        # fire new origins and forks (after the last time point the final rate is kept, so slow regions still finish):
        extended = extended or t >= T_orig
        initiate_forks = np.random.rand(L) < initiation_rate[:, min(t, T_orig - 1)]
        init_locs = np.nonzero(initiate_forks)[0]
        for init in init_locs:
            if last_rep[init] == 0:
                v = np.abs(np.random.normal(speed_mean, speed_std, 1)[0]/T_orig)
                l_forks = np.vstack((np.asarray(l_forks).reshape(-1, 6), [init, v, t, init, init, 0]))  # reshape: safe when no forks are active # start, v, t0, pos, new_pos, to_del
                r_forks = np.vstack((l_forks, [init, v, t, init, init, 0]))
                activated_origins.add(init)
        rep_dna_col[initiate_forks] = 1

        # Update replicated_dna and forks matrices:
        last_rep = rep_dna_col.reshape(L)
        rep_cols.append(last_rep.astype(np.int8)); lf_cols.append(l_forks_mat_col.reshape(L).astype(np.int8)); rf_cols.append(r_forks_mat_col.reshape(L).astype(np.int8))

        done = 100 * last_rep.mean(); bar.update(max(0, done - bar.n))      # progress = replicated fraction

        # check whether already replicated:
        if np.all(last_rep == 1):
            dna_is_replicated = True
            T_final = t
        t += 1

        if t >= max_t:
            bar.close()
            raise RuntimeError(f"The replication simulation did not finish within {max_t} steps ({100 * last_rep.mean():.0f}% replicated). "
                               "Increase REP_SPEED_SCALE or REP_INIT_RATE_SCALE, use fewer beads, or check that the timing data covers the region.")

    if extended: log.info(f"Replication needed {T_final} steps, more than REP_SIM_T = {T_orig}: the final initiation rate was kept for the late steps.")
    replicated_dna, l_forks_mat, r_forks_mat = (np.stack(c, axis=1) for c in (rep_cols, lf_cols, rf_cols))
    bar.update(100 - bar.n); bar.close()
    first = int(np.argmax(replicated_dna.any(axis=0)))                                  # S phase = first firing ... last locus replicated
    replicated_dna = replicated_dna[:, first:]
    l_forks_mat, r_forks_mat = (np.hstack((m[:, first + 1:], np.zeros((L, 1), dtype=m.dtype))) for m in (l_forks_mat, r_forks_mat))  # forks are recorded one step after firing: align them to it
    if first: log.info(f"Idle steps before the first origin firing removed: S phase starts with the first firing (step {first} of {T_final}).")
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

    return replicated_dna, l_forks_mat, r_forks_mat, T_final, rep_fract, (first, T_final)
