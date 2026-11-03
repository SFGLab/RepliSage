"""Validation of Replikator: average many independent replication simulations."""
import numpy as np
from scipy import stats
from .simulator_class import ReplicationSimulator
from ..utils.arrays import min_max_normalize
from ..logger import log

def run_Ntrials(N_trials, L, T, initiation_rate, speed_ratio, speed_mean=3):
    '''Run N_trials independent replication simulations and average them.'''
    sf = np.zeros((L, T), dtype=np.float64)
    for _ in log.progress(range(N_trials), desc='Replication trials'):
        repsim = ReplicationSimulator(L, T, initiation_rate, speed_ratio, speed_mean)
        f, *_ = repsim.run_simulator()
        sf += f
    return sf / N_trials

def correlate_with_experiment(sf, avg_fx):
    """Pearson / Spearman / Kendall correlation of simulated vs experimental timing."""
    sim = np.average(sf, axis=1)
    rows = [(n, f"{fn(sim, avg_fx)[0]:.3f}") for n, fn in
            [("Pearson", stats.pearsonr), ("Spearman", stats.spearmanr), ("Kendall tau", stats.kendalltau)]]
    log.table("Simulated vs experimental timing", rows, ("Correlation", "Value"))
    return rows
