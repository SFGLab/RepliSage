"""ReplicationSimulator class (wraps simulator())."""
import numpy as np
from .fork_simulator import simulator
from .rep_plots import visualize_simulation
from ..logger import log

class ReplicationSimulator:
    """Set parameters and run the replication simulation."""

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
        log.trace("Running replication simulation...")
        self.f, self.l_forks, self.r_forks, T_final, self.rep_fract, self.window = simulator(self.initiation_rate, self.speed_mean, self.speed_std)
        log.trace("Replication simulation finished.")

        return self.f, self.l_forks, self.r_forks, T_final, self.rep_fract

    def visualize_simulation(self, path=None):
        visualize_simulation(self.f, self.rep_fract, path)
