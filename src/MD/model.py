"""MD_MODEL: 3D chromatin model driven by the MC trajectory of loops (and replication / compartments).

One chain is simulated if there is no replication data, two chains (original + sister) otherwise.
"""
import numpy as np
from .forces import ForcesMixin
from .dynamics import DynamicsMixin
from .pipeline import PipelineMixin

class MD_MODEL(ForcesMixin, DynamicsMixin, PipelineMixin):
    def __init__(self, M, N, N_beads, burnin, MC_step, out_path, platform, rep_frac_T=None, t_rep=None, Cs=None):
        '''
        M, N (np arrays): positions (m,n) of the two legs of every cohesin; rows = LEFs, columns = time.
        N_beads (int): number of beads of one chain.
        rep_frac_T: replication matrix, time-major (T_rep, N_beads); None = no replication (single chain).
        Cs: Potts states (N_beads, T) or None (no compartments).
        '''
        self.M, self.N, self.Cs = M, N, Cs
        self.has_comps = Cs is not None and np.ndim(Cs) > 1
        self.replicated_dna = rep_frac_T
        self.run_repli = rep_frac_T is not None          # replication logic only if data is provided
        self.t_rep = t_rep if self.run_repli else 0
        self.rep_duration = rep_frac_T.shape[0] if self.run_repli else 0
        self.N_coh, self.N_steps = M.shape
        self.N_beads, self.step, self.burnin = N_beads, MC_step, burnin // MC_step
        self.out_path, self.platform, self.device = out_path, platform, None   # device: GPU index (CUDA/OpenCL)
        self.rw_l = np.sqrt(self.N_beads) * 0.1
        self.n_chains = 2 if self.run_repli else 1
        self.chain_idx = [c for c in range(self.n_chains) for _ in range(N_beads)]

    def compute_rep(self, i):
        '''Percentage of replicated DNA at sample i.'''
        if not self.run_repli: return 0
        t = i * self.step
        if t < self.t_rep: return 0
        if t < self.t_rep + self.rep_duration:
            return np.count_nonzero(self.replicated_dna[t - self.t_rep]) / self.N_beads * 100
        return 100

    def add_label(self, i):
        '''Phase label and index of sample i: BR / R / AR (or LE without replication).'''
        if not self.run_repli: return 'LE', i - self.burnin + 1
        t = i * self.step
        if t < self.t_rep: return 'BR', i - self.burnin + 1
        if t < self.t_rep + self.rep_duration: return 'R', i - self.t_rep // self.step + 1
        return 'AR', i - (self.t_rep + self.rep_duration) // self.step + 1
