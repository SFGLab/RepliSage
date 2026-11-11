"""Container of all parameters needed by the MC stage."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class MCParams:
    N_beads: int
    L: np.ndarray                    # left CTCF binding potential
    R: np.ndarray                    # right CTCF binding potential
    N_lef: int
    N_lef2: int = 0
    N_steps: int = 200000
    N_sweep: int = 1000
    MC_step: int = 200
    burnin: int = 1000
    T: float = 1.5
    f: float = 1.0                   # folding coefficient (family 1)
    f2: float = 0.0                  # folding coefficient (family 2)
    b: float = 1.0                   # binding coefficient
    kappa: float = 1.0               # crossing coefficient
    c_rep: float = 0.0               # replication penalty coefficient
    c_potts1: float = 0.0            # Potts field coefficient
    c_potts2: float = 0.0            # Potts interaction coefficient
    rw: bool = True
    p_rew: float = 0.5
    rep_fork_organizers: bool = True
    cohesin_blocks_condensin: bool = False
    random_spins: bool = True
    repfork_push: bool = True
    t_rep: int = 0
    rep_duration: int = 1
    f_rep_T: Optional[np.ndarray] = None   # replication matrix, time-major (T, N_beads), int8
    h: Optional[np.ndarray] = None         # Potts field (from replication timing)
    L_site: Optional[np.ndarray] = None    # bool mask of left CTCF sites (raw sites widened by the smoothing); None = L > 0
    R_site: Optional[np.ndarray] = None    # same for right sites
    bind_w: Optional[np.ndarray] = None    # LEF loading weight per bead (preferential binding); None = uniform

    @property
    def has_rep(self):
        return self.f_rep_T is not None
