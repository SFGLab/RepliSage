"""Energy normalisation constants (same formulas as the published RepliSage)."""
import numpy as np

def energy_norms(p):
    """Return (fold_norm, fold_norm2, bind_norm, k_norm, rep_norm, potts_norm1, potts_norm2) for MCParams p."""
    c_rep, c_p1, c_p2 = (p.c_rep, p.c_potts1, p.c_potts2) if p.has_rep else (0.0, 0.0, 0.0)
    fold = -p.N_beads * p.f / (p.N_lef * np.log(p.N_beads / p.N_lef))
    fold2 = -p.N_beads * p.f2 / (p.N_lef * np.log(p.N_beads / p.N_lef))
    bind = -p.N_beads * p.b / (np.sum(p.L) + np.sum(p.R))
    return fold, fold2, bind, p.kappa * 1e8, c_rep * 1e8, -c_p1, c_p2
