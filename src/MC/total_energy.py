"""Total energy and energy differences of a move (sums of the individual terms)."""
from numba import njit
from .loops_energy import E_bind, E_cross, E_fold, get_dE_bind, get_dE_fold, get_dE_cross
from .replication_energy import E_rep, get_dE_rep
from .potts_energy import E_potts, get_dE_potts_link

@njit(cache=True)
def get_E(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns, t, f_rep_T, has_rep,
          spins, h, ht, potts_norm1=0.0, potts_norm2=0.0, rep_fork_organizers=True, cohesin_blocks_condensin=False):
    '''The total energy.'''
    energy = E_bind(L, R, ms, ns, bind_norm) + E_cross(ms, ns, N_lef, N_lef2, k_norm, cohesin_blocks_condensin) + E_fold(ms, ns, fold_norm)
    if fold_norm2 != 0:
        energy += E_fold(ms[N_lef:N_lef + N_lef2], ns[N_lef:N_lef + N_lef2], fold_norm2)
    if has_rep and rep_norm != 0.0:
        energy += E_rep(f_rep_T, ms, ns, t, rep_norm)
    if potts_norm1 != 0.0 or potts_norm2 != 0.0:
        energy += E_potts(spins, ms, ns, h, ht, potts_norm1, potts_norm2, t, rep_fork_organizers)
    return energy

@njit(cache=True)
def get_dE_rewiring(N_lef, N_lef2, L, R, bind_norm, fold_norm, fold_norm2, k_norm, rep_norm, ms, ns, m_new, n_new, idx, t,
                    f_rep_T, has_rep, spins, potts_norm2=0.0, cohesin_blocks_condensin=False):
    '''Total energy difference of moving LEF idx to (m_new, n_new).'''
    dE = 0.0
    if idx < N_lef:
        dE += get_dE_fold(fold_norm, ms[idx], ns[idx], m_new, n_new)
    else:
        dE += get_dE_fold(fold_norm2, ms[idx], ns[idx], m_new, n_new)
    dE += get_dE_bind(L, R, bind_norm, ms, ns, m_new, n_new, idx)
    dE += get_dE_cross(ms, ns, m_new, n_new, idx, N_lef, N_lef2, k_norm, cohesin_blocks_condensin)
    if has_rep and rep_norm > 0.0:
        dE += get_dE_rep(f_rep_T, rep_norm, ms, ns, m_new, n_new, t, idx)
    if potts_norm2 > 0.0:
        dE += get_dE_potts_link(spins, ms, ns, m_new, n_new, ms[idx], ns[idx], potts_norm2)
    return dE
