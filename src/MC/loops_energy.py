"""Loop-extrusion energy terms: crossing, binding, folding (+ their differences)."""
import numpy as np
from numba import njit

@njit(cache=True)
def Kappa(mi, ni, mj, nj):
    '''Crossing function of LoopSage.'''
    k = 0.0
    if mi >= 0 and ni >= 0 and mj >= 0 and nj >= 0:
        if mi < mj < ni < nj: k += 1
        if mj < mi < nj < ni: k += 1
        if mj == ni or mi == nj or ni == nj or mi == mj: k += 1
    return k

@njit(cache=True)
def E_bind(L, R, ms, ns, bind_norm):
    '''The binding energy.'''
    binding = np.sum(L[ms[ms >= 0]] + R[ns[ns >= 0]])
    return bind_norm * binding

@njit(cache=True)
def E_cross(ms, ns, N_lef, N_lef2, k_norm, cohesin_blocks_condensin=False):
    '''The crossing energy.'''
    crossing = 0.0
    for i in range(N_lef + N_lef2):
        for j in range(i + 1, N_lef + N_lef2):
            if cohesin_blocks_condensin or (i < N_lef and j < N_lef) or (i >= N_lef and j >= N_lef):
                crossing += Kappa(ms[i], ns[i], ms[j], ns[j])
    return k_norm * crossing

@njit(cache=True)
def E_fold(ms, ns, fold_norm):
    '''The folding energy.'''
    return fold_norm * np.sum(np.log(ns - ms + 1))

@njit(cache=True)
def get_dE_bind(L, R, bind_norm, ms, ns, m_new, n_new, idx):
    '''Energy difference for binding energy.'''
    B_new = L[m_new] + R[n_new] if m_new >= 0 and n_new >= 0 else 0
    B_old = L[ms[idx]] + R[ns[idx]] if ms[idx] >= 0 and ns[idx] >= 0 else 0
    return bind_norm * (B_new - B_old)

@njit(cache=True)
def get_dE_fold(fold_norm, m_old, n_old, m_new, n_new):
    '''Energy difference for folding energy (one LEF; m_old/n_old are its current ends).'''
    return fold_norm * (np.log(n_new - m_new + 1) - np.log(n_old - m_old + 1))

@njit(cache=True)
def get_dE_cross(ms, ns, m_new, n_new, idx, N_lef, N_lef2, k_norm, cohesin_blocks_condensin=False):
    '''Energy difference for crossing energy.'''
    K1, K2 = 0.0, 0.0
    for i in range(N_lef + N_lef2):
        if i != idx:
            if cohesin_blocks_condensin or (idx < N_lef and i < N_lef) or (idx >= N_lef and i >= N_lef):
                K1 += Kappa(ms[idx], ns[idx], ms[i], ns[i])
                K2 += Kappa(m_new, n_new, ms[i], ns[i])
    return k_norm * (K2 - K1)
