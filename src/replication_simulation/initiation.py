"""Probability vector of origin firing (Gaussian-in-time initiation)."""
import numpy as np
from numba import njit

@njit
def gaussian(x, mu, sig):
    return np.exp(-(x - mu)*(x - mu) / (sig*sig) / 2)/np.sqrt(2*np.pi*sig)

@njit
def get_p_vector(T, mu, sig):
    """Creates a vector of probabilities for a given locus with desired initiation mu and standard deviation sig"""
    ps = [gaussian(0, mu, sig)]
    for i in range(1, T):
        ps.append(gaussian(i, mu, sig)/(1-ps[-1]))
    return ps
