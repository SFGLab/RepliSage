"""Order parameters of the Potts states."""
import pandas as pd
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from ..logger import log

def magnetization(S, q=5, viz=False, out_path=None):
    """
    Computes the magnetization over time.
    Formula: M = (q * max(N_s) - N) / (N * (q - 1))
    where N_s is the count of spins in the most common state.
    ---------------------------------------------------------
    Description:
    For the Potts model, the magnetization measures how much 
    the system prefers one particular state over metadatas. 
    This order parameter ranges from:
    
    M=0: Symmetric (disordered) phase (e.g., in the high-temperature regime),
    M=1: Fully ordered phase where all spins align in one state (e.g., in the low-temperature regime).
    """
    N, T = S.shape
    M = np.zeros(T)
    for t in range(T):
        state_counts = np.bincount(S[:, t], minlength=q)
        max_state = np.max(state_counts)
        M[t] = (q * max_state - N) / (N * (q - 1))
    np.save(out_path+'/metadata/graph_metrics/Mag_potts.npy',M)

    return M

def cluster_order(S, viz=False, out_path=None):
    """
    Computes the cluster-based order parameter over time.
    Formula: C = <S_max> / N, where S_max is the size of the largest cluster.
    -------------------------------------------------------------------------
    Description:
    In the disordered phase, clusters are small, and C is close to 0. In the ordered phase,
    the largest cluster spans the system, and C approaches 1.

    Note:
    Here we refer as clusters how they form linearly without taking ito account the interaction matrix.
    """
    N, T = S.shape
    C = np.zeros(T)
    for t in range(T):
        state_counts = np.bincount(S[:, t])
        largest_cluster = np.max(state_counts)
        C[t] = largest_cluster / N
    np.save(out_path+'/metadata/graph_metrics/cluster_order.npy',C)

    return C

def binder_cumulant(S, q=5, viz=False, out_path=None):
    """
    Computes the Binder cumulant over time.
    Formula: U = 1 - <M^4> / (3 * <M^2>^2)
    """
    N, T = S.shape
    U = np.zeros(T)
    for t in range(T):
        state_counts = np.bincount(S[:, t], minlength=q)
        probs = state_counts / N
        m2 = np.sum(probs**2)
        m4 = np.sum(probs**4)
        U[t] = 1 - m4 / (3 * m2**2)
    np.save(out_path+'/metadata/graph_metrics/binder_cumulant.npy',U)
    
    return U

def entropy_order(S, q=5, viz=False,out_path=None):
    """
    Computes the entropy over time.
    Formula: S = -sum(P_s * log(P_s)) where P_s is the probability of state s.
    -------------------------------------------------------------------------
    In the ordered phase:
    
    Ps​≈1 for one dominant state, and S→0S→0.

    In the disordered phase:

    Ps​≈1/q, and S→ln⁡(q)S→ln(q).
    """
    N, T = S.shape
    S_entropy = np.zeros(T)
    for t in range(T):
        state_counts = np.bincount(S[:, t], minlength=q)
        probs = state_counts / N
        S_entropy[t] = entropy(probs, base=np.e)
    np.save(out_path+'/metadata/graph_metrics/entropy.npy',S_entropy)

    return S_entropy

def overlap_order(S1, S2, out_path=None):
    """
    Computes the overlap between two configurations S1 and S2 over time.
    Formula: Q = (1/N) * sum(delta(s_i^1, s_i^2))
    """
    N, T = S1.shape
    Q = np.zeros(T)
    for t in range(T):
        Q[t] = np.mean(S1[:, t] == S2[:, t])
    np.save(out_path+'/metadata/graph_metrics/configuration_overlap.npy',Q)
    return Q
