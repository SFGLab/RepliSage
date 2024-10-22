import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from Replikator import Replikator
from numba import njit, prange

@njit
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@njit(parallel=True)
def compute_pseudolikelihood_gradient(S, J, h):
    N_exp, N_beads = S.shape
    dJ = np.zeros_like(J, dtype=np.float64)  # Gradient of J
    dh = np.zeros_like(h, dtype=np.float64)  # Gradient of h

    for n in prange(N_exp):
        state = np.ascontiguousarray(S[n])  # Ensure 'state' is contiguous
        for i in range(N_beads):
            # Make J[i, :] contiguous
            J_row = np.ascontiguousarray(J[i, :])  # Contiguous copy of the i-th row of J

            # Calculate local field (h_i + sum_j J_ij * S_j)
            local_field = h[i] + np.dot(J_row, state) - J[i, i] * state[i]  # Exclude self-interaction
            prob = sigmoid(2 * state[i] * local_field)  # P(S_i | S_{\not i})

            # Gradients for h and J
            dh[i] += state[i] * (1 - prob)
            for j in range(N_beads):
                if i != j:
                    dJ[i, j] += state[i] * state[j] * (1 - prob)

    return dJ / N_exp, dh / N_exp  # Normalize by number of experiments

@njit
def optimize_ising(S, iterations=1000, learning_rate=0.01):
    N_exp, N_beads = S.shape
    J = np.random.randn(N_beads, N_beads).astype(np.float64) * 0.1  # Random initialization
    h = np.random.randn(N_beads).astype(np.float64) * 0.1

    for _ in range(iterations):
        dJ, dh = compute_pseudolikelihood_gradient(S, J, h)

        # Update J and h using the gradients (in-place update for better performance)
        J += learning_rate * dJ
        h += learning_rate * dh
    
    return J, h

# Parameters
region, chrom = [0, 146259331], 'chr8'
N_beads, rep_duration = 50000, 5000

# Paths
rept_path = '/home/skorsak/Data/Replication/sc_timing/GM12878_single_cell_data_hg37.mat'

# Run simulation
rep = Replikator(rept_path, N_beads, rep_duration, chrom, region)
rep.process_matrix()

states = rep.mat
print('States dimension:',states.shape)
states[states == 0] = -1

# Ensure states has the correct dtype
states = states.astype(np.float64)
print('Optimizing Ising model...')
start = time.time()
J, h = optimize_ising(states)
end = time.time()
elapsed = end - start
print(f'Computation finished succesfully in {elapsed//3600:.0f} hours, {elapsed%3600//60:.0f} minutes and  {elapsed%60:.0f} seconds.')

# Save parameters
np.save('J.npy', J)
np.save('h.npy', h)