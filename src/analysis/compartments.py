"""A/B compartment switching from the Potts spin trajectory (A: spin > 0, B: spin < 0, N: neutral)."""
import numpy as np

A, N, B = 1, 0, -1          # compartment codes
LABELS = {A: "A", N: "N", B: "B"}
DEAD = 0.25                 # |window-mean spin| below this -> neutral

def window_state(spins, sl):
    """Compartment code of every bead from the mean spin in a window of samples."""
    m = spins[:, sl].mean(axis=1)
    return np.where(m > DEAD, A, np.where(m < -DEAD, B, N))

def s_phase_states(spins, ph, frac=0.1):
    """Compartments at the start and at the end of S phase (windows of `frac` of S)."""
    w = max(1, int(frac * (ph.s1 - ph.s0)))
    s0 = slice(ph.s0, min(ph.s0 + w, ph.n)); s1 = slice(max(ph.s1 - w, 0), ph.s1)
    return window_state(spins, s0), window_state(spins, s1), (s0, s1)

def transition_matrix(a, b):
    """3x3 counts [from B,N,A] x [to B,N,A]."""
    order = (B, N, A); M = np.zeros((3, 3), int)
    for i, u in enumerate(order):
        for j, v in enumerate(order): M[i, j] = np.sum((a == u) & (b == v))
    return M

def switch_rates(spins, lag):
    """Per-sample fraction of beads switching B->A and A->B (sign change over `lag` samples)."""
    s = np.sign(spins); a, b = s[:, :-lag], s[:, lag:]
    return (np.mean((a < 0) & (b > 0), axis=0), np.mean((a > 0) & (b < 0), axis=0))

def state_fractions(spins):
    """(5, T) fraction of beads in each Potts state -2..2."""
    return np.stack([(spins == k).mean(axis=0) for k in (-2, -1, 0, 1, 2)])

def switch_summary(a, b):
    """Counts of beads that switch / stay between two compartment assignments."""
    return {"B→A": int(np.sum((a == B) & (b == A))), "A→B": int(np.sum((a == A) & (b == B))),
            "A→A": int(np.sum((a == A) & (b == A))), "B→B": int(np.sum((a == B) & (b == B))),
            "via N": int(np.sum(((a == N) | (b == N)) & (a != b)))}
