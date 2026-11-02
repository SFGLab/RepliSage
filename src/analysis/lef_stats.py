"""Series and statistics of the loop extruding factors (LEFs)."""
import numpy as np

def bound_mask(Ms, Ns):
    """A LEF is bound when 0 <= m < n."""
    return (Ms >= 0) & (Ns > Ms)

def coverage(Ms, Ns, n_beads):
    """(beads, samples) number of loops covering every bead."""
    T = Ms.shape[1]; ok = bound_mask(Ms, Ns)
    t = np.broadcast_to(np.arange(T), Ms.shape)
    d = np.zeros((n_beads + 1) * T)
    np.add.at(d, Ms[ok] * T + t[ok], 1); np.add.at(d, Ns[ok] * T + t[ok], -1)
    return np.cumsum(d.reshape(n_beads + 1, T), axis=0)[:n_beads]

def lef_series(res, p):
    """Per-sample LEF observables."""
    Ms, Ns = res.Ms, res.Ns; ok = bound_mask(Ms, Ns)
    nb = np.maximum(ok.sum(0), 1)
    safe_m, safe_n = np.clip(Ms, 0, None), np.clip(Ns, 0, None)
    return {"loop": (Ns - Ms).mean(0),                                  # as published: all LEFs
            "loop_bound": np.where(ok, Ns - Ms, 0).sum(0) / nb,
            "bound": ok.mean(0),
            "on_L": ((p.L[safe_m] > 0) & ok).sum(0) / nb,
            "on_R": ((p.R[safe_n] > 0) & ok).sum(0) / nb}

def leg_occupancy(Ms, Ns, n_beads, seg):
    """Mean number of left / right legs per bead and sample."""
    ok = bound_mask(Ms, Ns)[:, seg]; m, n = Ms[:, seg][ok], Ns[:, seg][ok]
    T = max(1, Ms[:, seg].shape[1])           # legs per bead per sample
    return np.bincount(m, minlength=n_beads) / T, np.bincount(n, minlength=n_beads) / T
