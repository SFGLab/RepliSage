"""Load the 3D ensemble (+ MC annotations) into one trajectory object shared by the video and the HTML viewer."""
import os, re
from dataclasses import dataclass, field
import numpy as np
from ..io.structure_io import get_coordinates_cif

PHASE_OF = {"BR": "G1", "R": "S", "AR": "G2/M", "LE": "All"}

@dataclass
class Traj:
    V: np.ndarray            # (F, chains*N, 3) aligned coordinates (Angstrom)
    n_chains: int
    n_beads: int
    phase: list              # phase name per frame
    step: np.ndarray         # MC step of every frame
    spins: np.ndarray        # (F, N) Potts spin -2..2 (zeros if none)
    rep: np.ndarray          # (F, N) replicated state 0/1
    lefs: list               # per frame: (k, 2) array of bound LEF anchors (m, n)
    has_rep: bool
    files: list = field(default_factory=list)
    lef_fam: list = field(default_factory=list)   # per frame: 0 = cohesin, 1 = condensin for every LEF in lefs
    ctcf: tuple = (np.zeros(0, int), np.zeros(0, int))   # CTCF bead indices and orientation (+1 forward, -1 reverse)

def kabsch(P, Q):
    """Rotation matrix aligning the centred P onto the centred Q."""
    U, _, Vt = np.linalg.svd(P.T @ Q); d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt

def align(V):
    """Centre every frame and rotate it onto the previous one (removes tumbling, keeps motion smooth)."""
    V = V - V.mean(1, keepdims=True)
    for k in range(1, len(V)): V[k] = V[k] @ kabsch(V[k], V[k - 1])
    return V

def list_ensemble(d):
    """Ensemble files sorted by frame index: [(index, phase, path)]."""
    out = []
    for f in os.listdir(d):
        m = re.match(r"ensemble_(\d+)_(\w+)\.cif$", f)
        if m: out.append((int(m.group(1)), m.group(2), os.path.join(d, f)))
    rank = {"BR": 0, "LE": 0, "R": 1, "AR": 2}
    return sorted(out, key=lambda t: (rank.get(t[1], 0), t[0]))   # G1 -> S -> G2/M

def phase_at(step, t_rep, duration):
    """Cell-cycle phase of an MC step: G1 before the S window, S inside it, G2/M after it (independent of the sampling interval)."""
    return "G1" if step < t_rep else ("S" if step < t_rep + duration else "G2/M")

def load_traj(out_dir, p, res, rep=None, max_frames=None):
    """Build a Traj from out_dir/ensemble and the MC result (res) of the same replica."""
    items = list_ensemble(os.path.join(out_dir, "ensemble"))
    if max_frames and len(items) > max_frames: items = [items[i] for i in np.unique(np.linspace(0, len(items) - 1, max_frames).astype(int))]
    V = align(np.stack([get_coordinates_cif(f) for _, _, f in items]))
    n_ch = 2 if rep is not None else 1; N = p.N_beads
    ms = p.MC_step; idx = [p.burnin // ms + k - 1 for k, _, _ in items]   # file number = sample index (+1) counted from burn-in
    idx = np.minimum(np.array(idx), res.Ms.shape[1] - 1)
    sp = res.spin_traj[:, idx].T.astype(np.int8) if getattr(res, "spin_traj", None) is not None else np.zeros((len(idx), N), np.int8)
    r = np.zeros((len(idx), N), np.uint8)
    if rep is not None:
        rt = np.clip(idx * ms - p.t_rep, 0, rep.f_rep_T.shape[0] - 1); r = rep.f_rep_T[rt].astype(np.uint8)
        r[idx * ms < p.t_rep] = 0; r[idx * ms >= p.t_rep + p.rep_duration] = 1   # before S: none, after S: all replicated
    lefs, fam = [], []; kind = (np.arange(res.Ms.shape[0]) >= p.N_lef).astype(np.int8)   # LEF index >= N_lef -> condensin
    for j in idx:
        m, n = res.Ms[:, j], res.Ns[:, j]; ok = (m >= 0) & (n > m); lefs.append(np.stack([m[ok], n[ok]], 1)); fam.append(kind[ok])
    phase = [phase_at(s_, p.t_rep, p.rep_duration) for s_ in idx * ms] if rep is not None else [PHASE_OF.get(ph, ph) for _, ph, _ in items]   # phase from the MC step itself, not from file labels
    return Traj(V, n_ch, N, phase, idx * ms, sp, r, lefs, rep is not None, [f for _, _, f in items], fam, ctcf_sites(p))

def ctcf_sites(p, rel=.25):
    """CTCF beads (binding weight above rel x the maximum) with orientation as in the model: L (attracts the left LEF leg) = forward-facing motif, arrow along the chain (+1); R = reverse-facing, arrow against it (-1)."""
    L, R = np.asarray(p.L, float), np.asarray(p.R, float); i, d = [], []
    for v, o in ((L, 1), (R, -1)):
        k = np.flatnonzero(v > rel * v.max()) if v.max() > 0 else []; i += list(k); d += [o] * len(k)
    return np.array(i, int), np.array(d, int)


def replication_eyes(rp):
    """Replicated segments (a, b), inclusive bead indices, of one chain. A segment has a fork at a (moving left) and at b + 1 (moving right); a fully copied chain has none."""
    r = np.asarray(rp).astype(int); d = np.diff(np.r_[0, r, 0]); s = np.flatnonzero(d == 1); e = np.flatnonzero(d == -1) - 1
    return [(int(a), int(b)) for a, b in zip(s, e) if not (a == 0 and b == len(r) - 1)]
