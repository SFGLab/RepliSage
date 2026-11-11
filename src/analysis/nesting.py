"""Families of loops: nesting of the LEF loops per sample, per cell-cycle phase.

A loop B is inside loop A when A's left leg is at or left of B's and A's right leg is at or right of B's
(identical loops count once). The order of a loop is the number of loops that contain it:
order 0 = inside no other loop (a parent if it holds loops, an orphan if it holds none), order k = child of order k.
"""
from dataclasses import dataclass
import numpy as np
from .lef_stats import bound_mask
from .tables import Table, g

NAMES = ["orphans", "parents", "order 1", "order 2", "order 3+"]     # loop classes, see kind_of

@dataclass
class Nesting:
    valid: np.ndarray       # (K, T) real loop: bound, not zero-length, not a duplicate
    order: np.ndarray       # (K, T) number of loops containing this loop
    length: np.ndarray      # (K, T) loop length, beads
    n_desc: np.ndarray      # (K, T) number of loops inside this loop
    parent_len: np.ndarray  # (K, T) length of the smallest loop containing this one (NaN for order 0)

def nesting(ms, ns, chunk=None):
    """Nesting of all loops in every sample; ms, ns are (K, T) leg positions (equal legs = no loop)."""
    lo, hi = np.minimum(ms, ns), np.maximum(ms, ns); K, T = lo.shape
    chunk = chunk or max(1, int(2e7 // max(K * K, 1)))                    # keeps the (K, K, chunk) work arrays small
    valid_a, order_a, desc_a = (np.zeros((K, T), t) for t in (bool, int, int)); plen = np.full((K, T), np.nan)
    earlier = np.tri(K, k=-1, dtype=bool)[:, :, None]                     # j < i
    for a in range(0, T, chunk):
        l, h = lo[:, a:a + chunk], hi[:, a:a + chunk]
        same = (l[:, None] == l[None]) & (h[:, None] == h[None])
        valid = ~(same & earlier).any(axis=1) & (h > l)                    # first of identical loops only
        cont = ((l[:, None] <= l[None]) & (h[None] <= h[:, None]) & ~same) & valid[:, None]   # cont[j, i]: j contains i
        valid_a[:, a:a + chunk], order_a[:, a:a + chunk] = valid, cont.sum(axis=0)
        desc_a[:, a:a + chunk] = (cont & valid[None]).sum(axis=1)
        p = np.where(cont, (h - l)[:, None], np.inf).min(axis=0); plen[:, a:a + chunk] = np.where(np.isinf(p), np.nan, p)
    return Nesting(valid_a, order_a, (hi - lo).astype(float), desc_a, plen)

def nesting_of(res):
    """Nesting of all LEFs of an MC result (unbound LEFs are not loops)."""
    if getattr(res, "_nesting", None) is None:                             # computed once per result
        ok = bound_mask(res.Ms, res.Ns); res._nesting = nesting(np.where(ok, res.Ms, 0), np.where(ok, res.Ns, 0))
    return res._nesting

def kind_of(order, n_desc, max_order=3):
    """Loop class: 0 orphan, 1 parent, k + 1 child of order k (last class = max_order or deeper)."""
    return np.where(order == 0, np.where(n_desc == 0, 0, 1), np.minimum(order, max_order) + 1)

def order_counts(nst, sl=slice(None)):
    """(5, T) loops per sample in each class (orphans, parents, order 1, 2, 3+)."""
    k, v = kind_of(nst.order, nst.n_desc)[:, sl], nst.valid[:, sl]
    return np.array([((k == c) & v).sum(0) for c in range(5)])

def phase_summary(nst, sl, kb=None):
    """Headline numbers of one phase slice (samples)."""
    v = nst.valid[:, sl]; o, d, ln = nst.order[:, sl][v], nst.n_desc[:, sl][v], nst.length[:, sl][v]
    if not len(o): return {}
    k = kind_of(nst.order[:, sl], nst.n_desc[:, sl])[v]; snap = np.where(v, nst.order[:, sl], 0).max(0)
    s = dict(loops=float(v.sum(0).mean()), orphans=float((k == 0).mean()), parents=float((k == 1).mean()), children=float((o > 0).mean()),
             mean_order=float(o.mean()), deepest=float(snap.mean()), desc_per_parent=float(d[k == 1].mean()) if (k == 1).any() else 0.0,
             length=float(ln.mean() * (kb or 1)))
    for c in range(2, 5): s[f"len{c}"] = float(ln[k == c].mean() * (kb or 1)) if (k == c).any() else float("nan")
    return s

def nesting_table(nst, ph, kb=None):
    """Table of the nesting statistics per phase."""
    segs = ph.segments(); S = {n: phase_summary(nst, s, kb) for n, s in segs.items()}; u = "kb" if kb else "beads"
    spec = [("loops per sample", "loops", g), ("orphan loops (isolated), share", "orphans", lambda x: f"{x * 100:.0f} %"), ("parent loops, share", "parents", lambda x: f"{x * 100:.0f} %"),
            ("child loops (order >= 1), share", "children", lambda x: f"{x * 100:.0f} %"), ("mean nesting order", "mean_order", g), ("deepest order per sample", "deepest", g),
            ("loops inside a parent (mean)", "desc_per_parent", g), (f"mean loop length, {u}", "length", g), (f"mean length of order-1 children, {u}", "len2", g)]
    rows = [(lab, *(f(S[n][key]) if S[n] else "n/a" for n in segs)) for lab, key, f in spec]
    return Table("nesting", "Families of loops per phase", ("Quantity", *segs), rows,
                 "A loop is a child of order k when k other loops contain it; parents hold loops but sit inside none; orphans are isolated loops.")
