"""Anchor-level view of the CTCF input: position, strength, orientation and its confidence (and loop configuration for BEDPE)."""
from dataclasses import dataclass
import numpy as np

CONF_MIN = 0.2                                   # below this an orientation call counts as ambiguous
CONFIGS = ("convergent", "tandem", "divergent", "uncertain")

@dataclass
class Anchors:
    bp: np.ndarray               # anchor centre (bp)
    strength: np.ndarray         # weight carried by the anchor
    p_rev: np.ndarray            # P(reverse), -1 = unknown
    left_bp: np.ndarray          # per loop: left / right anchor centre (BEDPE only, else empty)
    right_bp: np.ndarray
    loop_strength: np.ndarray
    config: np.ndarray           # per loop: one of CONFIGS

    @property
    def confidence(self):
        """|2p - 1|: 0 = coin flip, 1 = certain (0 if unknown)."""
        return np.where(self.p_rev >= 0, np.abs(2 * self.p_rev - 1), 0.0)

    @property
    def orient(self):
        """'forward' / 'reverse' / 'ambiguous' / 'unknown' per anchor."""
        o = np.where(self.p_rev < .5, "forward", "reverse").astype(object)
        o[self.confidence < CONF_MIN] = "ambiguous"; o[self.p_rev < 0] = "unknown"
        return o

    @property
    def has_loops(self): return len(self.left_bp) > 0

    @property
    def has_orientation(self): return bool((self.p_rev >= 0).any())

def loop_config(p_left, p_right):
    """Loop configuration from the orientations of its left and right anchor."""
    sure = lambda p: (p >= 0) & (np.abs(2 * p - 1) >= CONF_MIN)
    ok = sure(p_left) & sure(p_right); lf, rf = p_left < .5, p_right < .5
    out = np.full(len(p_left), "uncertain", dtype=object)
    out[ok & lf & ~rf] = "convergent"; out[ok & ~lf & rf] = "divergent"; out[ok & (lf == rf)] = "tandem"
    return out

def anchors_of(binding):
    """Anchors from the reader output (Binding.anchors)."""
    a = binding.anchors; bp, w, p = (np.asarray(a[k], float) for k in ("bp", "w", "p"))
    if a["loop"] is None: e = np.zeros(0); return Anchors(bp, w, p, e, e, e, np.zeros(0, dtype=object))
    n = len(bp) // 2; c1, c2, p1, p2 = bp[:n], bp[n:], p[:n], p[n:]; swap = c1 > c2        # anchor 1 = the left one
    cl, cr = np.where(swap, c2, c1), np.where(swap, c1, c2); pl, pr = np.where(swap, p2, p1), np.where(swap, p1, p2)
    return Anchors(np.r_[cl, cr], w, np.r_[pl, pr], cl, cr, w[:n], loop_config(pl, pr))
