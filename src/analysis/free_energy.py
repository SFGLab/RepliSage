"""Energy budget and free-energy landscapes per cell-cycle phase.

Budget: every energy term of the model, recomputed from the state at each sample, as mean ± SD per phase.
Landscape: F(x) = -T ln P(x) along an order parameter x (T = MC temperature), up to an additive constant.
"""
import numpy as np
from .tables import Table, g
from .lef_stats import lef_series

def energy_terms(res, p):
    """Soft energy terms per sample (model units); 'Total' is their sum, recomputed from the state."""
    t = {"Folding": res.Fs, "Binding": res.Bs}
    if p.has_rep and (p.c_potts1 or p.c_potts2): t["Epigenetic"] = res.Es_potts
    t["Total"] = sum(t.values()); return t

def hard_terms(res, p):
    """Hard-wall terms as penalty counts per sample: crossing (energy / kappa x 1e8) and replication fork (energy / rep norm)."""
    from ..MC.normalization import energy_norms
    k, rep = energy_norms(p)[3], energy_norms(p)[4]; h = {"Crossing penalties": np.asarray(res.Ec, float) / k if res.Ec is not None and k else np.zeros_like(res.Es)}
    if p.has_rep: h["Fork penalties"] = np.asarray(res.Er, float) / rep if res.Er is not None and rep else np.zeros_like(res.Es)
    return h

def budget_table(res, p, ph):
    """Mean ± SD of each term per phase, and the change relative to G1."""
    segs = ph.segments(); rows = []
    def row(name, y, fmt=g):
        cells = [f"{fmt(y[s].mean())} ± {fmt(y[s].std())}" if len(y[s]) > 1 else "n/a" for s in segs.values()]
        if ph.has_rep and len(y[segs["G1"]]) > 1: base = y[segs["G1"]].mean(); cells += [fmt(y[segs[k]].mean() - base) for k in ("S", "G2/M")]
        rows.append((name, *cells))
    for name, y in energy_terms(res, p).items(): row(name if name != "Total" else "Total (soft terms)", y)
    for name, y in hard_terms(res, p).items(): row(name + " (count)", y)
    head = ("Energy term", *segs) + (("Δ S − G1", "Δ G2/M − G1") if ph.has_rep else ())
    return Table("energy_budget", "Energy budget per phase (model units)", head, rows,
                 "Soft terms are recomputed from the loop and spin state at every sample; Δ = change of the phase mean relative to G1; negative = favourable. "
                 "Crossing and replication-fork terms are hard walls (scaled by 1e8 x coefficient): they are shown as the number of penalty units, not as energy.")

def landscape(x, T, lo, hi, bins=16, min_count=5):
    """(centres, F) with F = -T ln P(x) shifted to min 0; bins with fewer than min_count samples are NaN."""
    c, e = np.histogram(x, bins=bins, range=(lo, hi)); P = c / max(c.sum(), 1)
    F = np.where(c >= min_count, -T * np.log(np.where(P > 0, P, 1)), np.nan)
    return (e[:-1] + e[1:]) / 2, F - np.nanmin(F) if np.isfinite(F).any() else F

def order_parameters(res, p):
    """{name: per-sample series} used for the landscapes."""
    o = {"Mean loop length (beads)": lef_series(res, p)["loop_bound"]}
    if p.has_rep and (p.c_potts1 or p.c_potts2): o["Mean epigenetic state"] = res.mags
    return o
