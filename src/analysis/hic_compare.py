"""Hi-C validation: simulated contact maps (MC loop map, MD ensemble) against a Hi-C map on the same bead grid."""
import numpy as np
from pathlib import Path
from .hic_maps import (ps_curve, fill_masked, observed_over_expected, oe_correlation, insulation_score, insulation_window, compartment_from_hic,
                       pc1, boundaries, boundary_overlap, smooth_track)
from .tables import Table, g
from ..plots.model_maps import lef_contacts, checkerboard, background
from ..logger import log

NAN = float("nan"); STATE_SMOOTH = 2.0
METRICS = [("log-contact r", "r_log", "Pearson r of log contacts over all bead pairs"),
           ("O/E r", "r_oe", "Pearson r after dividing each diagonal by its mean (decay removed)"),
           ("SCC", "scc", "mean Pearson r per diagonal (stratified correlation)"),
           ("P(s) r", "r_ps", "Pearson r of the log contact decay with distance"),
           ("insulation r", "r_ins", "Pearson r of the insulation scores (domain boundaries)"),
           ("boundary F1", "f1", "domain boundaries found in both maps (within 2 beads)"),
           ("correlation matrix r", "r_cm", "Pearson r of the O/E correlation matrices (shared contact patterns: compartments and domains)"),
           ("compartment r", "r_pc", "Pearson r of the Hi-C A/B eigenvector with the mean epigenetic state (epigenetic model on) or with the simulated eigenvector (sign free)")]

def _r(x, y):
    """Pearson r, NaN-safe."""
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 3 and np.std(x[ok]) > 1e-12 and np.std(y[ok]) > 1e-12 else NAN

def _scc(h, m, kmin=2):
    """Mean per-diagonal Pearson r, weighted by the number of pairs."""
    r, w = [], []
    for k in range(kmin, len(h) // 2):
        x, y = np.diagonal(h, k), np.diagonal(m, k); v = _r(x, y)
        if np.isfinite(v): r.append(v); w.append(len(x))
    return float(np.average(r, weights=w)) if r else NAN

def compare_hic(model, hic, window=5, tol=2, state=None, track=0.0, full=True):
    """Metrics (and, with full, the tracks behind them) of a simulated map against the Hi-C map (same shape, NaN = masked)."""
    n = len(hic); seen = np.isfinite(hic); h, m = fill_masked(hic), np.array(model, float)
    sep = np.abs(np.subtract.outer(np.arange(n), np.arange(n))); iu = (sep >= 2) & (np.triu(np.ones((n, n)), 1) > 0); up = iu & seen
    ok = up & (h > 0) & (m > 0); oe_h, oe_m = observed_over_expected(h), observed_over_expected(m)
    out = dict(r_log=_r(np.log10(h[ok]), np.log10(m[ok])), r_oe=_r(np.log2(oe_h[ok] + 1e-3), np.log2(oe_m[ok] + 1e-3)), scc=_scc(np.where(seen, h, np.nan), m), n_pairs=int(ok.sum()))
    if not full: return out
    ps_h, ps_m = ps_curve(h)[1:n // 2], ps_curve(m)[1:n // 2]; ps_h, ps_m = ps_h / ps_h.sum(), ps_m / ps_m.sum()
    ins_h, ins_m = insulation_score(h, window, track), insulation_score(m, window, track); sl = slice(window, n - window)
    ins_h, ins_m = (np.nan_to_num(np.log2(np.clip(v[sl], 1e-12, None) / v[sl].mean())) for v in (ins_h, ins_m))
    b_h, b_m = boundaries(ins_h), boundaries(ins_m); rec, prec = boundary_overlap(b_h, b_m, tol), boundary_overlap(b_m, b_h, tol)
    f1 = 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0
    c_h, c_m = oe_correlation(h), oe_correlation(m); p_h = compartment_from_hic(hic, smooth=track)
    if state is not None: st = smooth_track(np.asarray(state, float), STATE_SMOOTH); p_m, r_pc = st, _r(p_h, st)
    else: p_m = pc1(m, track); r_pc = abs(_r(p_h, p_m)); p_m = p_m * (1 if _r(p_h, p_m) >= 0 else -1)
    out.update(r_cm=_r(c_h[iu & seen], c_m[iu & seen]), r_ps=_r(np.log(ps_h), np.log(ps_m)), r_ins=_r(ins_h, ins_m), f1=float(f1) if np.isfinite(f1) else NAN, r_pc=r_pc,
               ps_data=ps_h, ps_model=ps_m, ins_data=ins_h, ins_model=ins_m, b_data=b_h + window, b_model=b_m + window, pc_data=p_h, pc_model=p_m, window=window, h=h, m=m)
    return out

def model_maps(res, p, ph, seg, kappa=1.5):
    """Simulated contact map of the samples in `seg` from the MC state: LEF contacts + polymer background, times the epigenetic checkerboard when it is on."""
    n = p.N_beads; c = lef_contacts(res.Ms, res.Ns, n, seg); c = np.clip(c / (np.percentile(c[c > 0], 99) if (c > 0).any() else 1), 0, 1)
    has_epi = p.has_rep and bool(np.any(res.spin_traj != 0)); m = background(n) + c
    if has_epi: S = checkerboard(res.spin_traj, seg); m = m * np.exp(kappa * S / max(np.abs(S).max(), 1e-9))
    return m, (res.spin_traj[:, seg].mean(1) / 2 if has_epi else None)

def md_map(out_dir, n):
    """Average inverse-distance heatmap of the 3D ensemble, when it exists and matches the bead grid."""
    f = Path(out_dir) / "metadata" / "structural_metrics" / "avg_heatmap.npy"
    if not f.is_file(): return None
    a = np.load(f); return a if a.shape == (n, n) else None

def validate_hic(ctx, p, res, ph, hic, out_dir, track=7.0):
    """Compare the model with the Hi-C map: pooled post-burn-in samples (a cell population), each phase, and the MD ensemble if present."""
    seg = slice(ph.g1, ph.n); window = insulation_window((ctx.region[1] - ctx.region[0]) // ctx.n_beads)
    m, state = model_maps(res, p, ph, seg); maps = {"MC loop map": m}; md = md_map(out_dir, p.N_beads)
    if md is not None: maps["MD ensemble"] = md
    R = {k: compare_hic(v, hic, window, state=state if k == "MC loop map" else None, track=track) for k, v in maps.items()}
    phases = {k: compare_hic(model_maps(res, p, ph, s)[0], hic, window, full=False) for k, s in ph.segments().items()} if ph.has_rep else {}
    return dict(hic=hic, maps=maps, metrics=R, phases=phases, window=window)

def hic_tables(V, info):
    """Tables: agreement metrics per model map, and the log-contact r of each phase."""
    names = list(V["metrics"]); rows = [(lab, *(g(V["metrics"][k][key], 3) for k in names)) for lab, key, _ in METRICS]
    t1 = Table("hic_validation", "Hi-C validation", ("Metric", *names), rows,
               f"Hi-C: {info['file']}, {info['resolution']:,} bp, {info['norm']} normalisation, {info['coverage'] * 100:.0f} % of beads usable. "
               + " ".join(f"{lab}: {d}." for lab, _, d in METRICS))
    out = [t1]
    if V["phases"]:
        ph = list(V["phases"]); out.append(Table("hic_phases", "Hi-C agreement per phase (MC loop map)", ("Metric", *ph),
                                               [(lab, *(g(V["phases"][k][key], 3) for k in ph)) for lab, key, _ in METRICS[:3]],
                                               "A Hi-C map averages an asynchronous population, so no single phase is expected to match best."))
    return out
