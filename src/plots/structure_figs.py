"""3D ensemble figures: structural metrics over the trajectory and averaged contact map."""
import os
import numpy as np
import matplotlib.pyplot as plt
from .style import PHASE_COL, phase_bands, phase_runs, PHASE_FILL, genome_axis, BLUE, ROYAL, TEAL, PURPLE, ORANGE, GREEN, rolling, save_fig, CMAP_SEQ, SLATE, GREY

METRICS = [("radius_of_gyration", "Radius of gyration"), ("mean_pairwise_distance", "Mean pair distance"), ("end_to_end_distance", "End-to-end distance"),
           ("asphericity", "Asphericity"), ("contact_number", "Contacts"), ("convex_hull_volume", "Hull volume")]

def _load(d, k):
    return np.load(d + k + ".npy") if os.path.exists(d + k + ".npy") else None

def _phases(d, n):
    """Phase label of every structure (None if unknown, or a single 'All' phase)."""
    ph = _load(d, "phases"); return list(ph) if ph is not None and len(ph) == n and len(set(ph)) > 1 else None

def fig_structure(out):
    """Structural metrics versus structure number, two figures of 3 panels (phases shaded)."""
    d = out + "/metadata/structural_metrics/"; res = None
    for part, (name, fid) in zip((METRICS[:3], METRICS[3:]), (("structural_metrics", "md_structure"), ("structural_metrics_2", "md_structure2"))):
        items = [(l, np.load(d + k + ".npy")) for k, l in part if os.path.exists(d + k + ".npy")]
        if not items: continue
        fig, axs = plt.subplots(1, len(items), figsize=(3.7 * len(items), 3.2), squeeze=False)
        for ax, (l, v), c in zip(axs[0], items, (BLUE, TEAL, PURPLE) if fid == "md_structure" else (ORANGE, GREEN, ROYAL)):
            v = np.asarray(v, float); ax.plot(v, color=c, lw=.7, alpha=.35); ax.plot(rolling(v, max(3, len(v) // 40)), color=c, lw=1.9); ax.set_title(l, fontsize=10.5, pad=16); ax.set_xlabel("Structure #")
            if _phases(d, len(v)): phase_bands(ax, _phases(d, len(v)))
        res = save_fig(fig, out, "structural_metrics", name, fid)
    return res

def fig_heatmap(mat, out, name, fid, title):
    """Averaged inverse-distance map of the ensemble."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4)); lo, hi = mat.mean() - mat.std(), mat.mean() + mat.std()
    im = ax.imshow(mat, cmap=CMAP_SEQ, vmin=max(lo, 0), vmax=hi, aspect="equal"); ax.grid(False); ax.set_title(title); genome_axis(ax, mat.shape[0], "x"); genome_axis(ax, mat.shape[0], "y")
    fig.colorbar(im, fraction=.046, pad=.03, label="1 / (d + 1)")
    return save_fig(fig, out, "structural_metrics", name, fid)

def fig_heatmaps_phases(mats, labels, ranges, out, cmap="turbo"):
    """Averaged inverse-distance maps of the phases in one row: phase in the title, shared colour scale (turbo, square-root stretch) and one colour bar."""
    from matplotlib.colors import PowerNorm
    off = [m[~np.eye(len(m), dtype=bool)] for m in mats]; lo, hi = np.percentile(np.concatenate(off), (2, 99.7))
    fig, axs = plt.subplots(1, len(mats), figsize=(4.6 * len(mats) + .9, 4.6), squeeze=False, gridspec_kw=dict(wspace=.12)); norm = PowerNorm(.5, vmin=max(lo, 0), vmax=hi)
    for ax, m, ph, (a, b) in zip(axs[0], mats, labels, ranges):
        im = ax.imshow(m, cmap=cmap, norm=norm, aspect="equal"); ax.grid(False); genome_axis(ax, m.shape[0], "x"); genome_axis(ax, m.shape[0], "y")
        if ax is not axs[0][0]: ax.set_ylabel(""); ax.tick_params(labelleft=False)   # shared y axis
        ax.set_title(f"{ph}  (structures {a}-{b})", color="white", fontsize=11, fontweight="bold", pad=8, bbox=dict(boxstyle="round,pad=.35", fc=PHASE_COL.get(ph, PURPLE), ec="none"))
    fig.colorbar(im, ax=list(axs[0]), fraction=.02, pad=.02, label="mean 1 / (d + 1)   (high = close)")
    return save_fig(fig, out, "structural_metrics", "mean_inv_heatmap_phases", "md_heat_phases")

# (file, title, unit conversion from Angstrom, normalise to G1)
COMPACT = [("radius_of_gyration", "Radius of gyration (one sister)", 0.1, "nm", True), ("convex_hull_volume", "Convex-hull volume (one sister)", 1e-3, "nm³", True),
           ("contact_number", "Contacts (one sister)", 1, "", True), ("sister_distance", "Sister centroid distance", 0.1, "nm", False)]

def fig_compaction(out):
    """Compaction through the cell cycle: traces with phase bands (top) and phase means relative to G1 (bottom)."""
    d = out + "/metadata/structural_metrics/"; cols = [(k, t, u, un, rel, _load(d, k)) for k, t, u, un, rel in COMPACT]; cols = [c for c in cols if c[-1] is not None]
    if not cols: return None
    ph = _phases(d, len(cols[0][-1])); rows = 2 if ph else 1
    fig, axs = plt.subplots(rows, len(cols), figsize=(3.5 * len(cols), 3.1 * rows + .4), squeeze=False, gridspec_kw=dict(hspace=.62 if rows > 1 else 0, wspace=.38))
    for j, (k, t, u, un, rel, v) in enumerate(cols):
        v = np.asarray(v, float) * u; c = [BLUE, ORANGE, GREEN, PURPLE][j]; a = axs[0, j]
        a.plot(v, color=c, lw=.7, alpha=.35); a.plot(rolling(v, max(3, len(v) // 40)), color=c, lw=1.9); a.set_title(t, fontsize=10, pad=16); a.set_xlabel("Structure #"); a.set_ylabel(un)
        if ph: phase_bands(a, ph)
        if not ph: continue
        b = axs[1, j]; runs = phase_runs(ph); m = np.array([v[x:y].mean() for _, x, y in runs]); sd = np.array([v[x:y].std() for _, x, y in runs])
        base = m[0] if rel and m[0] else 1.0; m, sd = m / base, sd / base; cc = [PHASE_FILL.get(n, PURPLE) for n, _, _ in runs]
        b.bar(range(len(runs)), m, yerr=sd, color=cc, alpha=.85, capsize=3, width=.62); b.set_xticks(range(len(runs)), [n for n, _, _ in runs])
        b.set_ylabel("relative to G1" if rel else un); b.set_title("phase mean ± s.d.", fontsize=9, color=SLATE)
        if rel:
            b.axhline(1, color=GREY, lw=.8, ls="--")
            for i, x in enumerate(m[1:], 1): b.annotate(f"{(x - 1) * 100:+.0f}%", (i, x + sd[i]), (0, 3), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold")
            b.set_ylim(0, max(m + sd) * 1.2)
    return save_fig(fig, out, "structural_metrics", "compaction", "md_compaction")
