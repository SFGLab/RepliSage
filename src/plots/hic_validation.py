"""Hi-C validation figures: maps side by side, triangle comparison, P(s), insulation and A/B tracks."""
import numpy as np
import matplotlib.pyplot as plt
from .style import CMAP_SEQ, ROYAL, BLUE, GREY, genome_axis, save_fig
from ..analysis.hic_maps import fill_masked

SUB = "hic_validation"

def _log(m):
    """log10 of a contact map, NaN-safe."""
    return np.log10(np.clip(np.nan_to_num(m, nan=0.0), 1e-9, None))

def _tri(h, m):
    """One matrix: upper triangle Hi-C, lower triangle model, each on its own robust 0-1 scale."""
    f = lambda a: np.clip((_log(a) - np.percentile(_log(a), 5)) / (np.percentile(_log(a), 99.5) - np.percentile(_log(a), 5) + 1e-9), 0, 1)
    out = np.tril(f(m), -1) + np.triu(f(h), 1); return out

def fig_hic(V, info, geo, out):
    """Row 1: Hi-C | model map(s) | triangle comparison. Row 2: P(s) | insulation | A/B compartments."""
    names = list(V["maps"]); k = len(names); R = V["metrics"][names[0]]; hic = V["hic"]; n = len(hic)
    fig = plt.figure(figsize=(5 * (2 + k) , 9.6)); gs = fig.add_gridspec(2, 2 + k, hspace=.35, wspace=.28)
    ims = [("Hi-C", fill_masked(hic))] + [(nm, V["maps"][nm]) for nm in names]
    for j, (t, M) in enumerate(ims):
        a = fig.add_subplot(gs[0, j]); L = _log(M); a.imshow(L, cmap=CMAP_SEQ, vmin=np.percentile(L, 5), vmax=np.percentile(L, 99.5), interpolation="nearest")
        a.set_title(t); a.grid(False); genome_axis(a, n, "x", 4); genome_axis(a, n, "y", 4)
    a = fig.add_subplot(gs[0, 1 + k]); a.imshow(_tri(fill_masked(hic), V["maps"][names[0]]), cmap=CMAP_SEQ, interpolation="nearest"); a.grid(False)
    a.set_title(f"Hi-C (upper) vs {names[0]} (lower)"); a.text(.97, .03, f"r = {R['r_log']:.2f}", transform=a.transAxes, ha="right", fontsize=10, color="white", fontweight="bold")
    genome_axis(a, n, "x", 4); genome_axis(a, n, "y", 4)
    a = fig.add_subplot(gs[1, 0]); s = np.arange(1, len(R["ps_data"]) + 1)
    a.loglog(s, R["ps_data"], color=GREY, lw=2, label="Hi-C"); [a.loglog(s, V["metrics"][nm]["ps_model"], lw=1.8, label=nm) for nm in names]
    a.set(xlabel="Separation (beads)", ylabel="Normalised P(s)", title=f"Contact decay (r = {R['r_ps']:.2f})"); a.legend(fontsize=8)
    w = V["window"]; x = np.arange(w, n - w)
    a = fig.add_subplot(gs[1, 1]); a.plot(x, R["ins_data"], color=GREY, lw=1.8, label="Hi-C"); a.plot(x, R["ins_model"], color=ROYAL, lw=1.4, label=names[0])
    a.vlines(R["b_data"], a.get_ylim()[0], a.get_ylim()[0] * .8, color=GREY, lw=1); a.vlines(R["b_model"], a.get_ylim()[1] * .8, a.get_ylim()[1], color=ROYAL, lw=1)
    a.set(xlabel="Bead", ylabel="log2 insulation", title=f"Domains (r = {R['r_ins']:.2f}, boundary F1 = {R['f1']:.2f})"); a.legend(fontsize=8)
    a = fig.add_subplot(gs[1, 2:]); a.plot(np.arange(n), R["pc_data"], color=GREY, lw=1.8, label="Hi-C A/B"); a.plot(np.arange(n), R["pc_model"], color=BLUE, lw=1.4, label="model")
    a.axhline(0, color=GREY, lw=.7); a.set(xlabel="Bead", ylabel="A (+) / B (−)", title=f"Compartments (r = {R['r_pc']:.2f})"); a.legend(fontsize=8)
    return save_fig(fig, out, SUB, "hic_validation", "hic_validation")
