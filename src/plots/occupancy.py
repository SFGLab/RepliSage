"""Preferential binding: loop occupancy profile with CTCF sites, orientation, strength and confidence."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from .style import LEFT, RIGHT, PURPLE, GREY, SLATE, ROYAL, save_fig
from ..preproc.bedpe import orientation_confidence

SUB = "MCMC_diagnostics"

def fig_occupancy(ctx, p, geo, out):
    """Top: occupancy (0-1, nested loops stack) with the LEF loading weight and CTCF strength. Bottom: orientation of each CTCF site."""
    occ = np.asarray(ctx.occupancy, float); x = geo.xs; L, R = p.L, p.R; tot = L + R; mx = max(tot.max(), 1e-9)
    score, conf = orientation_confidence(L, R); has_o = bool(np.any(conf > 0))
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1.5], hspace=.08))
    a.fill_between(x, occ, step="mid", color=PURPLE, alpha=.35, lw=0, label="loop occupancy (0-1)"); a.plot(x, occ, color=PURPLE, lw=1.2, drawstyle="steps-mid")
    if p.bind_w is not None: a.plot(x, p.bind_w / p.bind_w.max(), color=ROYAL, lw=1.1, ls="--", label="relative LEF loading weight")
    a.vlines(x[tot > 0], 0, tot[tot > 0] / mx, color=SLATE, lw=1.3, alpha=.8, label="CTCF strength (scaled to 0-1)")
    a.set_ylim(0, 1.08); a.set_ylabel("Normalised signal"); a.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.2), fontsize=8.5, frameon=False)
    cm = LinearSegmentedColormap.from_list("conf", ["#D5D5D5", "#D5D5D5"])                       # neutral colour for unknown orientation
    on = tot > 0; sz = 25 + 140 * tot[on] / mx; right = score[on] >= 0
    for m, sel, col in (("<", ~right, LEFT), (">", right, RIGHT)):
        if not sel.any(): continue
        c = np.array([col] * sel.sum()) if has_o else np.array(["#9A9A9A"] * sel.sum())
        b.scatter(x[on][sel], score[on][sel], s=sz[sel], marker=m, c=c, alpha=(.25 + .75 * conf[on][sel]) if has_o else .8, edgecolor="none", zorder=3)
    b.axhline(0, color=GREY, lw=.8); b.set_ylim(-1.15, 1.15); b.set_yticks([-1, 0, 1], ["left-ward", "mixed", "right-ward"]); b.grid(axis="y", alpha=.4)
    b.set_ylabel("CTCF orientation"); b.set_xlabel(geo.label)
    b.legend(handles=[Line2D([], [], marker="<", ls="", color=LEFT, label="left-ward CTCF"), Line2D([], [], marker=">", ls="", color=RIGHT, label="right-ward CTCF")], ncol=2, loc="lower left", fontsize=8.5, frameon=False)
    b.text(.995, .03, "marker size = strength, opacity = confidence |R - L| / (R + L)" if has_o else "no orientation in the file: confidence is 0 everywhere", transform=b.transAxes, ha="right", va="bottom", fontsize=8, color=GREY)
    return save_fig(fig, out, SUB, "ctcf_occupancy", "ctcf_occupancy")
