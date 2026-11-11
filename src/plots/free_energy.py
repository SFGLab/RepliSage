"""Energy budget and free-energy landscapes over the cell cycle."""
import numpy as np
import matplotlib.pyplot as plt
from .style import PHASE_COL, CYCLE, GREY, shade_phases, rolling, save_fig
from ..analysis.free_energy import energy_terms, landscape, order_parameters

SUB = "free_energy"

def fig_energy_budget(res, p, ph, out):
    """Mean energy of each term per phase (bars, SD whiskers), change versus G1, and the terms over time."""
    segs = ph.segments(); T = energy_terms(res, p); names = list(T); n = len(names)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(wspace=.32, width_ratios=[1.3, 1, 1.3])); w = .8 / len(segs)
    for i, (ph_name, s) in enumerate(segs.items()):
        m = [T[k][s].mean() for k in names]; sd = [T[k][s].std() for k in names]
        axs[0].bar(np.arange(n) + (i - (len(segs) - 1) / 2) * w, m, w, yerr=sd, color=PHASE_COL[ph_name], label=ph_name, capsize=2, edgecolor="white", error_kw=dict(lw=.8, color=GREY))
    axs[0].set_xticks(range(n), names, rotation=25, ha="right"); axs[0].axhline(0, color=GREY, lw=.8); axs[0].set(ylabel="Energy (model units)", title="Soft energy terms per phase"); axs[0].legend()
    if ph.has_rep:
        base = {k: T[k][segs["G1"]].mean() for k in names}; w2 = .38
        for i, k in enumerate(("S", "G2/M")):
            axs[1].bar(np.arange(n) + (i - .5) * w2, [T[q][segs[k]].mean() - base[q] for q in names], w2, color=PHASE_COL[k], label=f"{k} − G1", edgecolor="white")
        axs[1].set_xticks(range(n), names, rotation=25, ha="right"); axs[1].axhline(0, color=GREY, lw=.8); axs[1].set(ylabel="Δ energy (model units)", title="Change over the cell cycle"); axs[1].legend()
    else: axs[1].axis("off")
    x = ph.steps; w3 = max(3, len(x) // 80)
    for c, k in zip(CYCLE, names): axs[2].plot(x, rolling(T[k], w3), color=c, lw=2.2 if k == "Total" else 1.3, label=k)
    shade_phases(axs[2], ph, True); axs[2].set(xlabel="MC step", ylabel="Energy (model units)", xlim=(x[0], x[-1]), title="Terms over time (moving average)"); axs[2].legend(fontsize=8, ncol=2)
    return save_fig(fig, out, SUB, "energy_budget", "energy_budget_phases")

def fig_landscape(res, p, ph, out):
    """F(x) = -T ln P(x) of the order parameters, one curve per phase."""
    segs = ph.segments(); ops = order_parameters(res, p); k = len(ops)
    fig, axs = plt.subplots(1, k, figsize=(6.4 * k, 4.4), squeeze=False)
    for ax, (name, y) in zip(axs[0], ops.items()):
        y = np.asarray(y, float); r = y[ph.g1:]; lo, hi = np.min(r), np.max(r)
        if hi <= lo: hi = lo + 1
        for ph_name, s in segs.items():
            if len(y[s]) < 30: continue
            xc, F = landscape(y[s], p.T, lo, hi); ax.plot(xc, F, "-o", ms=3.5, lw=1.8, color=PHASE_COL[ph_name], label=f"{ph_name} (n={len(y[s])})")
        ax.set(xlabel=name, ylabel="F(x) = −T ln P(x)  (model units)", title=f"Free-energy landscape: {name.split(' (')[0].lower()}"); ax.legend()
    return save_fig(fig, out, SUB, "free_energy_landscape", "free_energy_landscape")
