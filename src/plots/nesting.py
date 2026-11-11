"""Families of loops: how nesting changes over the cell cycle."""
import numpy as np
import matplotlib.pyplot as plt
from .style import BLACK, PHASE_COL, PURPLE, GREY, BLUE, TEAL, GOLD, ROYAL, SLATE, shade_phases, kb, save_fig
from ..analysis.nesting import kind_of, order_counts, nesting_of, NAMES

SUB = "loop_families"
COLORS = [TEAL, SLATE, PURPLE, GOLD, ROYAL]      # orphans, parents, order 1, 2, 3+

def fig_nesting(nst, ph, out):
    """Loop counts per class over time, class shares per phase, nesting depth over time."""
    x = ph.steps; segs = ph.segments(); cnt = order_counts(nst)
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.4), gridspec_kw=dict(wspace=.3))
    axs[0].stackplot(x, cnt, colors=COLORS, labels=NAMES, alpha=.9, edgecolor="white", linewidth=.2); shade_phases(axs[0], ph, True)
    axs[0].set(xlabel="MC step", ylabel="Loops per sample", xlim=(x[0], x[-1]), title="Loops by family"); axs[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, -.18), fontsize=8)
    w = .8 / len(segs)
    for i, (name, sl) in enumerate(segs.items()):
        c = order_counts(nst, sl); frac = c.sum(1) / max(c.sum(), 1)
        axs[1].bar(np.arange(5) + (i - (len(segs) - 1) / 2) * w, frac, w, color=PHASE_COL[name], label=name, edgecolor="white")
    axs[1].set_xticks(range(5), ["orphan", "parent", "1", "2", "3+"]); axs[1].set(xlabel="Loop class (children: nesting order)", ylabel="Share of loops", title="Families per phase"); axs[1].legend()
    v = nst.valid; mean_o = (nst.order * v).sum(0) / np.maximum(v.sum(0), 1); deep = np.where(v, nst.order, 0).max(0)
    axs[2].plot(x, deep, color=ROYAL, lw=1, alpha=.6, label="deepest order"); axs[2].plot(x, mean_o, color=PURPLE, lw=1.4, label="mean order"); shade_phases(axs[2], ph, True)
    axs[2].set(xlabel="MC step", ylabel="Nesting order", ylim=(0, None), xlim=(x[0], x[-1]), title="Nesting depth"); axs[2].legend()
    return save_fig(fig, out, SUB, "loop_nesting", "loop_nesting")

def fig_parents_children(nst, ph, out):
    """Parent length vs loops inside, loop length per class, child/parent length ratio per phase."""
    segs = ph.segments(); fig, axs = plt.subplots(1, 3, figsize=(16.5, 4.4), gridspec_kw=dict(wspace=.3)); sl = slice(ph.g1, ph.n)
    v = nst.valid[:, sl]; ln, nd, o = kb(nst.length[:, sl][v]), nst.n_desc[:, sl][v], nst.order[:, sl][v]; kind = kind_of(nst.order[:, sl], nst.n_desc[:, sl])[v]; par = kind == 1
    axs[0].scatter(ln[par], nd[par], s=8, color=PURPLE, alpha=.25, edgecolor="none")
    if par.sum() > 20:
        e = np.quantile(ln[par], np.linspace(0, 1, 9)); idx = np.clip(np.digitize(ln[par], e[1:-1]), 0, 7); keep = [b for b in range(8) if (idx == b).any()]
        axs[0].plot([ln[par][idx == b].mean() for b in keep], [nd[par][idx == b].mean() for b in keep], "-o", color=BLACK, lw=2, ms=5, label="binned mean"); axs[0].legend()
    axs[0].set(xlabel="Parent loop length (kb)", ylabel="Loops inside the parent", title="Parents and their children")
    if len(ln):
        bins = np.logspace(np.log10(max(ln.min(), 1e-3) * .98), np.log10(ln.max() * 1.02), 25)
        for k in range(5):
            m = kind == k
            if m.sum() > 1: axs[1].hist(ln[m], bins=bins, density=True, histtype="step", lw=2, color=COLORS[k], label=f"{NAMES[k]} ({ln[m].mean():.0f} kb)")
        axs[1].set_xscale("log"); axs[1].legend(fontsize=8)
    axs[1].set(xlabel="Loop length (kb)", ylabel="Density", title="Loop length by family")
    for name, s in segs.items():
        vv = nst.valid[:, s]; r = (nst.length[:, s] / nst.parent_len[:, s])[vv & (nst.order[:, s] > 0)]; r = r[np.isfinite(r)]
        if len(r): axs[2].hist(r, bins=np.linspace(0, 1, 21), density=True, histtype="stepfilled", alpha=.35, color=PHASE_COL[name], edgecolor=PHASE_COL[name], label=f"{name} (mean {r.mean():.2f})")
    axs[2].set(xlabel="Child length / smallest enclosing loop", ylabel="Density", xlim=(0, 1), title="Child size per phase"); axs[2].legend()
    return save_fig(fig, out, SUB, "loop_parents_children", "loop_parents_children")
