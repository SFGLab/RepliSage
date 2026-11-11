"""CTCF input figures (BEDPE, BED or narrowPeak): anchor map, anchor statistics and binding profiles."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
from .style import save_fig, GREY, SLATE, ROSE, GOLD, PURPLE, GREEN, LEFT, RIGHT
from ..analysis.anchors import anchors_of, CONF_MIN, CONFIGS

SUB = "input"
ORIENT = dict(forward=LEFT, reverse=RIGHT, ambiguous=ROSE, unknown=GREY)      # L = forward, R = reverse CTCF
MARK = dict(forward=">", reverse="<", ambiguous="D", unknown="o")
CONF_COL = dict(convergent=PURPLE, tandem=GREEN, divergent=GOLD, uncertain=ROSE)
FMT = {"bedpe": "BEDPE loops", "bed": "BED peaks", "narrowpeak": "narrowPeak peaks"}

def _legend(ax, handles, **kw): ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=len(handles), fontsize=8.5, **kw)

def _box(ax, data, cols, labels):
    """Box plot with jittered points, one box per group."""
    bp = ax.boxplot(data, patch_artist=True, widths=.55, showfliers=False, medianprops=dict(color=SLATE, lw=1.4), whiskerprops=dict(color=GREY), capprops=dict(color=GREY))
    rng = np.random.default_rng(0)
    for k, (box, d, c) in enumerate(zip(bp["boxes"], data, cols), 1):
        box.set(facecolor=c, alpha=.3, edgecolor=c); ax.scatter(k + rng.uniform(-.18, .18, len(d)), d, s=12, color=c, alpha=.7, edgecolor="white", linewidth=.3)
    ax.set_xticks(range(1, len(data) + 1), labels)

def fig_anchor_map(ctx, geo, out):
    """Loop arcs (BEDPE), anchor strength with CTCF orientation, and how sure each orientation call is."""
    a = anchors_of(ctx.binding); x = a.bp / 1e6; known = a.has_orientation
    rows = (["arc"] if a.has_loops else []) + ["str"] + (["conf"] if known else [])
    ratios = {"arc": 2.2, "str": 1.3, "conf": .8}
    fig, axs = plt.subplots(len(rows), 1, figsize=(11.5, 2.3 + 2.1 * len(rows)), sharex=True, gridspec_kw=dict(height_ratios=[ratios[r] for r in rows], hspace=.5)); axs = np.atleast_1d(axs)
    ax = dict(zip(rows, axs)); lo, hi = ctx.region[0] / 1e6, ctx.region[1] / 1e6
    if a.has_loops:
        L, R = a.left_bp / 1e6, a.right_bp / 1e6; w = a.loop_strength / (a.loop_strength.max() or 1); top = (R - L).max()
        for i in np.argsort(-(R - L)):                                            # long loops first, short ones on top
            ax["arc"].add_patch(Arc(((L[i] + R[i]) / 2, 0), R[i] - L[i], 2 * (R[i] - L[i]), theta1=0, theta2=180, lw=.8 + 2.4 * w[i], color=CONF_COL[a.config[i]], alpha=.75))
        ax["arc"].set(ylim=(0, top * 1.05), ylabel="Loop length (Mb)", title=f"{len(L)} loops: height = length")
        if known: _legend(ax["arc"], [Line2D([], [], color=CONF_COL[c], lw=3, label=c) for c in CONFIGS])
    o, conf = a.orient, a.confidence
    for name in ("unknown", "ambiguous", "reverse", "forward"):
        m = o == name
        if not m.any(): continue
        ax["str"].vlines(x[m], 0, a.strength[m], color=ORIENT[name], lw=1.2, alpha=.75)
        ax["str"].scatter(x[m], a.strength[m], marker=MARK[name], s=26 + 40 * conf[m], color=ORIENT[name], edgecolor="white", linewidth=.5, alpha=.9, zorder=3)
    ax["str"].set(ylabel="Anchor strength", title=f"{len(x)} anchors: strength")
    if known: _legend(ax["str"], [Line2D([], [], marker=MARK[k], ls="", color=ORIENT[k], label=k, markersize=7) for k in ORIENT])
    else: ax["str"].text(.99, .95, "no orientation in the file: anchors count as both L and R", transform=ax["str"].transAxes, ha="right", va="top", fontsize=8.5, color=GREY)
    if known:
        m = a.p_rev >= 0; ax["conf"].bar(x[m], conf[m], width=geo.bead_kb / 1e3 * .8, color=[ORIENT[k] for k in o[m]])
        ax["conf"].axhline(CONF_MIN, color=GREY, ls="--", lw=.8); ax["conf"].set(ylim=(0, 1.05), ylabel="Confidence", title=f"Orientation confidence (dashed = {CONF_MIN})")
    axs[-1].set(xlim=(lo, hi), xlabel=f"{geo.chrom} position (Mb)")
    return save_fig(fig, out, SUB, "anchor_map", "anchor_map")

def fig_anchor_stats(ctx, out):
    """Strength by orientation, orientation confidence, and loop configuration (or orientation) counts."""
    a = anchors_of(ctx.binding)
    if not a.has_orientation: return
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(wspace=.3)); o = a.orient
    names = [n for n in ORIENT if (o == n).any()]
    _box(axs[0], [a.strength[o == n] for n in names], [ORIENT[n] for n in names], [f"{n}\n(n={(o == n).sum()})" for n in names]); axs[0].set(title="Strength by orientation", ylabel="Anchor strength")
    axs[1].hist(a.confidence[a.p_rev >= 0], bins=np.linspace(0, 1, 21), color=PURPLE, alpha=.75, edgecolor="white"); axs[1].axvline(CONF_MIN, color=ORIENT["forward"], ls="--", label=f"ambiguous below {CONF_MIN}")
    axs[1].set(title="Orientation confidence", xlabel="|2p − 1|  (0 = coin flip, 1 = certain)", ylabel="Anchors"); axs[1].legend(fontsize=8.5)
    keys, cols = (CONFIGS, CONF_COL) if a.has_loops else (names, ORIENT)
    cnt = [int((a.config == k).sum()) if a.has_loops else int((o == k).sum()) for k in keys]
    bars = axs[2].bar(keys, cnt, color=[cols[k] for k in keys], alpha=.85)
    for b, c in zip(bars, cnt): axs[2].text(b.get_x() + b.get_width() / 2, c, f"{c / max(sum(cnt), 1):.0%}", ha="center", va="bottom", fontsize=9, color=GREY)
    axs[2].set(title="Loop configuration" if a.has_loops else "Anchor orientation", ylabel="Loops" if a.has_loops else "Anchors")
    return save_fig(fig, out, SUB, "anchor_stats", "anchor_stats")

def _tracks(ax, x, L, R, bar=None):
    """L up / R mirrored down; filled profile (bar = stem width, for the raw sites)."""
    for v, c in ((L, LEFT), (-R, RIGHT)):
        if bar: ax.vlines(x[v != 0], 0, v[v != 0], color=c, lw=1.4)
        else: ax.fill_between(x, v, color=c, alpha=.35, lw=0); ax.plot(x, v, color=c, lw=.9)
    ax.axhline(0, color=GREY, lw=.8)
    ax.text(.01, .96, "L: left leg m", transform=ax.transAxes, color=LEFT, fontsize=9, va="top"); ax.text(.01, .04, "R: right leg n (mirrored)", transform=ax.transAxes, color=RIGHT, fontsize=9, va="bottom")

def fig_input(ctx, p, geo, out):
    """Binding profiles along the region (raw sites, then the smoothed model input) with strength and spacing histograms."""
    b, a = ctx.binding, anchors_of(ctx.binding); x = geo.xs
    fig = plt.figure(figsize=(14, 5.6)); gs = fig.add_gridspec(2, 2, width_ratios=[3, 1.1], hspace=.65, wspace=.22)
    top, bot = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0], sharex=None)
    _tracks(top, x, b.L0, b.R0, bar=True); top.set(ylabel="Signal", title=f"Raw sites on the beads ({b.n_sites} sites)")
    sg = f"smoothed, σ = {b.sigma:.1f} beads ({b.sigma * geo.bead_kb:.0f} kb)" if b.sigma > 0 else "not smoothed"
    _tracks(bot, x, p.L, p.R); bot.set(ylabel="Signal", xlabel=f"{geo.chrom} position (Mb)", title=f"Model input: {sg}")
    for ax in (top, bot): ax.set_xlim(x[0], x[-1] + geo.bead_kb / 1e3)
    h1, h2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    if a.has_loops:
        kb = (a.right_bp - a.left_bp) / 1e3; h1.hist(kb, bins=25, color=PURPLE, alpha=.75, edgecolor="white"); h1.axvline(kb.mean(), color=LEFT, ls="--", lw=1.5, label=f"mean {kb.mean():.0f} kb")
        h1.set(title="Loop lengths", xlabel="Loop length (kb)", ylabel="Loops"); h1.legend(fontsize=8.5)
    else:
        h1.hist(a.strength, bins=25, color=PURPLE, alpha=.75, edgecolor="white"); h1.set(title="Peak strength", xlabel="Strength", ylabel="Peaks")
    gap = np.diff(np.sort(a.bp)) / 1e3
    if len(gap) > 2: h2.hist(gap[gap > 0], bins=25, color=RIGHT, alpha=.75, edgecolor="white"); h2.axvline(np.median(gap), color=SLATE, ls="--", lw=1.5, label=f"median {np.median(gap):.0f} kb"); h2.legend(fontsize=8.5)
    h2.set(title="Spacing of neighbouring anchors", xlabel="Distance (kb)", ylabel="Anchors")
    return save_fig(fig, out, SUB, "ctcf_input", "loops_input")
