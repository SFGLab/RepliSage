"""LEF figures: trajectories, coverage, loop length, anchoring on CTCF sites."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from .style import kb, unit_label, genome_axis, bin_time, BLACK, ROYAL, GREY, ROSE, SLATE, PINK, BLUE, TEAL, GOLD, PURPLE, ORANGE, GREEN, LEFT, RIGHT, CMAP_SEQ, PHASE_COL, shade_phases, rolling, save_fig
from ..preproc.bedpe import load_region
from ..analysis.lef_stats import coverage, lef_series, leg_occupancy, bound_mask

SUB = "MCMC_diagnostics"
FAM1 = [BLACK, SLATE, GREY, "#B0B0B0"]                       # family 1: black / grey shades
FAM2 = [ROYAL, "#6B0B1F", ROSE, "#D6336C"]                   # family 2: red shades
MIX = [BLACK, ROYAL, GREY, "#6B0B1F", SLATE, ROSE, "#B0B0B0", "#D6336C"]   # red / black / grey cycle

def _stable(m, n, jump, min_len):
    """True where a LEF stays put (no jumps) for at least min_len samples."""
    good = np.ones(len(m), bool); good[1:] = (np.abs(np.diff(m)) < jump) & (np.abs(np.diff(n)) < jump)
    d = np.diff(np.r_[0, good.view(np.int8), 0])
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s < min_len: good[s:e] = False
    return good

def _front(f_rep_T, p, n_beads):
    """Single-valued replication front: MC step at which each bead is first replicated (median-smoothed along the genome)."""
    from scipy.ndimage import median_filter
    t = np.argmax(f_rep_T > 0, axis=0).astype(float); return p.t_rep + median_filter(t, size=max(5, n_beads // 25), mode="nearest")

def overlay_front(ax, p, ph, n_beads, f_rep_T):
    """Dashed line: MC step at which every bead gets replicated (the replication front)."""
    if f_rep_T is None: return
    y = np.arange(n_beads) + .5; fr = _front(f_rep_T, p, n_beads)
    ax.plot(fr, y, color="white", lw=3.4); ax.plot(fr, y, color=BLACK, lw=1.4, ls="--")

def _lef_colours(k):
    """k well-separated colours (hue-spread, shuffled so neighbours differ)."""
    from matplotlib.colors import hsv_to_rgb
    c = [hsv_to_rgb(((i * .382) % 1, .75 - .2 * (i % 2), .85 - .12 * (i % 3))) for i in range(k)]
    return c

def fig_kymograph(res, p, ph, f_rep_T, out, max_lefs=36):
    """LEF kymograph (one colour per LEF, vertical bar = loop at that time) above the mean loop size over time."""
    from matplotlib.collections import LineCollection
    Ms, Ns = res.Ms, res.Ns; ok = bound_mask(Ms, Ns); t = ph.steps; nT = Ms.shape[1]
    ids = np.unique(np.linspace(0, Ms.shape[0] - 1, min(max_lefs, Ms.shape[0])).astype(int)); cols = _lef_colours(len(ids))
    fig = plt.figure(figsize=(11.5, 8.4)); gs = fig.add_gridspec(2, 2, height_ratios=[3.4, 1.3], width_ratios=[14, 1.6], hspace=.1, wspace=.03)
    ax = fig.add_subplot(gs[0, 0]); side = fig.add_subplot(gs[0, 1], sharey=ax); bot = fig.add_subplot(gs[1, 0], sharex=ax); hist = fig.add_subplot(gs[1, 1], sharey=bot)
    lw = float(np.clip(520 / nT, .6, 3))                                       # bar width follows sample density
    for c, i in zip(cols, ids):
        r = ok[i]; seg = np.stack([np.c_[t[r], Ms[i, r]], np.c_[t[r], Ns[i, r]]], axis=1)
        ax.add_collection(LineCollection(seg, colors=[c], linewidths=lw, alpha=.55))
        ax.plot(t[r], Ms[i, r], ".", ms=1.4, color=c, alpha=.9); ax.plot(t[r], Ns[i, r], ".", ms=1.4, color=c, alpha=.9)   # legs
    shade_phases(ax, ph, labels=True); overlay_front(ax, p, ph, p.N_beads, f_rep_T)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(p.N_beads, 0); genome_axis(ax, p.N_beads); ax.grid(False); plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title(f"{len(ids)} of {Ms.shape[0]} LEFs, one colour each: a vertical bar joins the two legs of a bound LEF at that time", fontsize=9.5, color=GREY, loc="left", pad=24)
    b = np.arange(p.N_beads); side.barh(b, p.L, color=LEFT, height=1); side.barh(b, -p.R, color=RIGHT, height=1)
    side.set_title("CTCF", fontsize=9); side.set_xticks([]); side.grid(False); side.spines["left"].set_visible(False); plt.setp(side.get_yticklabels(), visible=False); side.tick_params(left=False)
    ls = lef_series(res, p); w = max(3, nT // 40); lb = kb(ls["loop_bound"]); la = kb(ls["loop"])
    shade_phases(bot, ph); bot.plot(t, la, color=GREY, lw=.7, alpha=.5); bot.plot(t, rolling(la, w), color=GREY, lw=1.8, label="all LEFs (unbound = 0)")
    bot.plot(t, lb, color=ROYAL, lw=.7, alpha=.35); bot.plot(t, rolling(lb, w), color=ROYAL, lw=2.2, label="bound LEFs only")
    bot.set_xlabel("MC step"); bot.set_ylabel(unit_label("Mean loop size")); bot.legend(frameon=False, fontsize=8, loc="upper right", ncol=2); bot.set_ylim(0, 1.3 * max(lb.max(), la.max()))
    v = lb[ph.g1:] if nT - ph.g1 > 5 else lb; hist.hist(v, bins=20, orientation="horizontal", color=ROYAL, alpha=.8)
    hist.set_xticks([]); hist.set_title("spread", fontsize=9); hist.spines["left"].set_visible(False); plt.setp(hist.get_yticklabels(), visible=False); hist.tick_params(left=False)
    return save_fig(fig, out, SUB, "LEFs", "lef_traj")

def _smooth_cov(m, s=(1.2, 1.0)):
    """Light Gaussian smoothing (beads, time) so the map shows structure, not single-sample noise."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(m, s)

def fig_coverage(res, p, ph, out, f_rep_T=None):
    """Number of loops covering each bead over time, and its time average."""
    cov = coverage(res.Ms, res.Ns, p.N_beads); cov_img = _smooth_cov(bin_time(cov.astype(float), 250)[0])
    fig, (ax, side) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[14, 2.2], wspace=.04), sharey=True)
    ext = [0, ph.x(ph.n), p.N_beads, 0]
    im = ax.imshow(cov_img, aspect="auto", cmap=CMAP_SEQ, extent=ext, vmax=max(.5, np.percentile(cov_img, 99.5)), interpolation="nearest")
    cax = ax.inset_axes([.0, 1.14, .22, .025]); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=7); ax.text(.235, 1.14, "loops over a bead", transform=ax.transAxes, fontsize=8, va="bottom", color=GREY)
    shade_phases(ax, ph, labels=True); overlay_front(ax, p, ph, p.N_beads, f_rep_T); ax.grid(False)
    ax.set_xlabel("MC step"); genome_axis(ax, p.N_beads)
    prof = cov[:, ph.g1:].mean(1); side.fill_betweenx(np.arange(p.N_beads), 0, prof, color=PURPLE, alpha=.45); side.plot(prof, np.arange(p.N_beads), color=PURPLE, lw=1)
    side.set_xlabel("Mean"); side.set_title("Profile", fontsize=9); side.set_xticks([round(float(prof.max()), 1)]); side.tick_params(axis="x", labelsize=8)
    return save_fig(fig, out, SUB, "loop_coverage", "lef_coverage")

def fig_loop_length(Ls, ph, out):
    """Mean loop length (95 % CI over LEFs) and its distribution (box plot per phase, or histogram without replication)."""
    Ls = kb(Ls); mean = Ls.mean(0); ci = 1.96 * Ls.std(0) / np.sqrt(Ls.shape[0]); t = ph.steps
    fig, (ax, vx) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[3, 1.2]))
    ax.fill_between(t, mean - ci, mean + ci, color=BLUE, alpha=.18, lw=0); ax.plot(t, mean, color=BLUE, lw=.7, alpha=.5)
    ax.plot(t, rolling(mean, max(3, ph.n // 60)), color=BLUE, lw=2); shade_phases(ax, ph, labels=True)
    ax.set_xlabel("MC step"); ax.set_ylabel(unit_label("Loop length")); ax.set_title("Mean loop length per LEF ± 95% CI (unbound LEFs count as 0)", pad=26, fontsize=11)
    _violin(vx, {k: Ls[:, s].ravel() for k, s in ph.segments().items()}, unit_label("Loop length"), log=True)
    return save_fig(fig, out, SUB, "loop_length", "loop_length")

def _box(ax, data, ylabel, log=False):
    """Box plot per phase: box = middle 50 %, line = median, dot = mean, whiskers = 5th-95th percentile.
    A single group becomes a histogram. log=True: log10 axis (zeros dropped)."""
    if log: data = {k: np.log10(v[v > 0]) for k, v in data.items()}
    labs = [k for k, v in data.items() if len(v) > 3]
    if len(labs) < 2:
        v = next(iter(data.values())); ax.hist(v[v > 0] if (v > 0).any() and not ylabel.startswith("Δ") else v, bins=40, color=BLUE, alpha=.75); ax.set_xlabel(ylabel); ax.set_ylabel("Count"); return
    bp = ax.boxplot([data[k] for k in labs], whis=(5, 95), showfliers=False, showmeans=True, patch_artist=True, widths=.55,
                    medianprops=dict(color=BLACK, lw=2), whiskerprops=dict(color=SLATE), capprops=dict(color=SLATE),
                    meanprops=dict(marker="o", markerfacecolor="white", markeredgecolor=BLACK, markersize=6))
    for b, k in zip(bp["boxes"], labs): b.set_facecolor(PHASE_COL.get(k, PURPLE)); b.set_alpha(.65); b.set_edgecolor(SLATE)
    ax.set_xticks(range(1, len(labs) + 1), labs); ax.set_ylabel(ylabel)
    if log:
        lo, hi = ax.get_ylim(); t = np.arange(np.ceil(lo), np.floor(hi) + 1); ax.set_yticks(t, [f"{10 ** v:g}" for v in t]); ax.set_ylabel(ylabel + ", log scale")
    try:
        ax.set_title(f"Kruskal p = {kruskal(*[data[k][::max(1, len(data[k]) // 2000)] for k in labs]).pvalue:.1e}", fontsize=9, loc="right", color=GREY)
    except ValueError: pass

_violin = _box   # old name kept for imports

def plot_loop_length(Ls, ph, out):
    """Loop-length figure plus the stand-alone box plot."""
    fig_loop_length(Ls, ph, out)
    if ph.has_rep:
        fig, ax = plt.subplots(figsize=(4.6, 3.8)); _violin(ax, {k: kb(Ls[:, s].ravel()) for k, s in ph.segments().items()}, unit_label("Loop length"), log=True)
        save_fig(fig, out, SUB, "loop_length_violin", "violin_loop_length")

def fig_binding(res, p, ph, out):
    """Fraction of LEFs bound / on CTCF over time, and per-bead leg occupancy against CTCF strength."""
    ls = lef_series(res, p); t = ph.steps; w = max(3, ph.n // 60)
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 6.8), gridspec_kw=dict(height_ratios=[1, 1.1], hspace=.5))
    for k, lab, c in (("bound", "Bound", PURPLE), ("on_L", "Left leg on CTCF", LEFT), ("on_R", "Right leg on CTCF", RIGHT)):
        a.plot(t, rolling(ls[k], w), color=c, label=lab)
    shade_phases(a, ph, labels=True); a.set_ylim(0, 1.02); a.set_xlabel("MC step"); a.set_ylabel("Fraction of LEFs"); a.legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.12))
    ol, orr = leg_occupancy(res.Ms, res.Ns, p.N_beads, slice(ph.g1, ph.n)); x = np.arange(p.N_beads)
    b.bar(x, ol, color=LEFT, width=1, label="Left leg"); b.bar(x, -orr, color=RIGHT, width=1, label="Right leg")
    sc = max(ol.max(), orr.max(), 1e-9) / max(p.L.max(), p.R.max(), 1e-9)
    b.scatter(x[p.L > 0], p.L[p.L > 0] * sc, marker="v", s=18, color=SLATE, edgecolor="none", label="CTCF (L)")
    b.scatter(x[p.R > 0], -p.R[p.R > 0] * sc, marker="^", s=18, color=GREY, edgecolor="none", label="CTCF (R)")
    genome_axis(b, p.N_beads, "x"); b.set_ylabel("Legs per sample"); b.legend(ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.0), fontsize=8)
    return save_fig(fig, out, SUB, "lef_binding", "lef_binding")

def fig_input(p, geo, out):
    """CTCF binding potentials along the region: left (up) and right (down) sites."""
    fig, ax = plt.subplots(figsize=(11, 2.8)); x = geo.xs
    ax.vlines(x, 0, p.L, color=LEFT, lw=1.4, label="Left sites (L)"); ax.vlines(x, 0, -p.R, color=RIGHT, lw=1.4, label="Right sites (R)")
    ax.axhline(0, color=GREY, lw=.8); ax.set_xlabel(geo.label); ax.set_ylabel("Binding strength"); ax.legend(ncol=2, loc="upper right")
    return save_fig(fig, out, SUB, "ctcf_input", "loops_input")

def fig_anchors(ctx, p, geo, out):
    """Loops of the BEDPE file as arcs, their anchors coloured by CTCF orientation, and the resulting left / right binding potentials."""
    from matplotlib.colors import LinearSegmentedColormap
    df = load_region(ctx.bedpe, ctx.region, ctx.chrom); c1, c2 = (df[1] + df[2]) / 2e6, (df[4] + df[5]) / 2e6; w = df[6].fillna(df[6].median()).values.astype(float)
    has_o = df.shape[1] > 8; o = np.r_[df[7], df[8]].astype(float) if has_o else np.full(2 * len(df), .5)     # share of right-ward CTCF per anchor
    cm = LinearSegmentedColormap.from_list("lr", [LEFT, "#CFCFCF", RIGHT]); wn = (w - w.min()) / (np.ptp(w) or 1)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw=dict(height_ratios=[2.6, 1.3], hspace=.1))
    th = np.linspace(0, np.pi, 60)
    for a, b, k in zip(c1, c2, wn):
        h = (abs(b - a) * 1e3) ** .6; ax.plot((a + b) / 2 + (b - a) / 2 * np.cos(th), h * np.sin(th), color=PURPLE, alpha=.18 + .5 * k, lw=.6 + 1.6 * k)
    xs = np.r_[c1, c2]; sz = 14 + 60 * np.r_[wn, wn]
    ax.scatter(xs, np.zeros_like(xs), s=sz, c=cm(o) if has_o else "#9A9A9A", edgecolor="white", linewidth=.6, zorder=3, clip_on=False)
    ax.set_ylim(-.04 * ax.get_ylim()[1], None); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.grid(False)
    ax.set_title(f"{len(df)} loops in the region: arc = loop, dot = anchor (size and darkness = loop strength)", fontsize=10.5, loc="left", pad=8)
    if has_o:
        for v, lab in ((0, "left-ward CTCF (L)"), (.5, "mixed"), (1, "right-ward CTCF (R)")): ax.scatter([], [], s=50, color=cm(float(v)), label=lab)
        ax.legend(ncol=3, loc="upper right", fontsize=8.5)
    else: ax.text(.99, .95, "no orientation in the file: anchors count as both L and R", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=GREY)
    bx.vlines(geo.xs, 0, p.L, color=LEFT, lw=1.4, label="L: where the left leg m can bind"); bx.vlines(geo.xs, 0, -p.R, color=RIGHT, lw=1.4, label="R: where the right leg n can bind")
    bx.axhline(0, color=GREY, lw=.8); bx.set_ylabel("Binding strength\non the beads"); bx.set_xlabel(geo.label); bx.legend(ncol=2, loc="upper right", fontsize=8.5)
    return save_fig(fig, out, SUB, "loop_anchors", "loop_anchors")
