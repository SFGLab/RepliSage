"""LEF figures: trajectories, coverage, loop length, anchoring on CTCF sites."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from .style import bin_time, BLACK, ROYAL, GREY, ROSE, SLATE, PINK, BLUE, TEAL, GOLD, PURPLE, ORANGE, GREEN, LEFT, RIGHT, CMAP_SEQ, PHASE_COL, shade_phases, rolling, save_fig
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

def overlay_front(ax, p, ph, n_beads, f_rep_T):
    """Dashed line: replication front (replicated state = 0.5) in MC steps."""
    if f_rep_T is None: return
    k = max(1, f_rep_T.shape[0] // 400); F = f_rep_T[::k].astype(float)
    t = p.t_rep + np.arange(0, f_rep_T.shape[0], k)
    ax.contour(t, np.arange(n_beads), F.T, levels=[.5], colors=["white"], linewidths=3.6)
    ax.contour(t, np.arange(n_beads), F.T, levels=[.5], colors=[BLACK], linewidths=1.5, linestyles="--")

def coh_traj_plot(ms, ns, N_beads, path, ph, p=None, f_rep_T=None, jump_threshold=200, min_stable_time=10):
    """Filled arcs of every LEF (between its legs) over time; jumps and short-lived loops are hidden."""
    N = len(ms); t = ph.steps
    fig, (ax, side) = plt.subplots(1, 2, figsize=(11, 6.5), gridspec_kw=dict(width_ratios=[14, 1], wspace=.02), sharey=True)
    n1 = p.N_lef if p is not None else N
    for i in range(N):
        m = np.ma.masked_array(ms[i], mask=~_stable(ms[i], ns[i], jump_threshold, min_stable_time)); n = np.ma.masked_array(ns[i], mask=m.mask)
        pal = MIX if (p is None or p.N_lef2 == 0) else (FAM1 if i < n1 else FAM2)
        c = pal[i % len(pal)]
        ax.fill_between(t, m, n, color=c, alpha=.22, lw=0)                 # soft loop body
        ax.plot(t, m, color=c, lw=.5, alpha=.9); ax.plot(t, n, color=c, lw=.5, alpha=.9)   # crisp legs
    shade_phases(ax, ph, labels=True)
    if p is not None: overlay_front(ax, p, ph, N_beads, f_rep_T)
    ax.set_xlabel("MC step"); ax.set_ylabel("Bead"); ax.set_ylim(N_beads, 0); ax.grid(False)
    if p is not None:
        b = np.arange(N_beads); side.barh(b, p.L, color=ROYAL, height=1); side.barh(b, -p.R, color=BLACK, height=1)
        side.set_title("CTCF", fontsize=9); side.set_xticks([]); side.grid(False); side.spines["left"].set_visible(False)
    return save_fig(fig, path, SUB, "LEFs", "lef_traj")

def fig_coverage(res, p, ph, out, f_rep_T=None):
    """Number of loops covering each bead over time, and its time average."""
    cov = coverage(res.Ms, res.Ns, p.N_beads); cov_img = bin_time(cov.astype(float))[0]
    fig, (ax, side) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw=dict(width_ratios=[14, 2.2], wspace=.04), sharey=True)
    ext = [0, ph.x(ph.n), p.N_beads, 0]
    im = ax.imshow(cov_img, aspect="auto", cmap=CMAP_SEQ, extent=ext, vmax=max(1, np.percentile(cov, 99)), interpolation="nearest")
    cax = ax.inset_axes([.02, .95, .22, .025]); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("Loops over bead", fontsize=8); cb.ax.tick_params(labelsize=7)
    shade_phases(ax, ph, labels=True); overlay_front(ax, p, ph, p.N_beads, f_rep_T); ax.grid(False)
    ax.set_xlabel("MC step"); ax.set_ylabel("Bead")
    prof = cov[:, ph.g1:].mean(1); side.fill_betweenx(np.arange(p.N_beads), 0, prof, color=PURPLE, alpha=.45); side.plot(prof, np.arange(p.N_beads), color=PURPLE, lw=1)
    side.set_xlabel("Mean"); side.set_title("Profile", fontsize=9); side.set_xticks([round(float(prof.max()), 1)]); side.tick_params(axis="x", labelsize=8)
    return save_fig(fig, out, SUB, "loop_coverage", "lef_coverage")

def fig_loop_length(Ls, ph, out):
    """Mean loop length (95 % CI over LEFs) and its distribution (violin per phase, or histogram without replication)."""
    mean = Ls.mean(0); ci = 1.96 * Ls.std(0) / np.sqrt(Ls.shape[0]); t = ph.steps
    fig, (ax, vx) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[3, 1.2]))
    ax.fill_between(t, mean - ci, mean + ci, color=BLUE, alpha=.18, lw=0); ax.plot(t, mean, color=BLUE, lw=.7, alpha=.5)
    ax.plot(t, rolling(mean, max(3, ph.n // 60)), color=BLUE, lw=2); shade_phases(ax, ph, labels=True)
    ax.set_xlabel("MC step"); ax.set_ylabel("Loop length (beads)"); ax.set_title("Mean ± 95% CI")
    _violin(vx, {k: Ls[:, s].ravel() for k, s in ph.segments().items()}, "Loop length")
    return save_fig(fig, out, SUB, "loop_length", "loop_length")

def _violin(ax, data, ylabel):
    """Violins (median line, IQR box) per phase; a single group becomes a histogram."""
    labs = [k for k, v in data.items() if len(v) > 3]
    if len(labs) < 2:
        v = next(iter(data.values())); ax.hist(v[v > 0] if (v > 0).any() and not ylabel.startswith("Δ") else v, bins=40, color=BLUE, alpha=.75); ax.set_xlabel(ylabel); ax.set_ylabel("Count"); return
    parts = ax.violinplot([data[k] for k in labs], showextrema=False, widths=.8)
    for b, k in zip(parts["bodies"], labs): b.set_facecolor(PHASE_COL.get(k, PURPLE)); b.set_edgecolor(SLATE); b.set_alpha(.6)
    for i, k in enumerate(labs, 1):
        q1, med, q3 = np.percentile(data[k], [25, 50, 75]); ax.vlines(i, q1, q3, color=SLATE, lw=5); ax.scatter(i, med, color="white", s=14, zorder=3)
    ax.set_xticks(range(1, len(labs) + 1), labs); ax.set_ylabel(ylabel)
    try:
        ax.set_title(f"Kruskal p = {kruskal(*[data[k][::max(1, len(data[k]) // 2000)] for k in labs]).pvalue:.1e}", fontsize=9, loc="right", color=GREY)
    except ValueError: pass

def plot_loop_length(Ls, ph, out):
    """Loop-length figure plus the stand-alone violin (kept from the published version)."""
    fig_loop_length(Ls, ph, out)
    if ph.has_rep:
        fig, ax = plt.subplots(figsize=(4.6, 3.8)); _violin(ax, {k: Ls[:, s].ravel() for k, s in ph.segments().items()}, "Loop length (beads)")
        save_fig(fig, out, SUB, "loop_length_violin", "violin_loop_length")

def fig_binding(res, p, ph, out):
    """Fraction of LEFs bound / on CTCF over time, and per-bead leg occupancy against CTCF strength."""
    ls = lef_series(res, p); t = ph.steps; w = max(3, ph.n // 60)
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 6.2), gridspec_kw=dict(height_ratios=[1, 1.1]))
    for k, lab, c in (("bound", "Bound", PURPLE), ("on_L", "Left leg on CTCF", LEFT), ("on_R", "Right leg on CTCF", RIGHT)):
        a.plot(t, rolling(ls[k], w), color=c, label=lab)
    shade_phases(a, ph, labels=True); a.set_ylim(0, 1.02); a.set_xlabel("MC step"); a.set_ylabel("Fraction of LEFs"); a.legend(ncol=1, loc="center right")
    ol, orr = leg_occupancy(res.Ms, res.Ns, p.N_beads, slice(ph.g1, ph.n)); x = np.arange(p.N_beads)
    b.bar(x, ol, color=LEFT, width=1, label="Left leg"); b.bar(x, -orr, color=RIGHT, width=1, label="Right leg")
    sc = max(ol.max(), orr.max(), 1e-9) / max(p.L.max(), p.R.max(), 1e-9)
    b.scatter(x[p.L > 0], p.L[p.L > 0] * sc, marker="v", s=18, color=SLATE, edgecolor="none", label="CTCF (L)")
    b.scatter(x[p.R > 0], -p.R[p.R > 0] * sc, marker="^", s=18, color=GREY, edgecolor="none", label="CTCF (R)")
    b.set_xlabel("Bead"); b.set_ylabel("Legs per sample"); b.legend(ncol=4, loc="upper right", fontsize=8)
    return save_fig(fig, out, SUB, "lef_binding", "lef_binding")

def fig_input(p, geo, out):
    """CTCF binding potentials along the region: left (up) and right (down) sites."""
    fig, ax = plt.subplots(figsize=(11, 2.8)); x = geo.xs
    ax.vlines(x, 0, p.L, color=LEFT, lw=1.4, label="Left sites (L)"); ax.vlines(x, 0, -p.R, color=RIGHT, lw=1.4, label="Right sites (R)")
    ax.axhline(0, color=GREY, lw=.8); ax.set_xlabel(geo.label); ax.set_ylabel("Binding strength"); ax.legend(ncol=2, loc="upper right")
    return save_fig(fig, out, SUB, "ctcf_input", "loops_input")
