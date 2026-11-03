"""Replication figures: fork trajectories, replicated fraction, initiation rate, timing validation."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from .style import fork_layer, s_window, BLACK, ROYAL, GREY, BLUE, TEAL, GOLD, PURPLE, ORANGE, SLATE, CMAP_REP, CMAP_SEQ, rolling, save_fig
from ..analysis.rep_stats import timing_profile, global_fraction, quantile_time, infer_origins

SUB = "replication_simulation"

def _t(p, T):
    """MC-step axis of the replication simulation (it spans the S phase)."""
    return p.t_rep + np.linspace(0, p.rep_duration, T)

def fig_forks(rep, p, geo, out):
    """Replicated DNA (left) and fork trajectories with inferred origins (right), genome × time."""
    rf = (rep.rep_frac > 0).astype(float); T = rf.shape[1]; t = _t(p, T); ext = [geo.xs[0], geo.xs[-1], t[0], t[-1]]
    tim = timing_profile(rep.rep_frac); org = infer_origins(tim)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True, gridspec_kw=dict(wspace=.06))
    a.imshow(rf.T, origin="lower", aspect="auto", cmap=CMAP_REP, extent=ext, vmin=0, vmax=1.15, interpolation="nearest"); a.set_title("Replicated DNA")
    forks = (np.asarray(rep.l_forks) > 0) | (np.asarray(rep.r_forks) > 0)
    img = fork_layer(forks)
    b.imshow(rf.T, origin="lower", aspect="auto", cmap=CMAP_REP, extent=ext, vmin=0, vmax=2.6, interpolation="nearest")
    b.imshow(img, origin="lower", aspect="auto", extent=ext, interpolation="nearest")
    b.scatter(geo.xs[org], t[0] + tim[org] * (t[-1] - t[0]), marker="^", s=38, color=ROYAL, edgecolor="white", linewidth=.7, zorder=3, label=f"Origins ({len(org)})")
    b.set_title("Forks"); b.legend(loc="lower right", fontsize=8.5)
    for ax in (a, b): ax.grid(False); ax.set_xlabel(geo.label); s_window(ax, t[0], t[-1], text=ax is b)
    a.set_ylabel("MC step")
    return save_fig(fig, out, SUB, "replication_forks", "rep_forks")

def fig_fraction(rep, p, out):
    """Replicated fraction over time and the number of active forks."""
    f = global_fraction(rep.rep_frac); T = len(f); t = _t(p, T); forks = ((np.asarray(rep.l_forks) > 0) | (np.asarray(rep.r_forks) > 0)).sum(0)
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.5, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1], hspace=.1))
    a.plot(t, f, color=ROYAL, lw=2.2)
    for q in (.1, .5, .9):
        x = t[0] + quantile_time(f, q) * (t[-1] - t[0]); a.plot([x], [q], "o", color=SLATE, ms=5); a.annotate(f"T{int(q * 100)}", (x, q), (6, -12), textcoords="offset points", fontsize=8.5)
    a.set_ylabel("Replicated fraction"); a.set_ylim(0, 1.03)
    fk = rolling(np.resize(forks, T), 7); b.fill_between(t, fk, color=BLUE, alpha=.25, lw=0); b.plot(t, fk, color=BLUE, lw=2)
    b.set_ylabel("Active forks"); b.set_xlabel("MC step"); s_window(a, t[0], t[-1], "x"); s_window(b, t[0], t[-1], "x", text=False)
    return save_fig(fig, out, SUB, "replication_fraction", "rep_fraction")

def fig_initiation(rep, p, geo, out):
    """Initiation rate I(x,t) with firing profile along the genome and in time."""
    I = np.asarray(rep.init_rate); T = I.shape[1]; t = _t(p, T); ext = [geo.xs[0], geo.xs[-1], t[0], t[-1]]
    tim = timing_profile(rep.rep_frac); org = infer_origins(tim)
    fig = plt.figure(figsize=(10.5, 6)); gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 4], hspace=.04, wspace=.04)
    ax, top, rt = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[0, 0], sharex=None), fig.add_subplot(gs[1, 1], sharey=None)
    im = ax.imshow(I.T, origin="lower", aspect="auto", cmap=CMAP_SEQ, extent=ext, vmax=np.percentile(I[I > 0], 99.5) if (I > 0).any() else 1, interpolation="nearest")
    ax.scatter(geo.xs[org], t[0] + tim[org] * (t[-1] - t[0]), marker="^", s=26, color="white", edgecolor=SLATE, linewidth=.8, label="Origins"); ax.legend(loc="lower right", fontsize=8)
    ax.grid(False); ax.set_xlabel(geo.label); ax.set_ylabel("MC step"); s_window(ax, t[0], t[-1])
    cax = fig.add_subplot(gs[0, 1]); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.set_label("I(x,t)", fontsize=9); cb.ax.tick_params(labelsize=8)
    top.fill_between(geo.xs, I.sum(1), color=ORANGE, alpha=.55, lw=0); top.set_xlim(ext[:2]); top.set_xticklabels([]); top.set_ylabel("Σ over t")
    rt.fill_betweenx(t, I.sum(0), color=PURPLE, alpha=.5, lw=0); s_window(rt, t[0], t[-1], text=False); rt.set_yticklabels([]); rt.set_xlabel("Σ over x"); rt.xaxis.set_label_position("top")
    return save_fig(fig, out, SUB, "initiation_rate", "rep_initiation")

def fig_timing(rep, geo, out):
    """Input timing (data) against the timing produced by the simulation."""
    sim = 1 - timing_profile(rep.rep_frac); inp = np.asarray(rep.input_timing)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw=dict(width_ratios=[3, 1.1]))
    a.plot(geo.xs, inp, color=BLUE, lw=2, label="Data"); a.plot(geo.xs, sim, color=ROYAL, lw=2, label="Simulation"); a.set_xlabel(geo.label); a.set_ylabel("Earliness"); a.legend(ncol=2, loc="lower right")
    b.scatter(inp, sim, s=10, c=geo.xs, cmap="viridis", alpha=.8, edgecolor="none"); b.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    rho = spearmanr(inp, sim).statistic if np.std(sim) > 0 else np.nan
    b.set_xlabel("Data"); b.set_ylabel("Simulation"); b.set_title(f"ρ = {rho:.2f}", loc="right", fontsize=10)
    return save_fig(fig, out, SUB, "timing_validation", "rep_timing")

def visualize_simulation(f, rep_fract, path=None):
    """Quick replication maps (kept for the stand-alone Replikator)."""
    if path is None: return
    from .style import apply_style; apply_style()
    fig, ax = plt.subplots(figsize=(9, 5)); im = ax.imshow(f.T, cmap=CMAP_REP, aspect="auto", origin="lower"); ax.grid(False)
    ax.set_xlabel("Simulation beads"); ax.set_ylabel("Replication time (simulation steps)"); fig.colorbar(im, label="Replicated"); fig.savefig(f"{path}/rep_simulation.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(rep_fract, color=ROYAL); ax.set_xlabel("Time"); ax.set_ylabel("Replicated fraction")
    fig.savefig(f"{path}/rep_frac.png", dpi=200, bbox_inches="tight"); plt.close(fig)
