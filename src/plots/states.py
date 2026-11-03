"""Epigenetic (Potts) figures: colour propagation, state fractions, A/B compartment switching."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from .style import (genome_axis, bin_time, CMAP_SPIN, BLACK, ROYAL, GREY, BLUE, TEAL, GOLD, PURPLE, ORANGE, SLATE, A_COL, B_COL, STATE_COL, CMAP_MAT, PHASE_COL, shade_phases, rolling, save_fig)
from .loops import overlay_front, _front, _violin
from ..analysis.compartments import (A, N, B, s_phase_states, transition_matrix, switch_rates, state_fractions, switch_summary)

SUB = "graph_metrics"
SPIN_CMAP = ListedColormap([STATE_COL[k] for k in range(-2, 3)]); SPIN_NORM = BoundaryNorm(np.arange(-2.5, 3), 5)
COMP_COL = {B: B_COL, N: "#EDEDED", A: A_COL}

def _state_legend(ax, **kw):
    ax.legend(handles=[Patch(color=STATE_COL[k], label=str(k)) for k in range(-2, 3)], title="Epigenetic state", ncol=5, fontsize=8, **kw)

def _spin_image(ax, fig, spins, ph):
    """Heatmap of the mean epigenetic state (time-binned and lightly smoothed): red = A-like, blue = B-like."""
    from scipy.ndimage import gaussian_filter
    m, f = bin_time(spins.astype(float), 150); m = gaussian_filter(m, (.7, .9)); ext = [0, ph.x(ph.n), spins.shape[0], 0]
    im = ax.imshow(m, aspect="auto", cmap=CMAP_SPIN, vmin=-2, vmax=2, extent=ext, interpolation="bilinear")
    cax = ax.inset_axes([1.01, .0, .015, 1]); cb = fig.colorbar(im, cax=cax); cb.set_label(f"Mean epigenetic state ({f} sample{'s' * (f > 1)} per column, smoothed)", fontsize=8); cb.ax.tick_params(labelsize=7)

def ising_traj_plot(spins, ph, out, p=None, f_rep_T=None):
    """Epigenetic state of every bead over time; dashed line = replication front."""
    fig, ax = plt.subplots(figsize=(11, 5.6)); _spin_image(ax, fig, spins, ph)
    shade_phases(ax, ph, labels=True); ax.grid(False)
    if p is not None: overlay_front(ax, p, ph, spins.shape[0], f_rep_T)
    ax.set_xlabel("MC step"); genome_axis(ax, spins.shape[0])
    return save_fig(fig, out, "MCMC_diagnostics", "potts_traj", "epi_traj")

def fig_timing_sorted(spins, timing, ph, out):
    """Same map with beads sorted from early to late replicating: shows the spread of the colour with replication."""
    order = np.argsort(timing, kind="stable"); fig, ax = plt.subplots(figsize=(11, 5))
    _spin_image(ax, fig, spins[order], ph); shade_phases(ax, ph, labels=True); ax.grid(False); ax.set_xlabel("MC step"); ax.set_ylabel("Simulation beads (early → late replicating)")
    return save_fig(fig, out, "MCMC_diagnostics", "potts_sorted_by_timing", "epi_sorted")

def fig_fractions(spins, ph, out):
    """Fraction of beads in each epigenetic state (stacked) and in A / B compartments."""
    fr = np.array([rolling(r, max(1, ph.n // 150)) for r in state_fractions(spins)]); t = ph.steps; sg = np.sign(spins)
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1]))
    a.stackplot(t, *fr, colors=[STATE_COL[k] for k in range(-2, 3)], labels=[str(k) for k in range(-2, 3)], lw=0)
    shade_phases(a, ph, labels=True); a.set_ylim(0, 1); a.set_ylabel("Beads per state"); a.legend(title="Epigenetic state", ncol=5, loc="lower center", bbox_to_anchor=(.5, -.55), fontsize=8)
    w = max(3, ph.n // 80)
    b.plot(t, rolling((sg > 0).mean(0), w), color=A_COL, label="A (state > 0)"); b.plot(t, rolling((sg < 0).mean(0), w), color=B_COL, label="B (state < 0)")
    shade_phases(b, ph); b.set_ylabel("Fraction of beads"); b.set_xlabel("MC step"); b.legend(loc="center right")
    return save_fig(fig, out, SUB, "state_fractions", "epi_fractions")

def fig_switching(spins, ph, out):
    """Rate of B→A and A→B switches over time, and the compartment transition matrix over S phase."""
    lag = max(1, ph.n // 200); rb, ra = switch_rates(spins, lag); t = ph.steps[:-lag]; w = max(3, ph.n // 80)
    fig = plt.figure(figsize=(11.5, 4.2)); gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.1], wspace=.28)
    ax = fig.add_subplot(gs[0]); ax.plot(t, rolling(rb, w), color=A_COL, label="B → A"); ax.plot(t, rolling(ra, w), color=B_COL, label="A → B")
    shade_phases(ax, ph, labels=True); ax.set_xlabel("MC step"); ax.set_ylabel("Switching beads (fraction)"); ax.legend(loc="upper right")
    hx = fig.add_subplot(gs[1])
    if ph.has_rep:
        a, b, _ = s_phase_states(spins, ph); M = transition_matrix(a, b)
        hx.imshow(M / np.maximum(M.sum(1, keepdims=True), 1), cmap=CMAP_MAT, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3): hx.text(j, i, M[i, j], ha="center", va="center", color="white" if M[i, j] > .5 * M.max() and M[i, j] > 0 else BLACK, fontweight="bold")
        hx.set_xticks(range(3), list("BNA")); hx.set_yticks(range(3), list("BNA")); hx.set_xlabel("S end"); hx.set_ylabel("S start"); hx.grid(False); hx.set_title("Beads (count)", fontsize=10)
    else: hx.axis("off")
    return save_fig(fig, out, SUB, "compartment_switching", "epi_switching")

def fig_switch_bars(spins, ph, out):
    """Bar plots: compartment content and switching between the start and the end of S phase."""
    a, b, _ = s_phase_states(spins, ph); n = len(a); sw = switch_summary(a, b)
    fig, (l, r) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw=dict(width_ratios=[1.1, 1.3]))
    x = np.arange(3); wd = .38; labs = ["B", "N", "A"]; codes = [B, N, A]
    for k, (st, lab, c) in enumerate(((a, "S start", TEAL), (b, "S end", PURPLE))):
        v = [np.mean(st == q) * 100 for q in codes]; bars = l.bar(x + (k - .5) * wd, v, wd, color=c, label=lab)
        l.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    l.set_xticks(x, labs); l.set_ylabel("Beads (%)"); l.legend(); l.set_title("Compartments")
    keys = ["B→A", "A→B", "B→B", "A→A"]; vals = [sw[k] / n * 100 for k in keys]
    bars = r.bar([f"{k}\n({sw[k]})" for k in keys], vals, color=[A_COL, B_COL, "#C9CED6", "#C9CED6"]); r.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    r.set_ylabel("Beads (%)"); r.set_title("Switching over S")
    return save_fig(fig, out, SUB, "switching_bars", "epi_switch_bars")

def _tercile_switch(a, b, timing):
    """% of beads switching B->A and A->B in the early / middle / late third of replication timing."""
    order = np.argsort(np.asarray(timing)); groups = np.array_split(order, 3)       # timing: small = early
    f = lambda g, x, y: 100 * np.mean((a[g] == x) & (b[g] == y))
    return [(f(g, B, A), f(g, A, B)) for g in groups]

def fig_switch_map(spins, timing, ph, geo, out):
    """Genome tracks: replication timing, compartments at the start / end of S, switched beads; below: switching in early / middle / late DNA."""
    a, b, _ = s_phase_states(spins, ph); ch = np.where((a == B) & (b == A), 2, np.where((a == A) & (b == B), 1, 0)); ext = [geo.xs[0], geo.xs[-1], 0, 1]
    fig = plt.figure(figsize=(10, 6.2)); gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.7], width_ratios=[4, 1.2], hspace=.5, wspace=.25)
    top = gs[0, 0].subgridspec(3, 1, height_ratios=[1.5, 1, .55], hspace=.14); axs = [fig.add_subplot(top[i]) for i in range(3)]; bx = fig.add_subplot(gs[1, 0]); lg = fig.add_subplot(gs[0, 1]); lg.axis("off")
    e = 1 - np.asarray(timing); axs[0].fill_between(geo.xs, e, color=GOLD, alpha=.35, lw=0); axs[0].plot(geo.xs, e, color=ORANGE, lw=1.3); axs[0].set_ylabel("Replicates\nearly (1) / late (0)", fontsize=8.5); axs[0].set_ylim(0, 1.02)
    cm3 = ListedColormap([COMP_COL[B], COMP_COL[N], COMP_COL[A]])
    axs[1].imshow(np.vstack([a, b]), aspect="auto", cmap=cm3, vmin=-1, vmax=1, extent=ext, interpolation="nearest"); axs[1].set_yticks([.75, .25], ["start of S", "end of S"]); axs[1].grid(False)
    axs[2].imshow(ch[None], aspect="auto", cmap=ListedColormap(["#F2F2F2", B_COL, A_COL]), vmin=0, vmax=2, extent=ext, interpolation="nearest"); axs[2].set_yticks([.5], ["switched"]); axs[2].grid(False)
    for x_ in axs[:2]: x_.set_xticklabels([])
    axs[2].set_xlabel(geo.label)
    lg.legend(handles=[Patch(color=A_COL, label="A (open)"), Patch(color=B_COL, label="B (closed)"), Patch(color="#EDEDED", label="undecided"), Patch(color=A_COL, label="switched B → A"), Patch(color=B_COL, label="switched A → B")], loc="center left", fontsize=8.5, frameon=False)
    v = _tercile_switch(a, b, timing); x = np.arange(3); w = .38
    bx.bar(x - w / 2, [t[0] for t in v], w, color=A_COL, label="B → A"); bx.bar(x + w / 2, [t[1] for t in v], w, color=B_COL, label="A → B")
    bx.set_xticks(x, ["earliest third", "middle third", "latest third"]); bx.set_ylabel("Beads switching (%)"); bx.legend(frameon=False, fontsize=8.5, ncol=2); bx.set_title("Do early or late replicating regions switch more?", fontsize=10, loc="left")
    return save_fig(fig, out, SUB, "switch_map", "epi_switch_map")

def fig_switch_trajectory(spins, ph, p, geo, f_rep_T, out):
    """Compartment (A / B / undecided) of every bead over time from window averages, with the shift of every bead over S and the share of A and B."""
    from scipy.ndimage import uniform_filter1d
    from ..analysis.compartments import DEAD
    N_ = spins.shape[0]; w = max(3, int(.1 * (ph.s1 - ph.s0))) if ph.has_rep else max(3, ph.n // 20)                  # window = 10% of S, as in the bar plots
    sm = uniform_filter1d(spins.astype(float), w, axis=1, mode="nearest"); k = max(1, sm.shape[1] // 300); sm = sm[:, ::k]; t = ph.x(np.arange(ph.n))[::k]
    st = np.where(sm > DEAD, A, np.where(sm < -DEAD, B, N)); a, b, (s0, s1) = s_phase_states(spins, ph); d = spins[:, s1].mean(1) - spins[:, s0].mean(1)
    fig = plt.figure(figsize=(11.5, 7.2)); gs = fig.add_gridspec(2, 2, width_ratios=[10, 1.5], height_ratios=[4, 1.6], hspace=.16, wspace=.06)
    ax, right, low = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
    ext = [t[0], t[-1], N_, 0]; ax.imshow(st, aspect="auto", cmap=ListedColormap([COMP_COL[B], COMP_COL[N], COMP_COL[A]]), vmin=-1, vmax=1, extent=ext, interpolation="nearest")
    shade_phases(ax, ph, labels=True); ax.grid(False); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(N_, 0); ax.set_xticklabels([]); genome_axis(ax, N_)
    if f_rep_T is not None:
        fr = _front(f_rep_T, p, N_); y = np.arange(N_) + .5; ax.plot(fr, y, color="white", lw=3.4); ax.plot(fr, y, color=BLACK, lw=1.3, ls="--", label="replication front (bead copied here)")
        ax.legend(handles=[Patch(color=A_COL, label="A (open)"), Patch(color=B_COL, label="B (closed)"), Patch(color="#EDEDED", label="undecided"), ax.lines[-1]], loc="upper center", bbox_to_anchor=(.5, -.01), ncol=4, fontsize=8.5, frameon=False)
    ax.set_title("Compartment of every bead over time (averaged over a short window)", fontsize=10.5, pad=24)
    right.barh(np.arange(N_) + .5, d, height=1, color=np.where(d >= 0, A_COL, B_COL), lw=0); right.axvline(0, color=GREY, lw=.8); right.set_ylim(N_, 0); right.set_yticks([]); right.set_title("shift over S\n(red: towards A)", fontsize=8.5); right.grid(axis="y", visible=False)
    shade_phases(low, ph); low.plot(t, (st == A).mean(0) * 100, color=A_COL, lw=2, label="beads in A"); low.plot(t, (st == B).mean(0) * 100, color=B_COL, lw=2, label="beads in B")
    low.set_xlim(ext[0], ext[1]); low.set_ylabel("% of beads"); low.set_xlabel("MC step"); low.legend(ncol=2, loc="upper left", fontsize=8.5, frameon=False)
    return save_fig(fig, out, SUB, "switch_trajectory", "epi_switch_traj")

def same_sign_series(Ms, Ns, spins):
    """Fraction of loops whose two legs have the same / opposite epigenetic sign (per sample)."""
    ok = (Ms >= 0) & (Ns >= 0); cols = np.arange(Ms.shape[1])[None, :]
    sm, sn = spins[np.clip(Ms, 0, None), cols], spins[np.clip(Ns, 0, None), cols]
    diff = ((sm > 0) & (sn < 0)) | ((sm < 0) & (sn > 0)); nv = np.maximum(ok.sum(0), 1)
    d = (diff & ok).sum(0) / nv
    return 1 - d, d

def compute_state_proportions_sign_based(Ms, Ns, spins, ph, out):
    """Same-sign loop fraction over time and per phase (violin kept from the published version)."""
    same, diff = same_sign_series(Ms, Ns, spins); t = ph.steps; w = max(3, ph.n // 80)
    fig, (ax, vx) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[3, 1.2]))
    ax.plot(t, same, color=TEAL, lw=.6, alpha=.35); ax.plot(t, rolling(same, w), color=TEAL, lw=1.8, label="Same sign"); ax.plot(t, rolling(diff, w), color=ORANGE, lw=1.8, label="Opposite sign")
    shade_phases(ax, ph, labels=True); ax.set_xlabel("MC step"); ax.set_ylabel("Fraction of loops"); ax.legend(loc="center right")
    _violin(vx, {k: same[s] for k, s in ph.segments().items()}, "Same-sign fraction")
    save_fig(fig, out, SUB, "same_diff_sign", "epi_same_sign")
    fig, ax = plt.subplots(figsize=(4.6, 3.8)); _violin(ax, {k: same[s] for k, s in ph.segments().items()}, "Same-sign fraction")
    save_fig(fig, out, SUB, "same_sign_fraction_violin", "violin_same_sign")
    return same, diff
