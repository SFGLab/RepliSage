"""Epigenetic (Potts) figures: colour propagation, state fractions, A/B compartment switching."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from .style import (bin_time, CMAP_SPIN, BLACK, ROYAL, GREY, BLUE, TEAL, GOLD, PURPLE, ORANGE, SLATE, A_COL, B_COL, STATE_COL, CMAP_MAT, PHASE_COL, shade_phases, rolling, save_fig)
from .loops import overlay_front, _violin
from ..analysis.compartments import (A, N, B, s_phase_states, transition_matrix, switch_rates, state_fractions, switch_summary)

SUB = "graph_metrics"
SPIN_CMAP = ListedColormap([STATE_COL[k] for k in range(-2, 3)]); SPIN_NORM = BoundaryNorm(np.arange(-2.5, 3), 5)
COMP_COL = {B: B_COL, N: "#EDEDED", A: A_COL}

def _state_legend(ax, **kw):
    ax.legend(handles=[Patch(color=STATE_COL[k], label=str(k)) for k in range(-2, 3)], title="Spin", ncol=5, fontsize=8, **kw)

def _spin_image(ax, fig, spins, ph):
    """Spin map; long runs are averaged in time bins (continuous colour = mean spin), short runs show exact states."""
    m, f = bin_time(spins.astype(float)); ext = [0, ph.x(ph.n), spins.shape[0], 0]
    if f == 1:
        ax.imshow(spins, aspect="auto", cmap=SPIN_CMAP, norm=SPIN_NORM, extent=ext, interpolation="nearest"); _state_legend(ax, loc="upper center", bbox_to_anchor=(.5, -.13))
    else:
        im = ax.imshow(m, aspect="auto", cmap=CMAP_SPIN, vmin=-2, vmax=2, extent=ext, interpolation="nearest")
        cax = ax.inset_axes([1.01, .0, .015, 1]); cb = fig.colorbar(im, cax=cax); cb.set_label(f"Mean spin ({f} samples per column)", fontsize=8); cb.ax.tick_params(labelsize=7)

def ising_traj_plot(spins, ph, out, p=None, f_rep_T=None):
    """Epigenetic state of every bead over time; dashed line = replication front."""
    fig, ax = plt.subplots(figsize=(11, 5.6)); _spin_image(ax, fig, spins, ph)
    shade_phases(ax, ph, labels=True); ax.grid(False)
    if p is not None: overlay_front(ax, p, ph, spins.shape[0], f_rep_T)
    ax.set_xlabel("MC step"); ax.set_ylabel("Bead")
    return save_fig(fig, out, "MCMC_diagnostics", "potts_traj", "epi_traj")

def fig_timing_sorted(spins, timing, ph, out):
    """Same map with beads sorted from early to late replicating: shows the spread of the colour with replication."""
    order = np.argsort(timing, kind="stable"); fig, ax = plt.subplots(figsize=(11, 5))
    _spin_image(ax, fig, spins[order], ph); shade_phases(ax, ph, labels=True); ax.grid(False); ax.set_xlabel("MC step"); ax.set_ylabel("Beads (early → late)")
    return save_fig(fig, out, "MCMC_diagnostics", "potts_sorted_by_timing", "epi_sorted")

def fig_fractions(spins, ph, out):
    """Fraction of beads in each spin state (stacked) and in A / B compartments."""
    fr = np.array([rolling(r, max(1, ph.n // 150)) for r in state_fractions(spins)]); t = ph.steps; sg = np.sign(spins)
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1]))
    a.stackplot(t, *fr, colors=[STATE_COL[k] for k in range(-2, 3)], labels=[str(k) for k in range(-2, 3)], lw=0)
    shade_phases(a, ph, labels=True); a.set_ylim(0, 1); a.set_ylabel("Beads per state"); a.legend(title="Spin", ncol=5, loc="lower center", bbox_to_anchor=(.5, -.55), fontsize=8)
    w = max(3, ph.n // 80)
    b.plot(t, rolling((sg > 0).mean(0), w), color=A_COL, label="A (spin > 0)"); b.plot(t, rolling((sg < 0).mean(0), w), color=B_COL, label="B (spin < 0)")
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

def fig_switch_map(spins, timing, ph, geo, out):
    """Genome tracks (3 panels): replication earliness, compartments at S start / S end, switched beads."""
    a, b, _ = s_phase_states(spins, ph); ch = np.where((a == B) & (b == A), 2, np.where((a == A) & (b == B), 1, 0)); ext = [geo.xs[0], geo.xs[-1], 0, 1]
    fig, axs = plt.subplots(3, 1, figsize=(10, 3.9), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1, .6], hspace=.14))
    e = 1 - np.asarray(timing); axs[0].fill_between(geo.xs, e, color=GOLD, alpha=.35, lw=0); axs[0].plot(geo.xs, e, color=ORANGE, lw=1.3); axs[0].set_ylabel("Early"); axs[0].set_ylim(0, 1.02)
    axs[1].imshow(np.vstack([a, b])[::-1], aspect="auto", cmap=ListedColormap([COMP_COL[B], COMP_COL[N], COMP_COL[A]]), vmin=-1, vmax=1, extent=ext, interpolation="nearest")
    axs[1].set_yticks([.25, .75], ["S end", "S start"]); axs[1].grid(False)
    axs[1].legend(handles=[Patch(color=A_COL, label="A"), Patch(color=B_COL, label="B"), Patch(color="#EDEDED", label="N")], loc="center left", bbox_to_anchor=(1.01, .5), fontsize=8)
    axs[2].imshow(ch[None], aspect="auto", cmap=ListedColormap(["#F2F2F2", B_COL, A_COL]), vmin=0, vmax=2, extent=ext, interpolation="nearest"); axs[2].set_yticks([.5], ["Switch"]); axs[2].grid(False); axs[2].set_xlabel(geo.label)
    axs[2].legend(handles=[Patch(color=A_COL, label="B→A"), Patch(color=B_COL, label="A→B")], loc="center left", bbox_to_anchor=(1.01, .5), fontsize=8)
    return save_fig(fig, out, SUB, "switch_map", "epi_switch_map")

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
