"""Replication overview of the modelled region: timing data, initiation rate, Replikator trajectory and compartments on one genome axis."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
from .style import fork_layer, ROYAL, BLUE, TEAL, GREY, SLATE, PURPLE, ORANGE, CMAP_SEQ, CMAP_SPIN, shade_phases, s_window, save_fig
from ..analysis.rep_stats import timing_profile, global_fraction, quantile_time, infer_origins
from .rep_plots_pro import _t

SUB = "replication_simulation"
CMAP_RT = LinearSegmentedColormap.from_list("rs_rt", [PURPLE, "#E9E2F1", "#F6E2B5", ORANGE])   # late (purple) -> early (orange)

def _strip(ax, vals, xs, cmap, vmin, vmax, label):
    """One-row colour strip along the genome."""
    ax.imshow(np.asarray(vals)[None], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, extent=[xs[0], xs[-1], 0, 1], interpolation="nearest")
    ax.set_yticks([.5], [label], fontsize=9); ax.grid(False); ax.tick_params(axis="y", length=0)

def fig_rt_overview(rep, res, p, ph, geo, out):
    """Five aligned views of replication in the region: input timing, initiation rate, trajectory (with fraction), compartments."""
    xs = geo.xs; inp = np.asarray(rep.input_timing, float); tim = timing_profile(rep.rep_frac); sim = 1 - tim; org = infer_origins(tim)
    rf = np.asarray(rep.rep_frac) > 0; I = np.asarray(rep.init_rate, float); T = rf.shape[1]; t = _t(p, T); ext = [xs[0], xs[-1], t[0], t[-1]]
    spin = np.asarray(res.spin_traj, float); has_epi = ph.has_rep and spin.size and np.any(spin != 0)
    rows = [("rt", 1.5), ("init", 1.9), ("traj", 3.2)] + [("comp", 1.5)] * bool(has_epi)
    fig = plt.figure(figsize=(11.5, 1.55 * sum(h for _, h in rows) + .8))
    gs = fig.add_gridspec(len(rows), 2, width_ratios=[1, .17], height_ratios=[h for _, h in rows], hspace=.5, wspace=.05)
    ax = {k: fig.add_subplot(gs[i, 0]) for i, (k, _) in enumerate(rows)}; side = {k: fig.add_subplot(gs[i, 1]) for i, (k, _) in enumerate(rows)}
    w = (xs[1] - xs[0]) if len(xs) > 1 else 1
    # 1) timing data (bars coloured early -> late) and the simulated timing
    a = ax["rt"]; a.bar(xs, inp, width=w, color=CMAP_RT(inp), lw=0, label="input (data)"); a.plot(xs, sim, color=ROYAL, lw=1.8, label="simulation (Replikator)")
    rho = spearmanr(inp, sim).statistic if np.std(sim) > 0 else np.nan
    a.set_ylim(0, 1.05); a.set_ylabel("Earliness"); a.set_title(f"1  Replication timing: data (bars) vs simulation (line), Spearman ρ = {rho:.2f}", fontsize=10.5, loc="left", pad=8)
    a.legend(ncol=2, loc="upper right", fontsize=8.5)
    sm = plt.cm.ScalarMappable(cmap=CMAP_RT, norm=plt.Normalize(0, 1)); side["rt"].axis("off"); cb = fig.colorbar(sm, cax=side["rt"].inset_axes([0, .05, .22, .9]), ticks=[0, 1]); cb.ax.set_yticklabels(["late", "early"], fontsize=8)
    # 2) initiation rate I(x,t)
    a = ax["init"]; vmax = np.percentile(I[I > 0], 99.5) if (I > 0).any() else 1
    im = a.imshow(I.T, origin="lower", aspect="auto", cmap=CMAP_SEQ, extent=ext, vmax=vmax, interpolation="nearest"); a.grid(False)
    a.scatter(xs[org], t[0] + tim[org] * (t[-1] - t[0]), marker="^", s=34, color="white", edgecolor=SLATE, linewidth=.9, label=f"inferred origins ({len(org)})", zorder=3)
    s_window(a, t[0], t[-1]); a.set_ylabel("MC step"); a.set_title("2  Initiation rate I(x,t), calculated from the timing data", fontsize=10.5, loc="left", pad=8)
    a.legend(loc="lower right", fontsize=8.5, bbox_to_anchor=(1, .1)); side["init"].axis("off"); cb = fig.colorbar(im, cax=side["init"].inset_axes([0, .05, .22, .9])); cb.set_label("I(x,t)", fontsize=8); cb.ax.tick_params(labelsize=7)
    # 3) trajectory: replicated DNA coloured by replication time of its bead, forks in blue
    a = ax["traj"]; img = np.ones((T, len(xs), 4)); img[..., :3] = CMAP_RT(sim)[None, :, :3]; img[~rf.T] = 1.0
    a.imshow(img, origin="lower", aspect="auto", extent=ext, interpolation="nearest")
    forks = (np.asarray(rep.l_forks) > 0) | (np.asarray(rep.r_forks) > 0); fk = fork_layer(forks)
    a.imshow(fk, origin="lower", aspect="auto", extent=ext, interpolation="nearest"); a.scatter(xs[org], t[0] + tim[org] * (t[-1] - t[0]), marker="^", s=34, color=ROYAL, edgecolor="white", linewidth=.8, zorder=3)
    s_window(a, t[0], t[-1]); a.grid(False); a.set_ylabel("MC step"); a.set_title("3  Replikator trajectory: replicated DNA (colour = timing), forks teal, origins ▲", fontsize=10.5, loc="left", pad=8)
    f = global_fraction(rep.rep_frac); b = side["traj"]; b.plot(f, t, color=ROYAL, lw=2.2); s_window(b, t[0], t[-1], text=False); b.set_xlim(0, 1.03); b.set_yticklabels([]); b.set_xticklabels([]); b.set_title("replicated fraction", fontsize=8.5)
    for q in (.1, .5, .9):
        yq = t[0] + quantile_time(f, q) * (t[-1] - t[0]); b.plot([q], [yq], "o", color=SLATE, ms=4); b.annotate(f"T{int(q * 100)}", (q, yq), (4, -10), textcoords="offset points", fontsize=7.5)
    # 4) compartments before and after S, against the timing
    if has_epi:
        a = ax["comp"]; g1, g2 = spin[:, ph.g1:ph.s0].mean(1), spin[:, ph.s1:].mean(1); a.axis("off")
        sub = gs[len(rows) - 1, 0].subgridspec(4, 1, hspace=.12); sa = [fig.add_subplot(sub[i]) for i in range(4)]
        _strip(sa[0], sim, xs, CMAP_RT, 0, 1, "timing"); _strip(sa[1], g1, xs, CMAP_SPIN, -2, 2, "G1"); _strip(sa[2], g2, xs, CMAP_SPIN, -2, 2, "G2/M"); _strip(sa[3], g2 - g1, xs, CMAP_SPIN, -2, 2, "change")
        for s_ in sa[:-1]: s_.set_xticklabels([])
        sa[0].set_title("4  Compartments vs timing: mean epigenetic state per bead in G1, G2/M and their change", fontsize=10.5, loc="left", pad=8)
        r1 = spearmanr(sim, g1).statistic; r2 = spearmanr(sim, g2).statistic
        side["comp"].axis("off"); side["comp"].text(0, .5, f"ρ(timing, state)\nG1: {r1:.2f}\nG2/M: {r2:.2f}", fontsize=8.5, va="center", color=SLATE)
        for s_ in sa: s_.set_xlim(ext[0], ext[1])
        sa[-1].set_xlabel(geo.label); bottom = sa[-1]
    else: bottom = ax["traj"]
    for k in ("rt", "init", "traj"): ax[k].set_xlim(ext[0], ext[1])
    for k in ("rt", "init"): ax[k].set_xticklabels([])
    if not has_epi: ax["traj"].set_xlabel(geo.label)
    else: ax["traj"].set_xticklabels([])
    return save_fig(fig, out, SUB, "rt_overview", "rt_overview")
