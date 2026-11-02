"""Overview figures: the cell-cycle story in one picture, and a phase comparison with honest error bars."""
import numpy as np
import matplotlib.pyplot as plt
from .style import kb, unit_label, ROYAL, BLUE, TEAL, GREY, BLACK, PURPLE, ORANGE, PHASE_COL, A_COL, B_COL, shade_phases, band, rolling, save_fig
from ..analysis.mc_stats import ess

SUB = "overview"

def _phase_means(y, ph):
    """{phase: mean} of a per-sample series (burn-in removed)."""
    return {k: float(np.mean(np.asarray(y, float)[s])) for k, s in ph.segments().items() if len(np.asarray(y)[s])}

def _tag(ax, ph, means, fmt="{:.2f}", col=None):
    """Phase means as small direct labels at the top of a panel."""
    tr = ax.get_xaxis_transform(); edges = {"G1": (ph.g1, ph.s0), "S": (ph.s0, ph.s1), "G2/M": (ph.s1, ph.n), "All": (ph.g1, ph.n)}
    for k, v in means.items():
        a, b = edges[k]; ax.text(ph.x((a + b) / 2), .96, fmt.format(v), transform=tr, ha="center", va="top", fontsize=9.5, fontweight="bold", color=col or PHASE_COL.get(k, BLACK))

def replicated_curve(rep, p, n_steps):
    """Fraction of the region replicated at every MC step (0 before S, 1 after)."""
    f = np.zeros(n_steps); t0, d = p.t_rep, p.rep_duration
    cur = np.asarray(rep.f_rep_T, float).mean(axis=1)[:d]
    f[t0:t0 + len(cur)] = cur[:max(0, n_steps - t0)]; f[t0 + len(cur):] = 1.0
    return f

def fig_story(obs, res, p, rep, ph, out):
    """Four stacked panels on one time axis: replication, loops, compartments, loop activity."""
    t = ph.steps; w = max(3, ph.n // 60); spin = np.asarray(res.spin_traj); has_epi = bool(rep is not None and spin.size and np.any(spin != 0))
    rows = [("rep", 1.0)] * (rep is not None) + [("loop", 1.5)] + [("comp", 1.5)] * has_epi + [("bound", 1.0)]
    fig, axs = plt.subplots(len(rows), 1, figsize=(10, 1.45 * sum(h for _, h in rows) + 1.6), sharex=True, gridspec_kw=dict(height_ratios=[h for _, h in rows], hspace=.62))
    axs = dict(zip([r for r, _ in rows], np.atleast_1d(axs))); first = True; first_ax = [next(iter(axs.values()))]
    def head(ax, title, sub=""):
        nonlocal first; shade_phases(ax, ph, labels=first); first = False
        ax.set_title(title + (f"   {sub}" if sub else ""), fontsize=11, pad=30 if ax is first_ax[0] else 8)
    if "rep" in axs:
        ax = axs["rep"]; xs = np.arange(p.N_steps); f = replicated_curve(rep, p, p.N_steps)
        head(ax, "Replication", "share of DNA that is copied"); ax.fill_between(xs, 0, f, color=TEAL, alpha=.25, lw=0); ax.plot(xs, f, color=TEAL, lw=2)
        ax.set_ylim(0, 1.05); ax.set_yticks([0, 1], ["0 %", "100 %"])
    y = kb(obs["Mean loop length"]); m = _phase_means(y, ph); ax = axs["loop"]
    sub = ""
    if "G1" in m and "S" in m and m["G1"] > 0: d = 100 * (m['S'] / m['G1'] - 1); sub = "S phase vs G1: no change" if abs(d) < .5 else f"S phase vs G1: {d:+.0f} %"
    head(ax, "Loops", unit_label("mean loop length") + (f"   ·   {sub}" if sub else "")); ax.plot(t, y, color=BLUE, lw=.5, alpha=.25); band(ax, t, y, w, BLUE); _tag(ax, ph, m)
    lo, hi = np.percentile(y[ph.g1:], [.5, 99.5]); d = (hi - lo) or 1; ax.set_ylim(max(0, lo - .35 * d), hi + .35 * d)
    if "comp" in axs:
        ax = axs["comp"]; a = (spin > 0).mean(0); b = (spin < 0).mean(0); head(ax, "Compartments", "share of beads in A-like (red) and B-like (blue) states; numbers = A-like share per phase")
        for y_, c, lab in ((a, A_COL, "A"), (b, B_COL, "B")):
            ax.plot(t, y_, color=c, lw=.5, alpha=.25); band(ax, t, y_, w, c); ax.text(t[-1], rolling(y_, w)[-1], f"  {lab}", color=c, va="center", fontweight="bold")
        _tag(ax, ph, _phase_means(a, ph), "{:.0%}", A_COL)
        r = np.r_[a[ph.g1:], b[ph.g1:]]; lo, hi = np.percentile(r, [.5, 99.5]); d = (hi - lo) or .1; ax.set_ylim(max(0, lo - .3 * d), min(1, hi + .45 * d))
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    y = np.asarray(obs["Bound LEFs"], float); ax = axs["bound"]; head(ax, "Loop extruders", "share of LEFs bound to DNA")
    ax.plot(t, y, color=PURPLE, lw=.5, alpha=.25); band(ax, t, y, w, PURPLE); ax.set_ylim(0, min(1.05, max(.2, y[ph.g1:].max() * 1.25))); ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axs["bound"].set_xlabel("Monte Carlo step (pseudotime)")
    for a in axs.values(): a.grid(axis="x", visible=False)
    return save_fig(fig, out, SUB, "story", "story")

def fig_phase_compare(obs, res, p, ph, out):
    """Per-phase means with 95 % intervals that account for autocorrelation (SE = SD / sqrt(ESS))."""
    segs = ph.segments(); spin = np.asarray(res.spin_traj); has_epi = bool(ph.has_rep and spin.size and np.any(spin != 0))
    ms = [("Mean loop length", unit_label("Loop length"), "{:.1f}"), ("Bound LEFs", "Bound LEFs", "{:.0%}")]
    series = {k: np.asarray(obs[k], float) for k, _, _ in ms}; series["Mean loop length"] = kb(series["Mean loop length"])
    if has_epi: series["A share"] = (spin > 0).mean(0); ms.append(("A share", "A-like beads", "{:.0%}"))
    if "Folding energy" in obs: series["Folding energy"] = np.asarray(obs["Folding energy"], float); ms.append(("Folding energy", "Folding energy", "{:.0f}"))
    fig, axs = plt.subplots(1, len(ms), figsize=(2.6 * len(ms) + .6, 3.3), gridspec_kw=dict(wspace=.55)); axs = np.atleast_1d(axs)
    for ax, (k, lab, fmt) in zip(axs, ms):
        y = series[k]
        for i, (name, s) in enumerate(segs.items()):
            v = y[s]
            if len(v) < 4: continue
            e = ess(v); e = len(v) if not e == e else e; se = v.std(ddof=1) / np.sqrt(max(e, 1)); c = PHASE_COL.get(name, PURPLE)
            ax.errorbar(i, v.mean(), yerr=1.96 * se, fmt="o", color=c, ms=8, capsize=4, lw=2); ax.text(i + .2, v.mean(), fmt.format(v.mean()), va="center", fontsize=9, color=c)
        ax.set_xticks(range(len(segs)), list(segs), fontsize=10); ax.set_xlim(-.5, len(segs) - .1); ax.set_title(lab, fontsize=10.5, pad=8); ax.grid(axis="x", visible=False)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo - .05 * (hi - lo), hi + .05 * (hi - lo))
    return save_fig(fig, out, SUB, "phase_compare", "phase_compare")
