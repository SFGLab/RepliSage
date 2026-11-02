"""Extra MC diagnostics (each figure has at most 3 panels): stationarity, MCSE, rolling stats, correlations, energy budget, LEF dynamics, domains."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp, spearmanr
from .style import kb, unit_label, BLACK, ROYAL, GREY, BLUE, TEAL, GOLD, PURPLE, ORANGE, GREEN, SLATE, PHASE_COL, A_COL, B_COL, CMAP_SEQ, shade_phases, rolling, band, centred, zoom, save_fig
from .mc_timeplots import SHORT, SUB
from ..analysis.mc_stats import ess, iat, diag_segment
from ..analysis.lef_stats import bound_mask

KEYS = ("Total energy", "Mean loop length", "Folding energy")
COLS = (BLUE, TEAL, PURPLE)

def _seg(obs, ph, k):
    seg, where = diag_segment(ph); return np.asarray(obs[k], float)[seg], seg, where

def fig_stationarity(obs, ph, out):
    """First half vs second half of the stationary part: the two densities should coincide."""
    keys = [k for k in KEYS if k in obs]; fig, axs = plt.subplots(1, len(keys), figsize=(3.7 * len(keys), 3.3)); axs = np.atleast_1d(axs)
    for ax, k in zip(axs, keys):
        v, _, where = _seg(obs, ph, k); v, pre = (v - np.median(v), "Δ") if abs(v.mean()) > 20 * v.std() else (v, ""); h = len(v) // 2; a, b = v[:h], v[h:2 * h]
        if h < 8 or v.std() == 0: continue
        g = np.linspace(v.min(), v.max(), 200)
        for s, c, lab in ((a, BLUE, "1st half"), (b, ROYAL, "2nd half")):
            d = gaussian_kde(s)(g); ax.fill_between(g, d, color=c, alpha=.25, lw=0); ax.plot(g, d, color=c, lw=2, label=lab)
        ax.set_title(f"{pre}{SHORT[k]}   KS={ks_2samp(a, b).statistic:.2f}"); ax.set_xlabel(pre + SHORT[k]); ax.set_yticks([]); ax.grid(False)
    axs[0].legend(); axs[0].set_ylabel("Density")
    return save_fig(fig, out, SUB, "stationarity", "mc_stationarity")

def fig_mcse(obs, ph, out):
    """Monte Carlo error vs number of samples, and effective samples gained."""
    keys = [k for k in KEYS if k in obs]; fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.5))
    for k, c in zip(keys, COLS):
        v, _, _ = _seg(obs, ph, k); n = np.unique(np.geomspace(20, len(v), 14).astype(int))
        if len(v) < 40 or v.std() == 0: continue
        e = np.array([ess(v[:i]) for i in n]); a.plot(n, v.std() / np.sqrt(e) / v.std(), "o-", color=c, ms=4, label=SHORT[k]); b.plot(n, e, "o-", color=c, ms=4)
    n0 = np.array([20, len(v)]); a.plot(n0, 1 / np.sqrt(n0), color=GREY, ls="--", label="1/√n (independent)")
    from matplotlib.ticker import NullFormatter, ScalarFormatter, LogLocator
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("Samples"); a.set_ylabel("MCSE / SD"); a.set_title("Error shrinks"); a.legend()
    b.plot(n0, n0, color=GREY, ls="--"); b.set_xscale("log"); b.set_yscale("log"); b.set_xlabel("Samples"); b.set_ylabel("Effective samples"); b.set_title("Information gained")
    for x in (a, b):
        for ax_ in (x.xaxis, x.yaxis): ax_.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=8)); ax_.set_major_formatter(ScalarFormatter()); ax_.set_minor_formatter(NullFormatter())
    return save_fig(fig, out, SUB, "mcse", "mc_mcse")

def fig_rolling(obs, ph, out):
    """Rolling mean ± SD of two observables and the rolling autocorrelation time."""
    t = ph.steps; w = max(5, ph.n // 40); fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.3))
    for ax, k, c in zip(axs[:2], ("Total energy", "Mean loop length"), (BLUE, TEAL)):
        if k not in obs: continue
        shade_phases(ax, ph); band(ax, t, np.asarray(obs[k], float), w, c); ax.set_title(SHORT[k]); ax.set_xlabel("MC step")
    y = np.asarray(obs["Total energy"], float); step = max(1, w // 2); idx = np.arange(0, ph.n - 2 * w, step)
    tau = [iat(y[i:i + 2 * w]) for i in idx]; shade_phases(axs[2], ph); axs[2].plot(t[idx + w], rolling(tau, 3), color=ORANGE, lw=2)
    axs[2].set_title("Autocorr. time τ"); axs[2].set_xlabel("MC step"); axs[2].set_ylabel("Samples")
    return save_fig(fig, out, SUB, "rolling", "mc_rolling")

def fig_correlation(obs, ph, out):
    """Rank correlation between observables, and energy vs loop length coloured by time."""
    seg = slice(ph.g1, ph.n); keys = list(obs); X = np.array([np.asarray(obs[k], float)[seg] for k in keys])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1.2], wspace=.55))
    R = np.nan_to_num(spearmanr(X.T)[0]) if len(keys) > 2 else np.corrcoef(X)
    im = a.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1); a.set_xticks(range(len(keys)), [SHORT.get(k, k) for k in keys], rotation=90)
    a.set_yticks(range(len(keys)), [SHORT.get(k, k) for k in keys]); a.grid(False); a.set_title("Rank correlation")
    for i in range(len(keys)):
        for j in range(len(keys)): a.text(j, i, f"{R[i, j]:.1f}", ha="center", va="center", fontsize=8, color="white" if abs(R[i, j]) > .6 else BLACK)
    plt.colorbar(im, ax=a, fraction=.046, pad=.03)
    sc = b.scatter(X[keys.index("Mean loop length")], X[keys.index("Total energy")], c=ph.steps[seg], cmap=CMAP_SEQ, s=12, alpha=.8)
    b.set_xlabel("Loop length"); b.set_ylabel("Energy"); b.set_title("Energy vs loops"); plt.colorbar(sc, ax=b, fraction=.046, pad=.04, label="MC step", format="%d")
    return save_fig(fig, out, SUB, "correlation", "mc_correlation")

def fig_budget(obs, ph, out):
    """Energy components over time (smoothed) and their mean per phase."""
    comp = {k: np.asarray(obs[k], float) for k in ("Folding energy", "Binding energy", "Potts energy") if k in obs}
    comp["Other"] = np.asarray(obs["Total energy"], float) - sum(comp.values())        # crossing + replication
    comp = {k: y - y[ph.g1:].mean() for k, y in comp.items()}                            # centre on post burn-in mean
    cols = dict(zip(comp, (BLUE, PURPLE, ORANGE, GREEN))); w = max(3, ph.n // 60)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    shade_phases(a, ph)
    for k, y in comp.items(): a.plot(ph.steps, rolling(y, w), color=cols[k], lw=2, label=SHORT.get(k, k))
    r = np.concatenate([rolling(y, w)[ph.g1:] for y in comp.values()]); lo, hi = np.percentile(r, [.5, 99.5]); a.set_ylim(lo - .2 * (hi - lo), hi + .3 * (hi - lo))
    a.set_xlabel("MC step"); a.set_ylabel("Energy − mean"); a.legend(ncol=2); a.set_title("Components")
    segs = ph.segments(); x = np.arange(len(comp)); wd = .8 / len(segs)
    for i, (lab, s) in enumerate(segs.items()): b.bar(x + i * wd, [comp[k][s].mean() for k in comp], wd, color=PHASE_COL[lab], label=lab)
    b.set_xticks(x + wd * (len(segs) - 1) / 2, [SHORT.get(k, k) for k in comp], rotation=20); b.set_title("Phase mean − overall mean"); b.axhline(0, color=GREY, lw=1); b.legend()
    return save_fig(fig, out, SUB, "energy_budget", "mc_budget")

def fig_lef_dynamics(res, ph, out):
    """Bound fraction over time and the lifetime of bound episodes."""
    ok = bound_mask(res.Ms, res.Ns); t = ph.steps; fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
    shade_phases(a, ph, labels=True); band(a, t, ok.mean(0), max(5, ph.n // 40), TEAL); a.set_ylim(0, 1.05); a.set_xlabel("MC step"); a.set_ylabel("Bound fraction"); a.set_title("Binding")
    life = []
    for r in ok:
        d = np.diff(np.r_[0, r.view(np.int8), 0]); life += list(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
    life = np.array(life) * ph.mc_step
    if len(life) > 5:
        bins = np.geomspace(max(1, life.min()), life.max() + 1, 25); b.hist(life, bins=bins, color=BLUE, alpha=.8); b.set_xscale("log")
        b.axvline(np.median(life), color=ROYAL, lw=2, label=f"median {np.median(life):.0f}"); b.legend()
    b.set_xlabel("Bound time (MC steps)"); b.set_ylabel("Episodes"); b.set_title("Lifetimes")
    return save_fig(fig, out, SUB, "lef_dynamics", "lef_dynamics")

def _domains(s):
    """Sizes of maximal runs of beads with the same compartment sign (neutral beads break runs)."""
    sg = np.sign(s); out = []; run = 0
    for i in range(len(sg)):
        if i and sg[i] == sg[i - 1] and sg[i] != 0: run += 1
        else:
            if run and sg[i - 1] != 0: out.append(run)
            run = 1 if sg[i] != 0 else 0
    if run: out.append(run)
    return np.array(out)

def fig_domains(spin_traj, ph, out):
    """Compartment domain sizes: distribution before / after S and mean size over time."""
    segs = ph.segments(); fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[1, 1.5]))
    for lab, s in segs.items():
        if s.stop - s.start < 3: continue
        d = np.concatenate([_domains(spin_traj[:, j]) for j in np.linspace(s.start, s.stop - 1, min(30, s.stop - s.start)).astype(int)])
        if len(d): a.hist(kb(d), bins=kb(np.arange(1, max(12, d.max()) + 2) - .5), density=True, histtype="step", lw=2, color=PHASE_COL[lab], label=lab)
    a.set_yscale("log"); a.set_xlabel(unit_label("Domain size")); a.set_ylabel("Density"); a.set_title("Domain sizes"); a.legend()
    idx = np.unique(np.linspace(0, ph.n - 1, min(ph.n, 300)).astype(int)); m = kb(np.array([_domains(spin_traj[:, j]).mean() for j in idx]))
    shade_phases(b, ph, labels=True); b.plot(ph.x(idx), m, color=GREY, lw=.8, alpha=.6); b.plot(ph.x(idx), rolling(m, 9), color=ROYAL, lw=2)
    b.set_xlabel("MC step"); b.set_ylabel(unit_label("Mean domain size")); b.set_title("Domains grow or shrink")
    return save_fig(fig, out, SUB, "domains", "epi_domains")
