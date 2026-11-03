"""MC diagnostics: traces with distributions, autocorrelation, convergence, running means, replicas."""
import numpy as np
import matplotlib.pyplot as plt
from .style import BLACK, ROYAL, GREY, BLUE, TEAL, GOLD, PURPLE, ORANGE, GREEN, SLATE, PHASE_COL, shade_phases, rolling, band, centred, zoom, save_fig
from ..analysis.mc_stats import autocorr, iat, ess, diag_segment, split_rhat

SUB = "MCMC_diagnostics"
SHORT = {"Total energy": "Energy", "Folding energy": "Folding", "Binding energy": "Binding", "Epigenetic energy": "Epigenetic",
         "Mean loop length": "Loop length", "Bound loop length": "Bound loop", "Bound LEFs": "Bound", "Average epigenetic state": "Mean state"}

def _traces(obs, ph, out, keys, name, fid, cols, size=2.1):
    """Stacked traces (light) with moving average (bold), at most 3 panels."""
    keys = [k for k in keys if k in obs]
    if not keys: return None
    fig, axs = plt.subplots(len(keys), 1, figsize=(10, size * len(keys) + .6), sharex=True); axs = np.atleast_1d(axs); t = ph.steps; w = max(3, ph.n // 80)
    for r, (ax, k) in enumerate(zip(axs, keys)):
        y = np.asarray(obs[k], float); shade_phases(ax, ph, labels=r == 0); lab = SHORT.get(k, k)
        y, pre = centred(y, ph); lab = pre + lab
        c = cols[r % len(cols)]; ax.plot(t, y, color=c, lw=.5, alpha=.25); band(ax, t, y, w, c); ax.set_ylabel(lab); zoom(ax, y, ph)
    axs[-1].set_xlabel("MC step")
    return save_fig(fig, out, SUB, name, fid)

def fig_energy(obs, ph, out):
    """Total, folding and binding energy over time."""
    return _traces(obs, ph, out, ("Total energy", "Folding energy", "Binding energy"), "energies", "mc_energy", (BLUE, TEAL, PURPLE))

def fig_energy_potts(obs, ph, out):
    """Epigenetic energy and average epigenetic state over time."""
    return _traces(obs, ph, out, ("Epigenetic energy", "Average epigenetic state"), "energies_potts", "mc_energy_potts", (ORANGE, GREEN))

def fig_distributions(obs, ph, out, keys=("Total energy", "Folding energy", "Mean loop length")):
    """Distribution of three observables per phase (violins) or after burn-in (histogram)."""
    from .loops import _violin
    keys = [k for k in keys if k in obs]; fig, axs = plt.subplots(1, len(keys), figsize=(3.8 * len(keys), 3.6), gridspec_kw=dict(wspace=.5)); axs = np.atleast_1d(axs)
    for ax, k in zip(axs, keys):
        y, pre = centred(obs[k], ph); _violin(ax, {l: y[s] for l, s in ph.segments().items()}, pre + SHORT.get(k, k))
    return save_fig(fig, out, SUB, "distributions", "mc_distributions")

def fig_acf(obs, ph, out, keys=("Total energy", "Mean loop length", "Folding energy"), max_lag=80):
    """Autocorrelation curves (3 panels): one per phase, or one per observable without replication."""
    keys = [k for k in keys if k in obs]; segs = ph.segments(); cols = [BLUE, ROYAL, TEAL]
    panels = list(segs.items()) if ph.has_rep else [(SHORT[k], k) for k in keys]
    fig, axs = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 3.4), sharey=True); axs = np.atleast_1d(axs)
    for pi, (ax, (title, sel)) in enumerate(zip(axs, panels)):
        ax.set_title(title, color=PHASE_COL.get(title, cols[pi % 3])); ax.axhline(0, color=GREY, lw=.8)
        for i, k in enumerate(keys if ph.has_rep else [sel]):
            s = sel if ph.has_rep else segs["All"]; v = np.asarray(obs[k], float)[s]
            if len(v) < 12 or v.std() == 0: continue
            a = autocorr(v)[:min(max_lag, len(v) // 3)]; ax.plot(a, color=cols[i % 3] if ph.has_rep else cols[pi % 3], label=f"{SHORT[k]}  T_ρ={iat(v):.0f}")
        ax.set_xlabel("Lag (samples)"); ax.legend(fontsize=8)
    axs[0].set_ylabel("Autocorrelation")
    return save_fig(fig, out, SUB, "autocorrelation", "mc_acf")

def fig_convergence(diag, where, out):
    """Geweke z, effective sample size and split R-hat for every observable."""
    names = [SHORT.get(d["name"], d["name"]) for d in diag]; y = np.arange(len(diag))[::-1]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 0.55 * len(diag) + 1.6), sharey=True, gridspec_kw=dict(wspace=.4))
    z = [d["geweke"] for d in diag]; axs[0].barh(y, np.nan_to_num(z), color=[ROYAL if abs(v) > 2 else TEAL for v in np.nan_to_num(z)])
    axs[0].axvspan(-2, 2, color=GREY, alpha=.2, lw=0); axs[0].set_title("Geweke z"); axs[0].set_yticks(y, names)
    axs[1].barh(y, [d["ess"] / d["n"] * 100 if d["ess"] == d["ess"] else 0 for d in diag], color=BLUE); axs[1].set_title("ESS / n (%)")
    r = np.nan_to_num([d["rhat"] for d in diag], nan=1.0)
    axs[2].barh(y, r - 1, left=1, color=[ROYAL if v > 1.05 else (GOLD if v > 1.01 else TEAL) for v in r]); axs[2].set_title("Split R̂")
    axs[2].axvline(1.01, color=GREY, ls="--", lw=1); axs[2].set_xlim(.99, min(1.5, max(1.06, r.max() + .01)))
    for ax, vals in zip(axs, ([f"{v:.1f}" for v in np.nan_to_num(z)], [f"{d['ess']:.0f}" for d in diag], [f"{v:.3f}" for v in r])):
        for yy, tx in zip(y, vals): ax.annotate(tx, (1, yy), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color=GREY)
    fig.suptitle(f"Chain diagnostics ({where})", x=.01, ha="left", fontsize=11, color=GREY)
    return save_fig(fig, out, SUB, "convergence", "mc_convergence")

def fig_running(obs, ph, out, keys=("Total energy", "Mean loop length", "Folding energy")):
    """Running mean after burn-in; band = final mean ± 2 MCSE."""
    keys = [k for k in keys if k in obs]; seg, where = diag_segment(ph)
    fig, axs = plt.subplots(1, len(keys), figsize=(3.1 * len(keys) + .5, 3), squeeze=False)
    for ax, k, c in zip(axs[0], keys, (BLUE, TEAL, PURPLE)):
        y, pre = centred(obs[k], ph); v = y[seg]; rm = np.cumsum(v) / np.arange(1, len(v) + 1); e = ess(v)
        m = v.mean(); se = v.std() / np.sqrt(e) if e == e else 0
        ax.plot(ph.x(np.arange(len(v)) + seg.start), v, color=c, lw=.4, alpha=.2); ax.plot(ph.x(np.arange(len(v)) + seg.start), rm, color=c, lw=2); ax.axhline(m, color=SLATE, lw=1)
        ax.axhspan(m - 2 * se, m + 2 * se, color=GREY, alpha=.3, lw=0); ax.set_title(pre + SHORT.get(k, k)); ax.set_xlabel("MC step")
        lo, hi = np.percentile(v, [2, 98]); d = (hi - lo) or 1; ax.set_ylim(min(lo, m - 2 * se) - .15 * d, max(hi, m + 2 * se) + .15 * d)
    axs[0, 0].set_ylabel("Running mean")
    return save_fig(fig, out, SUB, "running_means", "mc_running")

def fig_violations(res, ph, out):
    """Conflicts between loops and replication forks in S phase."""
    s = slice(ph.s0, ph.s1); t = ph.x(np.arange(ph.n))[s]
    fig, axs = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True)
    for ax, y, lab in zip(axs, (res.N_viols[s], res.S_viols[s]), ("Conflicts (count)", "Conflict strength")):
        ax.bar(t, y, width=ph.mc_step, color=ORANGE, alpha=.4); ax.plot(t, rolling(y, max(3, len(y) // 40)), color=ROYAL, lw=1.8); ax.set_ylabel(lab)
    axs[-1].set_xlabel("MC step (S phase)")
    return save_fig(fig, out, SUB, "fork_conflicts", "rep_violations")

def fig_replicas(results, ph, out, obs_list):
    """Overlay of replicas and cross-replica R-hat."""
    keys = ["Total energy", "Mean loop length"]; seg = slice(ph.g1, ph.n)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2), gridspec_kw=dict(width_ratios=[2, 2, 1.1]))
    for ax, k in zip(axs[:2], keys):
        for i, o in enumerate(obs_list): ax.plot(ph.steps, o[k], lw=.9, alpha=.8, label=f"#{i}")
        shade_phases(ax, ph); ax.set_title(SHORT[k]); ax.set_xlabel("MC step")
    axs[0].legend(ncol=2, title="Replica", fontsize=8)
    names = list(obs_list[0]); rh = [split_rhat([np.asarray(o[k], float)[seg] for o in obs_list]) for k in names]
    y = np.arange(len(names))[::-1]; axs[2].barh(y, np.nan_to_num(rh, nan=1) - 1, left=1, color=[ROYAL if v > 1.05 else TEAL for v in np.nan_to_num(rh, nan=1)])
    axs[2].set_yticks(y, [SHORT.get(n, n) for n in names]); axs[2].set_title("R̂ across replicas"); axs[2].axvline(1.01, color=GREY, ls="--")
    return save_fig(fig, out, SUB, "replicas", "mc_replicas")
