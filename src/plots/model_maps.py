"""Heatmaps from the Monte Carlo state alone: LEF contacts, epigenetic checkerboard and their product, per phase."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from .style import genome_axis, unit_label, CMAP_SEQ, CMAP_SPIN, GREY, save_fig
from ..analysis.lef_stats import bound_mask

SUB = "MCMC_diagnostics"

def lef_contacts(Ms, Ns, n, seg, sigma=.8):
    """Mean LEF contact map (beads x beads): each bound LEF contacts its two legs; lightly blurred."""
    ok = bound_mask(Ms, Ns)[:, seg]; m, k = Ms[:, seg][ok], Ns[:, seg][ok]
    C = np.bincount(m * n + k, minlength=n * n).reshape(n, n).astype(float) / max(1, Ms[:, seg].shape[1])
    return gaussian_filter(C + C.T, sigma)

def checkerboard(spins, seg):
    """Outer product of the mean compartment signal (state / 2, in [-1, 1]): + same compartment, - different."""
    s = spins[:, seg].mean(1) / 2
    return np.outer(s, s)

def background(n, alpha=1.):
    """Polymer baseline P(s) ~ s^-alpha (s = genomic distance in beads) so that the compartment pattern is visible away from loops."""
    s = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]); return 1. / (1 + s) ** alpha

def fig_model_maps(res, p, ph, out, kappa=1.5):
    """Rows: LEF contacts / epigenetic checkerboard / combined map; columns: cell-cycle phases."""
    n = p.N_beads; segs = ph.segments(); cols = list(segs); has_epi = bool(np.any(res.spin_traj != 0)) and p.has_rep
    lef = {k: lef_contacts(res.Ms, res.Ns, n, s) for k, s in segs.items()}; ref = max(np.percentile(c[c > 0], 99) if (c > 0).any() else 1 for c in lef.values())
    rows = ["LEF contacts"] + (["Epigenetic checkerboard", "Combined"] if has_epi else [])
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(3.9 * len(cols) + 1.2, 3.8 * len(rows) + .6), squeeze=False, gridspec_kw=dict(wspace=.12, hspace=.22))
    bg = background(n)
    for j, k in enumerate(cols):
        c = np.clip(lef[k] / ref, 0, 1)
        ims = [axs[0, j].imshow(np.log1p(30 * c) / np.log1p(30), cmap=CMAP_SEQ, vmin=0, vmax=1, interpolation="nearest")]
        if has_epi:
            S = checkerboard(res.spin_traj, segs[k]); ims.append(axs[1, j].imshow(S, cmap=CMAP_SPIN, vmin=-.6, vmax=.6, interpolation="nearest"))
            comb = (bg + c) * np.exp(kappa * S / max(np.abs(S).max(), 1e-9)); comb = np.log10(comb)
            ims.append(axs[2, j].imshow(comb, cmap=CMAP_SEQ, vmin=np.percentile(comb, 2), vmax=np.percentile(comb, 99.7), interpolation="nearest"))
        axs[0, j].set_title(k if k != "All" else "All samples", fontsize=12, fontweight="bold")
    for i, r in enumerate(rows):
        for j in range(len(cols)):
            a = axs[i, j]; a.grid(False); genome_axis(a, n, "x", 4); genome_axis(a, n, "y", 4)
            if i < len(rows) - 1: a.set_xticklabels([]); a.set_xlabel("")
            if j > 0: a.set_yticklabels([]); a.set_ylabel("")
        axs[i, 0].set_ylabel(f"{r}\n" + axs[i, 0].get_ylabel(), fontsize=10)
    fig.text(.5, .005, "Row 1: mean count of LEF loops joining two beads (Gaussian blur, log colour). Row 2: s_i·s_j, s = mean epigenetic state / 2. Row 3: (1/(1+|i−j|) + row 1) × exp(1.5 × row 2 / max).", ha="center", fontsize=8, color=GREY, wrap=True)
    return save_fig(fig, out, SUB, "model_maps", "model_maps")
