"""Potts order parameters over time (two figures of at most 3 panels)."""
import numpy as np
import matplotlib.pyplot as plt
from .style import PURPLE, TEAL, BLUE, ORANGE, GREEN, shade_phases, rolling, save_fig

COLS = (PURPLE, TEAL, BLUE, ORANGE, GREEN)

def _panel_fig(metrics, names, ph, out, name, fid):
    fig, axs = plt.subplots(len(names), 1, figsize=(10, 2 * len(names) + .6), sharex=True); axs = np.atleast_1d(axs)
    for i, (ax, k) in enumerate(zip(axs, names)):
        y = np.asarray(metrics[k], float); t = ph.steps[-len(y):]; c = COLS[(i + (3 if fid == "epi_order2" else 0)) % 5]
        ax.plot(t, y, color=c, lw=.7, alpha=.35); ax.plot(t, rolling(y, max(3, ph.n // 80)), color=c, lw=1.9)
        shade_phases(ax, ph, labels=i == 0); ax.set_ylabel(k)
    axs[-1].set_xlabel("MC step")
    return save_fig(fig, out, "graph_metrics", name, fid)

def fig_potts_order(metrics, ph, out):
    """Magnetization, Binder cumulant, synchronization; then cluster order and entropy."""
    _panel_fig(metrics, ["Magnetization", "Binder cumulant", "Synchronization"], ph, out, "potts_order", "epi_order")
    return _panel_fig(metrics, ["Cluster order", "Entropy"], ph, out, "potts_order_2", "epi_order2")
