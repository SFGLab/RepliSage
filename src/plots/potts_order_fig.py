"""Order of the epigenetic colouring over time (two figures of at most 3 panels), in plain terms."""
import numpy as np
import matplotlib.pyplot as plt
from .style import PURPLE, TEAL, BLUE, ORANGE, GREEN, GREY, shade_phases, rolling, save_fig

COLS = (PURPLE, TEAL, BLUE, ORANGE, GREEN)
# key -> (axis label, plain meaning shown above the panel, display scale, [(reference value, label)])
INFO = {"Magnetization": ("Dominance of the\nmost common colour", "0 = all five colours equally common, 1 = every bead has the same colour", 1, [(0, "even mix")]),
        "Binder cumulant": ("Binder cumulant", "low (2/3) = one colour dominates, high (about 0.93) = colours evenly mixed", 1, [(2 / 3, "one colour"), (.933, "even mix")]),
        "Synchronization": ("Agreement between\nlinked beads", "share of links (chain neighbours and loops) whose two ends have the same colour", 4, [(.2, "random colours")]),
        "Cluster order": ("Share of beads in the\nmost common colour", "1 = all beads share one colour", 1, [(.2, "even mix")]),
        "Entropy": ("Mixing of colours\n(entropy)", "0 = one colour only, 1.6 = all five colours equally common", 1, [(np.log(5), "even mix")])}

def _panel_fig(metrics, names, ph, out, name, fid):
    fig, axs = plt.subplots(len(names), 1, figsize=(10, 2.3 * len(names) + .6), sharex=True, gridspec_kw=dict(hspace=.55)); axs = np.atleast_1d(axs)
    for i, (ax, k) in enumerate(zip(axs, names)):
        lab, meaning, sc, refs = INFO.get(k, (k, "", 1, []))
        y = np.asarray(metrics[k], float) * sc; t = ph.steps[-len(y):]; c = COLS[(i + (3 if fid == "epi_order2" else 0)) % 5]
        ax.plot(t, y, color=c, lw=.7, alpha=.35); ax.plot(t, rolling(y, max(3, ph.n // 80)), color=c, lw=1.9)
        for v, rl in refs: ax.axhline(v, color=GREY, lw=.9, ls=":"); ax.text(1.005, v, rl, transform=ax.get_yaxis_transform(), fontsize=7.5, color=GREY, va="center")
        shade_phases(ax, ph, labels=i == 0); ax.set_ylabel(lab, fontsize=9); ax.set_title(meaning, fontsize=8.5, color=GREY, loc="left", pad=18 if i == 0 else 4)
    axs[-1].set_xlabel("MC step")
    return save_fig(fig, out, "graph_metrics", name, fid)

def fig_potts_order(metrics, ph, out):
    """Dominance, Binder cumulant, link agreement; then share of the most common colour and mixing."""
    _panel_fig(metrics, ["Magnetization", "Binder cumulant", "Synchronization"], ph, out, "potts_order", "epi_order")
    return _panel_fig(metrics, ["Cluster order", "Entropy"], ph, out, "potts_order_2", "epi_order2")
