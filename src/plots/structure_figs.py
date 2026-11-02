"""3D ensemble figures: structural metrics over the trajectory and averaged contact map."""
import os
import numpy as np
import matplotlib.pyplot as plt
from .style import BLUE, ROYAL, TEAL, PURPLE, ORANGE, GREEN, rolling, save_fig, CMAP_SEQ

METRICS = [("radius_of_gyration", "Radius of gyration"), ("mean_pairwise_distance", "Mean pair distance"), ("end_to_end_distance", "End-to-end distance"),
           ("asphericity", "Asphericity"), ("contact_number", "Contacts"), ("convex_hull_volume", "Hull volume")]

def fig_structure(out):
    """Structural metrics versus structure number, two figures of 3 panels."""
    d = out + "/metadata/structural_metrics/"; res = None
    for part, (name, fid) in zip((METRICS[:3], METRICS[3:]), (("structural_metrics", "md_structure"), ("structural_metrics_2", "md_structure2"))):
        items = [(l, np.load(d + k + ".npy")) for k, l in part if os.path.exists(d + k + ".npy")]
        if not items: continue
        fig, axs = plt.subplots(1, len(items), figsize=(3.7 * len(items), 3.2), squeeze=False)
        for ax, (l, v), c in zip(axs[0], items, (BLUE, TEAL, PURPLE) if fid == "md_structure" else (ORANGE, GREEN, ROYAL)):
            v = np.asarray(v, float); ax.plot(v, color=c, lw=.7, alpha=.35); ax.plot(rolling(v, max(3, len(v) // 40)), color=c, lw=1.9); ax.set_title(l, fontsize=10.5); ax.set_xlabel("Structure #")
        res = save_fig(fig, out, "structural_metrics", name, fid)
    return res

def fig_heatmap(mat, out, name, fid, title):
    """Averaged inverse-distance map of the ensemble."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4)); lo, hi = mat.mean() - mat.std(), mat.mean() + mat.std()
    im = ax.imshow(mat, cmap=CMAP_SEQ, vmin=max(lo, 0), vmax=hi, aspect="equal"); ax.grid(False); ax.set_title(title); ax.set_xlabel("Bead"); ax.set_ylabel("Bead")
    fig.colorbar(im, fraction=.046, pad=.03, label="1 / (d + 1)")
    return save_fig(fig, out, "structural_metrics", name, fid)
