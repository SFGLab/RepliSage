"""All Potts / graph metrics of an MC run (data only; figure in plots/potts_order_fig.py)."""
from .synchronization import get_synch_ensemble
from .potts_order import magnetization, cluster_order, binder_cumulant, entropy_order

def compute_potts_metrics(Ms, Ns, Cs, path):
    """Order parameters and synchronization per MC sample (saved as .npy in metadata/graph_metrics)."""
    import numpy as np
    sync = get_synch_ensemble(Ms, Ns, Cs, path); np.save(path + '/metadata/graph_metrics/sync.npy', sync)
    return {"Magnetization": magnetization(Cs + 2, q=5, out_path=path), "Cluster order": cluster_order(Cs + 2, out_path=path),
            "Binder cumulant": binder_cumulant(Cs + 2, q=5, out_path=path), "Entropy": entropy_order(Cs + 2, q=5, out_path=path),
            "Synchronization": sync}
