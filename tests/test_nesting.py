"""Families of loops (nesting)."""
import numpy as np
from src.analysis.nesting import nesting, kind_of, order_counts, nesting_table
from src.plots.style import Phases

def test_order_parent_orphan_and_duplicates():
    ms = np.array([[0], [10], [12], [50], [10], [0]]); ns = np.array([[100], [60], [30], [70], [60], [0]])   # last LEF unbound, one duplicate
    n = nesting(ms, ns)
    assert n.valid[:, 0].tolist() == [True, True, True, True, False, False]
    assert n.order[:3, 0].tolist() == [0, 1, 2] and n.order[3, 0] == 1                      # 12-30 in 10-60 in 0-100; 50-70 in 0-100
    k = kind_of(n.order, n.n_desc)[:, 0]; assert k[0] == 1 and k[1] == 2 and k[2] == 3
    assert order_counts(n)[:, 0].sum() == 4 and n.parent_len[2, 0] == 50

def test_table_per_phase():
    rng = np.random.default_rng(0); ms = rng.integers(0, 60, (12, 80)); ns = ms + rng.integers(0, 40, (12, 80))
    ph = Phases(80, 1, 10, 30, 60, True); t = nesting_table(nesting(ms, ns), ph, 7.0)
    assert t.header == ("Quantity", "G1", "S", "G2/M") and len(t.rows) == 9
