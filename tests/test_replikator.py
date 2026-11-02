"""Replication simulator on the built-in single-cell data."""
import numpy as np
from pathlib import Path
from src.replication_simulation.replikator import Replikator
from tests.conftest import REGION

DATA = str(Path(__file__).resolve().parents[1] / "src" / "data" / "replication_timing_data.parquet")

def test_replikator_runs_and_is_monotone():
    np.random.seed(0)
    sim = Replikator(DATA, 60, 40, "chr14", list(REGION), sc=True)
    f, lf, rf = sim.run()
    f = np.asarray(f)
    assert f.shape == (60, 40) and set(np.unique(f)) <= {0, 1}
    assert (np.diff(f, axis=1) >= 0).all()          # a replicated bead stays replicated
    assert f[:, -1].mean() > 0.5                    # most of the region is replicated at the end
