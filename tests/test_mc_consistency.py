"""The running Metropolis energy must equal the energy recomputed from the same state (replication and Potts on or off)."""
import numpy as np
import pytest
from src.MC.runner import run_mc
from tests.conftest import make_params

CASES = [dict(), dict(n_lef2=4), dict(rep=True), dict(rep=True, potts=True), dict(rep=True, potts=True, n_lef2=4),
         dict(rep=True, potts=True, n_lef2=4, cohesin_blocks_condensin=True), dict(rep=True, potts=True, rw=False, repfork_push=False)]

@pytest.mark.parametrize("kw", CASES)
def test_running_energy_equals_recomputed(kw):
    r = run_mc(make_params(N=120, steps=3000, sweep=150, mc=20, **kw), seed=3)[0]
    assert np.max(np.abs(r.Es - r.Et)) < 1e-6 * max(1.0, np.max(np.abs(r.Et)))
