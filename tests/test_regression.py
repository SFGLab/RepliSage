"""Golden values of the current kernel for fixed seeds (guards against accidental model changes)."""
import hashlib
import numpy as np
from src.MC.runner import run_mc
from tests.conftest import make_params

def digest(r):
    h = hashlib.sha256()
    for a in (r.Ms, r.Ns, r.Es, r.spin_traj): h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]

CASES = {
    "le": dict(),
    "rep_potts": dict(rep=True, potts=True, n_lef2=4),
}
GOLDEN = {"le": "f8dc82c4cfe85e7a", "rep_potts": "f8d12eca0861426f"}

def test_golden():
    for name, kw in CASES.items():
        r = run_mc(make_params(**kw), seed=2024)[0]
        assert digest(r) == GOLDEN[name], f"{name}: {digest(r)}"
