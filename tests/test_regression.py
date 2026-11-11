"""Golden values of the current kernel for fixed seeds (guards against accidental model changes)."""
import hashlib
import numpy as np
from src.MC.runner import run_mc
from tests.conftest import make_params

def digest(r):
    h = hashlib.sha256()
    for a in (r.Ms, r.Ns, np.round(r.Es, 6), r.spin_traj): h.update(np.ascontiguousarray(a).tobytes())   # Es rounded: round-off safe
    return h.hexdigest()[:16]

CASES = {
    "le": dict(),
    "rep_potts": dict(rep=True, potts=True, n_lef2=4),
}
GOLDEN = {"le": "de5f1783a21ed34c", "rep_potts": "3a387edb4f625f0d"}

def test_golden():
    for name, kw in CASES.items():
        r = run_mc(make_params(**kw), seed=2024)[0]
        assert digest(r) == GOLDEN[name], f"{name}: {digest(r)}"
