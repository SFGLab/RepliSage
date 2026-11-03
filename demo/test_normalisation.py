"""S-phase time normalisation (no simulation): the first firing is at the first step and the last locus is replicated at the last step."""
import sys, pathlib, types
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.replication_simulation.replikator import Replikator

def run(t_rep, T=1000):
    """Normalise a synthetic replicated-state matrix whose loci replicate at the given times (columns)."""
    L = len(t_rep); f = (np.arange(T)[None, :] >= np.asarray(t_rep)[:, None]).astype(np.int8)
    fk = np.zeros_like(f); fk[np.arange(L), np.minimum(t_rep, T - 1)] = 1            # a fork event at each locus' replication time
    stub = types.SimpleNamespace(initiation_rate=np.tile(np.arange(T, dtype=float), (L, 1)))
    g, l, r = Replikator.normalise_time(stub, f, fk, fk.copy()); return f, g, l, stub

rng = np.random.default_rng(0)
for a, b in ((250, 900), (0, 600), (400, 999), (100, 1998)):                          # late start, early end, both, and a tail cut at T
    t = rng.integers(a, min(b, 999) + 1, 300); t[0], t[1] = a, min(b, 999)             # make the extremes exist
    f, g, l, stub = run(t); T = f.shape[1]
    first = int(np.flatnonzero(g.any(0))[0]); last = int(np.flatnonzero(g.all(0))[0])
    assert g.shape == f.shape == l.shape, "shape changed"
    assert first <= 1, f"first firing at step {first}, expected the start of S"
    assert last >= T - 2, f"last locus replicated at step {last}, expected the end of S"
    assert (np.diff(g.astype(int), axis=1) >= 0).all(), "a replicated locus became unreplicated"
    assert stub.initiation_rate.shape == f.shape, "initiation rate not remapped"
    print(f"normalised profile OK: input window {a}-{min(b, 999)} -> first firing {first}, fully replicated {last} of {T}")
f, g, *_ = run(np.linspace(0, 999, 100).astype(int))                                  # already normalised: unchanged
assert (f == g).all(), "an already normalised profile must stay as it is"
print("normalisation test OK")

# replication-only check on the built-in data (no MC/MD): the simulator itself crops the idle steps before the first firing
from src.config.arg_defs import DEFAULT_REPT
rep = Replikator(DEFAULT_REPT, 200, 1000, 'chr14', [80835000, 98674700], sc=True)
f, l, r = rep.run()
assert f.any(0)[0] and f.all(0)[-1], "S phase must start with the first firing and end with the last replicated locus"
assert int(np.flatnonzero((l | r).astype(bool).any(0))[0]) <= 1, "forks must appear at the start of S"
assert rep.initiation_rate.shape == f.shape
print("simulator S-phase window OK")
