"""Invariants, determinism, replicas and qualitative physics of the MC."""
import numpy as np
import pytest
from src.MC.runner import run_mc
from tests.conftest import make_params

def run(p, seed=3, **kw):
    return run_mc(p, seed=seed, **kw)[0]

def test_deterministic_by_seed():
    p = make_params(rep=True, potts=True, n_lef2=3)
    a, b, c = run(p, 5), run(p, 5), run(p, 6)
    assert np.array_equal(a.Ms, b.Ms) and np.array_equal(a.Es, b.Es)
    assert not np.array_equal(a.Ms, c.Ms)

@pytest.mark.parametrize("rep,potts", [(False, False), (True, False), (True, True)])
def test_invariants(rep, potts):
    p = make_params(n_lef2=3 if rep else 0, rep=rep, potts=potts)
    r = run(p)
    bound = (r.Ms >= 0) & (r.Ns >= 0)
    assert (r.Ms[bound] <= r.Ns[bound]).all() and (r.Ns < p.N_beads).all() and (r.Ms < p.N_beads).all()
    assert set(np.unique(r.spin_traj)) <= {-2, -1, 0, 1, 2}
    assert 0 <= r.acceptance_rate <= 1
    assert np.isfinite(r.Es).all() and np.isfinite(r.Fs).all()

def test_no_replication_gives_flat_potts():
    r = run(make_params(rep=False))
    assert np.all(r.Es_potts == 0)

def test_threaded_replicas_equal_serial():
    p = make_params(rep=True, potts=True)
    ser = run_mc(p, n_replicas=3, n_threads=1, seed=10)
    thr = run_mc(p, n_replicas=3, n_threads=3, seed=10)
    for a, b in zip(ser, thr):
        assert np.array_equal(a.Ms, b.Ms) and np.array_equal(a.Es, b.Es)
    assert not np.array_equal(ser[0].Ms, ser[1].Ms)            # replicas are independent

def test_replica_seeds_are_consecutive():
    rs = run_mc(make_params(), n_replicas=2, seed=7)
    assert [r.seed for r in rs] == [7, 8]

def test_folding_coefficient_lengthens_loops():
    lo, hi = (np.mean(run(make_params(f=f, steps=4000)).Ns - run(make_params(f=f, steps=4000)).Ms) for f in (0.1, 5.0))
    assert hi > lo

def test_binding_coefficient_puts_legs_on_ctcf():
    def frac(b):
        p = make_params(b=b, f=0.5, steps=4000)
        r = run(p); on = ((p.L > 0)[r.Ms[r.Ms >= 0]]).mean()
        return on
    assert frac(8.0) > frac(0.0)

def test_higher_temperature_raises_acceptance():
    assert run(make_params(T=5.0)).acceptance_rate > run(make_params(T=0.5)).acceptance_rate

def test_second_family_inactive_before_replication():
    p = make_params(n_lef2=5, rep=True)
    r = run(p)
    k = p.N_lef                                    # family-2 LEFs come after the first N_lef rows
    pre = slice(0, p.t_rep // p.MC_step)
    assert (r.Ms[k:, pre] < 0).all() or (r.Ns[k:, pre] <= r.Ms[k:, pre]).all()

def test_replication_penalty_reduces_violations():
    v = lambda c: run(make_params(rep=True, c_rep=c, steps=3000, repfork_push=False)).S_viols.sum()
    assert v(10.0) <= v(0.0)
