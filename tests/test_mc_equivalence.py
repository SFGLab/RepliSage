"""The optimised kernel must reproduce the published kernel exactly (same seed -> identical arrays)."""
import numpy as np
import pytest
from src.MC.engine import run_energy_minimization as new_kernel
from src.MC.normalization import energy_norms
from tests.reference import legacy_kernel as old
from tests.conftest import make_params

NAMES = "Ms Ns Es Es_potts Fs Bs spin_traj mags acceptance N_viols S_viols".split()

def run_both(p, seed=1234):
    fold, fold2, bind, k, rep, p1, p2 = energy_norms(p)
    h = p.h if p.h is not None else np.zeros(p.N_beads)
    common = dict(N_steps=p.N_steps, N_sweep=p.N_sweep, N_lef=p.N_lef, N_lef2=p.N_lef2, N_beads=p.N_beads, MC_step=p.MC_step,
                  T=p.T, L=p.L, R=p.R, k_norm=k, fold_norm=fold, fold_norm2=fold2, bind_norm=bind, rep_norm=rep,
                  t_rep=p.t_rep, rep_duration=p.rep_duration, potts_norm1=p1, potts_norm2=p2, h=h, rw=p.rw, p_rew=p.p_rew,
                  rep_fork_organizers=p.rep_fork_organizers, cohesin_blocks_condensin=p.cohesin_blocks_condensin,
                  random_spins=p.random_spins, repfork_push=p.repfork_push)
    f_T = p.f_rep_T if p.has_rep else np.zeros((1, p.N_beads), np.int8)
    o = old.run_energy_minimization(f_rep=np.ascontiguousarray(f_T.T.astype(np.int64)), seed=seed, **common)
    n = new_kernel(f_rep_T=f_T, has_rep=p.has_rep, seed=seed, progress=np.zeros(1, np.int64), bind_cdf=np.zeros(1), **common)
    return o, n

def assert_identical(o, n):
    for name, a, b in zip(NAMES, o, n):
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"{name} differs from the published kernel"

@pytest.mark.parametrize("rep,potts,n_lef2", [(True, True, 5), (True, False, 0), (False, False, 0), (True, True, 0)])
def test_identical_to_published_kernel(rep, potts, n_lef2):
    # rep=False cannot run in the legacy kernel without a matrix, so it gets a dummy zero matrix with rep_norm = 0
    p = make_params(n_lef2=n_lef2, rep=rep, potts=potts)
    if not rep: p.f_rep_T = None
    o, n = run_both(p) if rep else run_both_no_rep(p)
    assert_identical(o, n)

def run_both_no_rep(p):
    """No replication: legacy needs an array -> dummy zeros (rep_norm = 0, no pushing effect), same trajectory expected."""
    p.t_rep, p.rep_duration = 10 ** 9, 1
    fold, fold2, bind, k, rep, p1, p2 = energy_norms(p)
    common = dict(N_steps=p.N_steps, N_sweep=p.N_sweep, N_lef=p.N_lef, N_lef2=p.N_lef2, N_beads=p.N_beads, MC_step=p.MC_step,
                  T=p.T, L=p.L, R=p.R, k_norm=k, fold_norm=fold, fold_norm2=fold2, bind_norm=bind, rep_norm=0.0,
                  t_rep=p.t_rep, rep_duration=p.rep_duration, potts_norm1=0.0, potts_norm2=0.0, h=np.zeros(p.N_beads), rw=True,
                  p_rew=0.5, rep_fork_organizers=True, cohesin_blocks_condensin=False, random_spins=True, repfork_push=False)
    o = old.run_energy_minimization(f_rep=np.zeros((p.N_beads, 1), np.int64), seed=77, **common)
    n = new_kernel(f_rep_T=np.zeros((1, p.N_beads), np.int8), has_rep=False, seed=77, progress=np.zeros(1, np.int64), bind_cdf=np.zeros(1), **common)
    return o, n

@pytest.mark.parametrize("rw,cbc,rfo", [(False, False, True), (True, True, False), (True, False, False)])
def test_identical_for_option_variants(rw, cbc, rfo):
    p = make_params(n_lef2=4, rep=True, potts=True, rw=rw, cohesin_blocks_condensin=cbc, rep_fork_organizers=rfo)
    o, n = run_both(p, seed=5)
    assert_identical(o, n)

def test_identical_without_fork_pushing_and_fixed_spins():
    p = make_params(n_lef2=3, rep=True, potts=True, repfork_push=False, random_spins=False, p_rew=0.8)
    o, n = run_both(p, seed=9)
    assert_identical(o, n)
