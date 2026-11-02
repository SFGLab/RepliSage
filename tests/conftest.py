"""Shared fixtures: synthetic systems, loops file, run helpers. Everything runs on CPU in seconds."""
import os
import sys
import pathlib
import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.MC.params import MCParams          # noqa: E402

REGION = (80835000, 98674700)               # chr14 region covered by the built-in single-cell data

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs OpenMM / the full pipeline (deselect with -m 'not slow')")

def make_ctcf(N, n_sites=40, seed=0):
    rng = np.random.default_rng(seed)
    L, R = np.zeros(N), np.zeros(N)
    idx = rng.choice(N, n_sites, replace=False)
    L[idx], R[idx] = rng.random(n_sites) * 10, rng.random(n_sites) * 10
    return L, R

def make_fork_matrix(N, dur, seed=0):
    """Replication state f[t, bead]: every bead flips 0 -> 1 once (monotone), int8, time-major."""
    rng = np.random.default_rng(seed)
    t_b = rng.permutation(np.sort(rng.integers(0, dur, N)))
    return np.ascontiguousarray((np.arange(dur)[:, None] >= t_b[None, :]).astype(np.int8))

def make_params(N=120, n_lef=12, n_lef2=0, rep=False, potts=False, steps=1200, sweep=150, mc=20, **kw):
    L, R = make_ctcf(N, max(5, N // 5))
    p = dict(N_beads=N, L=L, R=R, N_lef=n_lef, N_lef2=n_lef2, N_steps=steps, N_sweep=sweep, MC_step=mc,
             burnin=2 * mc, T=1.8, f=1.0, f2=2.0 if n_lef2 else 0.0, b=1.0, kappa=1.0)
    if rep:
        dur = 400
        rng = np.random.default_rng(1)
        p.update(t_rep=300, rep_duration=dur, f_rep_T=make_fork_matrix(N, dur), c_rep=1.0, h=2 * rng.random(N) - 1)
    if potts:
        p.update(c_potts1=1.0, c_potts2=1.0)
    p.update(kw)
    return MCParams(**p)

@pytest.fixture
def params():
    return make_params

@pytest.fixture(scope="session")
def bedpe_file(tmp_path_factory):
    """Synthetic loops (with CTCF orientation columns) on chr14 inside REGION."""
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(300):
        a = int(rng.integers(REGION[0], REGION[1] - 2e5)); b = int(a + rng.integers(2e4, 2e5))
        rows.append(f"chr14\t{a}\t{a + 2000}\tchr14\t{b}\t{b + 2000}\t{int(rng.integers(10, 60))}\t{rng.random():.4f}\t{rng.random():.4f}")
    f = tmp_path_factory.mktemp("data") / "loops.bedpe"
    f.write_text("\n".join(rows) + "\n")
    return str(f)

@pytest.fixture
def cli(bedpe_file, tmp_path):
    """Base command-line arguments of a tiny loop-extrusion-only run (extend/override per test)."""
    def build(**over):
        a = {"bedpe_path": bedpe_file, "rept_path": "", "use_sc_rept": "False", "chrom": "chr14",
             "region_start": str(REGION[0]), "region_end": str(REGION[1]), "n_beads": "100", "n_lef": "10", "n_lef2": "0",
             "n_steps": "1000", "mc_step": "50", "burnin": "100", "n_sweep": "100", "simulation_type": "",
             "save_plots": "False", "viz_heats": "False", "out_path": str(tmp_path / "results"), "seed": "1"}
        a.update({k: str(v) for k, v in over.items()})
        ini = tmp_path / f"cfg_{abs(hash(tuple(sorted(a.items()))))}.ini"      # parameters go through a config file
        ini.write_text("[all]\n" + "".join(f"{k} = {v}\n" for k, v in a.items()))
        return ["-c", str(ini)]
    return build
