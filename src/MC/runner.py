"""Run one or several independent MC chains (replicas), optionally in parallel threads.

A single Metropolis chain is inherently sequential (each move depends on the previous state),
so parallelism is over independent chains: the numba kernel releases the GIL (nogil) and
`n_threads` chains run at the same time.

Example (tiny synthetic system, LEF only):
    python src/MC/runner.py      (or: python -m src.MC.runner)
"""
if __package__ in (None, ""):  # run directly (e.g. VS Code "Run Python File"): make `src` importable
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "src.MC"

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .engine import run_energy_minimization
from .normalization import energy_norms
from .results import MCResult
from ..logger import log

def _kernel_args(p, seed, progress):
    fold, fold2, bind, k, rep, p1, p2 = energy_norms(p)
    f_T = p.f_rep_T if p.has_rep else np.zeros((1, p.N_beads), dtype=np.int8)
    h = p.h if p.h is not None else np.zeros(p.N_beads)
    cdf = np.cumsum(p.bind_w[:-1], dtype=np.float64) if p.bind_w is not None else np.zeros(1)      # loading CDF over beads 0..N-2 (size 1 = uniform)
    return dict(N_steps=p.N_steps, N_sweep=p.N_sweep, N_lef=p.N_lef, N_lef2=p.N_lef2, N_beads=p.N_beads,
                MC_step=p.MC_step, T=p.T, L=p.L, R=p.R, k_norm=k, fold_norm=fold, fold_norm2=fold2, bind_norm=bind,
                rep_norm=rep, t_rep=p.t_rep, rep_duration=p.rep_duration, f_rep_T=f_T, has_rep=p.has_rep,
                potts_norm1=p1, potts_norm2=p2, h=h, rw=p.rw, p_rew=p.p_rew, rep_fork_organizers=p.rep_fork_organizers,
                cohesin_blocks_condensin=p.cohesin_blocks_condensin, random_spins=p.random_spins,
                repfork_push=p.repfork_push, seed=seed, progress=progress, bind_cdf=cdf)

def _one_chain(p, seed, progress):
    t0 = time.time()
    out = run_energy_minimization(**_kernel_args(p, seed, progress))
    return MCResult(*out, seed=seed, elapsed=time.time() - t0)

def run_mc(p, n_replicas=1, n_threads=1, seed=None):
    """Run n_replicas independent chains; returns a list of MCResult."""
    seeds = [(-1 if seed is None else int(seed) + k) for k in range(n_replicas)]
    progs = [np.zeros(1, dtype=np.int64) for _ in range(n_replicas)]
    workers = max(1, min(int(n_threads), n_replicas))
    log.diag(f"MC: {n_replicas} replica(s), {workers} thread(s), seeds={seeds}")
    log.trace("The first call compiles the numba kernels (cached afterwards).")
    total = p.N_steps * n_replicas
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one_chain, p, s, pr) for s, pr in zip(seeds, progs)]
        bar = log.progress(total=total, desc="Monte Carlo", unit="step", unit_scale=True)
        done_prev = 0
        while not all(f.done() for f in futs):
            now = int(sum(pr[0] for pr in progs)); bar.update(now - done_prev); done_prev = now
            time.sleep(0.2)
        bar.update(total - done_prev); bar.close()
        return [f.result() for f in futs]   # re-raises any kernel error

if __name__ == "__main__":  # tiny demo: 200 beads, loop extrusion only
    from .params import MCParams
    from .results import summary_rows
    from ..logger import setup_logger
    setup_logger(1)
    rng = np.random.default_rng(0)
    N = 200; L = np.zeros(N); R = np.zeros(N); i = rng.choice(N, 30, replace=False); L[i] = R[i] = rng.random(30) * 5
    p = MCParams(N_beads=N, L=L, R=R, N_lef=20, N_steps=2000, N_sweep=200, MC_step=50, burnin=100, T=1.8)
    res = run_mc(p, n_replicas=2, n_threads=2, seed=1)
    log.table("MC demo", summary_rows(res[0], p, 30))
