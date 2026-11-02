"""Stage: Monte Carlo of loop extrusion (+ replication barriers + Potts states when replication data exists).

Standalone example (loops only; edit the BEDPE path):
    python -m src.pipeline.mc_stage
"""
if __package__ in (None, ""):  # run directly (e.g. VS Code "Run Python File"): make `src` importable
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "src.pipeline"

from pathlib import Path
import numpy as np
from ..MC.params import MCParams
from ..MC.runner import run_mc
from ..MC.results import summary_rows, check_acceptance, save_mc
from ..utils.run_dirs import make_run_dir
from ..logger import log

def build_mc_params(ctx, L, R, n_ctcf, rep):
    """MCParams from the arguments; replication/Potts terms are switched off without replication data."""
    a = ctx.args
    n_lef = a.N_LEF if a.N_LEF else 2 * n_ctcf
    p = MCParams(N_beads=ctx.n_beads, L=L, R=R, N_lef=n_lef, N_lef2=a.N_LEF2, N_steps=a.N_STEPS, N_sweep=a.N_SWEEP,
                 MC_step=a.MC_STEP, burnin=a.BURNIN, T=a.T_MC, f=a.FOLDING_COEFF, f2=a.FOLDING_COEFF2, b=a.BIND_COEFF,
                 kappa=a.CROSS_COEFF, c_rep=a.REP_COEFF, c_potts1=a.EPI_FIELD_COEFF, c_potts2=a.EPI_INTERACT_COEFF,
                 rw=a.LEF_RW, p_rew=a.P_REW, rep_fork_organizers=a.REP_FORK_EPIGENETIC_ORGANIZER,
                 cohesin_blocks_condensin=a.COHESIN_BLOCKS_CONDENSIN, random_spins=a.RANDOM_INIT_SPINS,
                 repfork_push=a.REP_FORK_PUSH, t_rep=a.REP_START_TIME, rep_duration=a.REP_TIME_DURATION,
                 f_rep_T=None if rep is None else rep.f_rep_T, h=None if rep is None else rep.h)
    if rep is not None and a.REP_START_TIME + a.REP_TIME_DURATION > a.N_STEPS:
        log.alert("REP_START_TIME + REP_TIME_DURATION exceeds N_STEPS: replication will not finish within the simulation.")
    return p

def replica_dirs(ctx, n):
    """Run folder, or run_folder/replica_XX for several replicas."""
    if n == 1: return [ctx.run_dir]
    return [make_run_dir(ctx.run_dir, f"replica_{k:02d}") for k in range(n)]

def run_mc_stage(ctx, L, R, n_ctcf, rep):
    """Run the MC; returns (params, [(result, replica_dir), ...])."""
    a = ctx.args
    log.step("Monte Carlo: loop extrusion" + (" + replication + compartments" if rep is not None else " only"))
    p = build_mc_params(ctx, L, R, n_ctcf, rep)
    res = run_mc(p, n_replicas=a.N_REPLICAS, n_threads=a.N_THREADS, seed=a.SEED)
    dirs = replica_dirs(ctx, a.N_REPLICAS)
    for k, (r, d) in enumerate(zip(res, dirs)):
        check_acceptance(r.acceptance_rate)
        log.table(f"MC summary (replica {k})" if len(res) > 1 else "MC summary", summary_rows(r, p, n_ctcf, (ctx.region[1] - ctx.region[0]) / ctx.n_beads / 1e3))
        if a.SAVE_MDT: save_mc(r, p, d)
    log.success(f"Monte Carlo finished ({len(res)} replica(s), {max(r.elapsed for r in res):.1f} s per chain).")
    return p, list(zip(res, dirs))

if __name__ == "__main__":
    from .context import build_context
    from .preproc_stage import run_preprocessing
    from ..config.loader import get_config
    args, _ = get_config(["--bedpe-file", "my_loops.bedpe", "--region", "chr14:80835000-98674700", "--n-beads", "500", "--n-steps", "20000",
                          "--name", "demo_mc", "-v"], overrides={"N_LEF": 50, "MC_STEP": 100})
    ctx = build_context(args)
    L, R, n = run_preprocessing(ctx)
    run_mc_stage(ctx, L, R, n, None)
