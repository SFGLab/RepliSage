"""Stage: replication timing -> replication fork simulation -> Potts field (compartments).

Skipped automatically when no replication timing data is given.

Standalone example (built-in single-cell data):
    python -m src.pipeline.replication_stage
"""
if __package__ in (None, ""):  # run directly (e.g. VS Code "Run Python File"): make `src` importable
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "src.pipeline"

from dataclasses import dataclass
import os
import numpy as np
from ..replication_simulation.replikator import Replikator
from ..MC.replication_matrix import to_time_major
from ..logger import log

@dataclass
class ReplicationData:
    rep_frac: np.ndarray      # (N_beads, REP_SIM_T) replication simulation
    f_rep_T: np.ndarray       # (REP_TIME_DURATION, N_beads) int8, time-major
    h: np.ndarray             # Potts field from replication timing
    state: np.ndarray         # Potts initial state from the replication simulation
    l_forks: np.ndarray = None      # (N_beads, REP_SIM_T) left-fork presence
    r_forks: np.ndarray = None      # (N_beads, REP_SIM_T) right-fork presence
    init_rate: np.ndarray = None    # (N_beads, REP_SIM_T) initiation rate I(x,t)
    input_timing: np.ndarray = None # (N_beads,) earliness from the data (1 = early)
    speed: float = 0.0              # mean fork speed (simulation units)

def rt_input_rows(ctx, rep):
    """Rows describing the replication-timing input after it was mapped onto the beads (earliness: 1 = early, 0 = late)."""
    x = np.asarray(rep.avg_fx, float); a, b = ctx.region; src = "single-cell parquet (GM12878)" if ctx.is_sc else "Repli-Seq signal, log-normalised"
    return [("source", f"{src}: {os.path.basename(ctx.rept_path)}"), ("region", f"{ctx.chrom}:{a:,}-{b:,} ({(b - a) / 1e6:.2f} Mb)"),
            ("bead size", f"{(b - a) / ctx.n_beads / 1e3:.2f} kb"), ("earliness min / median / mean / max", " / ".join(f"{v:.2f}" for v in (x.min(), np.median(x), x.mean(), x.max()))),
            ("early-replicating region (earliness > 0.5)", f"{(x > .5).mean() * 100:.0f} % of the region = {(x > .5).sum() * (b - a) / ctx.n_beads / 1e3:,.0f} kb"),
            ("earliest / latest bead", f"{ctx.chrom}:{a + int(np.argmax(x)) * (b - a) // ctx.n_beads:,} / {ctx.chrom}:{a + int(np.argmin(x)) * (b - a) // ctx.n_beads:,}")]

def run_replication(ctx):
    """Return ReplicationData, or None if replication is not modelled."""
    if not ctx.has_rep: return None
    a = ctx.args
    log.step("Replication simulation (Replikator)")
    rep = Replikator(ctx.rept_path, ctx.n_beads, a.REP_SIM_T, ctx.chrom, ctx.region_coords, Tstd_factor=a.REP_T_STD_FACTOR,
                     speed_factor=a.REP_SPEED_SCALE, sc=ctx.is_sc, out_path=ctx.run_dir + '/plots/replication_simulation',
                     bw_bin_size=a.BIGWIG_BIN_SIZE, bw_sigma=a.BIGWIG_SIGMA, bw_log_norm=a.BIGWIG_LOG_NORM)
    try:
        rep_frac, l_forks, r_forks = rep.run(scale=a.REP_INIT_RATE_SCALE)
    except Exception as e:
        raise ValueError("Something went wrong with the replication simulation. Probably the region of modelling is too large "
                         "or too short for the resolution (number of beads) specified.") from e
    log.table("Replication timing input", rt_input_rows(ctx, rep))
    if a.REP_TIME_DURATION % a.REP_SIM_T:
        log.warning(f"REP_TIME_DURATION is not a multiple of REP_SIM_T={a.REP_SIM_T}: the last "
                    f"{a.REP_TIME_DURATION % a.REP_SIM_T} steps keep no replication state (same as the published code).")
    f_T = to_time_major(rep_frac, a.REP_TIME_DURATION)
    h, state = rep.calculate_ising_parameters()
    mc = ctx.run_dir + '/metadata/MCMC_output/'
    for name, arr in (("rep_frac", rep_frac), ("l_forks", l_forks), ("r_forks", r_forks), ("init_rate", rep.initiation_rate), ("input_timing", rep.avg_fx)): np.save(mc + name + '.npy', arr)
    log.table("Replication simulation", [("beads x time points", rep_frac.shape), ("fully replicated beads", f"{(rep_frac[:, -1] > 0).mean() * 100:.1f} %"),
                                          ("fork speed (mean)", f"{rep.speed_avg:.3f}"), ("MC-time matrix", f"{f_T.shape} int8, {f_T.nbytes / 1e6:.0f} MB")])
    return ReplicationData(rep_frac, f_T, h, state, l_forks, r_forks, rep.initiation_rate, np.asarray(rep.avg_fx), float(rep.speed_avg))

if __name__ == "__main__":
    from .context import build_context
    from ..config.loader import get_config
    args, _ = get_config(["--bedpe-file", "my_loops.bedpe", "--region", "chr14:80835000-98674700", "--n-beads", "200", "--rt-file", "builtin",
                          "--name", "demo_replication", "-v"], overrides={"REP_TIME_DURATION": 20000})
    ctx = build_context(args)
    run_replication(ctx)
