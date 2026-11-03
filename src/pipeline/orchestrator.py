"""RepliSage pipeline: runs the stages that the given inputs allow.

  loops only (BEDPE)           -> preprocessing + Monte Carlo of loop extrusion (+ 1-chain 3D model)
  + replication timing data    -> + replication simulation, Potts compartments (+ 2-chain 3D model)
  SIMULATION_TYPE = MD / EM    -> + 3D structures with OpenMM
"""
import time
from .context import build_context, describe
from .preproc_stage import run_preprocessing
from .replication_stage import run_replication
from .mc_stage import run_mc_stage
from .md_stage import run_md_stage
from .viz3d_stage import run_viz3d
from .analysis_stage import run_analysis
from ..config.writer import write_config
from ..report.builder import write_summary_report
from ..plots.style import apply_style
from ..logger import log

def run_pipeline(args, command=""):
    """Run all stages for the resolved Args object (`command` is recorded in the report)."""
    apply_style()
    ctx = build_context(args)
    log.table("Run configuration", describe(ctx), level=1)
    write_config(args, ctx.run_dir + "/config_used.ini")
    log.trace("Resolved parameters: " + ", ".join(f"{a.name}={a.val}" for a in args))
    L, R, n_ctcf = run_preprocessing(ctx)
    rep = run_replication(ctx)
    p, runs = run_mc_stage(ctx, L, R, n_ctcf, rep)
    for res, out_dir in runs:
        if ctx.has_md: run_md_stage(ctx, p, res, out_dir, rep); run_viz3d(ctx, p, res, out_dir, rep)
        run_analysis(ctx, p, res, out_dir, rep, command)
    if args.REPORT and len(runs) > 1: write_summary_report(ctx, p, runs, command)
    log.success(f"RepliSage finished in {log.total_time()}. Results: {ctx.run_dir}")
    return ctx.run_dir
