"""Stage: preprocessing of the loop data (BEDPE -> CTCF binding potentials).

Standalone example (edit the paths):
    python -m src.pipeline.preproc_stage
"""
if __package__ in (None, ""):  # run directly (e.g. VS Code "Run Python File"): make `src` importable
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "src.pipeline"

from ..preproc.bedpe import preprocessing, loop_stats, input_stats, occupancy_profile
import numpy as np
from ..logger import log

def run_preprocessing(ctx):
    """Return (L, R, N_CTCF) for the region of the context."""
    log.step("Preprocessing: loop (BEDPE) data")
    L, R, n_ctcf = preprocessing(ctx.bedpe, ctx.region, ctx.chrom, ctx.n_beads)
    ctx.n_ctcf = n_ctcf
    ctx.occupancy = occupancy_profile(ctx.bedpe, ctx.n_beads, ctx.region, ctx.chrom) if ctx.args.PREFERENTIAL_BINDING else None
    if ctx.occupancy is not None: log.success(f"Preferential binding on: {np.mean(ctx.occupancy > 0) * 100:.0f}% of the beads lie inside at least one loop; LEFs load there about {1 / ctx.args.PREF_BIND_FLOOR:.0f}x more often than in loop-naked areas.")
    log.table("Input loops (BEDPE)", input_stats(ctx.bedpe, ctx.region, ctx.chrom, ctx.n_beads))
    log.table("CTCF binding potentials on the beads", loop_stats(L, R))
    return L, R, n_ctcf

if __name__ == "__main__":
    from .context import build_context
    from ..config.loader import get_config
    args, _ = get_config(["--bedpe-file", "my_loops.bedpe", "--region", "chr14:80835000-98674700", "--n-beads", "1000", "-v"])
    ctx = build_context(args, make_dirs=False)
    run_preprocessing(ctx)
