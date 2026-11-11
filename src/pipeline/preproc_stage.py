"""Stage: preprocessing of the loop data (BEDPE -> CTCF binding potentials).

Standalone example (edit the paths):
    python -m src.pipeline.preproc_stage
"""
if __package__ in (None, ""):  # run directly (e.g. VS Code "Run Python File"): make `src` importable
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "src.pipeline"

from ..preproc.bedpe import loop_stats, input_stats, occupancy_profile, ORIENT_WARN
from ..preproc.readers import read_binding
import numpy as np
from ..logger import log

def run_preprocessing(ctx):
    """Return (L, R, N_CTCF) for the region of the context; the Binding is kept in ctx.binding."""
    a = ctx.args; log.step(f"Preprocessing: CTCF input ({ctx.input_fmt})")
    motif = dict(genome=a.MOTIF_GENOME, pfm=a.MOTIF_PFM, pvalue=a.MOTIF_PVALUE, pseudocount=a.MOTIF_PSEUDOCOUNT, pad=a.MOTIF_PAD) if a.MOTIF_GENOME else None
    b = read_binding(ctx.bedpe, ctx.region, ctx.chrom, ctx.n_beads, a.SMOOTH_INPUT, a.SMOOTH_SIGMA, motif)
    ctx.binding, ctx.n_ctcf = b, b.n_sites
    if not b.has_orientation: log.alert(ORIENT_WARN)
    if b.sigma > 0: log.info(f"Input smoothed with a Gaussian of sigma = {b.sigma:.2f} beads ({b.sigma * (ctx.region[1] - ctx.region[0]) / ctx.n_beads / 1e3:.1f} kb).")
    pref = a.PREFERENTIAL_BINDING and ctx.input_fmt == "bedpe"
    if a.PREFERENTIAL_BINDING and not pref: log.alert("PREFERENTIAL_BINDING needs loops (a .bedpe file): it is switched off for peak input.")
    ctx.occupancy = occupancy_profile(ctx.bedpe, ctx.n_beads, ctx.region, ctx.chrom) if pref else None
    if ctx.occupancy is not None: log.success(f"Preferential binding on: {np.mean(ctx.occupancy > 0) * 100:.0f}% of the beads lie inside at least one loop; LEFs load there about {1 / a.PREF_BIND_FLOOR:.0f}x more often than in loop-naked areas.")
    log.table(f"Input ({ctx.input_fmt})", input_stats(ctx.bedpe, ctx.region, ctx.chrom, ctx.n_beads))
    log.table("CTCF binding potentials on the beads", loop_stats(b.L0, b.R0))
    return b.L, b.R, b.n_sites

if __name__ == "__main__":
    from .context import build_context
    from ..config.loader import get_config
    args, _ = get_config(["--bedpe-file", "my_loops.bedpe", "--region", "chr14:80835000-98674700", "--n-beads", "1000", "-v"])
    ctx = build_context(args, make_dirs=False)
    run_preprocessing(ctx)
