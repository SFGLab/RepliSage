"""RunContext: validated parameters + derived quantities shared by all stages."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ..utils.constants import chrom_sizes
from ..utils.run_dirs import resolve_out_root, run_name, make_run_dir
from ..logger import log

REPT_EXT = (".txt", ".bw", ".bigwig")

@dataclass
class RunContext:
    args: object
    chrom: str
    bedpe: str
    region: list            # [start, end] used for loops (whole chromosome if none given)
    region_given: bool      # False -> whole chromosome
    n_beads: int
    rept_path: str          # None if replication is not modelled
    is_sc: bool
    has_rep: bool
    has_md: bool
    run_dir: str = None
    n_ctcf: int = 0

    @property
    def region_coords(self):
        """Region for the replication data (None = whole chromosome)."""
        return self.region if self.region_given else None

def _chrom_end(chrom, bedpe):
    """Chromosome length: table of sizes, else the largest coordinate in the bedpe."""
    if chrom in chrom_sizes: return chrom_sizes[chrom]
    import pandas as pd
    df = pd.read_csv(bedpe, sep='\t', header=None, usecols=[0, 5])
    return int(df[df[0] == chrom][5].max())

def build_context(args, make_dirs=True):
    """Validate arguments and decide which stages run (loops always; replication/MD optionally)."""
    if not args.BEDPE_PATH: raise ValueError("BEDPE_PATH is required (the only mandatory input).")
    if not args.CHROM: raise ValueError("CHROM is required (e.g. chr14).")
    given = args.REGION_START is not None and args.REGION_END is not None
    region = [args.REGION_START, args.REGION_END] if given else [0, _chrom_end(args.CHROM, args.BEDPE_PATH)]
    if not given: log.alert(f"No REGION_START/REGION_END: modelling the WHOLE chromosome {args.CHROM} ({region[1]} bp).")
    n_beads = args.N_BEADS if args.N_BEADS else int(np.round((region[1] - region[0]) / 2000))

    rept = args.REPT_PATH
    if rept and not rept.lower().endswith(REPT_EXT + (".parquet",)):
        raise ValueError("REPT_PATH must be a .txt, .bw, .bigwig or single-cell .parquet file.")
    has_rep = bool(rept) or bool(args.USE_SC_REPT)
    rept_path = (rept or args.SC_REPT_PATH) if has_rep else None
    is_sc = has_rep and not rept_path.lower().endswith(REPT_EXT)

    sim = args.SIMULATION_TYPE
    if sim and sim not in ("MD", "EM"): raise ValueError("SIMULATION_TYPE must be 'MD', 'EM' or empty.")
    ctx = RunContext(args, args.CHROM, args.BEDPE_PATH, region, given, n_beads, rept_path, is_sc, has_rep, bool(sim))
    if make_dirs:
        ctx.run_dir = make_run_dir(resolve_out_root(args.OUT_PATH), run_name(args, has_rep, ctx.has_md))
    return ctx

def describe(ctx):
    """Rows summarising what will be run."""
    on = lambda b: "yes" if b else "no"
    return [("chromosome", ctx.chrom), ("region", f"{ctx.region[0]}-{ctx.region[1]}" if ctx.region_given else "whole chromosome"),
            ("beads", ctx.n_beads), ("loop extrusion (MC)", "yes"), ("replication + compartments", on(ctx.has_rep)),
            ("3D structures (MD/EM)", ctx.args.SIMULATION_TYPE if ctx.has_md else "no"),
            ("chains in 3D model", 2 if ctx.has_rep else 1), ("run folder", ctx.run_dir)]
