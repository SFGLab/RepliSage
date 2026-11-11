"""RunContext: validated parameters + derived quantities shared by all stages."""
from dataclasses import dataclass
from pathlib import Path
import os
import sys
import numpy as np
from ..utils.constants import chrom_sizes
from ..utils.run_dirs import resolve_out_root, run_name, make_run_dir
from ..logger import log
from ..preproc.readers import detect_format

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
    input_fmt: str = "bedpe"   # bedpe | bed | narrowpeak
    binding: object = None  # Binding of the input file (set by the preprocessing stage)

    @property
    def region_coords(self):
        """Region for the replication data (None = whole chromosome)."""
        return self.region if self.region_given else None

def _chrom_end(chrom, bedpe):
    """Chromosome length: table of sizes, else the largest coordinate in the input file."""
    if chrom in chrom_sizes: return chrom_sizes[chrom]
    import pandas as pd
    end = 5 if detect_format(bedpe) == "bedpe" else 2
    df = pd.read_csv(bedpe, sep='\t', header=None, usecols=[0, end], comment='#')
    return int(df[df[0] == chrom][end].max())

def _ask_rt(args):
    """No timing data given: ask (terminal only) whether to use the bundled GM12878 single-cell RT or loop extrusion alone."""
    msg = ("\nNo replication timing (RT) file was given. What should RepliSage do?\n"
           "  [1] use the built-in single-cell RT of GM12878 (parquet): loop extrusion + replication + compartments (G1, S, G2/M)\n"
           "  [2] loop extrusion only: no replication, no S and G2/M phases\n")
    print(msg)
    while True:
        try: ans = input("Choose 1 or 2 [default 2]: ").strip() or "2"
        except EOFError: ans = "2"
        if ans in ("1", "2"): break
        print("Please type 1 or 2.")
    return ans == "1"

def _resolve_rt(args):
    """Return (rept_path, has_rep); validates the file and handles the missing-file / no-file cases."""
    rept = args.REPT_PATH
    if rept and not rept.lower().endswith(REPT_EXT + (".parquet",)):
        raise ValueError("REPT_PATH must be a .txt, .bw, .bigwig or single-cell .parquet file.")
    if rept and not os.path.exists(rept):   # missing timing file: fall back to the bundled single-cell data
        log.alert(f"REPT_PATH '{rept}' not found; using the bundled GM12878 single-cell replication timing instead.")
        rept = None; args.USE_SC_REPT = True
    if not rept and not args.USE_SC_REPT:
        if args.ASK_RT and sys.stdin.isatty() and sys.stdout.isatty(): args.USE_SC_REPT = _ask_rt(args)
        if not args.USE_SC_REPT: log.note("No replication timing: loop extrusion only (no replication, S or G2/M phase).")
        else: log.info("Using the built-in GM12878 single-cell replication timing.")
    has_rep = bool(rept) or bool(args.USE_SC_REPT)
    return ((rept or args.SC_REPT_PATH) if has_rep else None), has_rep

BEAD_BP, MAX_BEADS = 7000, 10000   # ~10 beads per CTCF site at ~14 sites/Mb (7 kb/bead); cap = paper's chromosome-wide 10,000 beads

def default_beads(length):
    """7 kb per bead (scales with region length only); regions needing more than MAX_BEADS are coarsened (rounded to 1 kb)."""
    n = int(np.round(length / BEAD_BP))
    if n <= MAX_BEADS: return max(n, 1)
    bp = int(np.ceil(length / MAX_BEADS / 1000) * 1000)
    log.alert(f"Region of {length/1e6:.1f} Mb would need {n} beads at {BEAD_BP/1000:.0f} kb: using {MAX_BEADS} beads of ~{bp/1000:.0f} kb. Give --n-beads to change.")
    return MAX_BEADS

def check_resolution(bp):
    """Error for unusable bead sizes, alert outside the 1-10 kb range where loops (~100 kb) span >=10 beads."""
    if bp > 50_000 or bp < 200:
        raise ValueError(f"Bead size {bp:,.0f} bp is outside the usable range (200 bp - 50 kb): change --n-beads or the region.")
    if bp > 10_000: log.alert(f"Coarse resolution ({bp/1000:.0f} kb/bead): typical ~100 kb loops span fewer than 10 beads.")
    elif bp < 1000: log.alert(f"Very fine resolution ({bp:.0f} bp/bead): the MC needs many more steps to equilibrate.")

def build_context(args, make_dirs=True):
    """Validate arguments and decide which stages run (loops always; replication/MD optionally)."""
    if not args.BEDPE_PATH:
        raise ValueError("No input file given (--bedpe-file, or --input-file). RepliSage relies heavily on modelling loop extrusion, and without a "
                         "BEDPE, BED or narrowPeak file it does not know where the CTCF barriers are.")
    if not os.path.isfile(args.BEDPE_PATH): raise FileNotFoundError(f"Input file not found: {args.BEDPE_PATH}")
    fmt = detect_format(args.BEDPE_PATH)
    if args.HIC_FILE and not os.path.isfile(args.HIC_FILE): raise FileNotFoundError(f"Hi-C file (HIC_FILE) not found: {args.HIC_FILE}")
    if args.MOTIF_GENOME and not os.path.isfile(args.MOTIF_GENOME): raise FileNotFoundError(f"Reference genome (MOTIF_GENOME) not found: {args.MOTIF_GENOME}")
    if not args.CHROM: raise ValueError("CHROM is required (e.g. chr14), or give --region chr14:start-end.")
    given = args.REGION_START is not None and args.REGION_END is not None
    region = [args.REGION_START, args.REGION_END] if given else [0, _chrom_end(args.CHROM, args.BEDPE_PATH)]
    if given and region[1] <= region[0]: raise ValueError(f"Invalid region: end ({region[1]}) must be larger than start ({region[0]}).")
    if not given: log.alert(f"No REGION_START/REGION_END: modelling the WHOLE chromosome {args.CHROM} ({region[1]:,} bp).")
    n_beads = args.N_BEADS if args.N_BEADS else default_beads(region[1] - region[0])
    check_resolution((region[1] - region[0]) / n_beads)
    rept_path, has_rep = _resolve_rt(args)
    is_sc = has_rep and not rept_path.lower().endswith(REPT_EXT)

    sim = args.SIMULATION_TYPE
    if sim and sim not in ("MD", "EM"): raise ValueError("SIMULATION_TYPE must be 'MD', 'EM' or none.")
    ctx = RunContext(args, args.CHROM, args.BEDPE_PATH, region, given, n_beads, rept_path, is_sc, has_rep, bool(sim), input_fmt=fmt)
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
