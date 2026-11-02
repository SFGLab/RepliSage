"""Run folder naming and creation: results/<chrom>_<start>-<end>_<tag>/."""
import datetime
from pathlib import Path
from .constants import chrom_sizes
from ..logger import log

SUBFOLDERS = ['plots', 'metadata', 'ensemble', 'plots/MCMC_diagnostics', 'plots/structural_metrics',
              'plots/graph_metrics', 'plots/md_simulation', 'plots/replication_simulation',
              'metadata/energy_factors', 'metadata/MCMC_output', 'metadata/structural_metrics',
              'metadata/graph_metrics', 'metadata/md_dynamics']

def project_root():
    """Repo root if running from a source checkout (has setup.py), else the current folder."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "setup.py").exists() else Path.cwd()

def resolve_out_root(out_path):
    p = Path(out_path or "results")
    return p if p.is_absolute() else project_root() / p

def region_tag(chrom, start, end):
    """'chr14_40535000-60174700' or 'chr14_whole-chromosome'."""
    if start is None or end is None: return f"{chrom}_whole-chromosome"
    return f"{chrom}_{start}-{end}"

def mode_tag(has_rep, has_md):
    return "LE" + ("-RT" if has_rep else "") + ("-MD" if has_md else "")

def run_name(args, has_rep, has_md):
    if args.RUN_NAME: return args.RUN_NAME
    parts = [region_tag(args.CHROM, args.REGION_START, args.REGION_END), mode_tag(has_rep, has_md)]
    parts.append(args.RUN_TAG if args.RUN_TAG else datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return "_".join(parts)

def make_run_dir(root, name, subfolders=SUBFOLDERS):
    run = Path(root) / name
    existed = run.exists()
    for s in subfolders: (run / s).mkdir(parents=True, exist_ok=True)
    (log.warning if existed else log.info)(f"{'Reusing' if existed else 'Created'} run folder: {run}")
    return str(run)

def chrom_length(chrom):
    return chrom_sizes.get(chrom)
