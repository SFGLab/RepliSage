"""Load parameters: defaults <- config.ini <- command line (only the main options exist as flags)."""
import argparse
import configparser
import re
from pathlib import Path
from .arg_defs import build_args, ALIASES
from .stress import apply_stress
from ..logger import setup_logger, log
from ..banner import VERSION

DEFAULT_INI = Path(__file__).resolve().parents[1] / "config.ini"   # annotated template of every parameter
SC_SUFFIX = (".parquet",)

def str2bool(s):
    """true/false (also yes/no, 1/0) for flags that take a value."""
    s = str(s).strip().lower()
    if s in ("true", "1", "yes", "y", "on"): return True
    if s in ("false", "0", "no", "n", "off"): return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {s!r}")

def parse_region(s):
    """'chr14:80,835,000-98,674,700' -> (chrom, start, end); 'chr14' -> whole chromosome."""
    m = re.fullmatch(r"([\w.]+)(?::([\d,]+)-([\d,]+))?", s.strip())
    if not m: raise argparse.ArgumentTypeError("use CHROM or CHROM:START-END, e.g. chr14:80835000-98674700")
    c, a, b = m.groups(); return c, (int(a.replace(",", "")) if a else None), (int(b.replace(",", "")) if b else None)

def make_parser():
    p = argparse.ArgumentParser(
        prog="replisage", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="RepliSage: loop extrusion, epigenetic compartments and DNA replication in 3D chromatin.",
        epilog=("Examples:\n  replisage --bedpe-file loops.bedpe --region chr14:80835000-98674700\n"
                "  replisage --bedpe-file loops.bedpe --region chr14:80835000-98674700 --rt-file timing.bigWig --run-md true\n"
                "  replisage -c my_config.ini --seed 3\n\n"
                "Anything not listed here (MC, replication and MD energies, force field, plots...) is set in the\n"
                "config file: see src/config.ini and markdown/config.md. Command-line options win over the file."))
    opt = lambda g, *names, **kw: g.add_argument(*names, **kw)
    g = p.add_argument_group("config file")
    opt(g, "-c", "--config", "--config-file", dest="config_file", metavar="FILE", help="ini file with the parameters (see src/config.ini)")
    g = p.add_argument_group("input data")
    opt(g, "--bedpe-file", metavar="FILE", help="loops (.bedpe), CTCF orientation optional")
    opt(g, "--rt-file", metavar="FILE", help="replication timing: .bigWig/.bw, .txt, or .parquet (single cell); 'builtin' = shipped single-cell data. No file = no replication")
    opt(g, "--region", metavar="CHR[:START-END]", type=parse_region, help="modelled region, e.g. chr14:80835000-98674700 (no range = whole chromosome)")
    opt(g, "--n-beads", type=int, metavar="N", help="beads per chain (default: region length / 2 kb)")
    g = p.add_argument_group("what to run")
    opt(g, "--run-md", type=str2bool, metavar="{true,false}", help="build 3D structures with OpenMM (true) or stop after the Monte Carlo (false)")
    opt(g, "--md-mode", choices=["MD", "EM"], help="3D engine: molecular dynamics or repeated energy minimisation")
    opt(g, "--with-stress", action="store_true", help="replication stress: slower forks, more origins (default: off)")
    opt(g, "--n-steps", type=int, metavar="N", help="Monte Carlo steps")
    opt(g, "--n-replicas", type=int, metavar="N", help="independent Monte Carlo replicas")
    opt(g, "--seed", type=int, help="random seed (reproducible runs)")
    g = p.add_argument_group("output and hardware")
    opt(g, "--out", "--out-path", dest="out_path", metavar="DIR", help="results root folder (a run sub-folder is created inside)")
    opt(g, "--name", "--run-name", dest="run_name", help="exact name of the run folder (default: automatic)")
    opt(g, "--platform", choices=["CPU", "CUDA", "OpenCL"], help="OpenMM platform for the 3D part")
    opt(g, "--threads", type=int, metavar="N", help="CPU threads for parallel replicas")
    opt(g, "--no-report", action="store_true", help="skip report.html")
    opt(g, "--no-viewer", action="store_true", help="skip the interactive 3D viewer")
    opt(g, "--no-movie", action="store_true", help="skip the 3D movie (mp4 + gif)")
    opt(g, "-v", "--verbose", dest="verbose_count", action="count", default=0, help="-v tables, -vv diagnostics, -vvv everything")
    g.add_argument("--version", action="version", version=f"RepliSage {VERSION}")
    return p

def flags_to_params(ns):
    """Translate the command-line flags into {PARAMETER: value} (only the ones given)."""
    o = {}
    put = lambda k, v: o.__setitem__(k, v) if v is not None else None
    put("BEDPE_PATH", ns.get("bedpe_file")); put("N_BEADS", ns.get("n_beads")); put("N_STEPS", ns.get("n_steps"))
    put("N_REPLICAS", ns.get("n_replicas")); put("SEED", ns.get("seed")); put("OUT_PATH", ns.get("out_path")); put("RUN_NAME", ns.get("run_name"))
    put("PLATFORM", ns.get("platform")); put("N_THREADS", ns.get("threads"))
    if ns.get("region"): o["CHROM"], o["REGION_START"], o["REGION_END"] = ns["region"]
    rt = ns.get("rt_file")
    if rt is not None:
        if rt.lower() == "builtin": o.update(USE_SC_REPT=True, REPT_PATH="")
        elif rt.lower().endswith(SC_SUFFIX): o.update(USE_SC_REPT=True, SC_REPT_PATH=rt, REPT_PATH="")
        else: o.update(REPT_PATH=rt, USE_SC_REPT=False)
    if ns.get("run_md") is not None: o["SIMULATION_TYPE"] = (ns.get("md_mode") or "MD") if ns["run_md"] else ""
    elif ns.get("md_mode"): o["SIMULATION_TYPE"] = ns["md_mode"]
    if ns.get("with_stress"): o["REP_WITH_STRESS"] = True
    if ns.get("no_report"): o["REPORT"] = False
    if ns.get("no_viewer"): o["HTML_3D"] = False
    if ns.get("no_movie"): o["MOVIE"] = False
    return o

def read_ini(args, path):
    cp = configparser.ConfigParser(inline_comment_prefixes=(";",))
    if not cp.read(path): raise FileNotFoundError(f"Config file not found: {path}")
    for sec in cp.sections():
        for key, val in cp[sec].items():
            name = ALIASES.get(key.upper(), key.upper())
            if args.has(name): args.set(name, val)
            else: log.warning(f"Unknown key '{key}' in [{sec}] ignored.")

def get_config(argv=None, ini=None, overrides=None):
    """Resolved (Args, config path). overrides: {PARAMETER: value} applied last (Python API, tests)."""
    args = build_args()
    argv = [re.sub(r"^(--[\w-]+)", lambda m: m.group(1).replace("_", "-"), a) for a in (argv or [])]   # --rt_file == --rt-file
    ns = vars(make_parser().parse_args(argv))
    explicit = ns.get("config_file"); cfg = explicit or ini
    if cfg: read_ini(args, cfg)                       # a given file must exist
    for k, v in {**flags_to_params(ns), **(overrides or {})}.items(): args.set(k, v)   # command line wins
    args.to_python()
    setup_logger(max(ns["verbose_count"], args.VERBOSE or 0))
    apply_stress(args)
    return args, str(cfg or "(defaults)")

if __name__ == "__main__":
    a, c = get_config()
    log.table("Resolved parameters", [(x.name, x.val) for x in a], ("Name", "Value"), level=0)
