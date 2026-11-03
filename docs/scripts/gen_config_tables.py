"""Print the parameter tables of docs (one per config section) from src/config/arg_defs.py.

    python docs/scripts/gen_config_tables.py > tables.md      # run from the repository root
"""
import re, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config.arg_defs import build_args, DEFAULT_REPT, DEFAULT_FF

TITLES = {"general": "General", "input": "Input", "preprocessing": "Pre-processing", "replication": "Replication",
          "mc": "Monte Carlo", "md": "Molecular dynamics", "output": "Output"}
SPECIAL = {"CHROM": "--region", "REGION_START": "--region", "REGION_END": "--region", "REPT_PATH": "--rt-file", "SC_REPT_PATH": "--rt-file",
           "USE_SC_REPT": "--rt-file builtin", "ASK_RT": "--rt-file none", "SIMULATION_TYPE": "--run-md / --md-mode", "REPORT": "--no-report",
           "HTML_3D": "--no-viewer", "MOVIE": "--no-movie", "OUT_PATH": "--out-path", "RUN_NAME": "--run-name", "REP_WITH_STRESS": "--with-stress",
           "PREFERENTIAL_BINDING": "--preferential-binding", "REP_START_TIME": "--rep-start", "REP_TIME_DURATION": "--rep-duration"}
TYPES = {str: "text", int: "integer", float: "number", bool: "True/False"}

def flags():
    """{PARAMETER: --flag} read from the command-line parser (put("NAME", ns.get("flag")))."""
    src = (ROOT / "src/config/loader.py").read_text()
    f = {k: "--" + v.replace("_", "-") for k, v in re.findall(r'put\("(\w+)", ns\.get\("(\w+)"\)', src)}
    return {**f, **SPECIAL}

def clean(text):
    """Help text for a table cell (drop the [new] tag, escape pipes)."""
    return re.sub(r"^\[new\] ", "", text).replace("|", "/")

def default(a):
    d = a.default
    if d in ("", None): return "empty"
    return "built-in file" if str(d) in (DEFAULT_REPT, DEFAULT_FF) else str(d)

def tables():
    fl, out, secs = flags(), [], {}
    for a in build_args(): secs.setdefault(a.section, []).append(a)
    for sec, args in secs.items():
        out += [f"### {TITLES.get(sec, sec.title())} (`[{sec}]`)", "", "| Key | Flag | Type | Default | Meaning |", "|---|---|---|---|---|"]
        out += [f"| `{a.name.lower()}` | {('`' + fl[a.name] + '`') if a.name in fl else ''} | {TYPES.get(a.type, a.type.__name__)} | {default(a)} | {clean(a.help)} |" for a in args]
        out.append("")
    return "\n".join(out)

if __name__ == "__main__": print(tables())
