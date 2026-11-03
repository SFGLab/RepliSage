"""Config parity: every flag maps to an ini key, every parameter is in config.ini, ini-only parsing works."""
import configparser, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.config.arg_defs import build_args
from src.config.loader import make_parser, flags_to_params, get_config, DEFAULT_INI

names = {a.name for a in build_args()}
ns = {d.dest: True for d in make_parser()._actions}
ns.update(region=("chr1", 1, 2), rt_file="x.bw", run_md=True)
assert set(flags_to_params(ns)) <= names, set(flags_to_params(ns)) - names        # flag targets exist
cp = configparser.ConfigParser(); cp.read(DEFAULT_INI)
ini = {k.upper() for s in cp.sections() for k in cp[s]}
assert names <= ini, f"missing in config.ini: {names - ini}"                      # template lists everything
a, _ = get_config(["-c", str(DEFAULT_INI)])                                       # template parses
for w, sc, rt in (("none", False, None), ("builtin", True, None)):
    a, _ = get_config(["--rt-file", w]); assert (a.USE_SC_REPT, a.REPT_PATH) == (sc, rt), w
print("config parity OK:", len(names), "parameters")
