"""RepliSage entry point.

    replisage --bedpe-file loops.bedpe --region chr14:80835000-98674700             # loop extrusion only
    replisage --bedpe-file loops.bedpe --region chr14:80835000-98674700 --rt-file timing.bigWig --run-md true
    replisage -c config.ini                                                          # everything from a config file
    python src/run.py -h                                                             # all command-line options

Main options are flags; detailed parameters (MC, replication, MD, force field) live in config.ini.
Verbosity: -v statistics tables | -vv warnings + diagnostics | -vvv everything.
"""
if __package__ in (None, ""):          # executed as a script: make `src` importable
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    __package__ = "src"

import shlex
import sys
from .config.loader import get_config
from .pipeline.orchestrator import run_pipeline
from .logger import log

def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    command = "replisage " + " ".join(shlex.quote(a) for a in argv)   # recorded in report.html
    args, cfg = get_config(argv)
    log.banner()
    log.info(f"Config file: {cfg}")
    try:
        return run_pipeline(args, command)
    except (ValueError, FileNotFoundError) as e:
        log.error(str(e))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
