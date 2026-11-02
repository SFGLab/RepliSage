"""Colorful RepliSage logger: steps, tables, tqdm progress bars, 4 verbosity levels.

Verbosity (set with -v / -vv / -vvv or VERBOSE in config):
  0 : banner, step headers, progress bars, key results, errors, alerts
  1 : + statistics tables of input/output data          (-v)
  2 : + warnings and diagnostic messages                (-vv)
  3 : + every trace message (timings, shapes, config)   (-vvv)

Example:
    python src/logger.py -vv
"""
import sys
import time
import datetime
from contextlib import contextmanager
from tqdm.auto import tqdm

try:
    from .banner import ART, VERSION, get_quote
except ImportError:  # run as a script
    from banner import ART, VERSION, get_quote

# ANSI colors
C = dict(reset="\033[0m", bold="\033[1m", dim="\033[2m", red="\033[91m", green="\033[92m",
         yellow="\033[93m", blue="\033[94m", magenta="\033[95m", cyan="\033[96m", grey="\033[90m")

def col(txt, *styles):
    return "".join(C[s] for s in styles) + str(txt) + C["reset"]

class RepliLogger:
    def __init__(self, verbosity=0, color=True, stream=None):
        self.verbosity, self.color, self.stream = verbosity, color, stream or sys.stdout
        self._t0 = time.time()
        self._step = 0

    # ---- core output ----
    def _c(self, txt, *styles):
        return col(txt, *styles) if self.color else str(txt)

    def _stamp(self):  # timestamp has its own color (grey-cyan)
        return self._c(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"), "cyan", "dim")

    def _emit(self, tag, tag_style, msg, text_style=()):
        lines = str(msg).split("\n")
        for ln in lines:
            tqdm.write(f"{self._stamp()} {self._c(f'{tag:<5}', *tag_style)} {self._c(ln, *text_style)}", file=self.stream)

    def set_verbosity(self, v):
        self.verbosity = int(v)

    # ---- message levels ----
    def info(self, msg):    self._emit("INFO", ("blue",), msg)
    def success(self, msg): self._emit(" OK ", ("green", "bold"), msg, ("green",))
    def alert(self, msg):   self._emit("WARN", ("yellow", "bold"), msg, ("yellow",))   # always shown
    def error(self, msg):   self._emit("ERR ", ("red", "bold"), msg, ("red",))
    def warning(self, msg):                                                          # shown with -vv
        if self.verbosity >= 2: self._emit("WARN", ("yellow",), msg, ("yellow",))
    def diag(self, msg):
        if self.verbosity >= 2: self._emit("DIAG", ("magenta",), msg, ("magenta",))
    def trace(self, msg):
        if self.verbosity >= 3: self._emit("TRCE", ("grey",), msg, ("grey",))

    # ---- structure ----
    def banner(self):
        print(self._c(ART, "magenta", "bold"), file=self.stream)
        print(self._c(f"  Loop extrusion x replication x compartments   |   v{VERSION}", "cyan"), file=self.stream)
        print(self._c(f"  {get_quote()}\n", "grey"), file=self.stream)

    def step(self, title):
        self._step += 1
        bar = "=" * 70
        tqdm.write(self._c(f"\n{bar}\n  STEP {self._step} | {title}\n{bar}", "green", "bold"), file=self.stream)

    def section(self, title):
        tqdm.write(self._c(f"--- {title} ---", "blue", "bold"), file=self.stream)

    def table(self, title, rows, headers=("Quantity", "Value"), level=1):
        """Print a table of rows (list of tuples) if verbosity >= level."""
        if self.verbosity < level: return
        rows = [[str(x) for x in r] for r in rows]
        w = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
        sep = "+" + "+".join("-" * (x + 2) for x in w) + "+"
        fmt = lambda cells, st: "|" + "|".join(" " + self._c(f"{c:<{w[i]}}", *st) + " " for i, c in enumerate(cells)) + "|"
        out = [self._c(f" {title}", "magenta", "bold"), sep, fmt(headers, ("bold",)), sep]
        out += [fmt(r, ()) for r in rows] + [sep]
        tqdm.write("\n".join(out), file=self.stream)

    # ---- progress / timing ----
    def progress(self, iterable=None, total=None, desc="", **kw):
        """tqdm.auto progress bar (works in terminals and notebooks)."""
        return tqdm(iterable, total=total, desc=self._c(desc, "green") if self.color else desc,
                    file=self.stream, dynamic_ncols=True, **kw)

    @contextmanager
    def timer(self, label):
        t = time.time(); self.trace(f"{label}: started")
        yield
        self.success(f"{label} finished in {fmt_time(time.time() - t)}")

    def total_time(self):
        return fmt_time(time.time() - self._t0)

def fmt_time(s):
    return f"{s // 3600:.0f}h {s % 3600 // 60:.0f}m {s % 60:.1f}s"

# ---- global singleton ----
log = RepliLogger()

def setup_logger(verbosity=0, color=None):
    log.set_verbosity(verbosity)
    log.color = sys.stdout.isatty() if color is None else color
    return log

if __name__ == "__main__":  # demo
    v = sum(a.count("v") for a in sys.argv[1:] if a.startswith("-v"))
    L = setup_logger(v, color=True)
    L.banner(); L.step("Demo step")
    L.info("info message"); L.success("success"); L.alert("always-shown warning")
    L.warning("warning (needs -vv)"); L.diag("diagnostic (needs -vv)"); L.trace("trace (needs -vvv)")
    L.table("Input statistics", [("beads", 1000), ("LEFs", 100)])
    for _ in L.progress(range(50), desc="Demo bar"): time.sleep(0.01)
