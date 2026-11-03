"""Colorful RepliSage logger: steps, tables, tqdm progress bars, 4 verbosity levels.
(default: stages, progress bars, successes, alerts; -v: input / output tables; -vv: detailed info; -vvv: everything)

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

# ANSI styles (256 colours); the palette follows the RepliSage logo: teal, magenta, orange, violet
C = dict(reset="\033[0m", bold="\033[1m", dim="\033[2m", red="\033[91m", green="\033[92m", yellow="\033[93m", blue="\033[94m",
         magenta="\033[95m", cyan="\033[96m", grey="\033[90m")
fg = lambda n: f"\033[38;5;{n}m"
bg = lambda n: f"\033[48;5;{n}m"
PALETTE = (51, 213, 214, 84, 141, 209)                       # cyan, pink, amber, mint, violet, coral: one colour per stage
TAGS = {"INFO": (16, 45), "OK": (16, 84), "WARN": (16, 214), "ERR": (255, 197), "DIAG": (255, 99), "TRCE": (250, 240)}   # (text, background) of the level pills

def col(txt, *styles):
    return "".join(C[s] if s in C else s for s in styles) + str(txt) + C["reset"]

class RepliLogger:
    def __init__(self, verbosity=0, color=True, stream=None):
        self.verbosity, self.color, self.stream = verbosity, color, stream or sys.stdout
        self._t0 = time.time()
        self._step = 0

    # ---- core output ----
    def _c(self, txt, *styles):
        return col(txt, *styles) if self.color else str(txt)

    def _stamp(self):  # timestamp in dim grey
        return self._c(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"), fg(245), "dim")

    def _pill(self, tag):
        """Level tag as a coloured pill (plain text without colour)."""
        k = tag.strip()
        if not self.color: return f"{tag:<5}"
        t, b = TAGS.get(k, (16, 250)); return f"{bg(b)}{fg(t)}{C['bold']} {k:^4} {C['reset']}"

    def _emit(self, tag, tag_style, msg, text_style=()):
        for ln in str(msg).split("\n"):
            tqdm.write(f"{self._stamp()} {self._pill(tag)} {self._c(ln, *text_style)}" if self.color else f"{self._stamp()} {tag:<5} {ln}", file=self.stream)

    def set_verbosity(self, v):
        self.verbosity = int(v)

    # ---- message levels ----
    def note(self, msg):    self._emit("INFO", (), msg, (fg(117),))                              # always shown: what we are waiting for
    def info(self, msg):                                                             # shown with -vv
        if self.verbosity >= 2: self._emit("INFO", (), msg, (fg(117),))
    def success(self, msg): self._emit(" OK ", (), msg, (fg(84),))
    def alert(self, msg):   self._emit("WARN", (), msg, (fg(222),))   # always shown
    def error(self, msg):   self._emit("ERR ", (), msg, (fg(210), "bold"))
    def warning(self, msg):                                                          # shown with -vv
        if self.verbosity >= 2: self._emit("WARN", (), msg, (fg(222),))
    def diag(self, msg):
        if self.verbosity >= 2: self._emit("DIAG", (), msg, (fg(183),))
    def trace(self, msg):
        if self.verbosity >= 3: self._emit("TRCE", (), msg, (fg(244),))

    # ---- structure ----
    def banner(self):
        art = ART.split("\n")
        for i, ln in enumerate(art): print(self._c(ln, fg((51, 45, 39, 99, 135, 171, 213)[min(i, 6)]), "bold"), file=self.stream)   # teal -> magenta gradient
        print(self._c(f"  Loop extrusion x replication x compartments   |   v{VERSION}", fg(213)), file=self.stream)
        print(self._c(f"  {get_quote()}\n", fg(245)), file=self.stream)

    def step(self, title):
        self._step += 1; c = PALETTE[(self._step - 1) % len(PALETTE)]
        if not self.color: tqdm.write(f"\n{'=' * 70}\n  STEP {self._step} | {title}\n{'=' * 70}", file=self.stream); return
        tqdm.write(f"\n{bg(c)}{fg(16)}{C['bold']}  STEP {self._step} {C['reset']} {fg(c)}{C['bold']}{title}{C['reset']}\n{fg(c)}{'━' * 70}{C['reset']}", file=self.stream)

    def section(self, title):
        tqdm.write(self._c(f"▸ {title}", fg(213), "bold") if self.color else f"--- {title} ---", file=self.stream)

    def table(self, title, rows, headers=("Quantity", "Value"), level=1):
        """Print a table of rows (list of tuples) if verbosity >= level."""
        if self.verbosity < level: return
        rows = [[str(x) for x in r] for r in rows]
        w = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
        if not self.color:
            sep = "+" + "+".join("-" * (x + 2) for x in w) + "+"; fmt = lambda cells: "|" + "|".join(f" {c:<{w[i]}} " for i, c in enumerate(cells)) + "|"
            tqdm.write("\n".join([f" {title}", sep, fmt(headers), sep] + [fmt(r) for r in rows] + [sep]), file=self.stream); return
        g = fg(60); line = lambda l, m, r: g + l + m.join("─" * (x + 2) for x in w) + r + C["reset"]
        cell = lambda cells, st: g + "│" + C["reset"] + (g + "│" + C["reset"]).join(" " + col(f"{c:<{w[i]}}", *st) + " " for i, c in enumerate(cells)) + g + "│" + C["reset"]
        out = [col(f" ◆ {title}", fg(213), "bold"), line("┌", "┬", "┐"), cell(headers, (fg(117), "bold")), line("├", "┼", "┤")]
        out += [cell(r, (fg(252) if i % 2 == 0 else fg(246),)) for i, r in enumerate(rows)] + [line("└", "┴", "┘")]
        tqdm.write("\n".join(out), file=self.stream)

    # ---- progress / timing ----
    def progress(self, iterable=None, total=None, desc="", **kw):
        """tqdm.auto progress bar (works in terminals and notebooks)."""
        c = PALETTE[(self._step - 1) % len(PALETTE)] if self._step else 84
        if self.color: kw.setdefault("colour", "#%02x%02x%02x" % _RGB[c])
        return tqdm(iterable, total=total, desc=self._c(desc, fg(c)) if self.color else desc, file=self.stream, dynamic_ncols=True, **kw)

    @contextmanager
    def timer(self, label):
        t = time.time(); self.trace(f"{label}: started")
        yield
        self.success(f"{label} finished in {fmt_time(time.time() - t)}")

    def total_time(self):
        return fmt_time(time.time() - self._t0)

_RGB = {51: (0, 255, 255), 213: (255, 135, 255), 214: (255, 175, 0), 84: (95, 255, 135), 141: (175, 135, 255), 209: (255, 135, 95)}   # tqdm bar colours of the PALETTE

def fmt_time(s):
    return f"{s // 3600:.0f}h {s % 3600 // 60:.0f}m {s % 60:.1f}s"

# ---- global singleton ----
log = RepliLogger()

def setup_logger(verbosity=0, color=None):
    log.set_verbosity(verbosity)
    import warnings
    if verbosity < 2: warnings.simplefilter("ignore")   # library warnings only with -vv
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
