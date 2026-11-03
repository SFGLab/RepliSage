"""One palette and legend for every 3D view (PyVista movie / stills and the HTML viewer)."""
from ..plots.style import STATE_COL, ROYAL

SIS = ("#F08A24", "#3B7DD8")                 # sister chromatids
LEF = ("#2E9E5B", "#D81B60")                 # cohesin, condensin
LEF_NAME = ("cohesin", "condensin")
CTCF, FORK = "#5B2C83", "#00BFB3"            # CTCF arrows, replication forks
SPIN = {int(k): v for k, v in STATE_COL.items()}   # epigenetic state -2..2: blue (negative) .. red (positive)
REPL, UNREPL = ROYAL, "#9AA3AE"              # replicated / not yet

def rgb(h): return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

def as_dict():
    """Palette for the HTML viewer (JSON-friendly)."""
    return dict(sis=list(SIS), lef=list(LEF), lef_name=list(LEF_NAME), ctcf=CTCF, fork=FORK, spin={str(k): v for k, v in SPIN.items()}, repl=REPL, unrepl=UNREPL)
