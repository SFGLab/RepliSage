"""Figure style (light, black + royal red), shared helpers and the figure registry."""
import os
from dataclasses import dataclass
import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler

BLACK, GREY, LIGHT, SLATE = "#141414", "#7A7A7A", "#E6E6E6", "#3C4350"
ROYAL, BLUE, TEAL, GOLD = "#A50F2D", "#2B6CB0", "#1B9E8F", "#E3A008"
PURPLE, ORANGE, GREEN, ROSE, PINK = "#7B4FA3", "#E8702A", "#5BA33B", "#E27D8C", "#F6D3D9"
CYCLE = [BLUE, ROYAL, TEAL, GOLD, PURPLE, ORANGE, GREEN, SLATE]
PHASE_COL = {"G1": BLUE, "S": ROYAL, "G2/M": TEAL, "All": PURPLE}   # cell-cycle phases
A_COL, B_COL = ROYAL, BLUE                                          # compartments A (warm) / B (cool)
LEFT, RIGHT = ORANGE, TEAL                                          # left / right legs and CTCF sites
STATE_COL = {-2: "#1F4E99", -1: "#7FA7D9", 0: "#EDEDED", 1: "#F08A7E", 2: ROYAL}   # Potts spins -2..2

CMAP_SEQ = LinearSegmentedColormap.from_list("rs_seq", ["#FFFFFF", "#FFF0B3", ORANGE, ROYAL, "#3A0F3F"])
CMAP_REP = LinearSegmentedColormap.from_list("rs_rep", ["#FFFFFF", "#CDEBE6", TEAL])
CMAP_SPIN = LinearSegmentedColormap.from_list("rs_spin", [STATE_COL[k] for k in (-2, -1, 0, 1, 2)])   # mean spin of a bin
CMAP_MAT = LinearSegmentedColormap.from_list("rs_mat", ["#FFFFFF", "#D8C8EA", PURPLE])

DPI = 200
FIGS = {}   # figure id -> png path (filled by save_fig, read by the report)

def apply_style(dpi=None):
    """Light mode, white background, clean axes, black/red colour cycle."""
    global DPI
    if dpi: DPI = int(dpi)
    matplotlib.use("Agg", force=False)
    matplotlib.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "savefig.edgecolor": "white",
        "figure.max_open_warning": 0, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlepad": 16, "axes.titlelocation": "left",
        "axes.labelsize": 10.5, "axes.edgecolor": "#333333", "axes.labelcolor": BLACK, "text.color": BLACK,
        "xtick.color": "#333333", "ytick.color": "#333333", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.7, "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=CYCLE), "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 1.6, "figure.dpi": 100, "savefig.dpi": DPI, "svg.fonttype": "none"})

@dataclass
class Phases:
    """Phase boundaries in recorded-sample units (g1 = burn-in end, s0/s1 = S phase)."""
    n: int
    mc_step: int
    g1: int
    s0: int
    s1: int
    has_rep: bool

    def x(self, i): return np.asarray(i) * self.mc_step          # sample index -> MC step
    @property
    def steps(self): return self.x(np.arange(self.n))

    def segments(self):
        """{label: slice} of the analysed samples (burn-in removed)."""
        if not self.has_rep: return {"All": slice(self.g1, self.n)}
        return {"G1": slice(self.g1, self.s0), "S": slice(self.s0, self.s1), "G2/M": slice(self.s1, self.n)}

def make_phases(p, n):
    from ..MC.results import phase_bounds
    g1, s0, s1 = phase_bounds(p)
    return Phases(n, p.MC_step, min(g1, n), min(s0, n), min(s1, n), p.has_rep)

def centred(y, ph):
    """Remove a huge constant offset (using post burn-in median); returns (y, label prefix)."""
    y = np.asarray(y, float); r = y[ph.g1:] if ph.n - ph.g1 > 5 else y
    if r.std() > 0 and abs(r.mean()) > 20 * r.std(): return y - np.median(r), "Δ"
    return y, ""

def zoom(ax, y, ph, pad=.15):
    """Set y-limits from the post burn-in samples so the start-up spike does not squash the trace."""
    r = np.asarray(y, float)[ph.g1:] if ph.n - ph.g1 > 5 else np.asarray(y, float)
    lo, hi = np.percentile(r, [.5, 99.5]); d = (hi - lo) or 1; ax.set_ylim(lo - pad * d, hi + pad * d)

def shade_phases(ax, ph, labels=False):
    """Grey burn-in band, red S-phase band with dashed edges."""
    if ph.g1 > 0: ax.axvspan(0, ph.x(ph.g1), color=LIGHT, alpha=.9, lw=0, zorder=0)
    if not ph.has_rep: return
    ax.axvspan(ph.x(ph.s0), ph.x(ph.s1), color=ROYAL, alpha=.07, lw=0, zorder=0)
    for v in (ph.s0, ph.s1): ax.axvline(ph.x(v), color=ROYAL, lw=1, ls="--", alpha=.8)
    if labels:
        tr = ax.get_xaxis_transform()
        for lab, a, b in (("G1", ph.g1, ph.s0), ("S", ph.s0, ph.s1), ("G2/M", ph.s1, ph.n)):
            if b > a: ax.text(ph.x((a + b) / 2), 1.02, lab, transform=tr, ha="center", va="bottom", fontsize=9,
                              color=ROYAL if lab == "S" else GREY, fontweight="bold")

def bin_time(M, max_cols=400):
    """Average the time axis (last) of a matrix in equal bins so that at most max_cols columns remain; returns (matrix, factor)."""
    f = int(np.ceil(M.shape[1] / max_cols))
    if f <= 1: return M, 1
    n = M.shape[1] // f * f
    return M[:, :n].reshape(M.shape[0], -1, f).mean(axis=2), f

def rolling(y, w):
    """Centered moving average (edges use the available points)."""
    y = np.asarray(y, float); w = max(1, int(w))
    if w == 1 or len(y) < 3: return y
    k = np.ones(w); return np.convolve(y, k, "same") / np.convolve(np.ones(len(y)), k, "same")

def band(ax, x, y, w, c, lw=2, alpha=.18):
    """Smoothed mean (rolling window w) with a +-1 SD band."""
    y = np.asarray(y, float); m = rolling(y, w); sd = np.sqrt(np.maximum(rolling(y * y, w) - m * m, 0))
    ax.fill_between(x, m - sd, m + sd, color=c, alpha=alpha, lw=0); ax.plot(x, m, color=c, lw=lw)

@dataclass
class Geo:
    """Genomic coordinates of the beads (in Mb) for axis labels."""
    xs: np.ndarray
    chrom: str
    bead_kb: float

    @property
    def label(self): return f"{self.chrom} position (Mb)"

def make_geo(ctx):
    a, b = ctx.region
    return Geo(np.linspace(a, b, ctx.n_beads, endpoint=False) / 1e6, ctx.chrom, (b - a) / ctx.n_beads / 1e3)

def _inview(axis, lim):
    """Tick labels whose tick lies inside the axis limits."""
    lo, hi = sorted(lim)
    return [l for t, l in zip(axis.get_majorticklocs(), axis.get_majorticklabels()) if lo - 1e-9 <= t <= hi + 1e-9]

def overlaps(fig):
    """Pairs of visible texts / legends whose boxes intersect (layout check)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); items = []
    for ax in fig.axes:
        ts = [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label, *ax.texts, *_inview(ax.xaxis, ax.get_xlim()), *_inview(ax.yaxis, ax.get_ylim())]
        lg = ax.get_legend()
        if lg: ts += [lg.get_frame(), *lg.get_texts()]
        for t in ts:
            if t.get_visible() and (not hasattr(t, "get_text") or t.get_text().strip()):
                b = t.get_window_extent(r)
                if b.width > 1 and b.height > 1: items.append((getattr(t, "get_text", lambda: "legend")(), b, ax, t is (lg.get_frame() if lg else None)))
    bad = []
    for i, (a, ba, ax1, fa) in enumerate(items):
        for c, bc, ax2, fc in items[i + 1:]:
            if fa or fc: continue
            if ba.overlaps(bc) and min(ba.x1, bc.x1) - max(ba.x0, bc.x0) > 2 and min(ba.y1, bc.y1) - max(ba.y0, bc.y0) > 2: bad.append((a[:25], c[:25]))
    return bad

def save_fig(fig, out_dir, sub, name, fid=None, svg=True):
    """Save png (+svg) under <out_dir>/plots/<sub>/ and register it for the report."""
    import matplotlib.pyplot as plt
    d = os.path.join(out_dir, "plots", sub); os.makedirs(d, exist_ok=True)
    png = os.path.join(d, name + ".png")
    if os.environ.get("RS_CHECK"):
        for a, b in overlaps(fig): print(f"OVERLAP {fid or name}: '{a}' x '{b}'")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    if svg: fig.savefig(os.path.join(d, name + ".svg"), bbox_inches="tight")
    plt.close(fig)
    FIGS[fid or name] = png
    return png
