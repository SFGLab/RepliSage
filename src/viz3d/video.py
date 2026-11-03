"""PyVista movie (mp4 + gif) of the ensemble with interpolation, contact map and a cell-cycle bar."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from ..plots.style import CMAP_SEQ, PHASE_COL, A_COL, B_COL, ORANGE, GOLD, ROYAL, TEAL, BLACK
from ..logger import log

from .data import replication_eyes
from .palette import SIS, LEF as LEF_COL, LEF_NAME, CTCF as CTCF_COL, FORK as FORK_COL, SPIN, REPL, UNREPL, rgb as _rgb
SPIN_RGB = {k: _rgb(c) for k, c in SPIN.items()}

def bead_rgb(tr, f, mode):
    """(chains*N, 3) uint8 colours of frame f for a colour mode: sister / compartment / rainbow / replication."""
    N, C = tr.n_beads, tr.n_chains
    if mode == "sister": one = [np.tile(_rgb(SIS[c % 2]), (N, 1)) for c in range(C)]
    elif mode == "compartment":
        s = np.clip(tr.spins[f], -2, 2); one = [np.array([SPIN_RGB[int(v)] for v in s])] * C   # graded epigenetic state: blue (negative) .. red (positive)
    elif mode == "replication":
        col = np.array([_rgb(UNREPL)] * N); col[tr.rep[f] > 0] = _rgb(REPL); one = [col] * C
    else:
        import colorsys
        one = [np.array([[int(255 * v) for v in colorsys.hls_to_rgb(.78 * (1 - i / (N - 1)), .5, .8)] for i in range(N)])] * C
    return np.vstack(one).astype(np.uint8)

class Contacts:
    """Window-averaged soft contact map of the frames, cached per frame."""
    def __init__(self, tr, win=8, mult=3.0):
        self.tr, self.win, self.c = tr, win, {}
        V = tr.V[0]; self.rc = mult * np.median(np.linalg.norm(np.diff(V[:tr.n_beads], axis=0), axis=1))
    def frame(self, f):
        if f not in self.c:
            tr, N = self.tr, self.tr.n_beads; V = tr.V[f]; m = np.zeros((N, N))
            for a in range(tr.n_chains):
                for b in range(tr.n_chains): m += 1 / (1 + (cdist(V[a * N:(a + 1) * N], V[b * N:(b + 1) * N]) / self.rc) ** 4)   # soft contact: 1 below rc, decaying above
            self.c[f] = m / tr.n_chains ** 2
        return self.c[f]
    def avg(self, f):
        a = max(0, f - self.win + 1); return np.mean([self.frame(g) for g in range(a, f + 1)], axis=0)

CMAPS = ("turbo", "inferno", "magma", "viridis", "cividis", "rs")   # contact-map colour maps (turbo is the most variable)

def get_cmap(name):
    import matplotlib
    return CMAP_SEQ if name == "rs" else matplotlib.colormaps[name]

def _tri(d, x, y, r, direction, fill, outline=(255, 255, 255)):
    """Small triangle (right, left, up or down) centred at (x, y)."""
    pts = {"r": [(x - r, y - r), (x - r, y + r), (x + r, y)], "l": [(x + r, y - r), (x + r, y + r), (x - r, y)],
           "d": [(x - r, y - r), (x + r, y - r), (x, y + r)], "u": [(x - r, y + r), (x + r, y + r), (x, y - r)]}[direction]
    d.polygon(pts, fill=fill, outline=outline)

def _diamond(d, x, y, r, fill, outline=(255, 255, 255)): d.polygon([(x, y - r), (x + r, y), (x, y + r), (x - r, y)], fill=fill, outline=outline)

def heat_image(tr, cm, f, size, cmap="turbo"):
    """RGB square image: contact map (window average) with loop arcs, CTCF ticks, state / replication strips,
    cohesin (circles) and condensin (squares) anchors and forks (diamonds on the strips and the diagonal, short crosses near it)."""
    N = tr.n_beads; M = cm.avg(f); iu = np.abs(np.subtract.outer(np.arange(N), np.arange(N))) > 3
    vmax = max(.05, np.percentile(M[iu], 99.5)); img = (get_cmap(cmap)(np.clip(M / vmax, 0, 1) ** .5)[..., :3] * 255).astype(np.uint8)   # square-root stretch shows weak patterns
    ml = max(30, size // 22); arc = size // 10; mt = ml + arc; leg = max(26, size // 22); cb = size // 16
    S = min(size - ml - cb, size - mt - leg); u = S / N
    can = Image.new("RGBA", (size, size), (255, 255, 255, 255)); can.paste(Image.fromarray(img).resize((S, S), Image.NEAREST), (ml, mt))
    px = lambda i: ml + (i + .5) * u; py = lambda i: mt + (i + .5) * u
    d = ImageDraw.Draw(can); sp, rp = tr.spins[f], tr.rep[f]; eyes = replication_eyes(rp) if tr.has_rep else []
    for i in range(N):   # strips: epigenetic state (outer), replicated (inner)
        x0, x1 = ml + i * u, ml + (i + 1) * u + .5
        d.rectangle([x0, mt - 20, x1, mt - 13], fill=SPIN_RGB[int(np.clip(sp[i], -2, 2))]); d.rectangle([ml - 20, mt + i * u, ml - 13, mt + (i + 1) * u + .5], fill=SPIN_RGB[int(np.clip(sp[i], -2, 2))])
        col = _rgb(REPL) if rp[i] else (232, 232, 232)
        d.rectangle([x0, mt - 28, x1, mt - 21], fill=col); d.rectangle([ml - 28, mt + i * u, ml - 21, mt + (i + 1) * u + .5], fill=col)
    ci, cs = tr.ctcf
    for i, s_ in zip(ci, cs):   # CTCF ticks: right-pointing = forward, left-pointing = reverse
        _tri(d, px(i), mt - 7, 4, "r" if s_ > 0 else "l", _rgb(CTCF_COL)); _tri(d, ml - 7, py(i), 4, "d" if s_ > 0 else "u", _rgb(CTCF_COL))
    lay = Image.new("RGBA", (size, size), (0, 0, 0, 0)); ld = ImageDraw.Draw(lay); lefs = tr.lefs[f]; fam = tr.lef_fam[f] if tr.lef_fam else np.zeros(len(lefs), int)
    for (a, b), k in zip(lefs, fam):   # loop arcs above the map
        xa, xb = px(a), px(b); h = min(arc - 5, (arc - 5) * np.sqrt((b - a) / N) * 1.4); t = np.linspace(0, np.pi, 40)
        ld.line([(xa + (xb - xa) * (1 - np.cos(q)) / 2, mt - 31 - h * np.sin(q)) for q in t], fill=_rgb(LEF_COL[k]) + (200,), width=2)
    for a, b in eyes:   # a replication eye: thick band on the diagonal between its two forks
        ld.line([(ml + a * u, mt + a * u), (ml + (b + 1) * u, mt + (b + 1) * u)], fill=_rgb(FORK_COL) + (210,), width=5)
    can = Image.alpha_composite(can, lay); d = ImageDraw.Draw(can); r = max(3, size // 150) + 1
    FC = _rgb(FORK_COL)
    for a, b in eyes:   # forks: triangles on both strips pointing the way the fork moves, and a diamond on the diagonal
        for pos, side in ((a, "l"), (b + 1, "r")):
            if (side == "l" and a == 0) or (side == "r" and b + 1 >= N): continue
            x = y = ml + pos * u; y = mt + pos * u
            _tri(d, x, mt - 24, 5, side, FC); _tri(d, ml - 24, y, 5, "u" if side == "l" else "d", FC); _diamond(d, x, y, 5, FC)
    for (a, b), k in zip(lefs, fam):   # anchors at (a, b) and (b, a): cohesin = circle, condensin = square
        for x, y in ((a, b), (b, a)):
            cx, cy = px(x), py(y)
            if k == 0: d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_rgb(LEF_COL[0]), outline=(255, 255, 255), width=1)
            else: d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=_rgb(LEF_COL[1]), outline=(255, 255, 255), width=1)
    # colour bar and legend
    cx0 = size - cb + 4; grad = (get_cmap(cmap)(np.linspace(1, 0, S) ** 1)[:, :3] * 255).astype(np.uint8)[:, None, :].repeat(10, 1)
    can.paste(Image.fromarray(grad), (cx0, mt)); fs = max(11, size // 50); ft = _font(fs, False)
    d.text((cx0 - 2, mt - fs - 4), "high", font=ft, fill=(60, 60, 60)); d.text((cx0 - 2, mt + S + 3), "low", font=ft, fill=(60, 60, 60))
    ent = [("c", LEF_COL[0], "cohesin"), ("s", LEF_COL[1], "condensin"), ("t", CTCF_COL, "CTCF"), ("d", FORK_COL, "fork (arrow = direction)")] if tr.has_rep else [("c", LEF_COL[0], "cohesin"), ("s", LEF_COL[1], "condensin"), ("t", CTCF_COL, "CTCF")]
    x = ml; y = size - leg // 2 - 1
    for kind, c, lab in ent:
        c = _rgb(c)
        if kind == "c": d.ellipse([x, y - 4, x + 8, y + 4], fill=c)
        elif kind == "s": d.rectangle([x, y - 4, x + 8, y + 4], fill=c)
        elif kind == "t": _tri(d, x + 4, y, 4, "r", c, c)
        else: _diamond(d, x + 4, y, 5, c, c)
        d.text((x + 12, y), lab, font=ft, fill=(40, 40, 46), anchor="lm"); x += 12 + d.textlength(lab, font=ft) + 14
    return np.array(can.convert("RGB"))

def bar_image(tr, t, width, h=104):
    """Phase bar (one cell per recorded frame) with the replicated-fraction curve, MC-step ticks and a playhead at fractional frame t of the full trajectory."""
    F = len(tr.V); im = Image.new("RGB", (width, h), "white"); d = ImageDraw.Draw(im); bw = width / F; top, bot = 24, h - 26
    try: ft = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
    except OSError: ft = ImageFont.load_default()
    s = 0
    for f in range(F + 1):
        if f == F or tr.phase[f] != tr.phase[s]:
            col = _rgb(PHASE_COL.get(tr.phase[s], "#7B4FA3")); d.rectangle([s * bw, top, f * bw, bot], fill=col)
            d.text(((s + f) / 2 * bw, 2), tr.phase[s], fill=(20, 20, 20), font=ft, anchor="mt"); s = f
    if tr.has_rep:
        y = bot - tr.rep.mean(1) * (bot - top - 4) - 2; d.line([((i + .5) * bw, v) for i, v in enumerate(y)], fill="white", width=3)
    try: fs = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError: fs = ImageFont.load_default()
    for f in np.unique(np.linspace(0, F - 1, 6).astype(int)):   # MC-step ticks
        x = (f + .5) * bw; d.line([x, bot, x, bot + 5], fill=(90, 90, 90), width=1); d.text((x, bot + 7), f"{int(tr.step[f])}", fill=(70, 70, 70), font=fs, anchor="mt")
    t = float(np.clip(t, 0, F - 1)); x = (t + .5) * bw; d.rectangle([x - 2, top - 5, x + 2, bot + 3], fill=(20, 20, 20))
    d.text((x, h - 2), f"step {int(round(np.interp(t, np.arange(F), tr.step)))}", fill=(20, 20, 20), font=fs, anchor="mb"); return np.array(im)

def playhead(idx, ft):
    """Position in the full trajectory (frame units) of the interpolated movie position ft; idx = recorded frames used by the movie."""
    return float(np.interp(ft, np.arange(len(idx)), idx))

def _smooth(pts, col, k=4):
    """Cubic-spline the bead path (k points per bead) and interpolate colours along it, for smooth tubes."""
    from scipy.interpolate import CubicSpline
    n = len(pts); u = np.linspace(0, n - 1, k * (n - 1) + 1)
    return CubicSpline(np.arange(n), pts)(u), np.stack([np.interp(u, np.arange(n), col[:, i]) for i in range(3)], 1).astype(np.uint8)

def _light(pl, size, shadows=True):
    """Soft studio light: key light with shadows, ambient occlusion, anti-aliasing (each step optional)."""
    import pyvista as pv
    pl.remove_all_lights(); pl.add_light(pv.Light(position=(3, 4, 5), focal_point=(0, 0, 0), intensity=.9, light_type="scene light"))
    pl.add_light(pv.Light(position=(-4, -2, 3), intensity=.35, light_type="scene light"))
    for f in (lambda: pl.enable_anti_aliasing("ssaa"), lambda: pl.enable_ssao(radius=size / 55, bias=.6, kernel_size=128, blur=True), (lambda: pl.enable_shadows()) if shadows else (lambda: None)):
        try: f()
        except Exception: pass

BG = np.array([232, 237, 245.])   # backdrop colour that distant parts fade into

def _fade(pl, pts, rgb, ext, k=.35):
    """Depth cueing (as in ChimeraX): colours blend into the backdrop with distance from the camera, so far parts look far."""
    cam = np.array(pl.camera.position); fp = np.array(pl.camera.focal_point); v = fp - cam; v /= np.linalg.norm(v)
    t = np.clip(((np.asarray(pts) - cam) @ v - (np.linalg.norm(fp - cam) - ext)) / (2.2 * ext), 0, 1)[:, None] ** 1.2
    return ((1 - k * t) * np.asarray(rgb, float) + k * t * BG).astype(np.uint8)

def _add(pl, mesh, color, ext, **kw):
    """Add a mesh with a constant colour (hex) or per-point rgb, depth-cued."""
    rgb = np.tile(_rgb(color), (mesh.n_points, 1)) if isinstance(color, str) else color
    mesh["rgb"] = _fade(pl, mesh.points, rgb, ext); return pl.add_mesh(mesh, scalars="rgb", rgb=True, smooth_shading=True, **kw)

def _arrows(b, idx, sign, r, ctr):
    """Arrow glyphs for CTCF sites: placed off the fibre (away from the centre), pointing along (+1) or against (-1) the chain."""
    import pyvista as pv
    t = b[np.minimum(idx + 1, len(b) - 1)] - b[np.maximum(idx - 1, 0)]; t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    out = b[idx] - ctr; out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9; out -= (out * t).sum(1, keepdims=True) * t; out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    pos = b[idx] + out * r * 2.6; d = t * sign[:, None]; poly = pv.PolyData(pos - d * r * 2.5); poly["d"] = d   # centre the arrow on the site
    return poly.glyph(orient="d", scale=False, factor=r * 5, geom=pv.Arrow(tip_length=.42, tip_radius=.17, shaft_radius=.07, tip_resolution=16, shaft_resolution=16))

def _draw(pl, tr, V, col, fn, r, size, ext):
    """Add tubes (one per chain), cohesin / condensin anchors + links, CTCF arrows and forks of frame fn to the plotter (all depth-cued)."""
    import pyvista as pv
    N = tr.n_beads; sph = lambda q: pv.Sphere(radius=q, theta_resolution=24, phi_resolution=24); lefs, fam = tr.lefs[fn], (tr.lef_fam[fn] if tr.lef_fam else np.zeros(len(tr.lefs[fn]), int))
    for c in range(tr.n_chains):
        b = V[c * N:(c + 1) * N]; pts, cc = _smooth(b, col[c * N:(c + 1) * N], 6); M = len(pts)
        poly = pv.PolyData(pts); poly.lines = np.r_[M, np.arange(M)].astype(np.int_); tube = poly.tube(radius=r, n_sides=24)
        _add(pl, tube, cc[cKDTree(pts).query(tube.points)[1]], ext, specular=.6, specular_power=30, ambient=.25, diffuse=.85)   # colour of the nearest spline point
        for k in (0, 1):
            e = lefs[fam == k]
            if not len(e): continue
            ap = b[e.ravel()]; ch = pv.PolyData(ap); nl = len(e)
            ch.lines = np.c_[np.full(nl, 2), np.arange(0, 2 * nl, 2), np.arange(1, 2 * nl, 2)].ravel()
            _add(pl, ch.tube(radius=r * .3, n_sides=12), LEF_COL[k], ext, opacity=.85, specular=.4)
            _add(pl, pv.PolyData(ap).glyph(geom=sph(r * 1.25), scale=False, orient=False), LEF_COL[k], ext, specular=.7, specular_power=30)
        ci, cs = tr.ctcf
        if len(ci): _add(pl, _arrows(b, ci, cs, r, b.mean(0)), CTCF_COL, ext, specular=.5)
        if tr.has_rep:
            fk = np.flatnonzero(tr.rep[fn][:-1] != tr.rep[fn][1:])
            if len(fk): _add(pl, pv.PolyData(b[fk]).glyph(geom=sph(r * 2.0), scale=False, orient=False), FORK_COL, ext, specular=.6)

def _quiet_vtk():
    """Hide VTK / OpenGL warnings unless the run is very verbose (-vvv)."""
    import pyvista as pv
    if log.verbosity >= 3: return
    try: pv.vtk_verbosity("off")
    except Exception: pass
    try:
        import vtk; vtk.vtkObject.GlobalWarningDisplayOff()
    except Exception: pass

def _scene(tr, size, shadows=True):
    """Plotter with a soft white-to-grey backdrop (ChimeraX-like), light, and a slightly raised camera."""
    import pyvista as pv
    _quiet_vtk()
    ext = float(np.linalg.norm(tr.V, axis=2).max()); pl = pv.Plotter(off_screen=True, window_size=(size, size)); pl.set_background("#FFFFFF", top="#E6EBF3"); _light(pl, size, shadows)
    pl.camera_position = [(0, .35 * ext, 3.6 * ext), (0, 0, 0), (0, 1, 0)]
    return pl, ext, None

def _font(px, bold=True):
    try: return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", px)
    except OSError: return ImageFont.load_default()

def overlay(img, tr, f, mode, time_label=""):
    """Phase badge (top left) and colour legend (bottom left) drawn on a rendered frame."""
    im = Image.fromarray(img); d = ImageDraw.Draw(im, "RGBA"); S = im.size[0]; fs = max(13, S // 38); ft, fl = _font(int(fs * 1.25)), _font(fs, False); pad = fs // 2
    ph = tr.phase[f]; col = _rgb(PHASE_COL.get(ph, "#7B4FA3")); txt = f"{ph}" + (f"   {time_label}" if time_label else "")
    w = d.textlength(txt, font=ft); d.rounded_rectangle([pad, pad, pad * 3 + w, pad * 3 + fs * 1.25], radius=pad, fill=col + (235,)); d.text((pad * 2, pad * 2), txt, font=ft, fill="white")
    ent = []   # (kind, colour, label)
    if mode == "sister": ent += [("dot", _rgb(SIS[0]), "sister 1")] + ([("dot", _rgb(SIS[1]), "sister 2")] if tr.n_chains > 1 else [])
    elif mode == "compartment": ent += [("dot", SPIN_RGB[2], "A: positive states"), ("dot", SPIN_RGB[-2], "B: negative states")]
    elif mode == "replication": ent += [("dot", _rgb(REPL), "replicated"), ("dot", _rgb(UNREPL), "not yet")]
    fam = set(int(x) for a in tr.lef_fam for x in a) if tr.lef_fam else set()
    ent += [("dot", _rgb(LEF_COL[k]), LEF_NAME[k]) for k in sorted(fam)]
    if len(tr.ctcf[0]): ent.append(("arrow", _rgb(CTCF_COL), "CTCF (arrow = orientation)"))
    if tr.has_rep: ent.append(("dot", _rgb(FORK_COL), "replication fork"))
    h = fs * 1.5; y0 = S - pad - h * len(ent) - pad; wmax = max(d.textlength(e[2], font=fl) for e in ent) if ent else 0
    if ent: d.rounded_rectangle([pad, y0 - pad / 2, pad * 4 + fs + wmax, S - pad / 2], radius=pad, fill=(255, 255, 255, 190))
    for i, (kind, c, lab) in enumerate(ent):
        y = y0 + i * h + h / 2; x = pad * 2 + fs / 2
        if kind == "dot": d.ellipse([x - fs / 3, y - fs / 3, x + fs / 3, y + fs / 3], fill=c)
        else: d.polygon([(x - fs / 2.2, y - fs / 5), (x + fs / 8, y - fs / 5), (x + fs / 8, y - fs / 2.5), (x + fs / 2.2, y), (x + fs / 8, y + fs / 2.5), (x + fs / 8, y + fs / 5), (x - fs / 2.2, y + fs / 5)], fill=c)
        d.text((x + fs, y), lab, font=fl, fill=(30, 30, 36), anchor="lm")
    return np.array(im)

def render_still(tr, f, out_png, mode="sister", size=900, radius=None):
    """One high-quality picture of frame f (used for the report)."""
    pl, ext, floor = _scene(tr, size); r = radius or ext / 60; col = bead_rgb(tr, f, mode)
    _draw(pl, tr, tr.V[f], col, f, r, size, ext); img = pl.screenshot(return_img=True); pl.close()
    Image.fromarray(overlay(img, tr, f, mode, f"MC step {int(tr.step[f])}")).save(out_png); return out_png

def render_stills(tr, out_png, mode="sister", size=700):
    """Strip of stills (one per cell-cycle phase present) with phase labels, for the report."""
    first = {}
    for f, ph in enumerate(tr.phase): first.setdefault(ph, f)
    pick = {ph: (f + max(i for i, q in enumerate(tr.phase) if q == ph)) // 2 for ph, f in first.items()}   # middle frame of each phase
    tmp = []
    for ph, f in pick.items():
        render_still(tr, f, out_png + f".{ph.replace('/', '')}.png", mode, size); tmp.append(Image.open(out_png + f".{ph.replace('/', '')}.png").convert("RGB"))
    strip = Image.new("RGB", (size * len(tmp), size), "white")
    for i, im in enumerate(tmp): strip.paste(im, (i * size, 0))
    strip.save(out_png)
    import os
    for ph in pick: os.remove(out_png + f".{ph.replace('/', '')}.png")
    return out_png

def make_movie(tr, out_base, mode="sister", n_interp=6, fps=24, win=8, size=640, radius=None, gif=False, max_frames=120, cmap="turbo"):
    """Write out_base.mp4 (and .gif if asked). Coordinates are interpolated with a smooth monotone cubic (PCHIP) and colours are cross-faded."""
    import imageio
    from scipy.interpolate import PchipInterpolator
    idx = np.unique(np.linspace(0, len(tr.V) - 1, min(max_frames, len(tr.V))).astype(int)); K = len(idx)
    cm = Contacts(tr, win); pl, ext, floor = _scene(tr, size); r = radius or ext / 60
    path = PchipInterpolator(np.arange(K), tr.V[idx], axis=0) if K > 2 else None
    mp4 = imageio.get_writer(out_base + ".mp4", fps=fps, codec="libx264", quality=9, macro_block_size=2, pixelformat="yuv420p", ffmpeg_log_level="error")
    hcache = {}; frames = []; k = 0
    for a in log.progress(range(K), desc="Rendering movie"):
        for j in range(n_interp if a < K - 1 else 1):
            t = j / n_interp; ft = a + t; f0, f1 = idx[a], idx[min(a + 1, K - 1)]; fn = f0 if t < .5 else f1
            V = path(ft) if path is not None else (1 - t) * tr.V[f0] + t * tr.V[f1]; w = t * t * (3 - 2 * t)
            col = ((1 - w) * bead_rgb(tr, f0, mode) + w * bead_rgb(tr, f1, mode)).astype(np.uint8); pl.clear(); _draw(pl, tr, V, col, fn, r, size, ext)
            pl.camera.Azimuth(.6)
            left = overlay(pl.screenshot(return_img=True), tr, int(fn), mode, f"MC step {int(tr.step[fn])}")
            if fn not in hcache: hcache.clear(); hcache[fn] = heat_image(tr, cm, int(fn), left.shape[0], cmap)
            frame = np.vstack([np.hstack([left, hcache[fn]]), bar_image(tr, playhead(idx, ft), left.shape[1] * 2)])
            frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]; mp4.append_data(frame)
            if gif and k % 3 == 0: frames.append(Image.fromarray(frame).resize((frame.shape[1] // 2, frame.shape[0] // 2), Image.LANCZOS))
            k += 1
    mp4.close(); pl.close()
    if gif: frames[0].save(out_base + ".gif", save_all=True, append_images=frames[1:], duration=int(3000 / fps), loop=0, optimize=True)
    return out_base + ".mp4"
