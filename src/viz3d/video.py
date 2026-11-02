"""PyVista movie (mp4 + gif) of the ensemble with interpolation, contact map and a cell-cycle bar."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cdist
from ..plots.style import CMAP_SEQ, PHASE_COL, A_COL, B_COL, ORANGE, GOLD, ROYAL, TEAL, BLACK
from ..logger import log

SIS = ("#E8702A", "#E3A008")
def _rgb(h): return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

def bead_rgb(tr, f, mode):
    """(chains*N, 3) uint8 colours of frame f for a colour mode: sister / compartment / rainbow / replication."""
    N, C = tr.n_beads, tr.n_chains
    if mode == "sister": one = [np.tile(_rgb(SIS[c]), (N, 1)) for c in range(C)]
    elif mode == "compartment":
        s = tr.spins[f]; col = np.array([_rgb("#BDBDBD")] * N); col[s > 0] = _rgb("#C8213F"); col[s < 0] = _rgb("#2B6CB0"); one = [col] * C
    elif mode == "replication":
        col = np.array([_rgb("#9AA3AE")] * N); col[tr.rep[f] > 0] = _rgb(ROYAL); one = [col] * C
    else:
        import colorsys
        one = [np.array([[int(255 * v) for v in colorsys.hls_to_rgb(.78 * (1 - i / (N - 1)), .5, .8)] for i in range(N)])] * C
    return np.vstack(one).astype(np.uint8)

class Contacts:
    """Window-averaged contact map of the frames, cached per frame."""
    def __init__(self, tr, win=8, mult=2.0):
        self.tr, self.win, self.c = tr, win, {}
        V = tr.V[0]; self.rc = mult * np.median(np.linalg.norm(np.diff(V[:tr.n_beads], axis=0), axis=1))
    def frame(self, f):
        if f not in self.c:
            tr, N = self.tr, self.tr.n_beads; V = tr.V[f]; m = np.zeros((N, N))
            for a in range(tr.n_chains):
                for b in range(tr.n_chains): m += cdist(V[a * N:(a + 1) * N], V[b * N:(b + 1) * N]) < self.rc
            self.c[f] = m / tr.n_chains ** 2
        return self.c[f]
    def avg(self, f):
        a = max(0, f - self.win + 1); return np.mean([self.frame(g) for g in range(a, f + 1)], axis=0)

def heat_image(tr, cm, f, size):
    """RGB uint8 square image: contact map + compartment / replication strips + LEF dots + fork lines."""
    N = tr.n_beads; M = cm.avg(f); iu = np.abs(np.subtract.outer(np.arange(N), np.arange(N))) > 3
    vmax = max(.05, np.percentile(M[iu], 99.5)); img = (CMAP_SEQ(np.clip(M / vmax, 0, 1) ** .6)[..., :3] * 255).astype(np.uint8)
    m0 = max(10, size // 40); S = size - m0; big = np.array(Image.fromarray(img).resize((S, S), Image.NEAREST))
    can = np.full((size, size, 3), 255, np.uint8); can[m0:, m0:] = big; u = S / N
    sp = tr.spins[f]; rp = tr.rep[f]; cmp = lambda s: _rgb("#C8213F") if s > 0 else (_rgb("#2B6CB0") if s < 0 else (221, 221, 221))
    im = Image.fromarray(can); d = ImageDraw.Draw(im)
    for i in range(N):
        x = m0 + i * u; d.rectangle([x, 2, x + u, m0 - 3], fill=cmp(sp[i])); d.rectangle([2, x, m0 - 3, x + u], fill=_rgb(ROYAL) if rp[i] else (227, 227, 227))
    if tr.has_rep:
        for i in np.flatnonzero(rp[:-1] != rp[1:]):
            p = m0 + (i + .5) * u; d.line([p, m0, p, size], fill=_rgb(TEAL), width=2); d.line([m0, p, size, p], fill=_rgb(TEAL), width=2)
    r = max(2, size // 200)
    for a, b in tr.lefs[f]:
        for x, y in ((a, b), (b, a)): cx, cy = m0 + (x + .5) * u, m0 + (y + .5) * u; d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_rgb(BLACK), outline=(255, 255, 255))
    return np.array(im)

def bar_image(tr, t, width, h=90, fps_text=""):
    """Phase bar with the replicated-fraction curve and a playhead at fractional frame t."""
    F = len(tr.V); im = Image.new("RGB", (width, h), "white"); d = ImageDraw.Draw(im); bw = width / F; top, bot = 24, h - 8
    try: ft = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
    except OSError: ft = ImageFont.load_default()
    s = 0
    for f in range(F + 1):
        if f == F or tr.phase[f] != tr.phase[s]:
            col = _rgb(PHASE_COL.get(tr.phase[s], "#7B4FA3")); d.rectangle([s * bw, top, f * bw, bot], fill=col)
            d.text(((s + f) / 2 * bw, 2), tr.phase[s], fill=(20, 20, 20), font=ft, anchor="mt"); s = f
    if tr.has_rep:
        y = bot - tr.rep.mean(1) * (bot - top - 4) - 2; d.line([((i + .5) * bw, v) for i, v in enumerate(y)], fill="white", width=3)
    x = (t + .5) * bw; d.rectangle([x - 2, top - 5, x + 2, bot + 3], fill=(20, 20, 20)); return np.array(im)

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
    for f in (lambda: pl.enable_anti_aliasing("ssaa"), lambda: pl.enable_ssao(radius=size / 40, bias=.5, kernel_size=128, blur=True), (lambda: pl.enable_shadows()) if shadows else (lambda: None)):
        try: f()
        except Exception: pass

def _draw(pl, tr, V, col, fn, r, size, floor=None):
    """Add tubes (one per chain), LEF anchors / links and forks of frame fn to the plotter."""
    import pyvista as pv
    N = tr.n_beads
    if floor is not None: pl.add_mesh(floor, color="#F3F4F7", ambient=.8, diffuse=.2, specular=0)
    for c in range(tr.n_chains):
        pts, cc = _smooth(V[c * N:(c + 1) * N], col[c * N:(c + 1) * N]); M = len(pts)
        poly = pv.PolyData(pts); poly.lines = np.r_[M, np.arange(M)].astype(np.int_); poly["rgb"] = cc
        pl.add_mesh(poly.tube(radius=r, n_sides=16), scalars="rgb", rgb=True, smooth_shading=True, specular=.5, specular_power=25, ambient=.25, diffuse=.8)
        b = V[c * N:(c + 1) * N]
        if len(tr.lefs[fn]):
            ap = b[tr.lefs[fn].ravel()]; ch = pv.PolyData(ap); nl = len(ap) // 2
            ch.lines = np.c_[np.full(nl, 2), np.arange(0, 2 * nl, 2), np.arange(1, 2 * nl, 2)].ravel()
            pl.add_mesh(ch.tube(radius=r * .35, n_sides=8), color=ROYAL, opacity=.75, smooth_shading=True)
            pl.add_mesh(pv.PolyData(ap).glyph(geom=pv.Sphere(radius=r * 1.6, theta_resolution=16, phi_resolution=16), scale=False, orient=False), color="#2A2D34", smooth_shading=True, specular=.5)
        if tr.has_rep:
            fk = np.flatnonzero(tr.rep[fn][:-1] != tr.rep[fn][1:])
            if len(fk): pl.add_mesh(pv.PolyData(b[fk]).glyph(geom=pv.Sphere(radius=r * 2.0, theta_resolution=16, phi_resolution=16), scale=False, orient=False), color=TEAL, smooth_shading=True, specular=.6)

def _scene(tr, size, shadows=True):
    """Plotter with white background, light, and a slightly raised camera; returns (plotter, ext, floor=None)."""
    import pyvista as pv
    ext = float(np.linalg.norm(tr.V, axis=2).max()); pl = pv.Plotter(off_screen=True, window_size=(size, size)); pl.set_background("white"); _light(pl, size, shadows)
    floor = None   # a floor looked heavy on white; shading comes from key light, shadows and ambient occlusion
    pl.camera_position = [(0, .35 * ext, 3.6 * ext), (0, 0, 0), (0, 1, 0)]
    return pl, ext, floor

def render_still(tr, f, out_png, mode="sister", size=900, radius=None):
    """One high-quality picture of frame f (used for the report)."""
    pl, ext, floor = _scene(tr, size); r = radius or ext / 60; col = bead_rgb(tr, f, mode)
    _draw(pl, tr, tr.V[f], col, f, r, size, floor); img = pl.screenshot(return_img=True); pl.close()
    Image.fromarray(img).save(out_png); return out_png

def render_stills(tr, out_png, mode="sister", size=700):
    """Strip of stills (one per cell-cycle phase present) with phase labels, for the report."""
    first = {}
    for f, ph in enumerate(tr.phase): first.setdefault(ph, f)
    pick = {ph: (f + max(i for i, q in enumerate(tr.phase) if q == ph)) // 2 for ph, f in first.items()}   # middle frame of each phase
    tmp = []; fnt = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    for ph, f in pick.items():
        render_still(tr, f, out_png + f".{ph.replace('/', '')}.png", mode, size); im = Image.open(out_png + f".{ph.replace('/', '')}.png").convert("RGB")
        ImageDraw.Draw(im).text((24, 20), ph, fill=_rgb(PHASE_COL.get(ph, "#7B4FA3")), font=fnt); tmp.append(im)
    strip = Image.new("RGB", (size * len(tmp), size), "white")
    for i, im in enumerate(tmp): strip.paste(im, (i * size, 0))
    strip.save(out_png)
    import os
    for ph in pick: os.remove(out_png + f".{ph.replace('/', '')}.png")
    return out_png

def make_movie(tr, out_base, mode="sister", n_interp=6, fps=24, win=8, size=640, radius=None, gif=True, max_frames=120):
    """Write out_base.mp4 (and .gif). Coordinates are interpolated with a smooth monotone cubic (PCHIP) and colours are cross-faded."""
    import imageio
    from scipy.interpolate import PchipInterpolator
    idx = np.unique(np.linspace(0, len(tr.V) - 1, min(max_frames, len(tr.V))).astype(int)); K = len(idx)
    cm = Contacts(tr, win); pl, ext, floor = _scene(tr, size); r = radius or ext / 60
    path = PchipInterpolator(np.arange(K), tr.V[idx], axis=0) if K > 2 else None
    mp4 = imageio.get_writer(out_base + ".mp4", fps=fps, codec="libx264", quality=8, macro_block_size=2, ffmpeg_log_level="error")
    hcache = {}; frames = []; k = 0
    for a in log.progress(range(K), desc="Rendering movie"):
        for j in range(n_interp if a < K - 1 else 1):
            t = j / n_interp; ft = a + t; f0, f1 = idx[a], idx[min(a + 1, K - 1)]; fn = f0 if t < .5 else f1
            V = path(ft) if path is not None else (1 - t) * tr.V[f0] + t * tr.V[f1]; w = t * t * (3 - 2 * t)
            col = ((1 - w) * bead_rgb(tr, f0, mode) + w * bead_rgb(tr, f1, mode)).astype(np.uint8); pl.clear(); _draw(pl, tr, V, col, fn, r, size, floor)
            pl.camera.Azimuth(.6)
            left = pl.screenshot(return_img=True)
            if fn not in hcache: hcache.clear(); hcache[fn] = heat_image(tr, cm, int(fn), left.shape[0])
            frame = np.vstack([np.hstack([left, hcache[fn]]), bar_image(tr, ft, left.shape[1] * 2)])
            frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]; mp4.append_data(frame)
            if gif and k % 3 == 0: frames.append(Image.fromarray(frame).resize((frame.shape[1] // 2, frame.shape[0] // 2), Image.LANCZOS))
            k += 1
    mp4.close(); pl.close()
    if gif: frames[0].save(out_base + ".gif", save_all=True, append_images=frames[1:], duration=int(3000 / fps), loop=0, optimize=True)
    return out_base + ".mp4"
