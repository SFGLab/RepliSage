"""Stage: interactive HTML viewer and PyVista movie of the 3D ensemble."""
import os
from ..logger import log

def run_viz3d(ctx, p, res, out_dir, rep):
    """Write viewer_3d.html and movie_3d.mp4/.gif next to the ensemble (each step is optional and fails softly)."""
    a = ctx.args
    if not (a.HTML_3D or not a.SKIP_MOVIES): return
    from ..viz3d.data import load_traj
    try: tr = load_traj(out_dir, p, res, rep)
    except Exception as e: return log.alert(f"3D visualisation skipped: {e}")
    if a.HTML_3D:
        try:
            from ..viz3d.html_viewer import write_html_viewer
            log.success(f"3D viewer: {write_html_viewer(tr, os.path.join(out_dir, 'viewer_3d.html'), os.path.basename(ctx.run_dir), cmap=a.MOVIE_CMAP)}")
        except Exception as e: log.alert(f"3D viewer skipped: {e}")
    modes = movie_modes(a.MOVIE_COLOR, tr)
    if not a.SKIP_MOVIES:
        from ..viz3d.video import make_movie
        for m in modes:   # one mp4 per colouring
            try: log.success(f"Movie ({m}): {make_movie(tr, os.path.join(out_dir, f'movie_3d_{m}'), m, a.MOVIE_INTERP, cmap=a.MOVIE_CMAP)}")
            except Exception as e: log.alert(f"Movie ({m}) skipped: {e}")
    try:   # one still per phase for the report (first colouring)
        from ..viz3d.video import render_stills
        d = os.path.join(out_dir, "plots", "3d"); os.makedirs(d, exist_ok=True)
        for m in modes: render_stills(tr, os.path.join(d, f"stills_{m}.png"), m)
    except Exception as e: log.alert(f"3D stills skipped: {e}")

def movie_modes(choice, tr):
    """auto -> sister chromatids and compartment colourings (those that make sense for the data); otherwise the one requested."""
    if choice != "auto": return [choice]
    return (["sister"] if tr.has_rep else []) + (["compartment"] if tr.spins.any() else []) or ["rainbow"]
