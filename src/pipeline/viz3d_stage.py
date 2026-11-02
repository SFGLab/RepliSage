"""Stage: interactive HTML viewer and PyVista movie of the 3D ensemble."""
import os
from ..logger import log

def run_viz3d(ctx, p, res, out_dir, rep):
    """Write viewer_3d.html and movie_3d.mp4/.gif next to the ensemble (each step is optional and fails softly)."""
    a = ctx.args
    if not (a.HTML_3D or a.MOVIE): return
    from ..viz3d.data import load_traj
    try: tr = load_traj(out_dir, p, res, rep)
    except Exception as e: return log.alert(f"3D visualisation skipped: {e}")
    if a.HTML_3D:
        from ..viz3d.html_viewer import write_html_viewer
        path = write_html_viewer(tr, os.path.join(out_dir, "viewer_3d.html"), os.path.basename(ctx.run_dir)); log.success(f"3D viewer: {path}")
    if a.MOVIE:
        try:
            from ..viz3d.video import make_movie
            mode = a.MOVIE_COLOR if a.MOVIE_COLOR != "auto" else ("sister" if tr.has_rep else "compartment" if tr.spins.any() else "rainbow")
            log.success(f"Movie: {make_movie(tr, os.path.join(out_dir, 'movie_3d'), mode, a.MOVIE_INTERP)}")
        except Exception as e: log.alert(f"Movie skipped: {e}")
    try:   # one still per phase for the report
        from ..viz3d.video import render_stills
        d = os.path.join(out_dir, "plots", "3d"); os.makedirs(d, exist_ok=True)
        mode = a.MOVIE_COLOR if a.MOVIE_COLOR != "auto" else ("sister" if tr.has_rep else "compartment" if tr.spins.any() else "rainbow")
        render_stills(tr, os.path.join(d, "stills.png"), mode)
    except Exception as e: log.alert(f"3D stills skipped: {e}")
