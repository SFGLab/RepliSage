"""Self-contained interactive 3D viewer (three.js inlined): trajectory bar, colour modes, contact map."""
import base64, json, os
import numpy as np

def _b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

def write_html_viewer(traj, path, title="", max_frames=400):
    """Write the viewer HTML for a Traj. Frames are thinned to max_frames to keep the file small."""
    here = os.path.dirname(__file__); rd = lambda *a: open(os.path.join(here, *a), encoding="utf-8").read()
    keep = np.unique(np.linspace(0, len(traj.V) - 1, min(max_frames, len(traj.V))).astype(int))
    V = traj.V[keep]; sc = float(np.abs(V).max()) / 32000 or 1.0
    data = dict(ext=float(np.percentile(np.linalg.norm(V, axis=2), 98)), F=len(keep), C=traj.n_chains, N=traj.n_beads, scale=sc, hasRep=traj.has_rep,
                xyz=_b64(np.round(V / sc).astype(np.int16)), spin=_b64(traj.spins[keep].astype(np.int8)), rep=_b64(traj.rep[keep].astype(np.uint8)),
                phase=[traj.phase[i] for i in keep], step=[int(traj.step[i]) for i in keep],
                lefs=[traj.lefs[i].astype(int).ravel().tolist() for i in keep])
    html = rd("assets", "viewer.html")
    for k, v in (("/*CSS*/", rd("assets", "viewer.css")), ("/*TITLE*/", title), ("/*THREE*/", rd("vendor", "three.min.js")),
                 ("/*ORBIT*/", rd("vendor", "OrbitControls.js")), ("/*APP*/", rd("assets", "viewer.js")),
                 ("/*DATA*/", json.dumps(data, separators=(",", ":")))):
        html = html.replace(k, v)   # data last: it can't contain the other markers
    with open(path, "w", encoding="utf-8") as f: f.write(html)
    return path
