"""Quick checks of the input handling. Run from the repository root:  python demo/test_inputs.py"""
import subprocess, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ["--region", "chr14:40535000-60174700", "--n-beads", "100", "--n-steps", "3000", "--mc-step", "100", "--burnin", "300",
        "--out-path", str(ROOT / "demo" / "results"), "--no-viewer", "--no-movie", "--no-report"]

def run(*args, stdin=""):
    """Run replisage with the given flags; returns (exit code, output)."""
    r = subprocess.run([sys.executable, "-m", "src.run", *args], cwd=ROOT, input=stdin, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr

def check(name, ok): print(f"{'PASS' if ok else 'FAIL'}  {name}"); return ok

res = []
code, out = run(*BASE, "--run-name", "t_nobedpe")
res.append(check("no BEDPE -> error about CTCF barriers", code != 0 and "CTCF barriers" in out))
code, out = run(*BASE, "--bedpe-file", "demo/missing.bedpe", "--run-name", "t_missing")
res.append(check("missing BEDPE file -> error", code != 0 and "not found" in out))
code, out = run(*BASE, "--bedpe-file", "demo/loops_no_orientation.bedpe", "--rt-file", "none", "--run-name", "t_noorient", "-v")
res.append(check("no orientation -> warning with motif_finder link, still runs", code == 0 and "no CTCF orientation" in out and "motif_finder" in out))
res.append(check("-v prints input statistics in kb", "loop length, kb" in out and "bead size" in out))
code, out = run(*BASE, "--bedpe-file", "demo/loops_demo.bedpe", "--run-name", "t_nort")
res.append(check("no RT (not a terminal) -> loop extrusion only", code == 0 and "loop extrusion only" in out and "G2/M phase" in out))
code, out = run(*BASE, "--bedpe-file", "demo/loops_demo.bedpe", "--rt-file", "demo/nonexistent.bw", "--rep-start", "1000", "--rep-duration", "1500", "--run-name", "t_fallback")
res.append(check("missing RT file -> fallback to built-in single-cell RT", code == 0 and "not found" in out and "single-cell" in out))
print(f"\n{sum(res)}/{len(res)} checks passed"); sys.exit(0 if all(res) else 1)
