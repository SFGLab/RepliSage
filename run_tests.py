"""Run all RepliSage checks and print one summary.

  python run_tests.py            quick: demo checks + fast tests (about 1 min)
  python run_tests.py --all      also the slow tests (OpenMM and full pipelines)
  python run_tests.py -k kernel  only tests whose name matches (extra args go to pytest)
"""
import subprocess, sys, time, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
DEMOS = ["demo/test_config.py", "demo/test_inputs.py", "demo/test_normalisation.py"]

FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"          # spinner frames
DANCE = ["(•_•)", "(•_•)>⌐■-■", "(⌐■_■)", "♪┏(・o･)┛♪", "♪┗(･o･)┓♪"]      # a little dancer for long steps

def run(cmd, label=""):
    """Run a command with a spinner while it works; return (ok, seconds, output)."""
    import tempfile
    t = time.time(); tty = sys.stdout.isatty()
    with tempfile.TemporaryFile("w+") as out:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, text=True); i = 0
        while p.poll() is None:
            if tty: print(f"\r\033[36m{FRAMES[i % len(FRAMES)]}\033[0m {label:<28}{time.time() - t:6.0f} s  {DANCE[i // 4 % len(DANCE)]}   ", end="", flush=True)
            i += 1; time.sleep(.1)
        if tty: print("\r" + " " * 78 + "\r", end="")
        out.seek(0); return p.returncode == 0, time.time() - t, out.read()

def main(argv):
    full = "--all" in argv; extra = [a for a in argv if a != "--all"]
    steps = [(d, [sys.executable, d]) for d in DEMOS]
    steps.append(("fast tests", [sys.executable, "-m", "pytest", "-q", "--no-header", *extra]))
    if full: steps.append(("slow tests (OpenMM)", [sys.executable, "-m", "pytest", "-q", "--no-header", "-m", "slow", *extra]))
    rows, failed = [], []
    for name, cmd in steps:
        ok, sec, out = run(cmd, name); print(f"{'✔' if ok else '✘'} {name} ({sec:.0f} s)", flush=True); rows.append((name, ok, sec, out.strip().splitlines()[-1] if out.strip() else ""))
        if not ok: failed.append((name, out))
    print("\n" + "=" * 72)
    for name, ok, sec, last in rows: print(f"{'PASS' if ok else 'FAIL':5}{name:28}{sec:7.1f} s   {last[:60]}")
    for name, out in failed: print(f"\n--- {name}: output ---\n" + "\n".join(out.strip().splitlines()[-40:]))
    print("=" * 72 + f"\n{'ALL OK' if not failed else str(len(failed)) + ' step(s) failed'}")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
