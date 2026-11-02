"""Assemble report.html: model summary, statistics, figures with captions, command and all parameters."""
import datetime
import os
import platform
import re
import sys
from importlib import metadata
from pathlib import Path
from .theme import CSS, JS
from .text import TAGLINE, ABOUT, STAGES, EQ, READ
from .html_parts import esc, figure, table, section, chip, params_table, page
from ..analysis.tables import Table, has_potts
from ..plots.style import FIGS
from .._version import __version__
from ..logger import log

def _ver(name):
    try: return metadata.version(name)
    except Exception: return "n/a"

def environment_rows(ctx):
    a = ctx.args
    return [("RepliSage", __version__), ("Python", platform.python_version()), ("numpy / numba / scipy", f"{_ver('numpy')} / {_ver('numba')} / {_ver('scipy')}"),
            ("matplotlib", _ver("matplotlib")), ("OpenMM", _ver("openmm")), ("system", f"{platform.system()} {platform.machine()}, {os.cpu_count()} CPUs"),
            ("seed / threads / replicas", f"{a.SEED} / {a.N_THREADS} / {a.N_REPLICAS}"), ("finished", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ("run time", log.total_time())]

def _hero(ctx, p, extra=()):
    reg = f"{ctx.chrom}:{ctx.region[0]:,}-{ctx.region[1]:,}" if ctx.region_given else f"{ctx.chrom} (whole chromosome)"
    chips = [chip(Path(ctx.run_dir).name, "red"), chip(reg), chip(f"{ctx.n_beads} beads"), chip("Loop extrusion", "on"),
             chip("Replication", "on" if ctx.has_rep else "off"), chip("Compartments", "on" if has_potts(p) else "off"), chip("3D structures", "on" if ctx.has_md else "off"), *extra]
    return f'<header class="hero"><h1>Repli<b>Sage</b></h1><div class="tag">{TAGLINE}</div><div class="meta">{"".join(chips)}</div></header>'

def _about():
    st = "".join(f'<div class="card"><b>{esc(t)}</b><br>{esc(d)}</div>' for t, d in STAGES)
    eq = "".join(f'<div class="eq">{e}<small>{esc(c)}</small></div>' for e, c in EQ)
    return (f'<h3>What RepliSage does</h3>' + "".join(f"<p>{esc(t)}</p>" for t in ABOUT) + f'<div class="grid2">{st}</div><h3>Energy of the model</h3>{eq}'
            f'<div class="legend"><span><i class="sw" style="background:#E6E6E6"></i>burn-in</span><span><i class="sw" style="background:#F3D0D6"></i>S phase</span>'
            f'<span><i class="sw" style="background:#A50F2D"></i>A compartment</span><span><i class="sw" style="background:#2B6CB0"></i>B compartment</span></div><p class="note">{esc(READ)}</p>')

def _figs(F, ids): return "".join(figure(i, F[i]) for i in ids if i in F)

def _repro(ctx, out_dir, command):
    cfg = Path(out_dir) / "config_used.ini" if (Path(out_dir) / "config_used.ini").exists() else Path(ctx.run_dir) / "config_used.ini"
    rel = str(cfg.resolve()) if cfg.exists() else "config_used.ini"
    ini = cfg.read_text() if cfg.exists() else ""
    env = Table("env", "Environment", ("Item", "Value"), environment_rows(ctx))
    return (f'<h3>Command</h3><pre>{esc(command or "(started from Python, no command line)")}</pre><h3>Reproduce</h3><pre>replisage -c {esc(rel)}</pre>'
            f'{table(env)}<h3>All parameters</h3>{params_table(ctx.args)}<h3>Configuration file</h3><pre>{esc(ini)}</pre>')

def _viz3d(F, out_dir):
    """Links to the 3D viewer / movie / ChimeraX script and the inline movie preview."""
    d = Path(out_dir); names = [("viewer_3d.html", "interactive 3D viewer"), ("movie_3d.mp4", "movie (mp4)"), ("ensemble/view.cxc", "ChimeraX script")]
    links = " · ".join(f'<a href="{n}">{t}</a>' for n, t in names if (d / n).exists())
    gif = d / "movie_3d.gif"
    return (f"<p>{links}</p>" if links else "") + (figure("movie_gif", str(gif)) if gif.exists() and gif.stat().st_size < 4e6 else "")

def _md_extra(F, out_dir):
    f = Path(out_dir) / "plots" / "md_simulation" / "dynamics.png"
    if f.exists(): F["md_dynamics"] = str(f)
    heats = sorted((k for k in F if k.startswith("md_heat_")), key=lambda k: int(k.split("_")[2]))
    return heats

def write_report(ctx, p, res, out_dir, rep, ph, obs, tables, command=""):
    """Write <out_dir>/report.html and return its path."""
    F, T = dict(FIGS), {t.key: t for t in tables}; heats = _md_extra(F, out_dir); secs = []
    def add(sid, title, body, lead=""):
        if body.strip(): secs.append((sid, title, section(sid, title, body, lead)))
    add("about", "The model", _about())
    add("summary", "Run summary", "".join(table(T[k]) for k in ("overview", "lef", "replication", "compartments") if k in T), "Key numbers of this run.")
    add("input", "Input", _figs(F, ["loops_input"]), "Loop anchors extracted from your BEDPE file.")
    add("mcmc", "Monte Carlo diagnostics", _figs(F, ["mc_energy", "mc_energy_potts", "mc_distributions", "mc_budget"])
        + _figs(F, ["mc_convergence"]) + table(T["convergence"]) + _figs(F, ["mc_acf", "mc_running", "mc_stationarity", "mc_mcse", "mc_rolling", "mc_correlation"])
        + table(T["phases"]) + _figs(F, ["rep_violations"]), "Is the chain converged, and how do observables change between phases?")
    add("lef", "Loop extrusion", _figs(F, ["lef_traj", "lef_coverage", "loop_length", "violin_loop_length", "lef_binding", "lef_dynamics"]), "Where loops form, how long they are and how they anchor.")
    if rep is not None:
        add("replication", "Replication", _figs(F, ["rep_forks", "rep_fraction", "rep_initiation", "rep_timing"]) + table(T["replication"]), "Origins, forks and replication timing.")
    if has_potts(p):
        add("compartments", "Epigenetic states and compartments", _figs(F, ["epi_traj", "epi_sorted", "epi_fractions", "epi_switching", "epi_switch_bars", "epi_switch_map",
                                                                              "epi_same_sign", "violin_same_sign", "epi_domains", "epi_order", "epi_order2"]) + (table(T["compartments"]) if "compartments" in T else ""),
            "How epigenetic colour spreads and how beads switch between A and B compartments (B→A and A→B) during S phase.")
    if ctx.has_md:
        add("structure", "3D ensemble", _viz3d(F, out_dir) + _figs(F, ["md_structure", "md_structure2", *heats, "md_dynamics"]), "Structures built by molecular dynamics from the sampled states.")
    add("reproduce", "Command and parameters", _repro(ctx, out_dir, command), "Everything needed to repeat this run.")
    main = _hero(ctx, p) + "".join(s for _, _, s in secs) + f'<footer>RepliSage v{__version__} · generated {datetime.datetime.now():%Y-%m-%d %H:%M}</footer>'
    path = os.path.join(out_dir, "report.html")
    Path(path).write_text(page(f"RepliSage report - {Path(ctx.run_dir).name}", [(i, t) for i, t, _ in secs], main, CSS, JS), encoding="utf-8")
    log.success(f"Report written: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path

def write_summary_report(ctx, p, runs, command=""):
    """Top-level report of several replicas: comparison table, cross-replica R-hat and links."""
    from ..analysis.tables import observables
    from ..plots.mc_timeplots import fig_replicas
    from ..plots.style import make_phases
    ph = make_phases(p, runs[0][0].Es.shape[0]); obs = [observables(r, p) for r, _ in runs]
    fig_replicas([r for r, _ in runs], ph, ctx.run_dir, obs); F = dict(FIGS)
    rows = [(f"replica_{k:02d}", r.seed, f"{r.acceptance_rate * 100:.1f} %", f"{o['Total energy'][ph.g1:].mean():.2f}", f"{o['Mean loop length'][ph.g1:].mean():.2f}",
             f'report: {Path(d).name}/report.html') for k, ((r, d), o) in enumerate(zip(runs, obs))]
    t = Table("replicas", "Replicas", ("Replica", "Seed", "Acceptance", "Mean energy", "Mean loop length", "Details"), rows, "Each replica has its own full report in its folder.")
    secs = [("about", "The model", section("about", "The model", _about())),
            ("replicas", "Replicas", section("replicas", "Replicas", table(t) + figure("mc_replicas", F["mc_replicas"]) if "mc_replicas" in F else table(t), "Independent chains with different seeds.")),
            ("reproduce", "Command and parameters", section("reproduce", "Command and parameters", _repro(ctx, ctx.run_dir, command)))]
    main = _hero(ctx, p, [chip(f"{len(runs)} replicas", "on")]) + "".join(s for _, _, s in secs)
    path = os.path.join(ctx.run_dir, "report.html")
    Path(path).write_text(page("RepliSage report", [(i, t_) for i, t_, _ in secs], main, CSS, JS), encoding="utf-8")
    log.success(f"Summary report written: {path}")
    return path
