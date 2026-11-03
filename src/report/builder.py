"""Assemble report.html: model summary, statistics, figures with captions, command and all parameters."""
import datetime
import os
import platform
import re
import sys
from importlib import metadata
from pathlib import Path
from .theme import CSS, JS
from .text import TAGLINE, ABOUT, STAGES, EQ, READ, EPI_INTRO, glossary
from .mathml import rich
from .html_parts import esc, figure, video, table, section, chip, params_table, page, findings, group
from .findings import key_findings
from ..analysis.tables import Table, has_potts
from ..plots.style import FIGS, UNITS
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
    eq = "".join(f'<div class="eq">{rich(e)}<small>{rich(c)}</small></div>' for e, c in EQ)
    return (f'<h3>What RepliSage does</h3>' + "".join(f"<p>{rich(t)}</p>" for t in ABOUT) + f'<div class="grid2">{st}</div><h3>Energy of the model</h3>{eq}'
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

STILL_MODES = ("sister", "compartment", "rainbow", "replication")

def _viz3d(F, out_dir):
    """Links to the 3D viewer / movies / ChimeraX script, inline movies and the stills per phase."""
    d = Path(out_dir); names = [("viewer_3d.html", "interactive 3D viewer"), ("ensemble/view.cxc", "ChimeraX script")] + [(f"movie_3d_{m}.mp4", f"movie, {m} colouring (mp4)") for m in STILL_MODES]
    links = "".join(f'<a href="{n}">{t}</a>' for n, t in names if (d / n).exists())
    box = f'<div class="links">{links}</div><p class="note">These files sit next to report.html; send the whole results folder to share them. Everything else in this report is embedded in this single file.</p>' if links else ""
    movies = "".join(video(f"movie_{m}", f"movie_3d_{m}.mp4") for m in STILL_MODES if (d / f"movie_3d_{m}.mp4").exists())
    return box + movies + _figs(F, [f"md_stills_{m}" for m in STILL_MODES])

def _md_extra(F, out_dir):
    f = Path(out_dir) / "plots" / "md_simulation" / "dynamics.png"
    if f.exists(): F["md_dynamics"] = str(f)
    for m in STILL_MODES:
        f = Path(out_dir) / "plots" / "3d" / f"stills_{m}.png"
        if f.exists(): F[f"md_stills_{m}"] = str(f)
    heats = sorted((k for k in F if k.startswith("md_heat_") and k.split("_")[2].isdigit()), key=lambda k: int(k.split("_")[2])) + (["md_heat_phases"] if "md_heat_phases" in F else [])
    return heats

def write_report(ctx, p, res, out_dir, rep, ph, obs, tables, command="", diag=None, where=""):
    """Write <out_dir>/report.html and return its path."""
    F, T = dict(FIGS), {t.key: t for t in tables}; heats = _md_extra(F, out_dir); secs = []
    def add(sid, title, body, lead=""):
        if body.strip(): secs.append((sid, title, section(sid, title, body, lead)))
    try: fd = key_findings(ctx, p, res, rep, ph, obs, diag, where, tables)
    except Exception as e: log.alert(f"Key findings failed: {type(e).__name__}: {e}"); fd = []
    gl = "".join(f"<tr><td><b>{esc(t)}</b></td><td>{rich(d)}</td></tr>" for t, d in glossary(UNITS["bead_kb"]))
    terms = group("How the quantities are defined", f'<div class="tbl"><table><tbody>{gl}</tbody></table></div>', "loop length, bound LEFs, compartments, switching, replication")
    add("findings", "Key findings", (findings(fd) if fd else "") + terms + _figs(F, ["story", "phase_compare"]), "What this run says, in plain language.")
    add("input", "Input", _figs(F, ["loop_anchors", "ctcf_occupancy", "loops_input"]), "Loop anchors extracted from your BEDPE file.")
    add("lef", "Loop extrusion", _figs(F, ["lef_traj", "lef_coverage", "loop_length", "lef_binding", "model_maps"]), "Where loops form, how long they are and how they anchor.")
    if rep is not None:
        add("replication", "Replication", _figs(F, ["rt_overview"]) + table(T["replication"])
            + group("More replication plots", _figs(F, ["rep_forks", "rep_fraction", "rep_timing", "rep_initiation", "rep_violations"]), "forks, fraction, timing check, initiation rate, fork conflicts"), "Origins, forks and replication timing.")
    if has_potts(p):
        add("compartments", "Epigenetic states and compartments", f'<p class="intro">{rich(EPI_INTRO)}</p>' + _figs(F, ["epi_traj", "epi_switch_bars", "epi_switch_map", "epi_switch_traj"]) + (table(T["compartments"]) if "compartments" in T else "")
            + group("More compartment plots", _figs(F, ["epi_sorted", "epi_fractions", "epi_switching", "epi_domains", "epi_same_sign", "epi_order", "epi_order2"]), "state fractions, domains, order parameters"),
            "How epigenetic colour spreads and how beads switch between A and B compartments during S phase.")
    if ctx.has_md:
        add("structure", "3D ensemble", _viz3d(F, out_dir) + group("Structure statistics", _figs(F, ["md_compaction", "md_structure", "md_structure2", *heats, "md_dynamics"]), "size, shape, contact maps"),
            "Structures built by molecular dynamics from the sampled states.")
    diag_body = (_figs(F, ["mc_energy", "mc_energy_potts", "mc_budget", "mc_distributions", "mc_convergence"]) + table(T["convergence"]) + _figs(F, ["mc_acf", "mc_running", "mc_stationarity", "mc_mcse", "mc_rolling", "mc_correlation", "lef_dynamics"])
                 + table(T["phases"]) + "".join(table(T[k]) for k in ("overview", "lef") if k in T))
    add("mcmc", "Diagnostics", group("Monte Carlo diagnostics and tables", diag_body, "convergence, autocorrelation, energies"), "Open to check that the chain is converged and the numbers are reliable.")
    add("about", "The model", _about())
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
