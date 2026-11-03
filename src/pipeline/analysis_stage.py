"""Stage: statistics tables, figures, structural metrics, averaged heatmaps and the HTML report of one replica."""
import numpy as np
from ..plots.style import FIGS, make_phases, apply_style, set_units
from ..analysis.tables import (observables, overview_table, convergence_table, phase_table, lef_table, compartment_table,
                               rep_table, has_potts, save_csv)
from ..analysis.rep_stats import rep_summary
from ..analysis.ensemble_metrics import compute_metrics_for_ensemble
from ..analysis.heatmaps import get_avg_heatmap
from .figures_stage import make_figures
from ..report.builder import write_report
from ..utils.files import save_parameters
from ..logger import log

def statistics(ctx, p, res, rep, ph):
    """Tables (also saved as csv) and the observable series."""
    obs = observables(res, p); conv, diag = convergence_table(obs, ph)
    kb = (ctx.region[1] - ctx.region[0]) / ctx.n_beads / 1e3
    tables = [overview_table(res, p, ctx.n_ctcf, kb), conv, phase_table(obs, ph), lef_table(res, p, ph, ctx.n_ctcf, kb)]
    if rep is not None:
        tables.append(rep_table(rep_summary(rep.rep_frac, rep.input_timing, rep.speed, kb, p.rep_duration)))
        if has_potts(p): tables.append(compartment_table(res, ph))
    return obs, tables, diag

def heatmaps(ctx, out_dir, rep):
    """Averaged heatmaps of the 3D ensemble (per phase if replication was modelled)."""
    a, n_ch = ctx.args, 2 if rep is not None else 1
    n_all = (a.N_STEPS - a.BURNIN) // a.MC_STEP + 1
    if rep is None:
        get_avg_heatmap(out_dir, 1, n_all, n_chains=n_ch); return
    b1 = (a.REP_START_TIME - a.BURNIN) // a.MC_STEP + 1
    b2 = (a.REP_START_TIME + a.REP_TIME_DURATION - a.BURNIN) // a.MC_STEP + 1
    rng = ((1, b1), (b1, b2), (b2, n_all)); h = [get_avg_heatmap(out_dir, x, y, n_chains=n_ch, plot=False) for x, y in rng]
    if all(m is not None for m in h):
        try:
            from ..plots.structure_figs import fig_heatmaps_phases
            fig_heatmaps_phases(h, ["G1", "S", "G2/M"], rng, out_dir)   # the three phase maps in one row
        except Exception as e: log.alert(f"Phase heatmap figure failed: {e}")
        np.save(f"{out_dir}/metadata/structural_metrics/avg_heatmap.npy", sum(h) / 3.0)
        log.success("Saved average heatmap (over all phases) as avg_heatmap.npy")

def run_analysis(ctx, p, res, out_dir, rep, command=""):
    """Everything derived from one MC replica; returns the report path (or None)."""
    a = ctx.args
    log.step(f"Analysis, figures and report -> {out_dir}")
    apply_style(a.FIG_DPI); FIGS.clear(); set_units(ctx)
    if a.SAVE_MDT: save_parameters(out_dir + '/metadata/params.txt', **a.as_dict())
    ph = make_phases(p, res.Es.shape[0]); obs, tables, diag = statistics(ctx, p, res, rep, ph)
    for t in tables: log.table(t.title, t.rows, t.header)
    save_csv(tables, out_dir + "/metadata/statistics")
    from ..analysis.mc_stats import diag_segment
    where = diag_segment(ph)[1]
    if a.SAVE_PLOTS:
        make_figures(ctx, p, res, out_dir, rep, ph, obs, diag, where)
        if ctx.has_md:
            with log.timer("Structural metrics"): compute_metrics_for_ensemble(out_dir + '/ensemble', duplicated_chain=rep is not None, path=out_dir)
    if a.VIZ_HEATS and ctx.has_md:
        with log.timer("Averaged heatmaps"): heatmaps(ctx, out_dir, rep)
    return write_report(ctx, p, res, out_dir, rep, ph, obs, tables, command, diag, where) if a.REPORT else None
