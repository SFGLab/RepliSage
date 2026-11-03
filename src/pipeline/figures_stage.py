"""Stage: all figures of one MC replica (the report picks them up from the figure registry)."""
from ..plots.style import make_geo
from ..plots.mc_timeplots import fig_energy, fig_energy_potts, fig_distributions, fig_acf, fig_convergence, fig_running, fig_violations
from ..plots.diag_extra import fig_stationarity, fig_mcse, fig_rolling, fig_correlation, fig_budget, fig_lef_dynamics, fig_domains
from ..plots.story import fig_story, fig_phase_compare
from ..plots.loops import fig_kymograph, plot_loop_length, fig_coverage, fig_binding, fig_input, fig_anchors
from ..plots.model_maps import fig_model_maps
from ..plots.occupancy import fig_occupancy
from ..plots.states import (ising_traj_plot, fig_switch_trajectory, fig_timing_sorted, fig_fractions, fig_switching, fig_switch_bars, fig_switch_map,
                            compute_state_proportions_sign_based)
from ..plots.rep_plots_pro import fig_forks, fig_fraction, fig_initiation, fig_timing
from ..plots.rt_overview import fig_rt_overview
from ..plots.potts_order_fig import fig_potts_order
from ..analysis.potts_metrics import compute_potts_metrics
from ..analysis.rep_stats import timing_profile
from ..analysis.tables import has_potts
from ..logger import log

def make_figures(ctx, p, res, out, rep, ph, obs, diag, where):
    """Write every figure (png + svg) of the replica; failures of single figures are reported, not fatal."""
    geo = make_geo(ctx); f_T = None if rep is None else rep.f_rep_T
    steps = [("story", lambda: fig_story(obs, res, p, rep, ph, out)), ("phase comparison", lambda: fig_phase_compare(obs, res, p, ph, out)),
             ("loop anchors", lambda: fig_anchors(ctx, p, geo, out)), ("CTCF input", lambda: fig_input(p, geo, out)),
             ("MC energies", lambda: fig_energy(obs, ph, out)), ("distributions", lambda: fig_distributions(obs, ph, out)), ("autocorrelation", lambda: fig_acf(obs, ph, out)),
             ("convergence", lambda: fig_convergence(diag, where, out)), ("running means", lambda: fig_running(obs, ph, out)),
             ("stationarity", lambda: fig_stationarity(obs, ph, out)), ("MCSE", lambda: fig_mcse(obs, ph, out)), ("rolling stats", lambda: fig_rolling(obs, ph, out)),
             ("correlations", lambda: fig_correlation(obs, ph, out)), ("energy budget", lambda: fig_budget(obs, ph, out)), ("LEF dynamics", lambda: fig_lef_dynamics(res, ph, out)),
             ("LEF kymograph", lambda: fig_kymograph(res, p, ph, f_T, out)),
             ("loop coverage", lambda: fig_coverage(res, p, ph, out, f_T)), ("loop length", lambda: plot_loop_length(res.Ns - res.Ms, ph, out)),
             ("LEF anchoring", lambda: fig_binding(res, p, ph, out)), ("model heatmaps", lambda: fig_model_maps(res, p, ph, out))]
    if getattr(ctx, "occupancy", None) is not None: steps.insert(2, ("CTCF occupancy", lambda: fig_occupancy(ctx, p, geo, out)))
    if rep is not None:
        tim = timing_profile(rep.rep_frac)
        steps += [("replication overview", lambda: fig_rt_overview(rep, res, p, ph, geo, out)), ("replication forks", lambda: fig_forks(rep, p, geo, out)), ("replication fraction", lambda: fig_fraction(rep, p, out)),
                  ("initiation rate", lambda: fig_initiation(rep, p, geo, out)), ("timing validation", lambda: fig_timing(rep, geo, out)),
                  ("fork conflicts", lambda: fig_violations(res, ph, out))]
        if has_potts(p):
            steps += [("Epigenetic energy", lambda: fig_energy_potts(obs, ph, out)), ("epigenetic trajectory", lambda: ising_traj_plot(res.spin_traj, ph, out, p, f_T)),
                      ("spins by timing", lambda: fig_timing_sorted(res.spin_traj, tim, ph, out)), ("state fractions", lambda: fig_fractions(res.spin_traj, ph, out)),
                      ("compartment switching", lambda: fig_switching(res.spin_traj, ph, out)), ("switching bars", lambda: fig_switch_bars(res.spin_traj, ph, out)),
                      ("switching map", lambda: fig_switch_map(res.spin_traj, tim, ph, geo, out)), ("switching trajectory", lambda: fig_switch_trajectory(res.spin_traj, ph, p, geo, f_T, out)),
                      ("domains", lambda: fig_domains(res.spin_traj, ph, out)),
                      ("same-sign loops", lambda: compute_state_proportions_sign_based(res.Ms, res.Ns, res.spin_traj, ph, out)),
                      ("Potts order", lambda: fig_potts_order(compute_potts_metrics(res.Ms, res.Ns, res.spin_traj, out), ph, out))]
    for name, fn in log.progress(steps, desc="Figures"):
        try: fn()
        except Exception as e: log.alert(f"Figure '{name}' failed: {type(e).__name__}: {e}")
