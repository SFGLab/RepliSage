"""Canonical list of RepliSage parameters (name, help, type, default, section)."""
from pathlib import Path
from .arg_types import Arg, Args

PKG = Path(__file__).resolve().parents[1]
DEFAULT_REPT = str(PKG / "data" / "replication_timing_data.parquet")
DEFAULT_FF = str(PKG / "forcefields" / "classic_sm_ff.xml")

# sections: general | input | preprocessing | replication | mc | md | output
def A(sec, name, help, typ, default=""):
    return Arg(sec, name, help, typ, default if default != "" else "")

def build_args() -> Args:
    return Args([
        # ---------------- general ----------------
        A("general", "PLATFORM", "OpenMM platform: auto (default) tries CUDA, then OpenCL, then CPU and falls back on any error; or force CUDA, OpenCL or CPU.", str, "auto"),
        A("general", "DEVICE", "Device index for CUDA or OpenCL (count from 0).", str),
        A("general", "OUT_PATH", "Root folder of results. Each run gets its own sub-folder inside.", str, "results"),
        A("general", "RUN_NAME", "[new] Exact name of the run sub-folder. Empty = automatic name (chrom + region + time).", str),
        A("general", "RUN_TAG", "[new] Free label appended to the automatic run name.", str),
        A("general", "N_THREADS", "[new] Number of CPU threads used to run independent MC replicas in parallel.", int, "1"),
        A("general", "SEED", "[new] Random seed of the Monte Carlo (empty = random).", int),
        A("general", "VERBOSE", "[new] Verbosity 0-3 (same as -v, -vv, -vvv): 0 stages, progress bars and results; 1 + input / output tables; 2 + detailed info; 3 everything.", int, "0"),
        # ---------------- input ----------------
        A("input", "BEDPE_PATH", "A .bedpe file path with loops (CTCF anchors). It is required: without it the model does not know where the CTCF barriers are. Optional columns 8-9 hold CTCF orientation.", str),
        A("input", "REPT_PATH", "The replication timing dataset (.txt, .bw, .bigwig). If empty, no replication is modelled (unless USE_SC_REPT).", str),
        A("input", "SC_REPT_PATH", "The single cell replication timing dataset (.parquet).", str, DEFAULT_REPT),
        A("input", "USE_SC_REPT", "[new] Use the built-in single-cell replication timing when REPT_PATH is empty.", bool, "False"),
        A("input", "ASK_RT", "[new] Ask interactively what to do when no replication timing is given (only in a terminal).", bool, "True"),
        A("input", "CHROM", "Chromosome of the modelling region of interest.", str),
        A("input", "REGION_START", "Starting region coordinate. Empty = whole chromosome.", int),
        A("input", "REGION_END", "Ending region coordinate. Empty = whole chromosome.", int),
        A("input", "N_BEADS", "Number of Simulation Beads. Empty = region length / 7 kb (about 10 beads per CTCF site; max 10,000 beads).", int),
        # ---------------- preprocessing ----------------
        A("preprocessing", "REP_SIM_T", "[new] Number of time points of the replication simulation.", int, "1000"),
        A("preprocessing", "BIGWIG_BIN_SIZE", "[new] Bin size (bp) used to average a Repli-Seq bigWig.", int, "1000"),
        A("preprocessing", "BIGWIG_SIGMA", "[new] Gaussian smoothing (in bins) of a Repli-Seq bigWig.", float, "200.0"),
        A("preprocessing", "BIGWIG_LOG_NORM", "[deprecated] Repli-Seq signals are always log-normalised; this value is ignored.", bool, "True"),
        # ---------------- replication ----------------
        A("replication", "REP_WITH_STRESS", "True sets REP_T_STD_FACTOR, REP_SPEED_SCALE and REP_INIT_RATE_SCALE to replication-stress values.", bool, "False"),
        A("replication", "REP_T_STD_FACTOR", "Factor multiplying the standard deviation of the replication timing curve.", float, "0.1"),
        A("replication", "REP_SPEED_SCALE", "A scale that quantifies the speed of the replication forks.", float, "10"),
        A("replication", "REP_INIT_RATE_SCALE", "Multiplier of the experimentally estimated initiation rate.", float, "1.0"),
        A("replication", "REP_START_TIME", "MC step at which replication starts (G1 = the steps before it).", int, "50000"),
        A("replication", "REP_TIME_DURATION", "Duration of replication (in MC steps); G2/M = the steps after S phase.", int, "50000"),
        # ---------------- mc ----------------
        A("mc", "LEF_RW", "Random walk for cohesins instead of unidirectional sliding.", bool, "True"),
        A("mc", "COHESIN_BLOCKS_CONDENSIN", "Allow cohesins to block condensins during simulation.", bool, "False"),
        A("mc", "RANDOM_INIT_SPINS", "Use a random initial distribution of epigenetic states.", bool, "True"),
        A("mc", "P_REW", "Probability of proposing a rewiring move (LEF move).", float, "0.5"),
        A("mc", "REP_FORK_PUSH", "Hard-coded replication fork pushing in the MC moves.", bool, "True"),
        A("mc", "REP_FORK_EPIGENETIC_ORGANIZER", "Model replication forks as epigenetic organizers.", bool, "True"),
        A("mc", "PREFERENTIAL_BINDING", "[new] LEFs load preferentially where loops span (CTCF occupancy), and rarely in loop-naked areas. Also --preferential-binding.", bool, "False"),
        A("mc", "PREF_BIND_FLOOR", "[new] Loading weight of loop-naked beads relative to the most occupied bead (0-1) when PREFERENTIAL_BINDING is on.", float, "0.05"),
        A("mc", "PREF_BIND_POWER", "[new] Exponent applied to the occupancy before loading (1 = proportional, >1 = stronger preference).", float, "1.0"),
        A("mc", "N_STEPS", "Total number of Monte Carlo steps (default: G1 50000 + S 50000 + G2/M 50000).", int, "150000"),
        A("mc", "N_SWEEP", "Number of node or link state changes proposed per simulation step.", int, "1000"),
        A("mc", "N_LEF", "Number of loop extrusion factors (LEFs). Empty = half the number of CTCF sites (changed from twice in earlier versions).", int),
        A("mc", "N_LEF2", "Number of second-family LEFs (different speed).", int, "0"),
        A("mc", "MC_STEP", "Sampling frequency of MC steps (avoids autocorrelation).", int, "200"),
        A("mc", "BURNIN", "Burn-in period (steps before equilibrium).", int, "5000"),
        A("mc", "T_MC", "Temperature of the stochastic MCMC model.", float, "1.5"),
        A("mc", "FOLDING_COEFF", "Folding coefficient for the first family of LEFs.", float, "1.0"),
        A("mc", "FOLDING_COEFF2", "Folding coefficient for the second family of LEFs.", float, "0.0"),
        A("mc", "REP_COEFF", "Replication-barrier penalty (energy scale = coeff * 1e8; was 1.0 = 1e8 in the published v1.0).", float, "0.01"),
        A("mc", "EPI_INTERACT_COEFF", "Interaction coefficient of the epigenetic state model.", float, "1.0"),
        A("mc", "EPI_FIELD_COEFF", "Field coefficient of the epigenetic state model.", float, "1.0"),
        A("mc", "CROSS_COEFF", "Coefficient for LEF crossing.", float, "1.0"),
        A("mc", "BIND_COEFF", "Coefficient for CTCF binding.", float, "1.0"),
        A("mc", "N_REPLICAS", "[new] Independent MC chains; each is processed into its own replica_XX sub-folder.", int, "1"),
        # ---------------- md ----------------
        A("md", "INITIAL_STRUCTURE_TYPE", "rw, confined_rw, self_avoiding_rw, helix, circle, spiral, sphere, hilbert.", str, "rw"),
        A("md", "SIMULATION_TYPE", "3D engine: MD (one minimization + dynamics, the default: gives the replication trajectory) or EM (many minimizations). none/empty = stop after the Monte Carlo.", str, "MD"),
        A("md", "DCD_REPORTER", "Save a .dcd video of the 3D structure.", bool, "False"),
        A("md", "INTEGRATOR_TYPE", "langevin or brownian.", str, "langevin"),
        A("md", "INTEGRATOR_STEP", "Integrator time step (e.g. '100 femtosecond').", str, "100 femtosecond"),
        A("md", "FORCEFIELD_PATH", "Path to XML file with the forcefield.", str, DEFAULT_FF),
        A("md", "EV_P", "Probability that excluded volume of a bead is randomly disabled.", float, "0.01"),
        A("md", "TOLERANCE", "Stopping tolerance of the energy minimization.", float, "1.0"),
        A("md", "SIM_TEMP", "Temperature of the 3D simulation (e.g. '310 kelvin').", str, "310 kelvin"),
        A("md", "SIM_STEP", "MD steps performed each time the loop forces change.", int, "1000"),
        # ---------------- output ----------------
        A("output", "SAVE_PLOTS", "Save diagnostic plots.", bool, "True"),
        A("output", "SAVE_MDT", "Save metadata of the stochastic simulation.", bool, "True"),
        A("output", "REPORT", "[new] Write report.html (model summary, figures, tables, command and all parameters).", bool, "True"),
        A("output", "HTML_3D", "[new] Write an interactive 3D viewer (viewer_3d.html) of the ensemble.", bool, "True"),
        A("output", "SKIP_MOVIES", "[new] Skip the PyVista movies (mp4) of the ensemble. By default the movies are written (they take time: use --skip-movies to disable).", bool, "False"),
        A("output", "MOVIE_COLOR", "[new] Movie colouring: auto (sister + compartment movies), sister, compartment, rainbow or replication.", str, "auto"),
        A("output", "MOVIE_CMAP", "[new] Colour map of the contact map in the movie and the viewer: turbo (most variable), inferno, magma, viridis, cividis or rs.", str, "turbo"),
        A("output", "MOVIE_INTERP", "[new] Interpolated frames between two structures in the movie.", int, "5"),
        A("output", "FIG_DPI", "[new] Resolution (dpi) of the saved figures and of the report.", int, "200"),
        A("output", "VIZ_HEATS", "Save the averaged heatmaps of the 3D ensemble.", bool, "True"),
    ])

# old config keys -> current names
ALIASES = {"POTTS_INTERACT_COEFF": "EPI_INTERACT_COEFF", "POTTS_FIELD_COEFF": "EPI_FIELD_COEFF"}
