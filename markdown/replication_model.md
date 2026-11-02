# Replication model

When replication timing data are given, RepliSage first simulates S phase on its own: which origins fire, when, and how forks travel along the DNA. The result is a matrix that tells the other models which beads are already copied at every moment.

No timing data means no replication simulation, no forks and only one DNA chain in the model.

## Input data

Use `--rt-file` with one of:

| Format | Meaning | Notes |
|---|---|---|
| `.bigWig`, `.bw` | Bulk Repli-Seq signal (for example ENCODE) | Averaged in bins, log-transformed, smoothed with a Gaussian filter |
| `.txt` | Tab-separated table, value in the third column | One value per genomic window |
| `.parquet` | Single-cell replication timing matrix (columns starting with `SC_`) | The shipped file is `src/data/replication_timing_data.parquet` |
| `builtin` | The shipped single-cell data | Covers the chr14 test region; use `--rt-file builtin` |

In every case the signal is rescaled to $f_i \in [0, 1]$ along the $N$ beads, where $f_i$ close to 1 means "replicates early" and close to 0 means "replicates late".

## From timing to initiation

Each bead $i$ is a potential origin. It fires on average at time

$$\mu_i = T (1 - f_i)$$

where $T$ is the number of simulated time points (`REP_SIM_T`) and $f_i$ the timing signal. Early beads have a small $\mu_i$ and fire first. The firing time is spread around this mean with standard deviation

$$\sigma = \lambda T$$

where $\lambda$ is `REP_T_STD_FACTOR`.

A larger spread makes timing noisier from cell to cell. The probability density of firing at time $t$ is a Gaussian $g_i(t)$ with mean $\mu_i$ and width $\sigma$. Because a bead can only fire if it has not been copied yet, RepliSage converts the density into a discrete hazard (the probability to fire at step $t$ given that it has not fired before):

$$g_i(t) = \frac{1}{\sigma\sqrt{2\pi}} \, e^{-(t-\mu_i)^2 / (2\sigma^2)}, \qquad p_i(t) = \frac{g_i(t)}{1 - p_i(t - 1)}$$

This matrix $p_i(t)$ is the initiation rate shown in the report. It is multiplied by `REP_INIT_RATE_SCALE` before the simulation.

## Forks

At every time step each non-replicated bead fires with probability $p_i(t)$. A fired origin launches two forks, one moving left and one moving right. Each fork gets its own speed drawn from a normal distribution with mean set by `REP_SPEED_SCALE`, and moves that many beads per time step. A bead passed by a fork becomes replicated. When a left fork meets a right fork coming from a neighbouring origin, both stop and the stretch between them is marked replicated. The simulation runs until every bead is copied.

## What the model hands over

| Output | Shape | Used by |
|---|---|---|
| Replicated-DNA matrix | beads by time | Monte Carlo barrier and push, 3D sister tie |
| Left and right fork matrices | beads by time | Figures |
| Initiation rate $p_i(t)$ | beads by time | Figures |
| Potts field $h_i = 2 f_i - 1$ | per bead | Monte Carlo spins |
| Initial spin sign | per bead | Monte Carlo spins |

The matrix is stretched to `REP_TIME_DURATION` Monte Carlo steps and placed at `REP_START_TIME`, so S phase happens between steps `REP_START_TIME` and `REP_START_TIME + REP_TIME_DURATION`.

## Replication stress

`--with-stress` replaces three parameters at once with stress-like values:

| Parameter | Normal | With stress |
|---|---|---|
| `REP_T_STD_FACTOR` | 0.1 | 0.2 |
| `REP_SPEED_SCALE` | 10 | 1.0 |
| `REP_INIT_RATE_SCALE` | 1.0 | 2.0 |

Forks are slower, more origins fire and timing is noisier, which is what cells under replication stress show. Set `REP_WITH_STRESS = True` in the config file for the same effect.

## Parameters

| Name | Meaning | Default |
|---|---|---|
| `REP_SIM_T` | Number of time points of the replication simulation | 1000 |
| `REP_T_STD_FACTOR` | Width of the firing-time distribution as a fraction of $T$ | 0.1 |
| `REP_SPEED_SCALE` | Fork speed scale | 10 |
| `REP_INIT_RATE_SCALE` | Multiplier of the initiation rate | 1.0 |
| `REP_WITH_STRESS` | Use the stress values above | False |
| `REP_START_TIME` (`--rep-start`) | Monte Carlo step where S phase starts | 15000 |
| `REP_TIME_DURATION` (`--rep-duration`) | Length of S phase in Monte Carlo steps | 20000 |
| `BIGWIG_BIN_SIZE` | Bin size for Repli-Seq bigWig (bp) | 1000 |
| `BIGWIG_SIGMA` | Gaussian smoothing in bins | 200 |
| `BIGWIG_LOG_NORM` | Log-transform the bigWig signal | True |

`REP_TIME_DURATION` must be larger than `REP_SIM_T`.

## What to expect in the report

The "Replication" section shows fork trajectories, the replicated fraction over time (an S-shaped curve), the initiation rate map with inferred origins, and a comparison of the simulated timing with the input timing.

## In the code

- [`src/replication_simulation/replikator.py`](https://github.com/SFGLab/RepliSage/blob/main/src/replication_simulation/replikator.py): `Replikator`: loads timing data and turns it into initiation rates
- [`src/replication_simulation/data_sources.py`](https://github.com/SFGLab/RepliSage/blob/main/src/replication_simulation/data_sources.py): `SourceMixin`: bigWig, text, parquet (single cell) and built-in timing, converted to the signal $f(x)$
- [`src/replication_simulation/timing_stats.py`](https://github.com/SFGLab/RepliSage/blob/main/src/replication_simulation/timing_stats.py): `StatsMixin`: extrema and slopes of the timing curve
- [`src/replication_simulation/init_rate.py`](https://github.com/SFGLab/RepliSage/blob/main/src/replication_simulation/init_rate.py): `InitRateMixin`: initiation rate $I(x,t)$
- [`src/replication_simulation/simulator_class.py`](https://github.com/SFGLab/RepliSage/blob/main/src/replication_simulation/simulator_class.py): `ReplicationSimulator`: runs the stochastic simulation
- [`src/replication_simulation/fork_simulator.py`](https://github.com/SFGLab/RepliSage/blob/main/src/replication_simulation/fork_simulator.py): `simulator`: origin firing and fork loop, with a progress bar
- [`src/replication_simulation/fork_propagation.py`](https://github.com/SFGLab/RepliSage/blob/main/src/replication_simulation/fork_propagation.py): `propagate_forks`: moves forks and deletes the ones that collide
- [`src/replication_simulation/validation.py`](https://github.com/SFGLab/RepliSage/blob/main/src/replication_simulation/validation.py): `run_Ntrials`, `correlate_with_experiment`: average over trials and compare with data
- [`src/pipeline/replication_stage.py`](https://github.com/SFGLab/RepliSage/blob/main/src/pipeline/replication_stage.py): `run_replication(ctx)`: stage entry point, returns `ReplicationData`
- [`src/MC/replication_matrix.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MC/replication_matrix.py): `to_time_major`: converts the replicated-fraction matrix for the MC

| Function or class | What it does |
|---|---|
| `Replikator(rept_data_path, sim_L, sim_T, chrom, coords, Tstd_factor, speed_factor, sc, out_path, bw_bin_size, bw_sigma, bw_log_norm)` | `rept_data_path` timing file, `sim_L` beads, `sim_T` time points, `coords` region, `Tstd_factor` width of firing times, `speed_factor` fork speed, `sc` single-cell data, `bw_*` bigWig smoothing and log transform |
| `ReplicationSimulator(L, T, initiation_rate, speed_ratio, speed_mean)` | holds the parameters; `.run_simulator()` returns the replicated DNA, left and right fork matrices |
| `simulator(initiation_rate, speed_mean, speed_std)` | Bernoulli firing of origins at each time step, then fork propagation until all DNA is replicated |
| `get_p_vector(T, mu, sig)` | Gaussian firing probability over time for one locus (mean `mu`, SD `sig`) |

| Argument | Role |
|---|---|
| `--rt-file` | timing data file, or `builtin` |
| `BIGWIG_LOG_NORM`, `BIGWIG_BIN_SIZE`, `BIGWIG_SIGMA` | bigWig preprocessing (log transform, bin size, Gaussian smoothing) |
| `REP_T_STD_FACTOR`, `REP_SPEED_SCALE`, `REP_INIT_RATE_SCALE` | width of firing times, fork speed and initiation rate scales |
| `REP_START_TIME`, `REP_TIME_DURATION` | S phase start and length in MC steps |
| `--with-stress` | turns on replication stress (`REP_WITH_STRESS`, applied by `apply_stress` in `config/stress.py`) |
