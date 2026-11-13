# Replication model

When replication timing data are given, RepliSage first simulates S phase on its own: which origins fire, when, and how forks travel along the DNA. The result is a matrix that tells the other models which beads are already copied at every moment.

No timing data means no replication simulation, no forks and only one DNA chain in the model.

## Input data

Give timing with `--rt-file` (bigWig, txt, single-cell parquet, `builtin` or `none`); the formats are listed in [Input data](input_data.md#replication-timing). In every case the signal is rescaled to $f_i \in [0, 1]$ along the $N$ beads, where $f_i$ close to 1 means "replicates early" and close to 0 means "replicates late".

## From timing to initiation

Averaging single-cell timing over cells gives the replication fraction $f(x,t)$, a pseudo-bulk timing curve with mean replication time $\mu(x)$ at position $x$ (bead $i$ in the code). The **initiation rate** $I(x,t)$ is the probability per unit time that an origin at $x$ fires at time $t$. It is assumed Gaussian in time around $\mu(x)$:

$$I(x,t) \propto \exp\left(-\frac{(t - \mu(x))^2}{2\sigma_I^2}\right)$$

with the same $\sigma_I$ everywhere, proportional to the total replication time $T_r$. In the code, $\mu_i = T(1 - f_i)$ with $T$ = `REP_SIM_T`, and $\sigma_I = \lambda T$ with $\lambda$ = `REP_T_STD_FACTOR`. Early beads fire first, and a larger $\sigma_I$ makes timing noisier from cell to cell.

Because a bead can only fire while it is not yet copied, the Gaussian $g_i(t)$ is turned into a discrete firing probability per step, renormalised as in the paper so that the firing time distribution matches $I(x,t)$:

$$p_i(t) = \frac{g_i(t)}{1 - p_i(t-1)}$$

This matrix is multiplied by `REP_INIT_RATE_SCALE` and shown in the report.

## Forks (Replikator)

At each step every uncopied bead draws a random number and fires if it is below its initiation rate. Not all origins fire: the others are copied passively by forks from neighbours. A fired origin launches two forks (left and right) with a speed $v$ drawn from a normal distribution,

$$p(v) \propto \exp\left(-\frac{(v - \mu_v)^2}{2\sigma_v^2}\right)$$

where $\mu_v, \sigma_v$ are estimated from the slopes between consecutive extrema of the timing curve $f$ (`REP_SPEED_SCALE` sets the scale). Each fork then moves $r = v \cdot (t - t_0)$ beads from its origin, fired at $t_0$. Forks stop when they meet, and the run ends when everything is copied. Speed is constant in time, a simplification (see [limitations](limitations.md)). The check of the model is that the average of many runs reproduces the input timing curve.

## What the model hands over

| Output | Shape | Used by |
|---|---|---|
| Replicated-DNA matrix | beads by time | Monte Carlo barrier and push, 3D sister tie |
| Left and right fork matrices | beads by time | Figures |
| Initiation rate $p_i(t)$ | beads by time | Figures |
| Epigenetic field $h_i = 2 f_i - 1$ | per bead | Monte Carlo epigenetic states |
| Initial epigenetic state sign | per bead | Monte Carlo epigenetic states |

The time axis is **normalised** inside the simulator, before the matrices are rescaled to `REP_SIM_T` points: the idle steps before the first fired origin are removed, and the simulation stops at the step where the last locus is copied. So the first column of the matrix is the first firing (with its forks, which are aligned to the firing step) and the last column is the moment when everything is replicated. The initiation rate $I(x,t)$ used in the plots is mapped to the same axis, and `Replikator.normalise_time` stays as a safeguard that does nothing for a profile that is already normalised. The log reports how many idle steps were removed (with `-vv`). This only rescales time; the model itself is unchanged, and `demo/test_normalisation.py` checks it (no Monte Carlo needed). The matrix is then stretched to `REP_TIME_DURATION` Monte Carlo steps and placed at `REP_START_TIME`, so S phase happens between steps `REP_START_TIME` and `REP_START_TIME + REP_TIME_DURATION`.

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
| `REP_START_TIME` (`--rep-start`) | Monte Carlo step where S phase starts | 50000 |
| `REP_TIME_DURATION` (`--rep-duration`) | Length of S phase in Monte Carlo steps | 50000 |
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

## Robustness checks

The model is unchanged; these are implementation safeguards.

* **Input checks.** A missing chromosome, a region outside the timing data, a constant or empty timing signal, or fewer than four data points stop the run with a message that names the problem.
* **Fork speed on smooth curves.** The fork speed is estimated from the slopes between successive peaks and dips of the timing curve. If a small region or fine beads leave fewer than two extrema, the region ends are used as segment boundaries, so the speed is never zero.
* **Slow regions.** If DNA is still unreplicated after `REP_SIM_T` time steps, the initiation rate of the last time step is kept and the simulation continues until the region is replicated (the result is rescaled to `REP_SIM_T` columns as before). A hard cap of `20 * REP_SIM_T` steps raises a clear error.
* **Speed.** Fork matrices are stacked once instead of at every step. Results for a given random seed are identical to the earlier implementation whenever it finished.
