# Analysis

After the Monte Carlo (and the 3D part, if it ran) RepliSage analyses the run, draws the figures and writes everything into `report.html`. This page explains what the numbers mean, so you can judge whether a run is trustworthy.

## Observables

At every recorded sample the following are computed:

| Observable | Meaning |
|---|---|
| Total energy | Sum of all energy terms |
| Folding, binding energy | The loop-size and CTCF terms |
| Mean loop length | Average of $n_k - m_k$ over bound LEFs, in beads |
| Fraction of bound LEFs | LEFs with $m_k \ge 0$ and $n_k > m_k$ |
| Coverage | Number of loops that span each bead |
| Violations | LEFs that straddle a replication fork |
| Magnetisation, cluster order, Binder cumulant, entropy, synchronisation | Order parameters of the spins (only with replication data) |

## MCMC diagnostics

Samples of a Markov chain are correlated, so ten thousand samples are not ten thousand independent measurements. The diagnostics quantify this.

**Autocorrelation.** For a series $x_t$ with mean $\bar{x}$, the autocorrelation at lag $\tau$ is

$$\rho(\tau) = \frac{\text{cov}(x_t, x_{t+\tau})}{\text{var}(x_t)}$$

It starts at 1 and should fall to 0 within a few lags.

**Integrated autocorrelation time.**

$$\tau_{\text{int}} = 1 + 2 \sum_{\tau = 1}^{\tau_{\max}} \rho(\tau)$$

summed with Geyer's rule (stop at the first pair of lags whose sum is negative). It is the number of samples that count as one independent sample.

**Effective sample size.** With $n$ samples,

$$\text{ESS} = \frac{n}{\tau_{\text{int}}}$$

Aim for at least a few hundred. The Monte Carlo standard error of the mean is $\text{sd} / \sqrt{\text{ESS}}$.

**Geweke z-score.** Compares the mean of the first 10 percent of the series with the mean of the last 50 percent, each corrected by its own ESS. If the chain is stationary, $|z|$ stays below about 2. A large value means the series still drifts.

**Split R-hat.** Splits every chain into two halves and compares the variance between halves with the variance inside them. Values close to 1 (below 1.05) mean that the halves, and different replicas, agree. With one replica the two halves of the chain are compared.

**Which part is tested.** With replication, G1 after burn-in is used because it is the only stationary part (S phase changes the system on purpose). Without replication, everything after burn-in is used.

## Phase comparison

For each observable and phase (G1, S, G2/M) the report shows mean, standard deviation and a Kruskal-Wallis test across phases, so you can see whether replication changes the loops.

## Compartment switching

Beads are classified as A (mean spin above 0.25), B (below -0.25) or neutral in two windows: the first and the last tenth of S phase. The report shows

* a transition matrix between the three states,
* bar plots of the fractions of A, B and neutral at the start and the end of S,
* B to A and A to B switching counts and rates over time.

Colours are always red for A and blue for B.

## Replication statistics

Replicated fraction over time, the times at which given fractions of the DNA are replicated, inferred origins (local minima of the timing profile), fork speed and a comparison of the simulated timing profile with the input.

## Structural metrics (3D runs)

Computed on every structure and saved in `metadata/structural_metrics/`:

| Metric | Meaning |
|---|---|
| Radius of gyration | Size of the structure |
| Mean pairwise distance, maximum pairwise distance | Compactness |
| End-to-end distance | Distance between chain ends |
| Asphericity, planarity, ellipsoid ratio and volume | Shape |
| Convex hull volume, bounding box volume | Occupied space |
| Contact number | Number of bead pairs within contact distance |
| Mean and maximum distance to the centre of mass | Radial spread |

With replication only the first chain is analysed.

## Where the numbers are

| Location | Content |
|---|---|
| `metadata/statistics/*.csv` | `overview`, `convergence`, `phases`, `lef`, `compartments`, `replication` tables, also shown in the report |
| `metadata/MCMC_output/*.npy` | Raw arrays: LEF legs `Ms`, `Ns`, spins, energies, violations |
| `metadata/structural_metrics/*.npy` | One value per structure for each metric |
| `plots/` | Every figure as `png` and `svg` |

## Reading a run in five minutes

1. Open the report and check the convergence table: ESS in the hundreds, $|z| < 2$, R-hat close to 1.
2. Look at the energy trace: after burn-in it should fluctuate around a flat line.
3. In S phase look at violations: they should stay near zero.
4. Compare G1 and G2/M loop lengths and compartments.

## In the code

- [`src/analysis/mc_stats.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/mc_stats.py): chain diagnostics
- [`src/analysis/tables.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/tables.py): tables of the report
- [`src/analysis/lef_stats.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/lef_stats.py): LEF series and coverage
- [`src/analysis/compartments.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/compartments.py): compartment switching
- [`src/analysis/rep_stats.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/rep_stats.py): replication timing summaries
- [`src/analysis/folding_stats.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/folding_stats.py): compaction score
- [`src/analysis/potts_order.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/potts_order.py): Potts order parameters
- [`src/analysis/potts_metrics.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/potts_metrics.py): `compute_potts_metrics`: per-sample order and synchronisation
- [`src/analysis/synchronization.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/synchronization.py): graph synchronisation
- [`src/analysis/structure_metrics.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/structure_metrics.py): 3D shape metrics
- [`src/analysis/ensemble_metrics.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/ensemble_metrics.py): `compute_metrics_for_ensemble`
- [`src/analysis/heatmaps.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/heatmaps.py): averaged contact maps of the ensemble
- [`src/pipeline/analysis_stage.py`](https://github.com/SFGLab/RepliSage/blob/main/src/pipeline/analysis_stage.py): `run_analysis(ctx, p, res, out_dir, rep, command)` ties everything together

| Function or class | What it does |
|---|---|
| `autocorr(x)` | normalised autocorrelation $\rho(\tau)$ by FFT |
| `iat(x)` | integrated autocorrelation time $\tau_{int}$ (Geyer positive sequence) |
| `ess(x)` | effective sample size $n/\tau_{int}$ |
| `geweke(x, a, b)` | Geweke $z$ comparing the first fraction `a` with the last fraction `b` |
| `split_rhat(chains)` | split $\hat R$ |
| `diagnose(series, seg)` | all diagnostics on slice `seg` for every named series |
| `diag_segment(ph)` | the stationary part used for diagnostics |
| `observables(res, p)` | named per-sample series (energies, loop length, bound LEFs, magnetization) |
| `convergence_table(obs, ph)  /  phase_table(obs, ph)` | convergence and per-phase tables (mean ± SD, Cohen $d$) |
| `lef_series(res, p)  /  coverage(Ms, Ns, n_beads)` | LEF observables and loop coverage per bead |
| `switch_rates(spins, lag)  /  transition_matrix(a, b)` | compartment switching rates and counts |
| `timing_profile(rep_frac)  /  rep_summary(...)` | replication time per bead and its summary |
| `radius_of_gyration(V)  /  asphericity(V)  /  ...` | 3D shape metrics of a structure `V` (beads by 3) |

The tables are written as CSV by `save_csv(tables, folder)`; the arguments `--n-replicas` and `--seed` decide the chains that `diagnose` compares.
