# Analysis

After the Monte Carlo (and the 3D part, if it ran) RepliSage analyses the run, draws the figures and writes everything into `report.html`. This page explains what the numbers mean, so you can judge whether a run is trustworthy.

## Observables

At every recorded sample the following are computed. Every metric is defined mathematically on the [Metrics](metrics.md) page.

| Observable | Meaning |
|---|---|
| Total energy | Sum of all energy terms |
| Folding, binding energy | The loop-size and CTCF terms |
| Mean loop length | Average of $n_k - m_k$ over bound LEFs, in beads |
| Fraction of bound LEFs | LEFs with $m_k \ge 0$ and $n_k > m_k$ |
| Coverage | Number of loops that span each bead |
| Violations | LEFs that straddle a replication fork |
| Average epigenetic state (mean epigenetic state), cluster order, Binder cumulant, entropy, synchronisation | Order measures of the epigenetic states (only with replication data) |

## MCMC diagnostics

The ideas behind these diagnostics (autocorrelation, ergodic mean, burn-in) are explained in [Monte Carlo, MCMC and Metropolis](mcmc.md).

Samples of a Markov chain are correlated, so ten thousand samples are not ten thousand independent measurements. The diagnostics quantify this.

**Autocorrelation.** For a series $x_t$ with mean $\bar{x}$, the autocorrelation at lag $\tau$ is

$$\rho(\tau) = \frac{C(\tau)}{C(0)}$$

$$C(\tau) = \big\langle (x_t - \bar{x})(x_{t+\tau} - \bar{x}) \big\rangle$$

where $C(\tau)$ is the autocovariance and $\langle\cdot\rangle$ the average over $t$.

It starts at 1 and should fall to 0 within a few lags.

In G1 and G2/M it should fall quickly. In S phase expect a slow decay: the forks keep changing loops and epigenetic states in a semi-deterministic way, so this is a feature of the replication program and not a failed run (see [what to expect in each phase](mcmc.md#what-to-expect-in-each-phase)).

**Integrated autocorrelation time.**

$$T_{\rho} = 1 + 2 \sum_{\tau = 1}^{\tau_{\max}} \rho(\tau)$$

where $\tau_{\max}$ is the lag at which the sum is cut,

chosen with Geyer's rule (stop at the first pair of lags whose sum is negative). $T_{\rho}$ is the autocorrelation time: the number of samples that count as one independent sample.

**Effective sample size.** With $n$ samples,

$$n_{\mathrm{eff}} = \frac{n}{T_{\rho}}$$

where $n_{\mathrm{eff}}$ is the effective sample size (ESS).

Aim for at least a few hundred. The Monte Carlo standard error of the mean is

$$\mathrm{MCSE} = \frac{s}{\sqrt{n_{\mathrm{eff}}}}$$

where $s$ is the standard deviation of the series.

**Geweke z-score.** Compares the mean of the first 10 percent of the series with the mean of the last 50 percent, each corrected by its own ESS. If the chain is stationary, $|z|$ stays below about 2. A large value means the series still drifts.

**Split R-hat.** Splits every chain into two halves and compares the variance between halves with the variance inside them. Values close to 1 (below 1.05) mean that the halves, and different replicas, agree. With one replica the two halves of the chain are compared.

**Which part is tested.** With replication, G1 after burn-in is used because it is the only stationary part (S phase changes the system on purpose). Without replication, everything after burn-in is used.

## Phase comparison

For each observable and phase (G1, S, G2/M) the report shows the mean and the standard deviation, and the effect size of S relative to G1 (Cohen's $d$, see [Metrics](metrics.md)), so you can see whether replication changes the loops.

## Families of loops

Loops can sit inside each other, as in a TAD with sub-TADs. A loop $B$ is **inside** loop $A$ when the left leg of $A$ is at or left of the left leg of $B$ and the right leg of $A$ is at or right of the right leg of $B$ (identical loops count once). The **nesting order** of a loop is the number of loops that contain it. Loops of order 0 are **parents** if they hold other loops and **orphans** if they are isolated; a loop of order $k$ is a **child of order $k$**. The report counts the loops of every family at every sample, and compares G1, S and G2/M: the share of each family, the mean and deepest order, the number of loops inside a parent, and the ratio of the length of a child to the loop that encloses it. More nesting means a more hierarchical fold, and a change from G1 to S shows how replication reorganises it.

## Energy budget and free-energy landscapes

The Monte Carlo records every term of its energy at every sample, recomputed from the state (not accumulated during the run):

* **folding**, $\mathcal{H}_{\text{fold}}$: favours long loops;
* **binding**, $\mathcal{H}_{\text{bind}}$: LEF legs on CTCF sites;
* **epigenetic**, the Potts field and interaction (when replication is modelled);
* **crossing** and **replication fork**: hard walls (scaled by $10^{8}$ times their coefficient). They are listed as the number of penalty units, not as energy.

The report gives the mean ± SD of each soft term in G1, S and G2/M and the change relative to G1, so you can see which term replication perturbs. The **free-energy landscape** along an order parameter $x$ is

$$F(x) = -T \ln P(x),$$

where $P(x)$ is the histogram of $x$ over the samples of a phase and $T$ the Monte Carlo temperature (`T_MC`); $F$ is shifted so its minimum is 0. The order parameters are the mean loop length and, with replication, the mean epigenetic state. A shifted minimum between phases is a change of the preferred fold. Samples are correlated, so small differences are not significant.

## Hi-C validation

With `--hic file.hic` the model is compared with a Hi-C map on the same bead grid. The simulated map is built from the loop contacts of all post-burn-in samples (a population of cells in all phases) plus a polymer background, multiplied by the epigenetic checkerboard when it is on; with a 3D run the mean inverse-distance map of the ensemble is compared as well. The metrics are the Pearson $r$ of log contacts, of observed/expected contacts (each diagonal divided by its mean), the stratified correlation (SCC, mean $r$ per diagonal), $r$ of the decay $P(s)$ of contact with distance, $r$ of the insulation score (domain boundaries) and the F1 of the boundaries found in both maps, $r$ of the correlation matrices, and $r$ of the A/B eigenvector with the mean epigenetic state. The first three are also given for each phase. The comparison needs `hic-straw` (see [Pre-processing](preproc.md)); no optimisation against Hi-C is done.

## Compartment switching

Beads are classified as A (mean epigenetic state above 0.25), B (below -0.25) or neutral in two windows: the first and the last tenth of S phase. The report shows

* a transition matrix between the three states,
* bar plots of the fractions of A, B and neutral at the start and the end of S,
* B to A and A to B switching counts and rates over time.

Colours are always red for A and blue for B.

## LEF-fork encounters

During S phase the kernel counts, in every recorded window, the **encounters** between LEFs and forks: a proposed LEF move met by a fork, i.e. the fork pushes a leg back (`REP_FORK_PUSH`) or the LEF straddles a fork. An accepted slide in which a fork pushed a leg is a **push**; an accepted rebinding of a straddling LEF is a **removal** (the fork dislodged it). The report gives the totals, the **pushing rate** and **removal rate** (events per encounter, so independent of the number of sweeps), events per 1000 MC steps, and the **cumulative fraction** of encounters, pushes and removals over S-phase progress (0 = first firing, 1 = last locus replicated), with counts and rates per tenth of S phase. The counts are saved as `fork_events.npy` (samples x 3: encounters, pushed, removed).

## Replication statistics

Replicated fraction over time, the times at which given fractions of the DNA are replicated, inferred origins (local minima of the timing profile), fork speed and a comparison of the simulated timing profile with the input.

## Structural metrics (3D runs)

The formulas are on the [Metrics](metrics.md#metrics-from-the-3d-structures) page.

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

With replication the shape metrics above are computed on the first sister chromatid. In addition, the whole system (both sisters) gets its own radius of gyration and convex hull volume (`radius_of_gyration_total`, `convex_hull_volume_total`), `sister_distance` is the distance between the two sister centroids, and `phases.npy` stores the cell-cycle phase (G1, S, G2/M) of every structure.

**Contact maps by phase.** `plots/structural_metrics/mean_inv_heatmap_phases.png` puts the three averaged distance maps (G1, S, G2/M) in one row, each titled with its phase and the range of structures used. The value is the mean of $1/(d+1)$ over the structures of the phase, where $d$ is the distance between two beads (high = close). All three maps share one colour scale (the `turbo` colour map with a square-root stretch, clipped at the 2nd and 99.7th percentiles) so that differences between phases are real and weak patterns stay visible. The maps are also saved as `mean_inv_heatmap_<first>_<last>.npy`. Without replication there is a single map.

**Compaction figure.** `plots/structural_metrics/compaction.png` shows, with G1, S and G2/M shaded, the radius of gyration (root-mean-square distance of the beads from their centre of mass), the convex hull volume (volume of the smallest convex shape enclosing the beads), the contacts and the sister distance of every structure (top), and the mean of each phase relative to G1 (bottom). The expectation is decompaction in S, when replicated chromatin unfolds and the sisters form, and further compaction in G2/M. Lengths are in nm and volumes in nm³ of the model. The first G1 structures are still relaxing from the initial conformation, so a short run overstates the G1 size; longer runs make this negligible.

## Where the numbers are

| Location | Content |
|---|---|
| `metadata/statistics/*.csv` | `overview`, `convergence`, `phases`, `lef`, `compartments`, `replication` tables, also shown in the report |
| `metadata/MCMC_output/*.npy` | Raw arrays: LEF legs `Ms`, `Ns`, epigenetic states, energies, violations |
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
- [`src/analysis/nesting.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/nesting.py): families of loops
- [`src/analysis/free_energy.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/free_energy.py): energy budget and landscapes
- [`src/analysis/hic_compare.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/hic_compare.py): Hi-C validation
- [`src/analysis/compartments.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/compartments.py): compartment switching
- [`src/analysis/rep_stats.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/rep_stats.py): replication timing summaries
- [`src/analysis/folding_stats.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/folding_stats.py): compaction score
- [`src/analysis/potts_order.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/potts_order.py): epigenetic order measures
- [`src/analysis/potts_metrics.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/potts_metrics.py): `compute_potts_metrics`: per-sample order and synchronisation
- [`src/analysis/synchronization.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/synchronization.py): graph synchronisation
- [`src/analysis/structure_metrics.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/structure_metrics.py): 3D shape metrics
- [`src/analysis/ensemble_metrics.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/ensemble_metrics.py): `compute_metrics_for_ensemble`
- [`src/analysis/heatmaps.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/heatmaps.py): averaged contact maps of the ensemble
- [`src/pipeline/analysis_stage.py`](https://github.com/SFGLab/RepliSage/blob/main/src/pipeline/analysis_stage.py): `run_analysis(ctx, p, res, out_dir, rep, command)` ties everything together

| Function or class | What it does |
|---|---|
| `autocorr(x)` | normalised autocorrelation $\rho(\tau)$ by FFT |
| `iat(x)` | integrated autocorrelation time $T_{\rho}$ (Geyer positive sequence) |
| `ess(x)` | effective sample size $n/T_{\rho}$ |
| `geweke(x, a, b)` | Geweke $z$ comparing the first fraction `a` with the last fraction `b` |
| `split_rhat(chains)` | split $\hat R$ |
| `diagnose(series, seg)` | all diagnostics on slice `seg` for every named series |
| `diag_segment(ph)` | the stationary part used for diagnostics |
| `observables(res, p)` | named per-sample series (energies, loop length, bound LEFs, average epigenetic state) |
| `convergence_table(obs, ph)  /  phase_table(obs, ph)` | convergence and per-phase tables (mean ± SD, Cohen $d$) |
| `lef_series(res, p)  /  coverage(Ms, Ns, n_beads)` | LEF observables and loop coverage per bead |
| `switch_rates(spins, lag)  /  transition_matrix(a, b)` | compartment switching rates and counts |
| `timing_profile(rep_frac)  /  rep_summary(...)` | replication time per bead and its summary |
| `radius_of_gyration(V)  /  asphericity(V)  /  ...` | 3D shape metrics of a structure `V` (beads by 3) |

The tables are written as CSV by `save_csv(tables, folder)`; the arguments `--n-replicas` and `--seed` decide the chains that `diagnose` compares.
