# Metrics

A simulation produces a lot of numbers: loop positions, epigenetic states, energies, and thousands of 3D structures. A **metric** is a single number (or one number per sample) that summarises one aspect of them, so that runs, phases and parameter values can be compared. This page defines every metric RepliSage computes, in the notation of the paper, and says whether it is inferred from the **Monte Carlo** (the graph: loops and epigenetic states) or from the **molecular dynamics** (the 3D coordinates). The results are in the report and in `metadata/`; see [Analysis](analysis.md) for how to read a run.

!!! note "Two sources of metrics"
    **Monte Carlo (MC) metrics** are computed from the sampled graph: where the LEF legs are, which state each bead has, the energies and the replication fork positions. They are cheap and exist for every sample.

    **Molecular dynamics (MD) metrics** are computed from the 3D bead coordinates of each structure, which exist only when the 3D part ran. They describe size, shape and contacts, and are the ones you can compare with microscopy and Hi-C.

## Notation

| Symbol | Meaning |
|---|---|
| $N$ | number of beads (one bead = a genomic segment, see `bead_kb` in the report) |
| $T$ | number of recorded samples (one every `MC_STEP` Monte Carlo steps) |
| $(m_k, n_k)$ | the two legs of LEF $k$; $m_k < n_k$ when bound, $m_k = n_k = -1$ when unbound |
| $\ell_k = n_k - m_k$ | length of the loop of LEF $k$, in beads ($0$ if unbound) |
| $s_i \in \{-2,-1,0,1,2\}$ | epigenetic state of bead $i$ (positive = A-like, negative = B-like) |
| $r_i$ | 3D position of bead $i$ in one structure (a vector with three coordinates) |
| $\langle \cdot \rangle$ | average (over beads, samples or structures, stated in each case) |
| $\#\{\cdot\}$ | number of items that satisfy the condition inside |

All MC quantities are computed per recorded sample $t$, so each metric is a time series. Phase values (G1, S, G2/M) are means of that series over the samples of the phase, and the report also gives the standard deviation. Because consecutive samples are correlated, this standard deviation describes fluctuations and is not an uncertainty.

## Metrics from the Monte Carlo

### Loops

**Loop length.** For one sample, the *mean loop length* is the average of $n_k - m_k$ over **all** LEFs, so an unbound LEF counts as length 0 (this is the definition used in the paper). The *bound loop length* averages only over the bound LEFs. Multiply by the bead size to get kilobases. The report also gives the median and the 95th percentile of the bound loop lengths.

**Fraction of bound LEFs.** A LEF is bound when $m_k \ge 0$ and $n_k > m_k$, so

$$f_{\text{bound}} = \frac{\#\{\text{bound LEFs}\}}{N_{\text{lef}}}$$

**Legs on CTCF sites.** The fraction of bound LEFs whose left leg sits on a CTCF site that attracts left legs, $f_L$, and whose right leg sits on a site that attracts right legs, $f_R$ (the orientation of the sites is explained in the [biology page](biology.md)). They tell how strongly the loops are anchored.

**Coverage.** The number of loops that span bead $i$, that is the number of bound LEFs with $m_k \le i < n_k$. The report gives the fraction of beads covered by at least one loop and the mean coverage (loops per bead). A very high mean coverage with short loops is the signature of **stacking**, many loops piled on the same place, which is what happens when $T_{\text{mc}}$ is too low (see [the critical region](hamiltonian.md#temperature-in-replisage)).

### Energies

The Monte Carlo energy is split into its terms, each recorded per sample (the terms are defined in the [Monte Carlo model](mc_model.md)):

| Metric | Meaning |
|---|---|
| Total energy $\mathcal{H}$ | sum of all terms |
| Folding energy | loop-size term (favours long loops) |
| Binding energy | CTCF term (favours legs on CTCF sites) |
| Epigenetic energy | state-field, neighbour and loop-anchor terms (only with replication data) |

After burn-in the energy should fluctuate around a flat line. The acceptance rate (the fraction of accepted proposals) is reported too; the report warns when it is below 10 % or above 60 %.

### Replication and loops

Loops are not allowed to straddle a replication fork. Let $f_i(t) = 1$ if bead $i$ is replicated at time $t$ and $0$ if not. A loop $(m, n)$ is a **violation** if its two legs are in different replication states ($f_m \ne f_n$), or if both legs are replicated but some bead between them is not. The number of violations is counted in S phase, and each violating loop has a strength

$$\sigma = \min(r,\ \ell - r) + 1$$

where $\ell = n - m$ is the loop length and $r$ the number of replicated beads inside it. The reported strength is the sum of $\sigma$ over the violating loops divided by (their number + 1). A good S phase has violations near zero.

### Epigenetic states and compartments

**Average epigenetic state.** The mean state over all beads, $\langle s \rangle$. It is close to 0 when A-like and B-like beads balance, and its change in S phase shows a global shift.

**State fractions.** $p_k$ is the fraction of beads in state $k$, for $k = -2, \dots, 2$.

**Compartment of a bead.** In a window of samples (the first and the last 10 % of S phase), take the mean state $\bar{s}_i$ of each bead and classify it:

| Window-mean state | Compartment |
|---|---|
| $\bar{s}_i > 0.25$ | A (active) |
| $\bar{s}_i < -0.25$ | B (inactive) |
| in between | N (neutral) |

**Compartment switching.** Comparing the two windows gives a $3 \times 3$ table of how many beads go from B, N or A to B, N or A. The report lists the number of beads that switched B to A and A to B, the **net flux** (B to A minus A to B), the fraction of unchanged beads and the A and B fractions before and after. For the time series, the *switching rate* at lag $\tau$ is the fraction of beads that have a negative state at sample $t$ and a positive state at sample $t + \tau$ (B to A), and the reverse for A to B.

### Order parameters of the epigenetic states

These describe how ordered the states are at each sample (only with replication data). With $q = 5$ states, $p_k$ the fraction of beads in state $k$ and $p_{\max}$ the largest of them:

| Metric | Definition | Meaning |
|---|---|---|
| Magnetisation | $M = \dfrac{q\,p_{\max} - 1}{q - 1}$ | 0 if the states are evenly mixed, 1 if all beads share one state |
| Cluster order | $C = p_{\max}$ | the fraction of beads in the most common state (state counts, not neighbours along the chain) |
| Binder-type cumulant | $U = 1 - \dfrac{\sum_k p_k^4}{3\,(\sum_k p_k^2)^2}$ | a measure of the shape of the state distribution (built from its second and fourth moments) |
| Entropy | $H = -\sum_k p_k \ln p_k$ | 0 if all beads share a state, $\ln q$ if the states are evenly mixed |
| Synchronisation | $\dfrac{\text{fraction of edges whose two beads have the same state}}{q - 1}$ | agreement of the states of beads joined by an edge of the graph |

For the synchronisation, the graph has one edge between every pair of consecutive beads and one edge for every bound loop, so it measures how well neighbouring beads and the two anchors of a loop agree, which is what produces compartments. The fraction is divided by $q - 1$, so with $q = 5$ the value stays between 0 and 0.25. Use it for comparisons between phases and runs, not as an absolute fraction.

### Replication simulation

These come from the replication simulation (Replikator), not from the loops. With $f_i(t)$ as above (1 = replicated):

| Metric | Definition |
|---|---|
| Replicated fraction | $F(t)$ = fraction of beads with $f_i(t) = 1$ |
| Replication time of bead $i$ | $t_i$ = first time with $f_i = 1$, divided by the S-phase length (0 to 1) |
| $T_{10}$, $T_{50}$, $T_{90}$ | the times at which $F$ first reaches 10 %, 50 % and 90 % |
| S-phase span | $T_{90} - T_{10}$, how spread out the replication is |
| Inferred origins | local minima of the profile $t_i$ |
| Median origin spacing | median distance between neighbouring origins, in kb |
| Agreement with the input | Spearman correlation $\rho$ between $t_i$ and the input timing (early beads should replicate early) |

### Convergence diagnostics

These tell whether the Monte Carlo has run long enough, and are applied to the series above (energies, loop lengths, bound fraction, average state): autocorrelation $\rho(\tau)$, integrated autocorrelation time $T_\rho$, effective sample size $n_{\text{eff}} = n / T_\rho$, standard error $\mathrm{MCSE} = s / \sqrt{n_{\text{eff}}}$, the Geweke $z$ score and the split $\hat{R}$. They are defined in [Analysis](analysis.md#mcmc-diagnostics) and explained in [Monte Carlo, MCMC and Metropolis](mcmc.md).

**Effect size between phases.** To say how much an observable changes from G1 to S, the report gives Cohen's $d$,

$$d = \frac{\mu_{\text{S}} - \mu_{\text{G1}}}{\sigma_{\text{pooled}}}, \qquad \sigma_{\text{pooled}} = \sqrt{\frac{\sigma_{\text{G1}}^2 + \sigma_{\text{S}}^2}{2}}$$

where $\mu$ and $\sigma$ are the mean and standard deviation of the observable in each phase. As a rule of thumb, $d$ below 0.2 (in absolute value) is small and above 0.8 is large.

## Metrics from the 3D structures

For every structure the coordinates form a matrix $V$ of $N$ rows (beads) and three columns. With replication there are two chains (the two sister chromatids). The shape and size metrics below are computed on the **first sister**, and a few extra ones use both. Coordinates are in Å in the structure files, and the figures convert lengths to nm and volumes to nm$^3$.

### Size

Let $c$ be the centre of mass (the mean of all $r_i$), $D_i$ the distance of bead $i$ from it, and $d_{ij}$ the distance between beads $i$ and $j$.

| Metric | Definition | Meaning |
|---|---|---|
| Radius of gyration | $R_g = \sqrt{\langle D_i^2 \rangle}$ | root-mean-square distance from the centre: the overall size |
| Mean distance to the centre | $\langle D_i \rangle$ | radial spread |
| Maximum distance to the centre | $\max_i D_i$ | the farthest bead |
| Mean pairwise distance | $\langle d_{ij} \rangle$ over all pairs | compactness |
| Standard deviation of pairwise distances | spread of $d_{ij}$ over all pairs | how heterogeneous the structure is |
| Maximum pairwise distance | $\max d_{ij}$ | the diameter |
| End-to-end distance | $d_{1N}$ | distance between the two chain ends |

### Shape

The **gyration tensor** $G$ describes how the beads are spread in each direction. It is a $3 \times 3$ matrix whose entry $G_{ab}$ is the average over beads of (offset from the centre along axis $a$) times (offset along axis $b$). Its eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge 0$ are the squared extents along the three principal axes, and they add up to $R_g^2$. A sphere has three equal eigenvalues, a rod has one large, and a flat disc has one small.

| Metric | Definition | Meaning |
|---|---|---|
| Asphericity | $\dfrac{\lambda_1 - \lambda_3}{R_g^2}$ | 0 for a sphere, towards 1 for an elongated shape (the definition used in RepliSage, which differs from the usual $\lambda_1 - \frac{1}{2}(\lambda_2 + \lambda_3)$ form) |
| Planarity | $\dfrac{\lambda_3}{R_g^2}$ | small for a flat shape, up to $1/3$ for a sphere |
| Ellipsoid ratio | $\sqrt{\lambda_1 / \lambda_3}$ | longest over shortest principal extent |
| Ellipsoid volume | $\frac{4}{3}\pi \sqrt{\lambda_1 \lambda_2 \lambda_3}$ | volume of the ellipsoid whose semi-axes are the principal extents |

### Volume and contacts

| Metric | Definition | Meaning |
|---|---|---|
| Convex hull volume | volume of the smallest convex shape that contains all beads | the space the structure occupies; falls when the structure compacts |
| Bounding box volume | width $\times$ height $\times$ depth | volume of the axis-aligned box around the beads |
| Contact number | $\langle \#\{j \ne i : d_{ij} < d_c\} \rangle_i$ | mean number of beads closer than the cutoff $d_c$ (default $d_c = 1$ in coordinate units) |

### Two chains: sisters

With replication, the whole system of both sisters has its own size and volume, and the two chromatids are compared:

| Metric | Definition | Meaning |
|---|---|---|
| Total radius of gyration, total convex hull volume | $R_g$ and hull volume computed on all $2N$ beads | size and volume of the replicated system |
| Sister distance | distance between the centroids of the two chains | 0 when the sisters overlap, growing as they separate |

### Contact maps

For a set of structures (all of one phase), the **contact map** is the average of an inverse distance,

$$C_{ij} = \left\langle \frac{1}{d_{ij} + 1} \right\rangle$$

where $\langle\cdot\rangle$ averages over the structures and $d_{ij}$ is the distance between beads $i$ and $j$ in one structure (high value = close). It is the 3D counterpart of a Hi-C map and is computed separately for G1, S and G2/M. Averaging the maps is preferred to taking the map of the average structure, because a mean structure is blurred and its distances are not those of any single cell.

### Available but not computed by default

The code also contains the box-counting (fractal) dimension, the global distance fluctuation and a PCA-based structural anisotropy. They are slow on large structures and are off in the standard pipeline; you can call them from `src/analysis/structure_metrics.py`.

## Summary: what is computed and where

| Group | Metrics | Source | Saved in |
|---|---|---|---|
| Loops | mean and bound loop length, bound fraction, legs on CTCF, coverage | MC | `metadata/MCMC_output/`, `lef` table |
| Energies | total, folding, binding, epigenetic | MC | `metadata/energy_factors/` |
| Violations | number and strength (S phase) | MC + replication | `metadata/MCMC_output/` |
| Epigenetic states | average state, fractions, compartments, switching | MC | `metadata/MCMC_output/`, `compartments` table |
| Order parameters | magnetisation, cluster order, Binder-type cumulant, entropy, synchronisation | MC | `metadata/graph_metrics/` |
| Replication | $F(t)$, $T_{10}/T_{50}/T_{90}$, origins, Spearman $\rho$ | replication simulation | `replication` table |
| Convergence | autocorrelation, $T_\rho$, ESS, MCSE, Geweke, $\hat{R}$, Cohen's $d$ | MC series | `convergence` and `phases` tables |
| Size and shape | $R_g$, pair distances, asphericity, planarity, ellipsoid ratio and volume | MD | `metadata/structural_metrics/` |
| Volume and contacts | convex hull and bounding box volume, contact number | MD | `metadata/structural_metrics/` |
| Sisters | total $R_g$, total hull volume, sister distance | MD | `metadata/structural_metrics/` |
| Contact maps | $C_{ij}$ per phase | MD | `mean_inv_heatmap_<first>_<last>.npy` |

## Reading the metrics

* **Loops getting shorter in S phase** is expected, because forks act as barriers that loops may not cross; check that the violations stay near zero.
* **Compartment switching** (a net flux from B to A) is expected when forks act as epigenetic organisers (`REP_FORK_EPIGENETIC_ORGANIZER`).
* **$R_g$ and the convex hull volume** should rise in S (replicated chromatin unfolds and the sisters form) and fall in G2/M (condensation). The sister distance rises when the sisters separate.
* **Order parameters** help to spot when the system is too frozen or too noisy; they are one of the ways to choose $T_{\text{mc}}$ in the critical region.
* Trust **trends and ratios** between phases more than absolute values, because the model has arbitrary length and energy units, and in S phase the system is out of equilibrium (see [the Hamiltonian idea](hamiltonian.md)).

## In the code

* [`src/analysis/lef_stats.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/lef_stats.py): loop series and coverage
* [`src/analysis/compartments.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/compartments.py): compartments and switching
* [`src/analysis/potts_order.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/potts_order.py), [`synchronization.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/synchronization.py): order parameters
* [`src/analysis/rep_stats.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/rep_stats.py): replication statistics
* [`src/analysis/structure_metrics.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/structure_metrics.py), [`ensemble_metrics.py`](https://github.com/SFGLab/RepliSage/blob/main/src/analysis/ensemble_metrics.py): 3D metrics
