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
| $\mathbf{r}_i$ | 3D position of bead $i$ in one structure |
| $\langle \cdot \rangle$ | average over beads, over samples or over structures (stated in each case) |

All MC quantities are computed per recorded sample $t$, so each metric is a time series. Phase values (G1, S, G2/M) are means of that series over the samples of the phase, and the report also gives the standard deviation. Because consecutive samples are correlated, this standard deviation describes fluctuations and is not an uncertainty.

## Metrics from the Monte Carlo

### Loops

**Loop length.** For one sample, the *mean loop length* is

$$\langle \ell \rangle = \frac{1}{N_{\text{lef}}} \sum_{k=1}^{N_{\text{lef}}} (n_k - m_k)$$

which averages over all LEFs and counts an unbound LEF as length 0 (this is the definition used in the paper). The *bound loop length* only averages over bound LEFs,

$$\langle \ell \rangle_{\text{bound}} = \frac{\sum_{k \in \mathcal{B}} (n_k - m_k)}{|\mathcal{B}|}$$

where $\mathcal{B}$ is the set of bound LEFs. Multiply by the bead size to get kilobases. The report also gives the median and the 95th percentile of the bound loop lengths.

**Fraction of bound LEFs.**

$$f_{\text{bound}} = \frac{|\mathcal{B}|}{N_{\text{lef}}}, \qquad \mathcal{B} = \{k : m_k \ge 0,\ n_k > m_k\}$$

**Legs on CTCF sites.** The fraction of bound LEFs whose left leg sits on a CTCF site that attracts left legs, $f_L$, and whose right leg sits on a site that attracts right legs, $f_R$ (the orientation of the sites is explained in the [biology page](biology.md)). They tell how strongly the loops are anchored.

**Coverage.** The number of loops that span bead $i$,

$$c_i = \big|\{k \in \mathcal{B} : m_k \le i < n_k\}\big|$$

The report gives the fraction of beads with $c_i > 0$ (beads inside a loop) and the mean $\langle c_i \rangle$ (loops per bead). A very high $\langle c_i \rangle$ with short loops is the signature of **stacking**, many loops piled on the same place, which is what happens when $T_{\text{mc}}$ is too low (see [the critical region](hamiltonian.md#temperature-in-replisage)).

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

When a loop straddles a fork it breaks the rule that loops do not cross replication forks. Let $f_i(t) \in \{0, 1\}$ say whether bead $i$ is replicated at time $t$. A loop $(m, n)$ is a **violation** if its two legs are in different replication states, $f_m \ne f_n$, or if both are replicated but some bead between them is not. The number of violations $N_{\text{viol}}$ is counted in S phase, and each violating loop has a strength

$$\sigma = \min(r, \ell - r) + 1$$

where $\ell = n - m$ and $r$ is the number of replicated beads inside the loop. The reported strength is $\sum \sigma / (N_{\text{viol}} + 1)$. A good S phase has violations near zero.

### Epigenetic states and compartments

**Average epigenetic state.** The mean state over all beads, $\langle s \rangle = \frac{1}{N} \sum_i s_i$. It is close to 0 when A-like and B-like beads balance, and its changes in S phase show a global shift.

**State fractions.** $p_k = \frac{1}{N}\,|\{i : s_i = k\}|$ for $k = -2, \dots, 2$.

**Compartment of a bead.** In a window of samples (the first and the last 10 % of S phase), compute the window mean state $\bar{s}_i$ of each bead and classify

$$\text{compartment}_i = \begin{cases} \text{A} & \bar{s}_i > 0.25 \\ \text{B} & \bar{s}_i < -0.25 \\ \text{N (neutral)} & \text{otherwise} \end{cases}$$

**Compartment switching.** Comparing the two windows gives a $3 \times 3$ transition matrix between B, N and A. The report lists the number of beads that switched B to A and A to B, the **net flux** (B to A minus A to B), the fraction of unchanged beads and the A and B fractions before and after. For the time series, the *switching rate* at lag $\tau$ is the fraction of beads whose state sign changes between sample $t$ and $t + \tau$,

$$r_{B \to A}(t) = \frac{1}{N} \big|\{i : s_i(t) < 0,\ s_i(t+\tau) > 0\}\big|$$

and similarly for A to B.

### Order parameters of the epigenetic states

These describe how ordered the states are at each sample (only with replication data). With $q = 5$ states, $N_k$ the number of beads in state $k$ and $p_k = N_k / N$:

| Metric | Definition | Meaning |
|---|---|---|
| Magnetisation | $M = \dfrac{q\,N_{\max} - N}{N\,(q - 1)}$ | 0 if the states are evenly mixed, 1 if all beads share one state ($N_{\max}$ = size of the most populated state) |
| Cluster order | $C = N_{\max} / N$ | the fraction of beads in the most common state (this uses state counts, not neighbours along the chain) |
| Binder-type cumulant | $U = 1 - \dfrac{\sum_k p_k^4}{3 \left(\sum_k p_k^2\right)^2}$ | a measure of the shape of the state distribution, built from its second and fourth moments |
| Entropy | $H = -\sum_k p_k \ln p_k$ | 0 if all beads share a state, $\ln q$ if the states are evenly mixed |
| Synchronisation | $\Sigma = \dfrac{1}{|E|} \sum_{(u,v) \in E} \dfrac{\mathbb{1}[s_u = s_v]}{q - 1}$ | agreement of the states of beads joined by an edge of the graph |

For the synchronisation, the graph has one edge between every pair of consecutive beads and one edge for every bound loop, so it measures how well neighbouring beads and the two anchors of a loop agree, which is what produces compartments. The sum runs over the edges $E$ and each agreeing edge adds $1/(q-1)$, so with $q = 5$ the value stays between 0 and 0.25. Use it for comparisons between phases and runs, not as an absolute fraction.

### Replication simulation

These come from the replication simulation (Replikator), not from the loops. With $f_i(t)$ as above:

| Metric | Definition |
|---|---|
| Replicated fraction | $F(t) = \frac{1}{N} \sum_i f_i(t)$ |
| Replication time of bead $i$ | $t_i$ = first time with $f_i = 1$, divided by the S-phase length (0 to 1) |
| $T_{10}$, $T_{50}$, $T_{90}$ | the times at which $F$ first reaches 10 %, 50 % and 90 % |
| S-phase span | $T_{90} - T_{10}$, how spread out the replication is |
| Inferred origins | local minima of the profile $t_i$ |
| Median origin spacing | median distance between neighbouring origins, in kb |
| Agreement with the input | Spearman correlation $\rho$ between $t_i$ and the input timing (early beads should replicate early) |

### Convergence diagnostics

These tell whether the Monte Carlo has run long enough, and are applied to the series above (energies, loop lengths, bound fraction, average state): autocorrelation $\rho(\tau)$, integrated autocorrelation time $T_\rho$, effective sample size $n_{\text{eff}} = n / T_\rho$, standard error $\mathrm{MCSE} = s / \sqrt{n_{\text{eff}}}$, the Geweke $z$ score and the split $\hat{R}$. They are defined in [Analysis](analysis.md#mcmc-diagnostics) and explained in [Monte Carlo, MCMC and Metropolis](mcmc.md).

**Effect size between phases.** To say how much an observable $x$ changes from G1 to S, the report gives Cohen's $d$,

$$d = \frac{\bar{x}_{\text{S}} - \bar{x}_{\text{G1}}}{\sqrt{\left(\sigma_{\text{G1}}^2 + \sigma_{\text{S}}^2\right)/2}}$$

where $\bar{x}$ and $\sigma$ are the mean and standard deviation in each phase. As a rule of thumb, $|d|$ below 0.2 is small and above 0.8 is large.

## Metrics from the 3D structures

For every structure the coordinates form a matrix $V$ of $N$ rows (beads) and three columns. With replication there are two chains (the two sister chromatids). The shape and size metrics below are computed on the **first sister**, and a few extra ones use both. Coordinates are in Å in the structure files, and the figures convert lengths to nm and volumes to nm$^3$.

### Size

Let $\mathbf{r}_c = \frac{1}{N}\sum_i \mathbf{r}_i$ be the centre of mass and $d_{ij} = |\mathbf{r}_i - \mathbf{r}_j|$ the distance between two beads.

| Metric | Definition | Meaning |
|---|---|---|
| Radius of gyration | $R_g = \sqrt{\frac{1}{N}\sum_i \lvert\mathbf{r}_i - \mathbf{r}_c\rvert^2}$ | root-mean-square distance from the centre: the overall size |
| Mean distance to the centre | $\frac{1}{N}\sum_i \lvert\mathbf{r}_i - \mathbf{r}_c\rvert$ | radial spread |
| Maximum distance to the centre | $\max_i \lvert\mathbf{r}_i - \mathbf{r}_c\rvert$ | the farthest bead |
| Mean pairwise distance | $\frac{2}{N(N-1)} \sum_{i<j} d_{ij}$ | compactness |
| Standard deviation of pairwise distances | spread of $d_{ij}$ over all pairs $i < j$ | how heterogeneous the structure is |
| Maximum pairwise distance | $\max_{i<j} d_{ij}$ | the diameter |
| End-to-end distance | $\lvert\mathbf{r}_N - \mathbf{r}_1\rvert$ | distance between the two chain ends |

### Shape

The **gyration tensor** describes how the beads are spread in each direction,

$$G = \frac{1}{N} \sum_i (\mathbf{r}_i - \mathbf{r}_c)(\mathbf{r}_i - \mathbf{r}_c)^{\mathsf{T}}$$

It is a $3 \times 3$ matrix with eigenvalues $\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge 0$ (the squared extent along the three principal axes), and $R_g^2 = \lambda_1 + \lambda_2 + \lambda_3$. A sphere has three equal eigenvalues, a rod has one large, and a flat disc has one small.

| Metric | Definition | Meaning |
|---|---|---|
| Asphericity | $\dfrac{\lambda_1 - \lambda_3}{\lambda_1 + \lambda_2 + \lambda_3}$ | 0 for a sphere, towards 1 for an elongated shape (this is the definition used in RepliSage, which differs from the usual $\lambda_1 - \frac{1}{2}(\lambda_2 + \lambda_3)$ form) |
| Planarity | $\dfrac{\lambda_3}{\lambda_1 + \lambda_2 + \lambda_3}$ | small for a flat shape, up to $1/3$ for a sphere |
| Ellipsoid ratio | $\sigma_1 / \sigma_3$ with $\sigma_a = \sqrt{\lambda_a}$ | ratio of the longest to the shortest principal extent |
| Ellipsoid volume | $\frac{4}{3}\pi\, \sigma_1 \sigma_2 \sigma_3$ | volume of the ellipsoid whose semi-axes are the standard deviations along the principal axes |

### Volume and contacts

| Metric | Definition | Meaning |
|---|---|---|
| Convex hull volume | volume of the smallest convex shape that contains all beads | the space the structure occupies; falls when the structure compacts |
| Bounding box volume | $\prod_{a=x,y,z} (\max r_a - \min r_a)$ | volume of the axis-aligned box around the beads |
| Contact number | $\frac{1}{N} \sum_i \lvert\{j \ne i : d_{ij} < d_c\}\rvert$ | mean number of beads closer than the cutoff $d_c$ (default $d_c = 1$ in coordinate units) |

### Two chains: sisters

With replication, the whole system of both sisters has its own size and volume, and the two chromatids are compared:

| Metric | Definition | Meaning |
|---|---|---|
| Total radius of gyration, total convex hull volume | $R_g$ and hull volume computed on all $2N$ beads | size and volume of the replicated system |
| Sister distance | $\big\lvert \langle\mathbf{r}\rangle_{\text{sister 1}} - \langle\mathbf{r}\rangle_{\text{sister 2}} \big\rvert$ | distance between the two centroids: 0 when the sisters overlap, growing as they separate |

### Contact maps

For a set of structures (all of one phase), the **contact map** is the ensemble average of an inverse distance,

$$C_{ij} = \left\langle \frac{1}{d_{ij} + 1} \right\rangle_{\text{structures}}$$

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
