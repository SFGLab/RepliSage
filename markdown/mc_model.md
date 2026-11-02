# Monte Carlo model

The notation follows the RepliSage paper. A configuration of the system is

$$\mathcal{S} = \left(\{(m_i, n_i)\}_{i=1}^{N_{\text{lef}}},\ \{s_k\}_{k=1}^{N_{\text{beads}}}\right)$$

where LEF $i$ has its legs on beads $m_i < n_i$, and bead $k$ has an epigenetic state $s_k \in \{-2,-1,0,1,2\}$ (a five-state Potts model: positive = A, negative = B). The graph $\mathcal{G}$ links neighbouring beads $k, k\pm1$ and the two anchors $m_i, n_i$ of every LEF. States are sampled from a Boltzmann distribution,

$$p(\mathcal{S}) = \frac{1}{\mathcal{Z}} \exp\left(-\frac{\mathcal{H}}{T_{\text{mc}}}\right)$$

with the partition function $\mathcal{Z}$ (a normalisation constant) and the Monte Carlo temperature $T_{\text{mc}}$ (the Boltzmann constant is set to 1). For the intuition, see [the idea in a nutshell](intuition.md).

## Energy

$$\mathcal{H} = \mathcal{H}_{\text{le}} + \mathcal{H}_{\text{rep}} + \mathcal{H}_{\text{epi}}$$

$$\mathcal{H}_{\text{le}} = \mathcal{H}_{\text{fold}} + \mathcal{H}_{\text{cross}} + \mathcal{H}_{\text{bind}}$$

Loop extrusion ($\mathcal{H}_{\text{le}}$, as in LoopSage), replication barrier ($\mathcal{H}_{\text{rep}}$) and epigenetic states ($\mathcal{H}_{\text{epi}}$). Each coefficient is written $\mathcal{C}$ (capital, internal) and is built from a user parameter (lowercase, in the config file), see the [notation table](#notation-paper-and-config).

**Folding: loops want to be large.**

$$\mathcal{H}_{\text{fold}} = \mathcal{C}_{\text{fold}} \sum_{i=1}^{N_{\text{lef}}} \log(n_i - m_i + 1)$$

$$\mathcal{C}_{\text{fold}} = -\frac{N_{\text{beads}} f}{N_{\text{lef}} \log(N_{\text{beads}}/N_{\text{lef}})}$$

**Binding: legs like CTCF sites.** $L(\cdot)$ and $R(\cdot)$ are the binding potentials for left and right legs ([pre-processing](preproc.md)).

$$\mathcal{H}_{\text{bind}} = \mathcal{C}_{\text{bind}} \sum_{i=1}^{N_{\text{lef}}} \left(L(m_i) + R(n_i)\right)$$

$$\mathcal{C}_{\text{bind}} = -\frac{N_{\text{beads}} b}{\sum_k (L_k + R_k)}$$

**Crossing: loops should not cross.** $K = 1$ if two LEFs cross ($m_i < m_j < n_i < n_j$) or share an end, otherwise 0.

$$\mathcal{H}_{\text{cross}} = \mathcal{C}_{\text{cross}} \sum_{i,j} K(m_i,n_i;m_j,n_j)$$

$$\mathcal{C}_{\text{cross}} = \kappa \times 10^8$$

Only LEFs of the same family interact, unless `COHESIN_BLOCKS_CONDENSIN` is on. A second family of $N^{(2)}_{\text{lef}}$ LEFs (condensin) has its own folding coefficient $\mathcal{C}^{(2)}_{\text{fold}}$.

**Replication barrier.** $f_{\text{rep}}(x,t) \in \{0,1\}$ says whether bead $x$ is copied at time $t$ (output of [Replikator](replication_model.md)).

$$\mathcal{H}_{\text{rep}}(t) = \mathcal{C}_{\text{rep}} \sum_{i=1}^{N_{\text{lef}}} \mathcal{R}(m_i, n_i; t)$$

$$\mathcal{C}_{\text{rep}} = c_{\text{rep}} \times 10^8$$

| Situation of LEF $i$ | $\mathcal{R}$ |
|---|---|
| $f_{\text{rep}}(m_i) \ne f_{\text{rep}}(n_i)$ (legs on different sides of a fork) | 1 |
| both copied, but an uncopied bead lies between the legs | 1 |
| both uncopied | $\ell_{\text{rep}} / (n_i - m_i)$ |
| otherwise | 0 |

Here $\ell_{\text{rep}}$ is the number of copied beads inside the loop. A fork is a *moving barrier*: a large $c_{\text{rep}}$ is a solid wall, a small one lets some LEFs through. The default is `REP_COEFF = 0.01` (a scale of $10^6$); tests show anything from $10^3$ to $10^8$ blocks loops equally well.

**Epigenetic states** (Potts-like, active only with replication data):

$$\mathcal{H}_{\text{epi}} = \mathcal{C}_{p,1} \sum_{k=1}^{N_{\text{beads}}} \frac{h_k + h_{t,k}}{2} s_k + \mathcal{C}_{p,2} \sum_{k>l} J_{kl} |s_k - s_l|$$

$$\mathcal{C}_{p,1} = -c_{p,1},\ \mathcal{C}_{p,2} = c_{p,2}$$

* $h_k \in [-1, 1]$ is the **epigenetic field**, from replication timing ($h_k = 2 f_k - 1$ for the timing signal $f_k \in [0,1]$): early replicating beads go to A, late ones to B.
* $h_{t,k}$ is the **replication-driven field**: it is zero before replication, then each bead receives a one-off contribution when a fork passes it, positive for early and negative for late replication,

$$h_{t,k} = \sum_{t'=0}^{t_r}\left(1 - \frac{2 t'}{T_r}\right)\left(f_{\text{rep}}(k,t') - f_{\text{rep}}(k,t'-1)\right)$$

with $t_r$ the time since replication started and $T_r$ its total duration. It is switched off with `REP_FORK_EPIGENETIC_ORGANIZER = False`.
* $J_{kl} = 1$ if beads $k, l$ are neighbours ($l = k \pm 1$) or the two anchors of a LEF, otherwise 0. Beads in different states cost energy, so loops pull their anchors towards one state.

## Moves and the algorithm

1. Start from random LEF positions and (by default) random states.
2. Propose a change: with probability `P_REW` a **LEF move** (an unbound LEF binds at a random bead; a bound one rebinds at a random place, or slides by one bead $(m_i, n_i) \to (m_i \pm 1, n_i \pm 1)$), otherwise a **state change** (bead $k$ moves to one of the four other states). During S phase a leg next to a fork is pushed back (`REP_FORK_PUSH`).
3. Accept if $\Delta\mathcal{H} \le 0$, otherwise with probability $e^{-\Delta\mathcal{H}/T_{\text{mc}}}$. Here $\Delta\mathcal{H} = \Delta\mathcal{H}_{\text{link}} + \Delta\mathcal{H}_{\text{nodes}}$ (links changed by a LEF move, states changed by a flip).
4. A Monte Carlo step is $N_{\text{sweep}}$ such proposals. Repeat for $N_{\text{steps}}$ steps, record every `MC_STEP` steps, and discard the first `BURNIN` steps. See [analysis](analysis.md) for convergence checks.

Lower $T_{\text{mc}}$ gives a more ordered system, higher $T_{\text{mc}}$ a noisier one. During S phase the energy keeps changing, so the run is out of equilibrium there.

## Phases

| Phase | Steps | What changes |
|---|---|---|
| G1 | before `REP_START_TIME` | No fork; all beads unreplicated |
| S | `REP_START_TIME` to `REP_START_TIME + REP_TIME_DURATION` | Forks move; replication barrier active |
| G2/M | after S | All beads replicated; second LEF family may act |

## Notation: paper and config

| Paper | Meaning | Config / flag |
|---|---|---|
| $N_{\text{beads}}$ | beads in one chain | `N_BEADS` |
| $N_{\text{lef}}$, $N^{(2)}_{\text{lef}}$ | LEFs, second family | `N_LEF`, `N_LEF2` |
| $(m_i, n_i)$ | leg positions of LEF $i$ | |
| $s_k$ | epigenetic state of bead $k$ | |
| $T_{\text{mc}}$ | Monte Carlo temperature | `T_MC` |
| $N_{\text{sweep}}$, $N_{\text{steps}}$ | proposals per step, number of steps | `N_SWEEP`, `N_STEPS` |
| $f$, $f^{(2)}$ | folding strengths | `FOLDING_COEFF`, `FOLDING_COEFF2` |
| $b$ | binding strength | `BIND_COEFF` |
| $\kappa$ | crossing penalty | `CROSS_COEFF` |
| $c_{\text{rep}}$ | replisome barrier strength (permeability) | `REP_COEFF` |
| $c_{p,1}$, $c_{p,2}$ | epigenetic field and coupling strength | `EPI_FIELD_COEFF`, `EPI_INTERACT_COEFF` |
| $f_{\text{rep}}(x,t)$ | replicated (1) or not (0) | replicated-DNA matrix |
| $T_r$, $t_r$ | duration of replication, time since it started | `REP_TIME_DURATION`, `REP_START_TIME` |

## Parameters

| Name | Meaning | Default |
|---|---|---|
| `N_STEPS` | Total MC steps (`--n-steps`) | 50000 |
| `MC_STEP` | Recording interval (`--mc-step`) | 200 |
| `BURNIN` | Discarded steps (`--burnin`) | 1000 |
| `N_SWEEP` | Proposals per step | 1000 |
| `T_MC` | Temperature | 1.5 |
| `N_LEF` | Number of LEFs (empty = 2 times the number of CTCF sites) | auto |
| `N_LEF2` | Number of second-family LEFs | 0 |
| `P_REW` | Probability of a LEF move | 0.5 |
| `LEF_RW` | Random-walk sliding (False = one-directional) | True |
| `FOLDING_COEFF`, `FOLDING_COEFF2` | Folding strength of family 1, family 2 | 1.0, 0.0 |
| `BIND_COEFF` | CTCF binding strength | 1.0 |
| `CROSS_COEFF` | Crossing penalty | 1.0 |
| `COHESIN_BLOCKS_CONDENSIN` | LEFs of different families also avoid crossing | False |
| `REP_COEFF` | Replication barrier (scale is this times $10^8$) | 0.01 |
| `REP_FORK_PUSH` | Forks push loops | True |
| `REP_FORK_EPIGENETIC_ORGANIZER` | Forks act as epigenetic organisers | True |
| `EPI_FIELD_COEFF` | Potts field strength | 1.0 |
| `EPI_INTERACT_COEFF` | Potts coupling strength | 1.0 |
| `RANDOM_INIT_SPINS` | Random initial spins | True |
| `N_REPLICAS` | Independent chains | 1 |
| `SEED` | Random seed | random |

All of these belong to the config file; see [config.md](config.md).

## In the code

- [`src/MC/params.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MC/params.py): `MCParams`: every parameter of one run
- [`src/MC/runner.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MC/runner.py): `run_mc(p, n_replicas, n_threads, seed)`: runs the chains in parallel
- [`src/MC/engine.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MC/engine.py): `run_energy_minimization(...)`: the Metropolis loop itself (numba)
- [`src/MC/moves.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MC/moves.py): proposal moves
- [`src/MC/normalization.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MC/normalization.py): `energy_norms(p)`: the normalisation constants
- [`src/MC/loops_energy.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MC/loops_energy.py): fold, bind and cross terms
- [`src/MC/replication_energy.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MC/replication_energy.py): replication penalty
- [`src/MC/potts_energy.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MC/potts_energy.py): compartment (Potts) term
- [`src/MC/total_energy.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MC/total_energy.py): sums the terms: `get_E`, `get_dE_rewiring`
- [`src/MC/results.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MC/results.py): `MCResult`, `save_mc`, `phase_bounds`
- [`src/pipeline/mc_stage.py`](https://github.com/SFGLab/RepliSage/blob/main/src/pipeline/mc_stage.py): `build_mc_params`, `run_mc_stage`

| Function or class | What it does |
|---|---|
| `unbind_bind(N_beads)` | proposes a new random position for a LEF (rebinding) |
| `slide(m_old, n_old, N_beads, f_T, t, rw, push_condition)` | moves one or both legs by one bead; `rw` random walk, `push_condition` fork pushing |
| `initialize(N_lef, N_lef2, N_beads, random_init_spins)` | random initial state of LEFs and spins |
| `energy_norms(p)` | returns `fold_norm, fold_norm2, bind_norm, k_norm, rep_norm, potts_norm1, potts_norm2` |
| `E_fold(ms, ns, fold_norm)  /  get_dE_fold(...)` | folding energy and its change for one move |
| `E_bind(L, R, ms, ns, bind_norm)  /  get_dE_bind(...)` | CTCF binding energy and its change |
| `E_cross(ms, ns, N_lef, N_lef2, k_norm, cohesin_blocks_condensin)  /  get_dE_cross(...)` | loop-crossing penalty, with the function `Kappa(mi, ni, mj, nj)` |
| `E_rep(f_rep_T, ms, ns, t, rep_norm)  /  get_dE_rep(...)` | penalty for a LEF crossing a replication fork, with `Rep_Penalty(m, n, f)` |
| `E_potts(spins, ms, ns, h, ht, potts_norm1, potts_norm2, t, rep_fork_organizers)` | Potts energy (field plus couplings `J_pair`); `get_dE_node`, `get_dE_potts_link` give the changes |
| `get_E(...)  /  get_dE_rewiring(...)` | total energy and the change for a rewiring move |

| Argument | Role |
|---|---|
| `N_STEPS`, `MC_STEP`, `BURNIN`, `N_SWEEP` | chain length, sampling interval, burn-in, steps per sweep |
| `N_LEF`, `N_LEF2`, `LEF_RW`, `COHESIN_BLOCKS_CONDENSIN` | number of LEFs (two families) and their movement rules |
| `T_MC`, `P_REW` | temperature and the probability of a rewiring move |
| `FOLDING_COEFF`, `FOLDING_COEFF2`, `BIND_COEFF`, `CROSS_COEFF`, `REP_COEFF` | energy weights $f$, $f_2$, $b$, $\kappa$, $c_{rep}$ |
| `EPI_INTERACT_COEFF`, `EPI_FIELD_COEFF`, `RANDOM_INIT_SPINS` | Potts coupling, field and initial spins |
| `REP_FORK_PUSH`, `REP_FORK_EPIGENETIC_ORGANIZER` | forks push LEFs; forks act as compartment organizers |
| `--n-replicas`, `--seed`, `--threads` | independent chains, random seed, CPU threads |
