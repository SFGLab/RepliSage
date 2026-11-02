# Monte Carlo model

The Monte Carlo (MC) model samples two kinds of objects on a chain of $N$ beads:

* **Loop extruding factors (LEFs).** Each LEF $k$ is a ring with two legs on beads $m_k < n_k$ that form a loop. Legs slide apart (extrusion) and stop at CTCF sites.
* **Epigenetic states.** Every bead $i$ carries a spin $s_i \in \{-2, -1, 0, 1, 2\}$ (a five-state Potts model). Positive spins give the active A compartment, negative spins the inactive B compartment.

A configuration is $(m_k, n_k)$ for all LEFs plus $(s_i)$ for all beads. The model prefers low-energy configurations and explores them with the Metropolis rule.

## Energy

$$E = E_{\text{fold}} + E_{\text{bind}} + E_{\text{cross}} + E_{\text{rep}} + E_{\text{potts}}$$

### Folding: loops want to be large

$$E_{\text{fold}} = c_f \cdot \frac{-N}{M \ln (N / M)} \sum_k \ln (n_k - m_k + 1)$$

Here $M$ is the number of LEFs and $c_f$ is `FOLDING_COEFF` (`FOLDING_COEFF2` for the second family). The logarithm rewards long loops with diminishing returns, and the prefactor keeps the scale independent of $N$ and $M$.

### Binding: legs like CTCF sites

$$E_{\text{bind}} = c_b \cdot \frac{-N}{\sum_i L_i + \sum_i R_i} \sum_k \left( L_{m_k} + R_{n_k} \right)$$

with $L$, $R$ the [binding potentials](preproc.md) and $c_b$ = `BIND_COEFF`. A leg sitting on a strong site lowers the energy.

### Crossing: loops should not cross

$$E_{\text{cross}} = 10^8 c_\kappa \sum_{k < l} K(k, l)$$

$K(k, l)$ is 1 when loops $k$ and $l$ cross ($m_k < m_l < n_k < n_l$ or the mirror case) or share an end, and 0 otherwise. $c_\kappa$ is `CROSS_COEFF`. Only LEFs of the same family interact, unless `COHESIN_BLOCKS_CONDENSIN` is on.

### Replication barrier

When replication is on, a loop must not straddle a replication fork. For each LEF with legs on replicated state $f(m_k)$ and $f(n_k)$ (1 = copied, 0 = not copied) the penalty is

$$\rho_k = [f(m_k) \ne f(n_k)] + [f(m_k) = f(n_k) = 1 \text{ and some bead between them is not copied}] + \frac{\text{copied beads inside}}{n_k - m_k} \cdot [f(m_k) = f(n_k) = 0]$$

and

$$E_{\text{rep}} = 10^8 c_{\text{rep}} \sum_k \rho_k$$

with $c_{\text{rep}}$ = `REP_COEFF`. A fork is therefore a soft wall for loops. The default is `REP_COEFF = 0.01`, an energy scale of $10^6$. Version 1.0 used 1.0 (a scale of $10^8$); tests show that anything from $10^3$ to $10^8$ blocks loops equally well, while the huge value only made energy numbers unreadable.

### Epigenetic state (Potts) energy

Only active when replication data are given:

$$E_{\text{potts}} = -c_1 \sum_i H_i s_i + c_2 \sum_{i < j} J_{ij} |s_i - s_j|$$

* The field $H_i$ comes from replication timing: $h_i = 2 f_i - 1$, so early-replicating beads are pushed to A and late ones to B. With `REP_FORK_EPIGENETIC_ORGANIZER` on, a time-dependent term is added so that a bead gets its field when a fork passes it.
* The coupling $J_{ij}$ is 1 for neighbouring beads and counts the LEFs that connect $i$ and $j$. Loops therefore pull their two anchors toward the same state, which spreads epigenetic colour along loops.
* $c_1$ = `EPI_FIELD_COEFF` and $c_2$ = `EPI_INTERACT_COEFF`.

## Moves

At each step the sampler proposes `N_SWEEP` changes. Each proposal is either

* with probability `P_REW`, a **LEF move**: an unbound LEF binds at a random bead; a bound LEF either unbinds and rebinds or slides each leg by -1, 0 or +1 beads (`LEF_RW`), or
* otherwise a **spin flip**: a random bead moves to one of the four other states.

During S phase, a leg next to a fork is pushed back (`REP_FORK_PUSH`), so forks sweep loops ahead of them.

## Acceptance

A move changing the energy by $\Delta E$ is accepted with probability

$$\min \left( 1, e^{-\Delta E / T} \right)$$

where $T$ is `T_MC`. Lower $T$ gives a more ordered system, higher $T$ a more fluctuating one.

## Sampling

The state is recorded every `MC_STEP` steps. The first `BURNIN` steps are discarded in the analysis. The [analysis](analysis.md) page explains how to check that the chain has converged.

## Phases

| Phase | Steps | What changes |
|---|---|---|
| G1 | before `REP_START_TIME` | No fork; all beads unreplicated |
| S | `REP_START_TIME` to `REP_START_TIME + REP_TIME_DURATION` | Forks move; replication barrier is active |
| G2/M | after S | All beads replicated; the final matrix applies |

## Parameters

| Name | Meaning | Default |
|---|---|---|
| `N_STEPS` | Total MC steps | 50000 |
| `MC_STEP` | Recording interval | 200 |
| `BURNIN` | Discarded steps | 1000 |
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
