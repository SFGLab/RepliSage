# Molecular dynamics model

The Monte Carlo gives a movie of abstract states: which beads are held together by loops, which beads are in A or B, and which beads are replicated. The 3D model turns each sampled state into a real 3D structure with [OpenMM](https://openmm.org).

It runs by default as molecular dynamics (`MD`), because we expect a trajectory of replication. Switch it off with `--run-md false` (or `simulation_type = none`), or use `EM` for a faster, static version. The platform is picked automatically: CUDA, then OpenCL, then CPU, with a fallback to the next one on any error.

## One chain or two

| Situation | Chains |
|---|---|
| No replication data | One chain of $N$ beads |
| With replication data | Two chains of $N$ beads, the original and its sister chromatid. They are chains A and B in the CIF files |

## Forces

In the notation of the paper, the total potential is

$$U = U_{\text{bk}} + U_{\text{ev}} + U_{\text{le}}(t) + U_{\text{rep}}(t) + U_{\text{pull}} + U_{\text{block}}(t)$$

backbone (bonds and stiffness), excluded volume, loops, sister tie, pulling force and compartment attraction. The Langevin temperature is 310 K and is unrelated to the Monte Carlo temperature $T_{\text{mc}}$.

Beads have a mass of a 10 kb chromatin piece. Distances are in nanometres and the typical bond length is $r_0 = 0.1$. All forces below act on every chain.

| Force | Form | Purpose |
|---|---|---|
| Excluded volume, $U_{\text{ev}}$ | Repulsive $( \frac{\epsilon^2 \sigma^2}{r + r_s} )^3$ with cutoff 0.2 | Beads do not overlap |
| Bonds, $U_{\text{bk}}$ | Harmonic, $r_0 = 0.1$, $k = 5 \times 10^5$ | Neighbouring beads stay connected |
| Stiffness | Harmonic angle around $\pi$, $k = 200$ | Chromatin is not infinitely flexible |
| Loops, $U_{\text{le}}$ | Harmonic, $r_0 = 0.1$, $k = 5 \times 10^4$ between legs $m_k$ and $n_k$ of every bound LEF | Loop extrusion shapes the structure. Updated at every sample |
| Compartments, $U_{\text{block}}$ | Gaussian attraction between beads of the same chain and the same epigenetic state | A and B beads cluster separately |
| Sister tie, $U_{\text{rep}}$ | Harmonic bond between bead $i$ on chain A and bead $i$ on chain B | Keeps the copies together (see below) |
| Pulling, $U_{\text{pull}}$ | Constant force along $x$, opposite on the two chains | Separates the sisters after S phase |

### Compartment strength

The attraction depth depends on the epigenetic state of the two beads:

| Epigenetic state | 2 | 1 | 0 | -1 | -2 |
|---|---|---|---|---|---|
| Depth | -0.2 | -0.4 | -0.6 | -0.8 | -1.0 |

The width of the Gaussian is $\sigma = r_w / 2$ and its centre is at 0.2, where $r_w = 0.1 \sqrt{N}$ is the random-walk length scale of the chain.

### Sister tie through the cell cycle

| When | Replicated bead | Not replicated bead |
|---|---|---|
| G1 | (n/a) | Held at distance 0 (strong tie: the copies sit on top of each other) |
| S | Loose tie at distance $r_w / 4$, $D = 1000$ | Strong tie at distance 0 |
| After S | Free; the pulling force separates the sisters | Free |

The pulling force is switched on at the end of S and off again once the centres of the two chains are further apart than the major axis of the structure, so the sisters part but do not drift away forever.

## Dynamics

At every sampled state, RepliSage does the following:

1. Updates loops, compartment states and sister ties from the MC state.
2. Randomly disables the excluded volume of each bead with probability `EV_P`, which lets chains pass through each other occasionally, a cheap stand-in for topoisomerase.
3. Either runs `SIM_STEP` integrator steps (`MD` mode) or minimises the energy (`EM` mode).
4. Writes the structure to `ensemble/ensemble_<k>_<label>.cif`.

`MD` mode minimises once at the start and then keeps integrating, so consecutive structures are continuous in time. `EM` mode minimises from scratch each time and is useful for fast, noise-free structures.

The integrator is Langevin (friction $0.1 / fs$) or Brownian, at the temperature `SIM_TEMP`.

## Labels in file names

| Label | Phase |
|---|---|
| `BR` | Before replication (G1) |
| `R` | During replication (S) |
| `AR` | After replication (G2/M) |
| `LE` | No replication (loop extrusion only) |

## Starting structure

`INITIAL_STRUCTURE_TYPE` chooses how the chain starts: `rw` (random walk), `confined_rw`, `self_avoiding_rw`, `helix`, `circle`, `spiral`, `sphere` or `hilbert`. The second chain starts as a copy displaced slightly from the first.

## CIF files

Each structure is an mmCIF file with C-alpha-only beads, one chain per DNA copy and covalent bonds between neighbours. The B-factor column holds the position along the chain, so `color bfactor` in ChimeraX gives a rainbow. See [visualization.md](visualization.md).

## Parameters

| Name | Meaning | Default |
|---|---|---|
| `SIMULATION_TYPE` | `MD`, `EM` or empty (no 3D) | empty |
| `PLATFORM` | `CPU`, `CUDA` or `OpenCL` | CPU |
| `DEVICE` | GPU index | empty |
| `INITIAL_STRUCTURE_TYPE` | Starting shape | rw |
| `INTEGRATOR_TYPE` | `langevin` or `brownian` | langevin |
| `INTEGRATOR_STEP` (`--md-step-size`) | Time step | 100 femtosecond |
| `SIM_TEMP` | Temperature | 310 kelvin |
| `SIM_STEP` (`--md-steps`) | Integrator steps per sampled state (MD mode) | 1000 |
| `TOLERANCE` | Minimiser tolerance | 1.0 |
| `EV_P` | Probability that a bead's excluded volume is off | 0.01 |
| `FORCEFIELD_PATH` | OpenMM XML force field | `src/forcefields/classic_sm_ff.xml` |
| `DCD_REPORTER` | Also write a `.dcd` trajectory | False |

All of these are advanced parameters set in [config.ini](config.md); only `--run-md`, `--md-mode` and `--platform` exist as flags.

## In the code

- [`src/MD/model.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MD/model.py): `MD_MODEL`: the OpenMM model, built from the MC result
- [`src/MD/forces.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MD/forces.py): `ForcesMixin`: backbone, loop, replication and compartment forces
- [`src/MD/dynamics.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MD/dynamics.py): `DynamicsMixin`: changes the forces from one MC sample to the next
- [`src/MD/pipeline.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MD/pipeline.py): `PipelineMixin`: runs MD or minimisation over all samples
- [`src/MD/structures.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MD/structures.py): starting shapes: `compute_init_struct(N_beads, mode)`, `helix_structure`, `spiral_structure`, `generate_hilbert_curve`, ...
- [`src/MD/units.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MD/units.py): `parse_quantity`: `'310 kelvin'` becomes an OpenMM quantity
- [`src/MD/md_plots.py`](https://github.com/SFGLab/RepliSage/blob/main/src/MD/md_plots.py): `plot_md_energies`
- [`src/pipeline/md_stage.py`](https://github.com/SFGLab/RepliSage/blob/main/src/pipeline/md_stage.py): `run_md_stage(ctx, p, res, out_dir, rep)`: stage entry point
- [`src/io/structure_io.py`](https://github.com/SFGLab/RepliSage/blob/main/src/io/structure_io.py): CIF, PSF and ChimeraX writers

| Argument | Role |
|---|---|
| `MD_MODEL(M, N, N_beads, burnin, MC_step, out_path, platform, rep_frac_T, t_rep, Cs)` | `M`, `N` leg positions (LEFs by time), `rep_frac_T` replication matrix, `t_rep` S start, `Cs` compartment spins, `platform` OpenMM platform |
| `--run-md`, `--md-mode` | MD on or off, `MD` or `EM` (repeated energy minimisation) |
| `--platform` | `CPU`, `CUDA` or `OpenCL` |
| `INITIAL_STRUCTURE_TYPE` | starting shape for `compute_init_struct` |
| `SIMULATION_TYPE`, `INTEGRATOR_TYPE`, `INTEGRATOR_STEP`, `SIM_TEMP`, `SIM_STEP` | engine, integrator, time step, temperature, steps per sample |
| `FORCEFIELD_PATH`, `EV_P`, `TOLERANCE`, `DCD_REPORTER` | force field XML, excluded volume strength, minimiser tolerance, DCD trajectory output |
