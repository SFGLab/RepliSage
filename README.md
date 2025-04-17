# RepliSage
A simulation software for modeling the motion of cohesin during the replication process. This tool explores the interaction between cohesin, or more generally loop extrusion factors (LEFs), with replication forks and chromatin compartmentalization. It employs a sophisticated force-field that integrates MCMC Metropolis and molecular dynamics methodologies. The output is a 3D chromatin trajectory, providing a dynamic visualization of DNA replication and the formation of two identical copies.

![image](https://github.com/user-attachments/assets/703859c1-3e0d-4609-891c-dbce55576331)


## Simulation pipeline
RepliSage is composed by three distinct parts:

### Replication simulation (Replikator.py)

This is a simplistic *Monte Carlo simulation*, where we import single cell replication timing data to model replication. The average replication timing curves can show us the percentage of cells that have been replicated, and from them we estimate the *initiation rate* $I(x,t)$ which represents the probability fires at time $t$ in loci $x$. Then we employ the Monte Carlo simulation where in each step an origin fires with a probability derived from the initiation rate. When an origin fires, the replication fork start propagating bi-directionally with velocity $v$. The vlocity's mean and standard deviation is derived by calculating the slopes of consecutive optima of the averaged replication timing curve. Replikator outputs the trajectories of the replication forks.

### Stochastic simulation that models the interplay of loop extrusion with other factors

In this part we import the previously produced trajectories of replication forks and we model them as moving barriers for the loop extrusion factors. In this way this simulation is composed by three distinct parts:

#### Loop Extrusion
We use the energy landscape of LoopSage for that. We assume that there are two basic players: LEFs which follow a random difussive motion in 1D and CTCF whose locations are dervied from ChIA-PET data, which are filtered where is a CTCF motif in their sequence. Therefore, assuming that we have $N_{\text{lef}}$ LEFs with two degrees of freedom $(m_i,n_i)$ and epigenetic color states $s_i$ for each monomer, we can write down the following formula,  
    
$$E_{\text{le}} =  c_{\text{fold}}\sum_{i=1}^{N_{\text{coh}}}\log(n_i-m_i)+c_{\text{cross}}\sum_{i,j}K(m_i,n_i;m_j,n_j)+c_{\text{bind}}\sum_{i=1}^{N_{\text{coh}}}\left(L(m_i)+R(n_i)\right).$$
    
In this equation the first term models the folding of chromatin (how fast a loop extrudes, the higher $f$ the more tendency to extrude), the second term is a penalty for cohesin crossing, and the last one minimizes the energy when a LEF encounters a CTCF.
    
#### Compartmentalization

It is modelled by using a five stats ($s_i\in[-2,-1,0,1,2]$) Potts model 

$$E_{\text{potts}} = C_{p,1} \sum_{k} \left(\dfrac{h_k + h_{t_{r} k}}{2} \right) s_k +C_{p,2}\sum_{i>j} J_{ij}| s_i - s_j |.$$

The second term includes the interaction matrix $J_{ij}$, which is 1 when there is a LEF connecting $i$ with $j$ and 0 if not. The other term represents an epigeetic field. There is an averaged term $h_k$, which represents the averaged replication timing, and a time dependent term $h_{t_{r} k}$ which represents the spread of the epigenetic state due to a single replication fork.
    
#### Replication
This term models the interaction between the LEFs and replication forks, $$E_{\text{rep}}=C_{\text{rep}}\sum_{i=1}^{N_{lef}} \mathcal{R}(m_i,n_i ;f_{\text{rep}})$$ and in general penalizes inapropriate configurations between LEFs and replication forks (read the paper).

Therefore, the stochastic simulation integrates the sum of these energies $E = E_{le}+E_{rep}+E_{potts}$ and uses MCMC Metropolis method.

### Molecular Dynamics

This parts takes as input the states produced by the stochastic simulation and outputs 3D structures by using a potential in OpenMM. The molecular modeling approach assumes two molecular chains, each consisting of $N_{\text{beads}}$ monomers, where $N_{\text{beads}}$ reflects the granularity of the stochastic simulation. The total potential governing the system is expressed as: $$U = U_{\text{bk}} + U_{\text{le}}(t) + U_{\text{rep}}(t) + U_{\text{block}}(t)$$, where each term corresponds to a specific contribution. The backbone potential ($U_{\text{bk}}$) includes strong covalent bonds between consecutive beads, angular forces, and excluded volume effects to maintain chain integrity. The loop-formation potential ($U_{\text{le}}$) is a time-dependent term introducing harmonic bonds to model loop formation. These bonds are weaker than the backbone interactions and act between dynamically changing pairs of beads, $m_i(t)$ and $n_i(t)$. The last term models compartmentalization with block-copolymer potential.

For more details of the implementation, we suggest to our users to read the method paper of RepliSage.

## Requirements

RepliSage is a computationally and biophysically demanding project. It requires significant computing resources, both CPU and GPU. We recommend running RepliSage on high-performance workstations or HPC clusters equipped with a strong CPU and a GPU supporting CUDA (preferred) or at least OpenCL.

RepliSage is tested and supported on Debian-based Linux distributions.

Please note that even on powerful hardware, simulating a full cell cycle for a single chromosome with a polymer of 10,000 beads can take several hours to over a day. While the installation process is straightforward and thoroughly documented in our manual, running simulations will require patience and proper resources.

## Installation
It is needed to have at least python 3.10 and run,

```
pip install -r requirements.txt
```

Or more easily (do not forget to install it with python 3.10 or higher),

```
pip install pyRepliSage
```

## How to use?
The usage is very simple. To run this model you need to specify the parameters and the input data. Then RepliSage can do everything for you. 

Note that to run replisage it is needed to have `GM12878_single_cell_data_hg37.mat` data, for single-cell replication timing. These data are not produced by our laboratory, but they are alredy published in the paper of D. S. Massey et al. Please do not forget to cite them. We have uploaded the input data used in the paper here: https://drive.google.com/drive/folders/1PLA147eiHenOw_VojznnlC_ZywhLzGWx?usp=sharing.

### Python API
```python
from RepliSage.stochastic_model import *

# Set parameters
N_beads, N_lef, N_lef2 = 1000, 100, 20
N_steps, MC_step, burnin, T, T_min, t_rep, rep_duration = int(8e4), int(4e2), int(1e3), 1.6, 1.0, int(1e4), int(2e4)

f, f2, b, kappa= 1.0, 5.0, 1.0, 1.0
c_state_field, c_state_interact, c_rep = 2.0, 1.0, 1.0
mode, rw, random_spins, rep_fork_organizers = 'Metropolis', True, True, True
Tstd_factor, speed_scale, init_rate_scale, p_rew = 0.1, 20, 1.0, 0.5
save_MDT, save_plots = True, True

# Define data and coordinates
region, chrom =  [80835000, 98674700], 'chr14'

# Data
bedpe_file = '/home/skorsak/Data/method_paper_data/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe'
rept_path = '/home/skorsak/Data/Replication/sc_timing/GM12878_single_cell_data_hg37.mat'
out_path = '/home/skorsak/Data/Simulations/RepliSage_whole_chromosome_14'

# Run simulation
sim = StochasticSimulation(N_beads, chrom, region, bedpe_file, out_path, N_lef, N_lef2, rept_path, t_rep, rep_duration, Tstd_factor, speed_scale, init_rate_scale)
sim.run_stochastic_simulation(N_steps, MC_step, burnin, T, T_min, f, f2, b, kappa, c_rep, c_state_field, c_state_interact, mode, rw, p_rew, rep_fork_organizers, save_MDT)
if show_plots: sim.show_plots()
sim.run_openmm('OpenCL',mode='MD')
if show_plots: sim.compute_structure_metrics()

# Save Parameters
if save_MDT:
    params = {k: v for k, v in locals().items() if k not in ['args','sim']}
    save_parameters(out_path+'/other/params.txt',**params)
```

### Bash command

An even easier way that you can avoid all python coding is by running the command,

```
replisage -c config.ini
```

The configuration file has the usual form,

```
[Main]

; Input Data and Information
BEDPE_PATH = /home/blackpianocat/Data/method_paper_data/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe
REPT_PATH = /home/blackpianocat/Data/Replication/sc_timing/GM12878_single_cell_data_hg37.mat
REGION_START = 80835000
REGION_END = 98674700
CHROM = chr14
PLATFORM = CUDA
OUT_PATH = /home/blackpianocat/Data/Simulations/RepliSage_test

; Simulation Parameters
N_BEADS = 2000
N_LEF = 200
BURNIN = 1000
T_INIT = 1.8
T_FINAL = 1.0
METHOD = Metropolis
LEF_RW = True
RANDOM_INIT_SPINS = True

; Molecular Dynamics
INITIAL_STRUCTURE_TYPE = rw
SIMULATION_TYPE = MD 
TOLERANCE = 1.0
EV_P=0.01
```

You can define these parameters based on the table of simulation parameters.

## Parameter table

### General Settings
| Parameter Name         | Type      | Default Value   | Description                                                                 |
|-------------------------|-----------|-----------------|-----------------------------------------------------------------------------|
| PLATFORM                | str       | CPU             | Specifies the computational platform to use (e.g., CPU, CUDA).             |
| DEVICE                  | str       | None            | Defines the specific device to run the simulation (e.g., GPU ID).          |
| OUT_PATH                | str       | ../results      | Directory where simulation results will be saved.                          |
| SAVE_PLOTS              | bool      | True            | Enables saving of simulation plots.                                        |
| SAVE_MDT                | bool      | True            | Enables saving of molecular dynamics trajectories.                         |
| VIZ_HEATS               | bool      | True            | Enables visualization of heatmaps.                                         |

### Input Data
| Parameter Name         | Type      | Default Value   | Description                                                                 |
|-------------------------|-----------|-----------------|-----------------------------------------------------------------------------|
| BEDPE_PATH              | str       | None            | Path to the BEDPE file containing CTCF loop data.                          |
| REPT_PATH               | str       | None            | Path to the replication timing data file.                                  |
| REGION_START            | int       | None            | Start position of the genomic region to simulate.                          |
| REGION_END              | int       | None            | End position of the genomic region to simulate.                            |
| CHROM                   | str       | None            | Chromosome identifier for the simulation.                                  |

### Simulation Parameters
| Parameter Name         | Type      | Default Value   | Description                                                                 |
|-------------------------|-----------|-----------------|-----------------------------------------------------------------------------|
| N_BEADS                 | int       | None            | Number of beads in the polymer chain.                                      |
| N_LEF                   | int       | None            | Number of loop extrusion factors (LEFs).                                   |
| N_LEF2                  | int       | 0               | Number of secondary loop extrusion factors.                                |
| LEF_RW                  | bool      | True            | Enables random walk for loop extrusion factors (LEFs).                     |
| LEF_DRIFT               | bool      | False           | Enables drift for loop extrusion factors.                                  |
| RANDOM_INIT_SPINS       | bool      | True            | Randomizes initial Potts model spin states.                                |
| REP_START_TIME          | int       | 50000           | Start time for replication in simulation steps.                            |
| REP_TIME_DURATION       | int       | 50000           | Duration of the replication process in simulation steps.                   |
| REP_T_STD_FACTOR        | float     | 0.1             | Standard deviation factor for replication timing.                          |
| REP_SPEED_SCALE         | float     | 20              | Scaling factor for replication fork speed.                                 |
| REP_INIT_RATE_SCALE     | float     | 1.0             | Scaling factor for replication initiation rate.                            |
| N_STEPS                 | int       | 200000          | Total number of simulation steps.                                          |
| MC_STEP                 | int       | 200             | Number of steps per Monte Carlo iteration.                                 |
| BURNIN                  | int       | 1000            | Number of burn-in steps before data collection.                            |
| METHOD                  | str       | Annealing       | Simulation method (e.g., Metropolis, Annealing).                           |

### Stochastic Energy Coefficients
| Parameter Name         | Type      | Default Value   | Description                                                                 |
|-------------------------|-----------|-----------------|-----------------------------------------------------------------------------|
| FOLDING_COEFF           | float     | 1.0             | Coefficient controlling chromatin folding.                                 |
| FOLDING_COEFF2          | float     | 0.0             | Secondary coefficient for chromatin folding.                               |
| REP_COEFF               | float     | 1.0             | Coefficient for replication-related energy terms.                          |
| POTTS_INTERACT_COEFF    | float     | 1.0             | Coefficient for Potts model interaction energy.                            |
| POTTS_FIELD_COEFF       | float     | 1.0             | Coefficient for Potts model field energy.                                  |
| CROSS_COEFF             | float     | 1.0             | Coefficient penalizing LEF crossing.                                       |
| BIND_COEFF              | float     | 1.0             | Coefficient for LEF binding energy.                                        |

### Annealing Parameters
| Parameter Name         | Type      | Default Value   | Description                                                                 |
|-------------------------|-----------|-----------------|-----------------------------------------------------------------------------|
| T_INIT                  | float     | 2.0             | Initial temperature for annealing.                                         |
| T_FINAL                 | float     | 1.0             | Final temperature for annealing.                                           |

### Molecular Dynamics
| Parameter Name         | Type      | Default Value   | Description                                                                 |
|-------------------------|-----------|-----------------|-----------------------------------------------------------------------------|
| INITIAL_STRUCTURE_TYPE  | str       | rw              | Type of initial structure (e.g., rw for random walk).                      |
| SIMULATION_TYPE         | str       | None            | Type of simulation to run (e.g., MD or EM).                                |
| INTGRATOR_TYPE          | str       | langevin        | Type of integrator for molecular dynamics.                                 |
| INTEGRATOR_STEP         | Quantity  | 10 femtosecond  | Time step for the molecular dynamics integrator.                           |
| FORCEFIELD_PATH         | str       | default_xml_path| Path to the force field XML file.                                          |
| EV_P                    | float     | 0.01            | Excluded volume parameter for molecular dynamics.                          |
| TOLERANCE               | float     | 1.0             | Tolerance for energy minimization.                                         |
| SIM_TEMP                | Quantity  | 310 kelvin      | Temperature for molecular dynamics simulation.                             |
| SIM_STEP                | int       | 10000           | Number of steps for molecular dynamics simulation.                         |

## Output and results

The output is organized in the following folders,

```
.
├── ensemble
│   ├── ensemble_100_BR.cif
...
├── LE_init_struct.cif
├── metadata
│   ├── asphs.npy
│   ├── binder_cumulant.npy
│   ├── Bs.npy
│   ├── cluster_order.npy
│   ├── CNs.npy
│   ├── convex_hull_volume.npy
│   ├── eeds.npy
│   ├── entropy.npy
│   ├── Es.npy
│   ├── Es_potts.npy
│   ├── fractal_dims.npy
│   ├── Fs.npy
│   ├── gdfs.npy
...
├── minimized_model.cif
└── plots
    ├── asphericity.svg
    ├── autoc.pdf
    ├── autoc.png
    ├── autoc.svg
    ├── bind_energy.pdf
    ├── bind_energy.png
    ├── bind_energy.svg
    ├── binder_cumulant.png
...
```

Therefore:
* In the first directory `ensemble` you can find ensembles of 3D structures produced by RepliSage. The index indicates pseudo-time and the tag `BR`, `R` or `AR` it has to do about the phase of cell cycle. `BR` means before replication (G1 phase), `R` during replication (S phase), and `AR` after replication (G2/M phase).
* In the `metadata` you can find a lot of arrays that are produced during simulation. There are all the energy factors, and the metrics of the 3D structure of polymer, as a functions of time. Moreover, there are saved the parameters that were used for input and the some files that are appropriate for visualization in UCSF chimera (for example `psf` and `dcd`).
* In `plots` there are output plots. Some of the most important ones are: the digram of the trajectories of LEFs, the diagram of potts states, the average length during time. It is also important to track the autocorrelations of the MCMC algorithm. Heatmaps are produced as well for comparisons with experimental data.


## Citation

Please cite the preprint of our paper in case of usage of this software

* S. Korsak et al, Chromatin as a Coevolutionary Graph: Modeling the Interplay of Replication with Chromatin Dynamics, bioRxiv, 2025-04-04
* D. J. Massey and A. Koren, “High-throughput analysis of single human cells reveals the complex nature of dna replication timing control,” Nature Communications, vol. 13, no. 1, p. 2402, 2022.
