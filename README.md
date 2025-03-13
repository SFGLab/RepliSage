# RepliSage
A simulation software for modeling the motion of cohesin during the replication process. This tool explores the interaction between cohesin, or more generally loop extrusion factors (LEFs), with replication forks and chromatin compartmentalization. It employs a sophisticated force-field that integrates MCMC Metropolis and molecular dynamics methodologies. The output is a 3D chromatin trajectory, providing a dynamic visualization of DNA replication and the formation of two identical copies.

## Simulation pipeline
RepliSage is composed by three distinct parts:

1. **Replication simulation (Replikator.py):** This is a simplistic *Monte Carlo simulation*, where we import single cell replication timing data to model replication. The average replication timing curves can show us the percentage of cells that have been replicated, and from them we estimate the *initiation rate* $I(x,t)$ which represents the probability fires at time $t$ in loci $x$. Then we employ the Monte Carlo simulation where in each step an origin fires with a probability derived from the initiation rate. When an origin fires, the replication fork start propagating bi-directionally with velocity $v$. The vlocity's mean and standard deviation is derived by calculating the slopes of consecutive optima of the averaged replication timing curve. Replikator outputs the trajectories of the replication forks.
2. **Stochastic simulation that models the interplay of loop extrusion with other factors:** In this part we import the previously produced trajectories of replication forks and we model them as moving barriers for the loop extrusion factors. In this way this simulation is composed by three distinct parts:
    * *Loop Extrusion*: We use the energy landscape of LoopSage for that. We assume that there are two basic players: LEFs which follow a random difussive motion in 1D and CTCF whose locations are dervied from ChIA-PET data, which are filtered where is a CTCF motif in their sequence. Therefore, assuming that we have $N_{\text{lef}}$ LEFs with two degrees of freedom $(m_i,n_i)$ and epigenetic color states $s_i$ for each monomer, we can write down the following formula,  $$E_{\text{le}} =  c_{\text{fold}}\sum_{i=1}^{N_{\text{coh}}}\log(n_i-m_i)+c_{\text{cross}}\sum_{i,j}K(m_i,n_i;m_j,n_j)+c_{\text{bind}}\sum_{i=1}^{N_{\text{coh}}}\left(L(m_i)+R(n_i)\right).$$ In this equation the first term models the folding of chromatin (how fast a loop extrudes, the higher $f$ the more tendency to extrude), the second term is a penalty for cohesin crossing, and the last one minimizes the energy when a LEF encounters a CTCF.
    * *Compartmentalization*: It is modelled by using a five stats ($s_i\in[-2,-1,0,1,2]$) Potts model $$E_{\text{potts}} = C_{p,1} \sum_{k} \left(\dfrac{h_k + h_{t_{r} k}}{2} \right) s_k +C_{p,2}\sum_{i>j} J_{ij}| s_i - s_j |$$. The second term includes the interaction matrix $J_{ij}$, which is 1 when there is a LEF connecting $i$ with $j$ and 0 if not. The other term represents an epigeetic field. There is an averaged term $h_k$, which represents the averaged replication timing, and a time dependent term $h_{t_{r} k}$ which represents the spread of the epigenetic state due to a single replication fork.
    * *Replication*: This term models the interaction between the LEFs and replication forks, $$E_{\text{rep}}=C_{\text{rep}}\sum_{i=1}^{N_{lef}} \mathcal{R}(m_i,n_i ;f_{\text{rep}})$$ and in general penalizes inapropriate configurations between LEFs and replication forks (read the paper).

Therefore, the stochastic simulation integrates the sum of these energies $E = E_{le}+E_{rep}+E_{potts}$ and uses MCMC Metropolis method.

  3. **Molecular Dynamics:** This parts takes as input the states produced by the stochastic simulation and outputs 3D structures by using a potential in OpenMM. The molecular modeling approach assumes two molecular chains, each consisting of $N_{\text{beads}}$ monomers, where $N_{\text{beads}}$ reflects the granularity of the stochastic simulation. The total potential governing the system is expressed as: $$U = U_{\text{bk}} + U_{\text{le}}(t) + U_{\text{rep}}(t) + U_{\text{block}}(t)$$, where each term corresponds to a specific contribution. The backbone potential ($U_{\text{bk}}$) includes strong covalent bonds between consecutive beads, angular forces, and excluded volume effects to maintain chain integrity. The loop-formation potential ($U_{\text{le}}$) is a time-dependent term introducing harmonic bonds to model loop formation. These bonds are weaker than the backbone interactions and act between dynamically changing pairs of beads, $m_i(t)$ and $n_i(t)$. The last term models compartmentalization with block-copolymer potential.

For more details of the implementation, we suggest to our users to read the method paper of RepliSage.

## Installation
It is needed to have at least python 3.10 and run,

```
pip install -r requirements.txt
```

## How to use?
The usage is very simple. To run this model you need to specify the parameters and the input data. Then RepliSage can do everything for you. 

Note that to run replisage it is needed to have `GM12878_single_cell_data_hg37.mat` data, for single-cell replication timing. These data are not produced by our laboratory, but they are alredy published in the paper,

* D. J. Massey and A. Koren, “High-throughput analysis of single human cells reveals the complex nature of dna replication timing control,” Nature Communications, vol. 13, no. 1, p. 2402, 2022.

Do not forget to cite them in case you use this software. We have uploaded the input data used in the paper here: https://drive.google.com/drive/folders/1PLA147eiHenOw_VojznnlC_ZywhLzGWx?usp=sharing.

```python
def main():
    # Set parameters
    N_beads = None
    N_steps, MC_step, burnin, T, T_min, t_rep, rep_duration = int(4e4), int(5e2), int(1e3), 2.0, 0.0, int(1e4), int(2e4)
    f, b, kappa, c_rep = 1.0, 0.5, 1.0, 2.0
    c_ising1, c_ising2 = 1.0, 4.0
    mode, rw, random_spin_state = 'Metropolis', True, True

    # Define data and coordinates
    region, chrom =  [178421513, 182421513], 'chr1'
    bedpe_file = '/home/skorsak/Data/method_paper_data/ENCSR184YZV_CTCF_ChIAPET/LHG0052H_loops_cleaned_th10_2.bedpe'
    rept_path = '/home/skorsak/Data/Replication/sc_timing/GM12878_single_cell_data_hg37.mat'
    out_path = '../output'
    
    # Run simulation
    sim = StochasticSimulation(N_beads,chrom,region, bedpe_file, out_path, None, rept_path, t_rep, rep_duration)
    sim.run_stochastic_simulation(N_steps, MC_step, burnin, T, T_min, f, b, kappa, c_rep, c_ising1, c_ising2, mode, rw, random_spin_state)
    sim.show_plots()
    sim.run_openmm('OpenCL')
```

## Citation
