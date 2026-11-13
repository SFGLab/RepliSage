# Examples

Small recipes that you can copy. All commands run from the repository root (`python -m src.run ...`) or after installation (`replisage ...`). Replace the data paths with your own.

The test region used in most examples is chr14:80835000-98674700, which is covered by the shipped single-cell replication timing (`--rt-file builtin`).

## Command-line recipes

**1. Loop extrusion only.** The fastest run; no replication, no 3D.

```bash
replisage --bedpe-file loops.bedpe --region chr14:80835000-98674700
```

**2. Add replication timing, Monte Carlo only.** Replication is simulated, loops and compartments are sampled across G1, S and G2/M.

```bash
replisage --bedpe-file loops.bedpe --region chr14:80835000-98674700 --rt-file builtin --run-md false
```

**3. Full 3D run on a GPU.** Two sister chromatids, viewer and movie are written.

```bash
replisage --bedpe-file loops.bedpe --region chr14:80835000-98674700 --rt-file GM12878.bigWig \
          --run-md true --platform CUDA
```

**4. Replication stress.**

```bash
replisage --bedpe-file loops.bedpe --region chr14:80835000-98674700 --rt-file builtin --with-stress
```

**5. Several replicas in parallel, for convergence checks.**

```bash
replisage --bedpe-file loops.bedpe --region chr14:80835000-98674700 --n-replicas 4 --threads 4 --seed 1
```

**6. A whole chromosome.**

```bash
replisage --bedpe-file loops.bedpe --region chr14 --n-beads 2000
```

**7. From a config file, with one flag changed.**

```bash
replisage -c examples/replication_3d.ini --seed 5 --name seed5
```

## Real data: GM12878 loops and Repli-seq

A CTCF ChIA-PET loop file and an ENCODE Repli-seq bigWig of GM12878, on a small 2 Mb region. The bigWig is log-transformed by default (`BIGWIG_LOG_NORM = True` in the config).

```bash
replisage --bedpe-file LHG0052H_loops_cleaned_th10_2.bedpe --rt-file ENCFF001GNK.bigWig \
          --region chr14:90000000-92000000 --n-beads 200 --run-md false --name gm12878_small
```

## Annotated config files

| File | Use case |
|---|---|
| [`loops_only.ini`](https://github.com/SFGLab/RepliSage/blob/main/examples/loops_only.ini) | Loop extrusion without replication |
| [`replication_mc.ini`](https://github.com/SFGLab/RepliSage/blob/main/examples/replication_mc.ini) | Replication and compartments, Monte Carlo only |
| [`replication_3d.ini`](https://github.com/SFGLab/RepliSage/blob/main/examples/replication_3d.ini) | Full cell cycle with 3D structures |
| [`replication_stress.ini`](https://github.com/SFGLab/RepliSage/blob/main/examples/replication_stress.ini) | Slow forks, extra origins |
| [`two_lef_families.ini`](https://github.com/SFGLab/RepliSage/blob/main/examples/two_lef_families.ini) | Two kinds of extruders with different folding strength |
| [`production_replicas.ini`](https://github.com/SFGLab/RepliSage/blob/main/examples/production_replicas.ini) | Long runs with several replicas |

Edit the data paths at the top of each file before running.

## Only a config file

If the file defines the input, `-c` is all you need:

```ini
[input]
bedpe_path = loops.bedpe
chrom = chr14
region_start = 80835000
region_end = 83835000
rept_path = builtin          ; a file, builtin or none
[mc]
preferential_binding = True
[md]
simulation_type = MD         ; none = no 3D
```

```bash
replisage -c my.ini
```

## A 3 Mb region with 3D structures

Beads are set automatically (7 kb per bead gives about 430 beads):

```bash
replisage --bedpe-file loops.bedpe --region chr14:80835000-83835000 --rt-file builtin --run-md true --name chr14_3Mb_md
```

## Writing a config file

Start small: the CTCF file and the region, plus whatever you want to change.

```ini
; my_config.ini: loops + replication + 3D structures for a 17.8 Mb region
[general]
out_path = results
run_name = chr14_demo
n_threads = 4
seed = 7

[input]
bedpe_path   = loops.bedpe
rept_path    = GM12878_timing.bigWig    ; a file, builtin or none
chrom        = chr14
region_start = 80835000
region_end   = 98674700
n_beads      = 1000                     ; empty = 7 kb per bead (max 10,000)

[replication]
rep_start_time    = 50000               ; MC step where S phase starts
rep_time_duration = 50000               ; length of S phase in MC steps

[mc]
n_steps = 150000
mc_step = 200                           ; record every 200th step
burnin  = 5000                          ; steps discarded at the start
t_mc    = 1.5                           ; fixed temperature, no annealing

[md]
simulation_type = MD                    ; none = no 3D
platform = CUDA
```

Keep `t_mc` in the critical region: too low and loops freeze, too high and they fall off; see [the Hamiltonian idea](hamiltonian.md#temperature-in-replisage). Run with `replisage -c my_config.ini -v` and open `config_used.ini` in the run folder to confirm your changes were read.

Common mistakes: a typo in a key (ignored with a warning), coordinates in kb instead of bp, a region too small for `n_beads` (bead size should be a few kb or more), and `rep_time_duration` not larger than `rep_sim_t` or an S phase that does not fit inside `n_steps`.
