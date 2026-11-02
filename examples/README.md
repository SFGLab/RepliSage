# Examples

Small recipes that you can copy. All commands run from the repository root (`python src/run.py ...`) or after installation (`replisage ...`). Replace the data paths with your own.

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

## Annotated config files

| File | Use case |
|---|---|
| [`loops_only.ini`](loops_only.ini) | Loop extrusion without replication |
| [`replication_mc.ini`](replication_mc.ini) | Replication and compartments, Monte Carlo only |
| [`replication_3d.ini`](replication_3d.ini) | Full cell cycle with 3D structures |
| [`replication_stress.ini`](replication_stress.ini) | Slow forks, extra origins |
| [`two_lef_families.ini`](two_lef_families.ini) | Two kinds of extruders with different folding strength |
| [`production_replicas.ini`](production_replicas.ini) | Long runs with several replicas |

Edit the data paths at the top of each file before running.
