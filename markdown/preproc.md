# Pre-processing

Pre-processing turns a list of loops into the quantities the Monte Carlo needs: how strongly each bead attracts a loop extruder from the left and from the right.

## Input: a BEDPE file

A tab-separated file, one loop per line:

| Column | Content |
|---|---|
| 1 | Chromosome of the first anchor |
| 2, 3 | Start and end of the first anchor |
| 4 | Chromosome of the second anchor |
| 5, 6 | Start and end of the second anchor |
| 7 | Loop strength (for example ChIA-PET or HiChIP score); negative values are ignored |
| 8, 9 | Optional CTCF orientation of each anchor, between 0 and 1 (0 = forward only, 1 = reverse only, negative = no motif) |

Only loops that lie fully inside the chosen region and chromosome are kept. If none remain, RepliSage stops with a clear message.

## From base pairs to beads

The region is cut into `N_BEADS` equal pieces. With region length $\ell$, one bead spans $\ell / N$ base pairs. An anchor is assigned to the bead at the middle of its interval.

## Binding potentials

For every loop with strength $w$ and anchors on beads $x$ and $y$, the left potential $L$ and the right potential $R$ receive contributions at the two anchor beads.

Without orientation information:

$$\begin{aligned} L_x &\leftarrow L_x + w, & R_x &\leftarrow R_x + w \\ L_y &\leftarrow L_y + w, & R_y &\leftarrow R_y + w \end{aligned}$$

With orientation $o_x, o_y$ in columns 8 and 9:

$$L_x \leftarrow L_x + w\,(1 - o_x), \qquad R_x \leftarrow R_x + w\,o_x$$

and the same for $y$. In words: a forward-facing CTCF site attracts the left leg of an extruder, a reverse-facing one attracts the right leg.

The number of CTCF sites $N_{CTCF}$ is the larger of the numbers of non-zero entries of $L$ and $R$. If you do not set `N_LEF`, RepliSage uses $2 N_{CTCF}$ extruders.

## Parameters

| Name | Where | Meaning | Default |
|---|---|---|---|
| `BEDPE_PATH` | `--bedpe-file` | Loops file | none (required) |
| `CHROM`, `REGION_START`, `REGION_END` | `--region` | Region to model | whole chromosome |
| `N_BEADS` | `--n-beads` | Number of beads | region length / 2 kb |

Parameters for reading replication timing (`BIGWIG_BIN_SIZE`, `BIGWIG_SIGMA`, `BIGWIG_LOG_NORM`, `REP_SIM_T`) are in the [replication model](replication_model.md) and the [config page](config.md).

## Output

The potentials $L$ and $R$ are plotted in the report (section "Input") and used directly by the [Monte Carlo](mc_model.md).

## In the code

The stage is called from `run_preprocessing(ctx)` in - [`src/pipeline/preproc_stage.py`](https://github.com/SFGLab/RepliSage/blob/main/src/pipeline/preproc_stage.py): stage entry point, returns `(L, R, N_CTCF)`

| Function or class | What it does |
|---|---|
| `preprocessing(bedpe_file, region, chrom, N_beads)` | reads the BEDPE file, bins it on the bead grid and returns the left and right CTCF binding potentials `L`, `R` and the number of CTCF sites |
| `binding_vectors_from_bedpe(bedpe_file, N_beads, region, chrom, normalization, viz)` | does the binning; `normalization` rescales the potentials, `viz` draws the input figure |
| `loop_stats(L, R)` | rows of the loop-input statistics table in the report |

Related arguments (see [Configuration](config.md)): `--bedpe-file` (`BEDPE_PATH`), `--region` (`CHROM`, `REGION_START`, `REGION_END`), `--n-beads` (`N_BEADS`).
