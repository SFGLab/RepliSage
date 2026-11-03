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

The region is cut into `N_BEADS` equal pieces. For a region of $G$ base pairs, one bead spans $G / N_{\text{beads}}$ base pairs. An anchor is assigned to the bead at the middle of its interval.

## Binding potentials

For every bead $k = 1, \dots, N_{\text{beads}}$ pre-processing computes two numbers, the **left potential** $L(k)$ and the **right potential** $R(k)$. They tell how strongly bead $k$ attracts the left leg $m_i$ and the right leg $n_i$ of a loop extruding factor (LEF) $i$. They enter the binding energy of the [Monte Carlo model](mc_model.md):

$$\mathcal{H}_{\text{bind}} = \mathcal{C}_{\text{bind}} \sum_{i=1}^{N_{\text{lef}}} \left(L(m_i) + R(n_i)\right)$$

Since $\mathcal{C}_{\text{bind}} < 0$, a LEF with legs on beads with large $L$ and $R$ lowers the energy, which pulls loops towards the CTCF loops of your data. Only the shape of $L$ and $R$ matters, not the scale of $w_\lambda$, because $\mathcal{C}_{\text{bind}}$ divides by $\sum_k (L(k) + R(k))$.

**Building $L$ and $R$.** Start from $L(k) = R(k) = 0$ and go through the loops $\lambda = 1, 2, \dots$ of the BEDPE file. Each loop brings three ingredients:

* two **anchor beads** $a_\lambda$ and $b_\lambda$, the beads that hold the two ends of the loop;
* a **strength** $w_\lambda \ge 0$, the score in column 7: how strongly the experiment supports this loop (for example the number of ChIA-PET read pairs or a HiChIP score). A strong loop makes its anchors more attractive for LEF legs, and a weak loop only slightly. If your file has no meaningful score, give every loop the same value, for example 1. Loops with a missing or negative score are skipped;
* optionally, a **CTCF orientation** $o_c \in [0, 1]$ for each anchor $c$ (columns 8 and 9).

For each anchor bead $c \in \{a_\lambda, b_\lambda\}$ add:

* Without orientation information, the anchor attracts both legs equally:

$$L(c) \leftarrow L(c) + w_\lambda$$

$$R(c) \leftarrow R(c) + w_\lambda$$

* With the CTCF orientation $o_c \in [0, 1]$ from columns 8 and 9 (0 = forward only, 1 = reverse only):

$$L(c) \leftarrow L(c) + w_\lambda (1 - o_c)$$

$$R(c) \leftarrow R(c) + w_\lambda\, o_c$$

In words: a forward-facing CTCF site ($o_c = 0$) attracts the left leg of a LEF, and a reverse-facing site ($o_c = 1$) attracts the right leg; values in between split the attraction. An anchor with a negative orientation (no motif) adds nothing. Anchors from different loops that fall on the same bead add up, so beads that anchor many strong loops get the deepest potential wells.

The number of CTCF sites is $N_{\text{CTCF}} = \max\left(\#\{k : L(k) > 0\},\ \#\{k : R(k) > 0\}\right)$. If you do not set `N_LEF`, RepliSage uses $N_{\text{lef}} = N_{\text{CTCF}} / 2$ extruders, one per two CTCF sites. (Earlier versions used $2 N_{\text{CTCF}}$; with many CTCF sites per bead that can exceed the number of beads, and then no loops form, so set `N_LEF` below the number of beads, at most a quarter of it; at or above half of it is an error.)

## Parameters

| Name | Where | Meaning | Default |
|---|---|---|---|
| `BEDPE_PATH` | `--bedpe-file` | Loops file | none (required) |
| `CHROM`, `REGION_START`, `REGION_END` | `--region` | Region to model | whole chromosome |
| `N_BEADS` | `--n-beads` | Number of beads | region length / 7 kb (max 10,000 beads) |

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
