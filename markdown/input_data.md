# Input data

What RepliSage reads, and in which format. Only the CTCF file and a region are required; everything else is optional. How the data are turned into model quantities is explained in [Pre-processing](preproc.md) and the [replication model](replication_model.md).

| Input | Flag (config key) | Required | Used for |
|---|---|---|---|
| [CTCF sites](#ctcf-sites) | `--bedpe-file` (`BEDPE_PATH`) | yes | loop anchors and CTCF barriers |
| [Region](#region-and-beads) | `--region` (`CHROM`, `REGION_START`, `REGION_END`) | yes (or a whole chromosome) | the modelled stretch of DNA |
| [Replication timing](#replication-timing) | `--rt-file` (`REPT_PATH`) | no | S phase, forks, compartments |
| [Reference genome](#reference-genome-and-motif) | `--motif-genome` (`MOTIF_GENOME`) | no | CTCF orientation from DNA |
| [Hi-C map](#hi-c-map) | `--hic` (`HIC_FILE`) | no | validation of the model |

## Region and beads

The region is a chromosome and, optionally, a start and an end in **base pairs**, for example `--region chr14:80835000-98674700`. A bare chromosome (`--region chr14`) models the whole chromosome. The region is cut into `N_BEADS` equal beads (default: region length / 7 kb, at most 10,000); bead sizes outside 200 bp - 50 kb are an error.

## CTCF sites

A tab-separated file; the format follows the extension (`.gz` is accepted). Only rows fully inside the region are used. **CTCF orientation** is the probability $o$ that a site faces *reverse*: 0 = forward only, 1 = reverse only, negative = unknown.

**`.bedpe`: loops** (one per line)

| Column | Content |
|---|---|
| 1-3 | Chromosome, start, end of the first anchor |
| 4-6 | Chromosome, start, end of the second anchor |
| 7 | Loop strength $w$ (for example a ChIA-PET or HiChIP score); empty = 1, negative = 0 |
| 8, 9 | Optional: orientation $o$ of each anchor |

**`.bed` and `.narrowPeak`: ChIP-seq peaks** (no loops, so loop figures and `PREFERENTIAL_BINDING` are off)

| Format | Weight $w$ | Position | Orientation (optional) |
|---|---|---|---|
| `.bed` | column 5 (score; 1 if missing) | middle of the interval | column 7 = $o$, or the strand in column 6 (`+` = forward, `-` = reverse) |
| `.narrowPeak` | column 7 (signalValue; the score in column 5 if all zero) | summit (column 10), else the middle | extra column 11 (or 6) with the probability of *forward* orientation |

Without any orientation the model still runs, treating every site as a barrier for both legs; give a [reference genome](#reference-genome-and-motif) to read it from the sequence.

## Replication timing

Without timing there is no replication, no forks and one DNA chain. Give it with `--rt-file`:

| Format | Content | Notes |
|---|---|---|
| `.bigWig`, `.bw` | Bulk Repli-Seq signal (for example ENCODE) | averaged in bins, log-transformed, Gaussian-smoothed |
| `.txt` | Tab-separated table, value in the third column | one value per genomic window |
| `.parquet` | Single-cell timing matrix (columns starting with `SC_`) | the shipped file is `src/data/replication_timing_data.parquet` |
| `builtin` | The shipped single-cell GM12878 data | covers the chr14 test region |
| `none` | no replication | loop extrusion only |

In a terminal, with no timing given, RepliSage asks what to do (`ASK_RT = False` switches the question off). In every case the signal becomes $f_i \in [0, 1]$ along the beads: close to 1 replicates early, close to 0 late.

## Reference genome and motif

* **Genome** (`--motif-genome`): an uncompressed FASTA. An index `.fai` is created next to it if the folder is writable. Chromosome names with or without `chr` both work.
* **Motif** (`MOTIF_PFM`, optional): a position frequency matrix, 4 rows of counts for A, C, G, T of equal length (JASPAR `A [ ... ]` lines are accepted). Empty = the built-in JASPAR CTCF motif MA0139.1.

## Hi-C map

A `.hic` file (`--hic`) read with `hic-straw` (`pip install "pyRepliSage[hic]"`). The chromosome must be in the file. Bin size and normalisation are chosen automatically, with fallbacks (see [Pre-processing](preproc.md#hi-c-map-for-validation)); the region must fit in at most 6000 bins of the chosen size.

## What runs depends on the inputs

| You give | What runs |
|---|---|
| CTCF file and region | loop extrusion, then 3D structures (MD) and the report |
| + replication timing | + replication, sister chromatids, A/B epigenetic states |
| + reference genome | + CTCF orientation from the motif |
| + Hi-C map | + validation against Hi-C |

Switch the 3D part off with `--run-md false` (config: `SIMULATION_TYPE = none`).
