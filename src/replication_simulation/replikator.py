"""Replikator: replication timing data -> initiation rate -> fork simulation.

Example (needs the bundled data):
    python -m src.replication_simulation.replikator
"""
if __package__ in (None, ""):  # run directly (e.g. VS Code "Run Python File"): make `src` importable
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = "src.replication_simulation"

import re
import numpy as np
import pandas as pd
from .data_sources import SourceMixin
from .timing_stats import StatsMixin
from .init_rate import InitRateMixin
from .timing_io import sanitize_chr_dataframe
from .simulator_class import ReplicationSimulator
from ..utils.arrays import min_max_normalize
from ..logger import log

class Replikator(SourceMixin, StatsMixin, InitRateMixin):
    def __init__(self, rept_data_path: str, sim_L: int, sim_T: int, chrom: str, coords=None, Tstd_factor=0.1,
                 speed_factor=20, sc=True, out_path=None, bw_bin_size=1000, bw_sigma=200.0, bw_log_norm=True):
        '''
        Initialization of the data preprocessing.
        ------------------------------------------
        rept_data_path: path of the replication timing data (parquet single-cell, .txt or bigWig).
        sim_L: number of beads of the simulation; sim_T: number of replication time points.
        chrom: chromosome of interest; coords: region [start, end] or None for the whole chromosome.
        bw_*: binning / smoothing / log-normalisation parameters of Repli-Seq bigWig files.
        '''
        self.out_path = out_path
        self.rept_data_path = rept_data_path
        self.chrom, self.sc = chrom, sc
        self.is_region = coords is not None and not any(c is None for c in coords)
        self.coords = np.array(coords) if self.is_region else None
        self.region_label = f"{self.coords[0]}-{self.coords[1]}" if self.is_region else "whole-chromosome"
        self.bw_bin_size, self.bw_sigma, self.bw_log_norm = bw_bin_size, bw_sigma, bw_log_norm
        digits = re.sub(r'\D', '', self.chrom)
        chrom_nr = int(digits) if digits else self.chrom.replace('chr', '')
        if sc:
            self.data = pd.read_parquet(rept_data_path)
            sel = self.data['chromosome'].astype(str) == str(chrom_nr)
            self.gen_windows = self.data[sel][['start', 'end', 'center']].values
            self.chrom_size = int(np.max(self.gen_windows))
            single_cell_cols = [col for col in self.data.columns if col.startswith('SC_')]
            self.mat = self.data[sel][single_cell_cols].T.values
        else:
            low = rept_data_path.lower()
            if low.endswith(".txt"):
                self.data = pd.read_csv(rept_data_path, sep='\t', header=None).fillna(0)
                self.data = sanitize_chr_dataframe(self.data)
                if not self.data[0].astype(str).str.startswith('chr').all():
                    self.data[0] = 'chr' + self.data[0].astype(str)
            elif low.endswith((".bw", ".bigwig")):
                import pyBigWig
                self.data = pyBigWig.open(rept_data_path)
        self.L, self.T = sim_L, sim_T
        self.sigma_t = self.T * Tstd_factor
        self.speed_factor = speed_factor

    def prepare_data(self):
        '''Prepare the data and compute the initiation rate (calls steps in the correct order).'''
        low = self.rept_data_path.lower()
        if self.sc:
            self.process_matrix(); self.compute_f()
        elif low.endswith(".txt"):
            self.process_txt()
        elif low.endswith((".bw", ".bigwig")):
            self.preprocess_bigwig(self.bw_bin_size, self.bw_sigma, self.bw_log_norm)
        else:
            raise ValueError("Unsupported file format. Please provide a .txt, .bigWig or single-cell .parquet file.")
        self.compute_peaks()
        self.compute_slopes()
        self.compute_init_rate()

    def run(self, scale=1):
        '''Run the replication simulation.'''
        self.prepare_data()
        repsim = ReplicationSimulator(self.L, self.T, scale * self.initiation_rate, self.speed_ratio, self.speed_avg)
        self.sim_f, l_forks, r_forks, T_final, rep_fract = repsim.run_simulator()
        if self.out_path is not None: repsim.visualize_simulation(self.out_path)
        return self.sim_f, l_forks, r_forks

    def calculate_ising_parameters(self):
        '''Compartmentalization data: link early/late replication timing with Potts field and states.'''
        magnetic_field = 2 * self.avg_fx - 1
        state = np.where(min_max_normalize(np.average(self.sim_f, axis=1), -1, 1) > 0, 1, -1)
        return np.array(magnetic_field, dtype=np.float64), np.array(state, dtype=np.int32)

if __name__ == "__main__":  # small demo: built-in single-cell data, chr14 region, 200 beads
    from ..config.arg_defs import DEFAULT_REPT
    from ..logger import setup_logger
    setup_logger(1)
    rep = Replikator(DEFAULT_REPT, 200, 1000, 'chr14', [80835000, 98674700], sc=True)
    f, _, _ = rep.run()
    log.table("Replication demo", [("beads x time", f.shape), ("mean replicated", f"{f.mean():.3f}")])
