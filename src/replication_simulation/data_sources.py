"""Import replication timing (single-cell parquet, .txt, bigWig) -> avg_fx on simulation beads."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ..utils.arrays import reshape_array, min_max_normalize
from ..logger import log

class SourceMixin:
    """Methods that turn raw replication timing data into the signal avg_fx(x)."""

    def preprocess_bigwig(self, bin_size=1000, sigma=200.0, log_norm=True, viz=False):
        """
        Load a BigWig signal, bin it, log-transform it, smooth it.

        Parameters
        ----------
        bin_size : int, default=1000
            Size of genomic bins in base pairs.
            The raw BigWig signal is averaged within consecutive bins of this size,
            reducing base-pair resolution to a coarse-grained signal.

        sigma : float, default=100
            Standard deviation of the Gaussian kernel, expressed in number of bins.
        viz : bool, default=False
            If True, plot the smoothed signal in a professional, wide-format figure.
        """
        # Extract base-pair resolution values
        if self.is_region:
            values = self.data.values(self.chrom, self.coords[0], self.coords[1])
        else:
            values = self.data.values(self.chrom)
        self.data.close()

        # Convert to NumPy and clean NaNs
        signal = np.asarray(values, dtype=float)
        signal = np.nan_to_num(signal)

        # Bin the signal (mean per bin)
        n_bins = len(signal) // bin_size
        signal_binned = (
            signal[:n_bins * bin_size]
            .reshape(n_bins, bin_size)
            .mean(axis=1)
        )
        # Log-transform
        log_signal = np.log(signal_binned+1) if log_norm else signal_binned
        
        # Smooth
        signal_smooth = gaussian_filter1d(log_signal, sigma=sigma) if sigma > 0 else log_signal

        # Reshape it appropriately
        self.avg_fx = reshape_array(signal_smooth, self.L)
        self.avg_fx = min_max_normalize(self.avg_fx)
        log.info(f"Succefully preprocessed BigWig data for {self.chrom}.")
        
        if viz:
            plt.figure(figsize=(12, 3))  # wide, paper-style
            plt.plot(self.avg_fx, color="#1f77b4", linewidth=1.5)
            plt.xlabel("Bin index", fontsize=10)
            plt.ylabel("Signal intensity", fontsize=10)
            plt.title(f"{self.chrom} smoothed signal (bin={bin_size}, sigma={sigma})", fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.close()

    def process_txt(self):
        self.data = self.data[self.data[0] == self.chrom].reset_index(drop=True)
        self.gen_windows = self.data[1].values
        self.avg_fx = self.data[2].values
        self.avg_fx = (self.avg_fx - np.min(self.avg_fx)) / (np.max(self.avg_fx) - np.min(self.avg_fx))
        if self.is_region:
            start, end = self.coords[0], self.coords[1]
            self.avg_fx = self.avg_fx[(self.gen_windows >= start) & (self.gen_windows <= end)]
        self.avg_fx = reshape_array(self.avg_fx, self.L)
        self.avg_fx = min_max_normalize(self.avg_fx)
        log.info(f"Succefully preprocessed .txt data for {self.chrom}.")

    def process_matrix(self):
        '''
        Import and rescale the matrices of single cell replication timing.
        '''
        min_value = np.min(np.nan_to_num(self.mat[self.mat>0]))
        self.mat = np.nan_to_num(self.mat,nan=min_value)
        self.mat = (self.mat-np.min(self.mat))/(np.max(self.mat)-np.min(self.mat))

    def compute_f(self):
        '''
        Compute the averages and standard deviations across the single cell replication matrix.
        Here we compute both averages over cell cycle time and over spartial dimensions.
        '''
        afx, sfx, aft, sft = np.average(self.mat,axis=0), np.std(self.mat,axis=0), np.average(self.mat,axis=1), np.std(self.mat,axis=1)
        min_avg, avg_std = np.min(afx[afx>0]), np.average(sfx)
        afx[afx<=0], sfx[sfx<=0] = min_avg, avg_std
        self.avg_fx = afx
        self.std_fx = sfx
        self.avg_ft = aft
        self.std_ft = sft
        if self.is_region:
            starts, ends = self.gen_windows[:, 0], self.gen_windows[:, 1]  # vector of bin starts and ends
            bin_size = starts[1] - starts[0]  # assume regular spacing
            first_bin_start = starts[0]
            last_bin_end = self.gen_windows[-1, 1]

            # compute mask
            mask = (starts >= self.coords[0]) & (starts < self.coords[1])
            self.avg_fx = self.avg_fx[mask]
            self.std_fx = self.std_fx[mask]
            
            # pad left if needed
            if self.coords[0] < first_bin_start:
                pad_bins = (first_bin_start - self.coords[0]) // bin_size
                self.avg_fx = np.concatenate([np.zeros(pad_bins), self.avg_fx])
                self.std_fx = np.concatenate([np.zeros(pad_bins), self.std_fx])

            # optionally pad right (if region extends beyond last bin)
            if self.coords[1] > last_bin_end:
                pad_bins = (self.coords[1] - last_bin_end) // bin_size
                self.avg_fx = np.concatenate([self.avg_fx, np.zeros(pad_bins)])
                self.std_fx = np.concatenate([self.std_fx, np.zeros(pad_bins)])

        self.avg_fx = reshape_array(self.avg_fx, self.L)
        self.std_fx = reshape_array(self.std_fx, self.L)
        self.avg_ft = reshape_array(self.avg_ft, self.T)
        self.std_ft = reshape_array(self.std_ft, self.T)

        # Plot avg_fx
        if self.out_path is not None:
            plt.figure(figsize=(10, 4))

            x = np.arange(len(self.avg_fx))
            y = self.avg_fx

            plt.plot(x, y, color='black', lw=1.5, label="Replication Signal")

            # Fill A compartment (early)
            plt.fill_between(x, y, where=(y > 0.5), color='red', alpha=0.3, label='A compartment (early)')

            # Fill B compartment (late)
            plt.fill_between(x, y, where=(y <= 0.5), color='blue', alpha=0.3, label='B compartment (late)')

            plt.title(f"Replication signal {self.chrom}: {self.region_label}", fontsize=14)
            plt.xlabel("Genomic bin", fontsize=12)
            plt.ylabel("Normalized Replication Fraction", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(f"{self.out_path}/avg_fx_{self.region_label.replace(' ','')}.png", dpi=150)
            plt.close()
