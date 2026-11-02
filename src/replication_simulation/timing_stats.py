"""Peaks and fork-speed estimates from the replication timing curve."""
import numpy as np
from scipy.signal import find_peaks
from ..logger import log

class StatsMixin:
    """Extrema and slopes of the replication timing curve."""

    def compute_peaks(self,prominence=0.01):
        '''
        Here we compute peaks and dips of the replication timing curves.
        ----------------------------------------------------------------
        Input:
        prominence: it is the prominence parameter from the scipy function: find_peaks().
        '''
        self.peaks, _ = find_peaks(self.avg_fx,prominence=prominence)
        self.dips, _ = find_peaks(-self.avg_fx,prominence=prominence)

    def compute_slopes(self):
        '''
        Here the slopes between successive maxima of the replication curves are estimated.
        Slopes of replication timing curve should correlate with the speed of replication forks.
        '''
        extrema_indices = np.sort(np.concatenate((self.peaks, self.dips)))
        extrema_indices_sorted = np.sort(extrema_indices)
        log.info('Computing slopes of replication curves...')

        avg_slopes, std_slopes = np.zeros(self.L), np.zeros(self.L)
        for i, extr in enumerate(extrema_indices_sorted[:-1]):
            start_idx = extrema_indices_sorted[i]
            end_idx = extrema_indices_sorted[i + 1]
            delta_fx = self.avg_fx[end_idx] - self.avg_fx[start_idx]
            delta_x = (end_idx - start_idx)
            # Avoid division by zero or near-zero to prevent infinite slopes
            if np.isclose(delta_fx, 0):
                segment_slope = 0.0
                sigma_slope = 0.0
            else:
                segment_slope = delta_x / delta_fx
                sigma_slope = delta_x * np.sqrt(2 * (self.sigma_t / self.T) ** 2) / (delta_fx ** 2)
            avg_slopes[extr] = np.abs(segment_slope)
            std_slopes[extr] = sigma_slope
        self.speed_avg = self.speed_factor * np.average(avg_slopes)
        self.speed_std = self.speed_factor * np.average(std_slopes)
        self.speed_ratio = self.speed_std / self.speed_avg if self.speed_avg != 0 else 0
        
        log.info('Done!')
