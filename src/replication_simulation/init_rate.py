"""Initiation rate function I(x,t)."""
import numpy as np
import matplotlib.pyplot as plt
from .initiation import get_p_vector
from ..plots.style import CMAP_SEQ
from ..logger import log

class InitRateMixin:
    """Estimation of the initiation rate."""

    def compute_init_rate(self,viz=False):
        '''
        Estimation of the initiation rate function I(x,t).
        '''
        self.initiation_rate = np.zeros((self.L, self.T))
        log.info('Computing initiation rate...')
        mus = self.T*(1-self.avg_fx)

        for ori in log.progress(range(len(mus)), desc='Initiation rate'):
            m = int(mus[ori])
            p_i = get_p_vector(self.T, m, sig=self.sigma_t)
            self.initiation_rate[ori, :] = p_i
        
        if self.out_path is not None:
            plt.figure(figsize=(15, 8), dpi=100)
            vmax = np.mean(self.initiation_rate) + np.std(self.initiation_rate)
            im = plt.imshow(self.initiation_rate.T, cmap=CMAP_SEQ, aspect='auto', vmax=vmax)
            
            plt.title('Initiation Rate Function', fontsize=24, pad=20)
            plt.xlabel('Genomic Distance', fontsize=20)
            plt.ylabel('Pseudo-Time', fontsize=20)
            
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            cbar = plt.colorbar(im)
            cbar.set_label("Initiation Rate", fontsize=16, labelpad=10)
            cbar.ax.tick_params(labelsize=12)

            plt.gca().invert_yaxis()

            plt.savefig(f"{self.out_path}/initiation_rate_function.png", dpi=100, bbox_inches='tight')
            plt.close()
        log.info('Computation Done! <3\n')
