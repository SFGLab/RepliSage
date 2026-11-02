"""Loop (BEDPE) preprocessing: CTCF binding potentials."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..logger import log

# NOTE: the dense N x N coupling matrix J of the old version was never used here
# (it was always zero); the Potts couplings are now handled sparsely in MC/.
def binding_vectors_from_bedpe(bedpe_file,N_beads,region,chrom,normalization=False,viz=False):
    '''
    Definition of left and right CTCF binding potential.

    Input:
    bedpe_file (str): path with bepde file with loops
    region (list): a list with two integers [start,end], which represent the start and end point of the region of interest.
    chrom (str): chromosome of interest.
    normalization (bool): in case that it is needed to normalize to numpy arrays that represent left and right CTCF binding potential.
    viz (bool): If True is vizualizes the distribution of distances of loops from the diagonal and the binding potentials as functions of simulated polymer distance.

    Output:
    L (numpy array): left CTCF binding potential.
    R (numpy array): right CTCF binding potential.
    '''
    # Read file and select the region of interest
    df = pd.read_csv(bedpe_file,sep='\t',header=None)
    df = df[(df[1]>=region[0])&(df[2]>=region[0])&(df[4]>=region[0])&(df[5]>=region[0])&(df[5]<region[1])&(df[4]<region[1])&(df[1]<region[1])&(df[2]<region[1])&(df[0]==chrom)].reset_index(drop=True)
    if df.empty:
        raise ValueError("The selected region or chromosome is invalid: the final set does not contain any loops.")

    # Convert hic coords into simulation beads
    resolution = (region[1]-region[0])//N_beads
    df[1], df[2], df[4], df[5] = (df[1]-region[0])//resolution, (df[2]-region[0])//resolution, (df[4]-region[0])//resolution, (df[5]-region[0])//resolution
    
    # Check if columns 7 and 8 exist
    has_col_7_8 = df.shape[1] > 8
    if has_col_7_8:
        log.success("The input bedpe file contains CTCF orientation! It will run taking it into account.")
    else:
        log.alert("WARNING: The input bedpe file does not contain CTCF orientation and thus it is not taken into account. If you would like to add CTCF orientation check the documentation.")

    # Compute the matrix
    distances = list()
    L, R = np.zeros(N_beads, dtype=np.float64), np.zeros(N_beads, dtype=np.float64)
    for i in range(len(df)):
        x, y = min((df[1][i] + df[2][i]) // 2, N_beads - 1), min((df[4][i] + df[5][i]) // 2, N_beads - 1)
        distances.append(distance_point_line(x, y))
        strength = df[6][i]
        # If strength is None or negative, skip adding signal
        if pd.isna(strength) or strength is None or strength < 0:
            continue
        if has_col_7_8:
            if df[7][i] >= 0:
                L[x] += strength * (1 - df[7][i])
            if df[8][i] >= 0:
                L[y] += strength * (1 - df[8][i])
            if df[7][i] >= 0:
                R[x] += strength * df[7][i]
            if df[8][i] >= 0:
                R[y] += strength * df[8][i]
        else:
            L[x] += strength
            L[y] += strength
            R[x] += strength
            R[y] += strength
    
    # Normalize (if neccesary): it means to convert values to probabilities
    if normalization:
        L, R = L/np.sum(L), R/np.sum(R)

    if viz:
        fig, axs = plt.subplots(2, figsize=(15, 10))
        axs[0].plot(L,'g-')
        axs[0].set_ylabel('Left potential',fontsize=16)
        axs[1].plot(R,'r-')
        axs[1].set_ylabel('Right potential',fontsize=16)
        axs[1].set_xlabel('Genomic Distance (with simumation beads as a unit)',fontsize=16)
        plt.close(fig)

    return L, R

def distance_point_line(x0,y0,a=1,b=-1,c=0):
    return np.abs(a*x0+b*y0+c)/np.sqrt(a**2+b**2)


def preprocessing(bedpe_file: str, region: list, chrom: str, N_beads: int):
    """Return left/right CTCF binding potentials and the number of CTCF sites."""
    L, R = binding_vectors_from_bedpe(bedpe_file, N_beads, region, chrom, False, False)
    N_CTCF = int(np.max(np.array([np.count_nonzero(L), np.count_nonzero(R)])))
    return L, R, N_CTCF

def loop_stats(L, R):
    """Rows for a statistics table of the loop input."""
    return [("beads", len(L)), ("CTCF sites (L)", int(np.count_nonzero(L))),
            ("CTCF sites (R)", int(np.count_nonzero(R))), ("sum L", f"{L.sum():.2f}"), ("sum R", f"{R.sum():.2f}")]
