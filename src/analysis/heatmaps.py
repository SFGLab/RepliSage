"""Contact/distance heatmaps from 3D structures."""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import distance
from ..io.structure_io import get_coordinates_mm, get_coordinates_pdb, get_coordinates_cif
from ..logger import log

def get_heatmap(mm_vec,save_path=None,th=1,save=False):
    '''
    It returns the corrdinate matrix V (N,3) of a .pdb file.
    The main problem of this function is that coordiantes are not always in 
    the same column position of a .pdb file. Do changes appropriatelly,
    in case that the data aren't stored correctly.
    
    Input:
    file (Openmm Qunatity): an OpenMM vector of the form 
    Quantity(value=[Vec3(x=0.16963918507099152, y=0.9815883636474609, z=-1.4776774644851685),
    Vec3(x=0.1548253297805786, y=0.9109517931938171, z=-1.4084612131118774),
    Vec3(x=0.14006929099559784, y=0.8403329849243164, z=-1.3392155170440674),
    Vec3(x=0.12535107135772705, y=0.7697405219078064, z=-1.269935131072998),
    ...,
    unit=nanometer)
    
    Output:
    H (np.array): a heatmap of the 3D structure.
    '''
    V = get_coordinates_mm(mm_vec)
    mat = distance.cdist(V, V, 'euclidean') # this is the way \--/
    mat = 1/(mat+1)

    if save_path!=None:
        figure(figsize=(25, 20))
        plt.imshow(mat,cmap="Reds")
        if save: plt.savefig(save_path,format='svg',dpi=500)
        plt.close()
        if save: np.save(save_path.replace("svg", "npy"),mat)
    return mat

def heats_to_prob(heats,path,burnin,q=0.15):
    q_dist = np.quantile(np.array(heats),1-q)
    prob_mat = np.zeros(heats[0].shape)

    norm = np.zeros(len(heats[0]))
    for heat in heats:
        for i in range(len(heats[0])):
            norm[i]+=(np.average(np.diagonal(heat,offset=i))+np.average(np.diagonal(heat,offset=-i)))/2
    norm = norm/len(heats)

    for i in range(burnin,len(heats)):
        prob_mat[heats[i]>q_dist] += 1
    
    prob_mat = prob_mat/len(heats)
    for i in range(len(prob_mat)):
        for j in range(0,len(prob_mat)-i):
            prob_mat[i,i+j]=prob_mat[i,i+j]/norm[j]
            prob_mat[i+j,i]=prob_mat[i+j,i]/norm[j]
    
    figure(figsize=(10, 10))
    plt.imshow(prob_mat,cmap="Reds")
    plt.colorbar()
    plt.title(f'Normalized Probability distribution that distance < {q} quantile', fontsize=13)
    plt.savefig(path,format='png',dpi=500)

def binned_distance_matrix(idx,folder_name,input=None,th=23):
    '''
    This function calculates the mean distance through models, between two specific beads.
    We do that for all the possible beads and we take a matrix/heatmap.
    This one may take some hours for many beads or models.
    This works for .pdb files.
    '''
    
    V = get_coordinates_pdb(folder_name+f'/pdbs/SM{idx}.pdb')
    mat = distance.cdist(V, V, 'euclidean') # this is the way \--/ 

    figure(figsize=(25, 20))
    plt.imshow(mat,cmap=LinearSegmentedColormap.from_list("bright_red",[(1,0,0),(1,1,1)]), vmin=0, vmax=th)
    plt.savefig(folder_name+f'/heatmaps/SM_bindist_heatmap_idx{idx}.png',format='png',dpi=500)
    plt.close()

    np.save(folder_name+f'/heatmaps/binned_dist_matrix_idx{idx}.npy',mat)
    
    return mat

def average_binned_distance_matrix(folder_name,N_steps,step,burnin,th1=0,th2=23):
    '''
    This function calculates the mean distance through models, between two specific beads.
    We do that for all the possible beads and we take a matrix/heatmap.
    This one may take some hours for many beads or models.
    smoothing (str): You can choose between 'Nearest Neighbour', 'bilinear', 'hanning', 'bicubic'.
    '''
    sum_mat = 0
    for i in log.progress(range(0,N_steps,step), desc='Avg distance matrix'):
        V = get_coordinates_pdb(folder_name+f'/pdbs/SM{i}.pdb')
        if i >= burnin*step:
            sum_mat += distance.cdist(V, V, 'euclidean') # this is the way \--/ 
    new_N = N_steps//step
    avg_mat = sum_mat/new_N
    
    figure(figsize=(25, 20))
    plt.imshow(avg_mat,cmap=LinearSegmentedColormap.from_list("bright_red",[(1,0,0),(1,1,1)]), vmin=th1, vmax=th2)
    plt.savefig(folder_name+f'/plots/SM_avg_bindist_heatmap.png',format='png',dpi=500)
    np.save(folder_name+'/plots/average_binned_dist_matrix.npy',avg_mat)

    return avg_mat

def get_avg_heatmap(path, N1, N2, method='mean_heatmap', epsilon=1.0, n_chains=2):
    """
    Computes the mean inverse distance heatmap for a set of 3D structures.

    Parameters:
    -----------
    n_chains : int
        Number of polymer chains in each structure (2 with replication, 1 without).
    path : str
        Path to directory containing CIF structures in /ensemble/.
    N1 : int
        Starting index (inclusive) of structure files to include.
    N2 : int
        Ending index (exclusive) of structure files to include.
    method : str
        Either 'mean_heatmap' (default) or 'mean_structure':
            - 'mean_heatmap': computes the average inverse distance matrix over all structures.
              Recommended for comparison with experimental data (e.g. Hi-C, FISH).
            - 'mean_structure': computes the inverse distance matrix of the mean 3D structure.
              Faster but less biologically realistic.
    epsilon : float
        Small value to avoid division by zero in inverse distance.

    Returns:
    --------
    mean_heat : np.ndarray
        Averaged inverse distance heatmap.
    """

    os.makedirs(path + '/metadata/structural_metrics', exist_ok=True)
    os.makedirs(path + '/plots/structural_metrics', exist_ok=True)

    first_file = glob.glob(path + f'/ensemble/ensemble_{N1}_*.cif')
    if not first_file:
        log.info(f"No starting structure found at ensemble_{N1}_*.cif")
        return None

    V_init = get_coordinates_cif(first_file[0])
    N_beads = len(V_init) // n_chains  # 2 chains if replication was modelled, else 1

    structure_sum = np.zeros((N_beads, 3), dtype=np.float64)
    heatmap_sum = np.zeros((N_beads, N_beads), dtype=np.float64)
    count = 0

    for i in log.progress(range(N1, N2), desc='Averaging heatmap'):
        file_list = glob.glob(path + f'/ensemble/ensemble_{i}_*.cif')
        if not file_list:
            log.info(f"Warning: No file for index {i}")
            continue
        try:
            V = get_coordinates_cif(file_list[0])[:N_beads]
            structure_sum += V
            count += 1

            if method == 'mean_heatmap':
                D = distance.cdist(V, V, 'euclidean')
                heatmap_sum += 1.0 / (D + epsilon)

        except Exception as e:
            log.info(f"Error reading {file_list[0]}: {e}")
            continue

    if count == 0:
        log.info("No valid structures loaded.")
        return None

    mean_V = structure_sum / count
    np.save(path + f'/metadata/structural_metrics/mean_structure_{N1}_{N2}.npy', mean_V)

    if method == 'mean_structure':
        D_mean = distance.cdist(mean_V, mean_V, 'euclidean')
        mean_heat = 1.0 / (D_mean + epsilon)
    else:
        mean_heat = heatmap_sum / count

    np.save(path + f'/metadata/structural_metrics/mean_inv_heatmap_{N1}_{N2}.npy', mean_heat)

    try:
        from ..plots.structure_figs import fig_heatmap
        fig_heatmap(mean_heat, path, f'mean_inv_heatmap_{N1}_{N2}', f'md_heat_{N1}_{N2}', 'Contact map')
    except Exception as e:
        log.info(f"Plotting failed: {e}")

    return mean_heat
