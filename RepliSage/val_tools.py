import os
import re
import numpy as np
import hicstraw as hc
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy import ndimage
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from matplotlib.pyplot import figure
from matplotlib_venn import venn2
from tqdm import tqdm
from utils import *
from common import *

epsilon = np.finfo(float).eps

def get_heatmap(V,viz=False,save_path=None,th=1,duplicated_chain=False):
    '''
    It returns the corrdinate matrix V (N,3) of a structure.
    ...,
    Input:
    V (np.array): (N,3) matrix with the coordinates of each atom.
    
    Output:
    H (np.array): a heatmap of the 3D structure.
    '''
    if duplicated_chain: V = V[:len(V)//2]
    mat = distance.cdist(V, V, 'euclidean') # this is the way \--/
    mat = 1/(mat+1)

    if viz:
        figure(figsize=(25, 20))
        plt.imshow(mat,cmap="Reds",vmax=np.average(mat)+3*np.std(mat))
        if save_path!=None: plt.savefig(save_path,format='svg',dpi=500)
        if save_path!=None: np.save(save_path.replace("svg", "npy"),mat)
        plt.show()
    return mat

def compute_averaged_heatmap(ensembles_path,viz=False,save_path=None,duplicated_chain=False):
    '''
    It needs the directory with cif ensembles as input and it outputs the
    averaged distance heatmap of the ensemble of structures.
    '''
    cif_files = list_files_in_directory(ensembles_path)
    N = len(cif_files)
    heat = 0

    print('Computing heatmaps of the ensemble...')
    for cif in tqdm(cif_files):
        V = get_coordinates_cif(ensembles_path+'/'+cif)
        if duplicated_chain: V = V[:len(V)//2]
        heat += get_heatmap(V)
    avg_heat = heat/N
    print('Done')

    if viz:
        figure(figsize=(25, 20))
        plt.imshow(avg_heat,cmap="Reds",vmax=np.average(avg_heat)+3*np.std(avg_heat))
        plt.colorbar()
        if save_path!=None: plt.savefig(save_path,format='svg',dpi=500)
        if save_path!=None: np.save(save_path.replace("svg", "npy"),avg_heat)
        plt.show()
    return avg_heat

def compute_experimental_heatmap(hic_file,chrom,region,resolution=5000,viz=False,save_path=None):
    '''
    Imports the path of a .hic file and exports the heatmap as a numpy array.
    '''
    hic = hc.HiCFile(hic_file)
    hic_mat_obj = hic.getMatrixZoomData(chrom, chrom, "observed", "NONE", "BP", resolution)
    numpy_hic = hic_mat_obj.getRecordsAsMatrix(region[0], region[1], region[0], region[1])
    if viz:
        figure(figsize=(25, 20))
        plt.imshow(numpy_hic,cmap="Reds",vmax=np.average(numpy_hic)+3*np.std(numpy_hic))
        plt.colorbar()
        if save_path!=None: plt.savefig(save_path,format='svg',dpi=500)
        if save_path!=None: np.save(save_path.replace("svg", "npy"),numpy_hic)
    return numpy_hic

def rescale_matrix(matrix, target_size):
    # Rescale or coarse-grain the matrix to the target size (nxn)
    N = matrix.shape[0]
    indices = np.linspace(0, N - 1, target_size, dtype=int)
    rescaled_matrix = matrix[np.ix_(indices, indices)]
    return rescaled_matrix

def min_max_scaling(matrix):
    '''
    Applies min-max scaling to normalize the matrix values between 0 and 1.
    
    Input:
    matrix (np.ndarray): The input matrix.
    
    Output:
    scaled_matrix (np.ndarray): The scaled matrix with values between 0 and 1.
    '''
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    scaled_matrix = (matrix - min_val) / (max_val - min_val)
    return scaled_matrix

def compare_matrices(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Compares two square matrices by resizing the larger matrix via average pooling,
    and then computes Pearson, Spearman, and Kendall Tau correlations between them.
    
    Input:
    mat1 (np.ndarray): First square matrix.
    mat2 (np.ndarray): Second square matrix.
    
    Output:
    Prints the correlations (Pearson, Spearman, Kendall Tau) and their p-values.
    '''
    size1, size2 = mat1.shape[0], mat2.shape[0]
    
    # Determine which matrix is larger
    if size1 > size2:
        mat1 = rescale_matrix(mat1, size2)
    elif size2 > size1:
        mat2 = rescale_matrix(mat2, size1)

    # Apply min-max scaling to both matrices
    mat1 = min_max_scaling(mat1)
    mat2 = min_max_scaling(mat2)
    
    # Flatten the matrices to 1D arrays for correlation computation
    mat1_flat = mat1.flatten()
    mat2_flat = mat2.flatten()
    
    # Pearson correlation
    pearson_corr, pearson_pval = pearsonr(mat1_flat, mat2_flat)
    
    # Spearman correlation
    spearman_corr, spearman_pval = spearmanr(mat1_flat, mat2_flat)
    
    # Kendall Tau correlation
    kendall_corr, kendall_pval = kendalltau(mat1_flat, mat2_flat)
    
    # Print the results
    print(f"Pearson correlation: {pearson_corr:.4f}, p-value: {pearson_pval:.4e}")
    print(f"Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_pval:.4e}")
    print(f"Kendall Tau correlation: {kendall_corr:.4f}, p-value: {kendall_pval:.4e}")
    return pearson_corr, spearman_corr, kendall_corr

def remove_diagonals(matrix):
    """
    Remove the main diagonal and the two diagonals adjacent to it from the given matrix.
    
    Parameters:
    - matrix: 2D numpy array.

    Returns:
    - Modified matrix with specified diagonals removed.
    """
    # Get the shape of the matrix
    n = matrix.shape[0]
    
    # Create a copy of the matrix to avoid modifying the original
    modified_matrix = np.copy(matrix)
    
    # Remove the main diagonal and the two adjacent diagonals
    for i in range(n):
        # Set the main diagonal
        modified_matrix[i, i] = 0
        # Set the first diagonal above
        if i < n - 1:
            modified_matrix[i, i + 1] = 0
        # Set the first diagonal below
        if i > 0:
            modified_matrix[i, i - 1] = 0

        # Set the second diagonal above
        if i < n - 2:
            modified_matrix[i, i + 2] = 0
        # Set the second diagonal below
        if i > 1:
            modified_matrix[i, i - 2] = 0
            
    return modified_matrix

def find_local_maxima(image_array, neighborhood_size=5, threshold=None, min_distance=1):
    """
    Find local maxima with stricter criteria for continuous heatmaps, incorporating a minimum distance between peaks.
    
    Parameters:
    - image_array (np.ndarray): The 2D heatmap array.
    - neighborhood_size (int): Size of the neighborhood for detecting local maxima.
    - threshold (float): A threshold to filter out low-intensity maxima.
    - min_distance (int): Minimum number of pixels separating local maxima.
    
    Returns:
    - maxima_coords (np.ndarray): An array of coordinates (row, col) of the local maxima.
    """
    # Debugging print statement
    print(f"Neighborhood size: {neighborhood_size}, Image array shape: {image_array.shape}")
    
    if neighborhood_size is None or neighborhood_size <= 0:
        raise ValueError("neighborhood_size must be a positive integer.")
    
    # Use maximum filter
    local_max = ndimage.maximum_filter(image_array, size=neighborhood_size)
    maxima = (image_array == local_max)

    # Apply threshold if provided
    if threshold is not None:
        maxima &= (image_array > threshold)

    # Get local maxima positions with min_distance constraint
    maxima_coords = peak_local_max(image_array, min_distance=min_distance, threshold_abs=threshold)
    
    return maxima_coords

def plot_heatmap_with_maxima(matrix, maxima_locs):
    '''
    Plots the heatmap and overlays the maxima locations.
    
    Input:
    matrix (np.ndarray): The heatmap matrix to be plotted.
    maxima_locs (np.ndarray): The coordinates (i, j) of the maxima in the matrix.
    '''
    plt.figure(figsize=(25, 20), dpi=200)
    plt.imshow(matrix, cmap='Reds', vmax=np.average(matrix) + 3 * np.std(matrix))
    plt.colorbar(label='Intensity')
    
    # Ensure maxima_locs is not empty before plotting
    if maxima_locs.size > 0:
        plt.scatter(maxima_locs[:, 1], maxima_locs[:, 0], marker='x', color='blue', label="Local Maxima", s=100)  # Plot maxima locations as blue crosses
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()


# Main function to handle everything
def process_heatmap(matrix, neighborhood_size=5, threshold=None, min_distance=None):
    '''
    Processes the heatmap by removing the diagonal, finding maxima, and plotting.
    
    Input:
    matrix (np.ndarray): The input heatmap matrix.
    '''
    # Remove diagonal
    matrix = remove_diagonals(matrix)
    
    # Find maxima
    if threshold==None: threshold = np.average(matrix)
    if min_distance==None: min_distance = len(matrix)//50
    maxima_locs = find_local_maxima(matrix, neighborhood_size, threshold, min_distance)
    
    # Plot heatmap with maxima
    plot_heatmap_with_maxima(matrix, maxima_locs)

def plot_venn_diagram(peaks1, peaks2, common_peaks):
    """
    Plot a Venn diagram for the peaks.
    """
    set1 = set(map(tuple, peaks1))
    set2 = set(map(tuple, peaks2))
    set_common = set(map(tuple, common_peaks))

    plt.figure(figsize=(8, 6))
    venn2([set1, set2], ('Heatmap 1', 'Heatmap 2'))
    plt.title('Venn Diagram of Local Maxima')
    plt.show()

def find_common_peaks(peaks1, peaks2, proximity=2):
    """
    Find common peaks within a given proximity.
    """
    common_peaks = []

    print('Finding common peaks...')
    for peak in tqdm(peaks1):
        for peak2 in peaks2:
            if np.linalg.norm(peak - peak2) <= proximity:
                common_peaks.append(tuple(peak))
                break

    return list(set(common_peaks))

def smooth_heatmap(heatmap, sigma=1):
    """
    Smooth the heatmap using Gaussian filtering.
    
    Parameters:
    - heatmap: 2D numpy array representing the heatmap.
    - sigma: Standard deviation for Gaussian kernel. Higher values result in more smoothing.

    Returns:
    - Smoothed heatmap.
    """
    return gaussian_filter(heatmap, sigma=sigma)

def compare_heatmap_peaks(heatmap1, heatmap2, neighborhood_size=5, proximity=2, threshold=None):
    """
    Main function to process two heatmaps and plot the Venn diagram.
    """
    # Jusitify the threshold
    if threshold==None:
        th1, th2 = np.average(heatmap1), np.average(heatmap2)

    # Find local maxima
    peaks1 = find_local_maxima(heatmap1, neighborhood_size, threshold)
    peaks2 = find_local_maxima(heatmap2, neighborhood_size, threshold)

    # Find common peaks
    common_peaks = find_common_peaks(peaks1, peaks2, proximity)

    # Plot Venn diagram
    plot_venn_diagram(peaks1, peaks2, common_peaks)