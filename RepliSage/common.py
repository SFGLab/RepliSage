import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import zoom
from numba import njit, int32, float64
from scipy.ndimage import binary_dilation, binary_erosion
import os
import re

chrom_sizes = {'chr1':248387328,'chr2':242696752,'chr3':201105948,'chr4':193574945,
               'chr5':182045439,'chr6':172126628,'chr7':160567428,'chr8':146259331,
               'chr9':150617247,'chr10':134758134,'chr11':135127769,'chr12':133324548,
               'chr13':113566686,'chr14':101161492,'chr15':99753195,'chr16':96330374,
               'chr17':84276897,'chr18':80542538,'chr19':61707364,'chr20':66210255,
               'chr21':45090682,'chr22':51324926,'chrX':154259566,'chrY':62460029}

def save_parameters(filepath, **params):
    with open(filepath, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

def rescale_matrix(matrix: np.array, new_cols: int, fork_mat: bool=False):
    """
    Rescales a binary replicated_dna matrix to a new number of columns, preserving
    the transition point in each row where zeros switch to ones or the matrix preserving
    the replication forks positions.
    
    Parameters:
        matrix (np.ndarray): Binary input matrix with rows containing
                             leading zeros followed by ones (replicated_dna).
        new_cols (int): Desired number of columns in the rescaled matrix.
        fork_mat (bool): Whether the matrix is describing the fork positions (True) or the replicated_dna (False).
    
    Returns:
        np.ndarray: Rescaled binary matrix.
    """
    original_cols = matrix.shape[1]
    rescaled_matrix = np.zeros((matrix.shape[0], new_cols), dtype=int)
    
    for i, row in enumerate(matrix):
        if not all(row == 0):
            transition_index = np.argmax(row == 1)
            new_transition_index = int(round(transition_index * (new_cols - 1) / (original_cols - 1)))
            if fork_mat:
                rescaled_matrix[i, new_transition_index] = 1
            else:
                rescaled_matrix[i, new_transition_index:] = 1
    
    return rescaled_matrix

def sharpen_edges(matrix):
    """
    Sharpens the binary matrix by retaining edge-like features while removing duplicates.
    The result is binary and highlights sharp edges.

    Parameters:
        matrix (np.ndarray): A binary numpy array (values 0 or 1).
    
    Returns:
        np.ndarray: A sharpened binary matrix.
    """
    # Ensure the input is binary
    matrix = (matrix > 0).astype(np.int32)

    # Apply morphological operations: edge detection
    dilated = binary_dilation(matrix)
    eroded = binary_erosion(matrix)
    sharpened = (dilated ^ eroded)  # XOR operation to keep edges

    # Return binary result
    return sharpened.astype(np.int32)

def expand_columns(array, new_columns, repeat=True):
    """
    Expand each column of a given array by repeating its elements to fit the desired number of columns.
    
    Parameters:
        array (numpy.ndarray): The input array of shape (N, T1).
        new_columns (int): The desired number of columns (T2 > T1).
    
    Returns:
        numpy.ndarray: The expanded array of shape (N, new_columns).
    """
    N, T1 = array.shape
    
    if new_columns <= T1:
        raise ValueError("Number of new columns (T2) must be greater than the original number of columns (T1).")
    
    # Compute the number of times to repeat each element within a column
    repeat_factor = new_columns // T1
    
    # Create an expanded array with repeated elements
    expanded_array = np.zeros((N, new_columns), dtype=array.dtype)
    for i in range(T1):
        for k in range(repeat_factor):
            expanded_array[:, i * repeat_factor + k] = array[:, i]
    
    return expanded_array

def min_max_normalize(matrix, Min=0, Max=1):
    # Calculate the minimum and maximum values of the matrix
    matrix = np.nan_to_num(matrix)
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    # Normalize the matrix using the min-max formula
    normalized_matrix = Min + (Max - Min) * ((matrix - min_val) / (max_val - min_val))

    return normalized_matrix


def reshape_array(input_array, new_dimension, interpolation_kind='cubic'):
    """
    Reshape the input numpy array by computing averages across windows or interpolation.

    Parameters:
    input_array (numpy.ndarray): Input array of dimension M.
    new_dimension (int): Desired new dimension N for the reshaped array.
    interpolation_kind (str): Type of interpolation for upsampling (default: 'cubic').

    Returns:
    numpy.ndarray: Reshaped array of dimension N.
    """
    # Validate inputs
    if not isinstance(new_dimension, int) or new_dimension <= 0:
        raise ValueError("new_dimension must be a positive integer.")
    if len(input_array) == 0:
        raise ValueError("Input array cannot be empty.")

    input_len = len(input_array)

    # If the dimensions are the same, return the original array
    if new_dimension == input_len:
        return input_array

    # Reshape using averages if downsampling
    if new_dimension < input_len:
        window_size = input_len // new_dimension
        reshaped_array = np.array([
            np.mean(input_array[i * window_size : (i + 1) * window_size])
            if i < new_dimension - 1 else np.mean(input_array[i * window_size:])
            for i in range(new_dimension)
        ])
    else:
        # Reshape using spline interpolation if upsampling
        original_indices = np.linspace(0, input_len - 1, input_len)
        new_indices = np.linspace(0, input_len - 1, new_dimension)
        spline_interpolation = interp1d(original_indices, input_array, kind=interpolation_kind)
        reshaped_array = spline_interpolation(new_indices)

    return reshaped_array

def remove_duplicate_ones(matrix):
    """
    Removes duplicate `1`s in the columns of a binary matrix based on nearest neighbors.
    
    Parameters:
        matrix (np.ndarray): A binary numpy array (values 0 or 1).
        
    Returns:
        np.ndarray: A new binary matrix with no duplicated `1`s in columns.
    """
    # Create a copy of the matrix to avoid modifying the original
    result = np.zeros_like(matrix)

    # Iterate over each column
    for col in range(matrix.shape[1]):
        # Get the indices of rows where the column has a 1
        ones_indices = np.where(matrix[:, col] == 1)[0]
        if len(ones_indices) > 0:
            # Select the row closest to the column midpoint for the single `1`
            closest_index = ones_indices[np.argmin(np.abs(ones_indices - np.median(ones_indices)))]
            result[closest_index, col] = 1

    return result

def natural_sort_key(s):
    # Splits string into parts of digits and non-digits
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def list_files_in_directory(directory: str):
    """
    Returns a naturally sorted list of all file names in the given directory.
    
    Input:
    directory (str): the path of the directory.
    
    Output:
    files_list (list): a naturally sorted list of file names.
    """
    
    # List all files in the directory
    files_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Sort files using natural order
    files_list.sort(key=natural_sort_key)
    
    return files_list

def expand_array(arr, L):
    arr = np.array(arr)  # Ensure it's a numpy array
    original_shape = arr.shape
    
    if len(original_shape) == 1:
        return np.interp(np.linspace(0, N-1, L), np.arange(N), arr)
    
    elif len(original_shape) == 2:  # 2D array
        N, M = original_shape  # Original size (NxM)
        
        # Calculate the zoom factors for both dimensions
        zoom_factor_row = L / N
        zoom_factor_col = L / M
        
        # Use scipy's zoom to interpolate both dimensions
        expanded_arr = zoom(arr, (zoom_factor_row, zoom_factor_col), order=1)  # Order 1 for linear interpolation
        
        return expanded_arr
    
    else:
        raise ValueError("Only 2D arrays are supported for this function.")

class SparseMatrix:
    def __init__(self, dense_matrix):
        """
        Initialize the SparseMatrix by storing only non-zero elements in a dictionary.
        The input dense matrix is removed from RAM after processing.
        """
        # Validate input
        if not isinstance(dense_matrix, np.ndarray):
            raise ValueError("Input must be a numpy array.")
        
        # Store non-zero elements and remove dense matrix
        self.shape = dense_matrix.shape
        self.data = {(i, j): dense_matrix[i, j] for i, j in zip(*np.nonzero(dense_matrix))}
        del dense_matrix  # Free RAM

    def __getitem__(self, index):
        """
        Get the value at a specific index (i, j). Returns 0 if the index is not in the data.
        """
        return self.data.get(index, 0)

    def __setitem__(self, index, value):
        """
        Set the value at a specific index (i, j). Removes the entry if value is 0.
        """
        if value == 0:
            self.data.pop(index, None)
        else:
            self.data[index] = value

    def __add__(self, other):
        """
        Add two sparse matrices (element-wise). Returns a new SparseMatrix object.
        """
        if not isinstance(other, SparseMatrix) or self.shape != other.shape:
            raise ValueError("Both matrices must be SparseMatrix objects of the same shape.")
        
        result = SparseMatrix(np.zeros(self.shape))  # Create an empty SparseMatrix
        for key in self.data:
            result[key] = self[key] + other[key]
        for key in other.data:
            if key not in self.data:
                result[key] = other[key]
        return result

    def to_dense(self):
        """
        Convert the sparse matrix back to a dense numpy array.
        """
        dense = np.zeros(self.shape, dtype=float)
        for (i, j), value in self.data.items():
            dense[i, j] = value
        return dense

    def non_zero_elements(self):
        """
        Get a list of non-zero elements and their indices.
        """
        return list(self.data.items())

    def __repr__(self):
        """
        String representation for debugging.
        """
        return f"SparseMatrix(shape={self.shape}, non_zero={len(self.data)})"

class NumbaSparseMatrix:
    def __init__(self, dense_matrix):
        """
        Initialize the sparse matrix using arrays for indices and values.
        """
        if not isinstance(dense_matrix, np.ndarray):
            raise ValueError("Input must be a numpy array.")
        
        self.shape = dense_matrix.shape
        non_zero = dense_matrix != 0
        self.indices = np.array(np.nonzero(non_zero), dtype=np.int64).T  # (i, j) pairs as int32
        self.values = dense_matrix[non_zero].astype(np.float64)          # Non-zero values as float64
        del dense_matrix  # Free RAM

    def to_dense(self):
        """
        Convert the sparse representation back to a dense matrix.
        """
        dense = np.zeros(self.shape, dtype=np.float64)
        for (i, j), value in zip(self.indices, self.values):
            dense[i, j] = value
        return dense

    def __repr__(self):
        """
        String representation for debugging.
        """
        return f"NumbaSparseMatrix(shape={self.shape}, non_zero={len(self.values)})"