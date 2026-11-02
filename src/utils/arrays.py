"""Array helpers (rescaling, normalisation, interpolation)."""
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import zoom, binary_dilation, binary_erosion

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

def min_max_normalize(x, new_min=0.0, new_max=1.0):
    """
    Min-max normalize a 1D array to [new_min, new_max], ignoring NaNs.
    Returns NaN where input is NaN.
    """
    x = np.asarray(x, dtype=float)
    x_norm = np.full_like(x, np.nan)  # output placeholder

    finite_mask = np.isfinite(x)
    if not np.any(finite_mask):
        return x_norm  # all NaNs

    x_finite = x[finite_mask]
    xmin, xmax = x_finite.min(), x_finite.max()

    # Handle constant array
    if np.isclose(xmax, xmin):
        x_norm[finite_mask] = 0.5 * (new_min + new_max)
    else:
        # Linear scaling
        x_norm[finite_mask] = new_min + (x_finite - xmin) * (new_max - new_min) / (xmax - xmin)

    # Clip for safety
    x_norm[finite_mask] = np.clip(x_norm[finite_mask], new_min, new_max)

    return x_norm

def reshape_array(input_array, new_dimension, interpolation_kind='cubic'):
    """
    Reshape the input numpy array to a new dimension via interpolation or averaged downsampling.

    Parameters:
        input_array (np.ndarray): 1D array of shape (M,)
        new_dimension (int): Target length (N)
        interpolation_kind (str): Interpolation type for upsampling (default: 'cubic')

    Returns:
        np.ndarray: Reshaped array of length N
    """
    if not isinstance(new_dimension, int) or new_dimension <= 0:
        raise ValueError("new_dimension must be a positive integer.")
    if len(input_array) == 0:
        raise ValueError("Input array cannot be empty.")

    input_len = len(input_array)

    if new_dimension == input_len:
        return input_array

    if new_dimension < input_len:
        # Downsampling: average over equal-width segments
        edges = np.linspace(0, input_len, new_dimension + 1, endpoint=True).astype(int)
        reshaped_array = np.array([
            np.mean(input_array[edges[i]:edges[i+1]]) if edges[i] < edges[i+1]
            else input_array[edges[i]]
            for i in range(new_dimension)
        ])
    else:
        # Upsampling: spline interpolation
        x_old = np.linspace(0, 1, input_len)
        x_new = np.linspace(0, 1, new_dimension)
        interpolator = interp1d(x_old, input_array, kind=interpolation_kind)
        reshaped_array = interpolator(x_new)

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

def expand_array(arr, L):
    arr = np.array(arr)  # Ensure it's a numpy array
    original_shape = arr.shape
    
    if len(original_shape) == 1:
        N = original_shape[0]
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
