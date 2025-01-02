import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import zoom
import os
import re

chrom_sizes = {'chr1':248387328,'chr2':242696752,'chr3':201105948,'chr4':193574945,
               'chr5':182045439,'chr6':172126628,'chr7':160567428,'chr8':146259331,
               'chr9':150617247,'chr10':134758134,'chr11':135127769,'chr12':133324548,
               'chr13':113566686,'chr14':101161492,'chr15':99753195,'chr16':96330374,
               'chr17':84276897,'chr18':80542538,'chr19':61707364,'chr20':66210255,
               'chr21':45090682,'chr22':51324926,'chrX':154259566,'chrY':62460029}

def expand_columns(array, new_columns):
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