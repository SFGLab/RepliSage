import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

chrom_sizes = {'chr1':248387328,'chr2':242696752,'chr3':201105948,'chr4':193574945,
               'chr5':182045439,'chr6':172126628,'chr7':160567428,'chr8':146259331,
               'chr9':150617247,'chr10':134758134,'chr11':135127769,'chr12':133324548,
               'chr13':113566686,'chr14':101161492,'chr15':99753195,'chr16':96330374,
               'chr17':84276897,'chr18':80542538,'chr19':61707364,'chr20':66210255,
               'chr21':45090682,'chr22':51324926,'chrX':154259566,'chrY':62460029}

def min_max_normalize(matrix, Min=0, Max=1):
    # Calculate the minimum and maximum values of the matrix
    matrix = np.nan_to_num(matrix)
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    # Normalize the matrix using the min-max formula
    normalized_matrix = Min + (Max - Min) * ((matrix - min_val) / (max_val - min_val))

    return normalized_matrix

def reshape_array(input_array, new_dimension):
    """
    Reshape the input numpy array by computing averages across windows.

    Parameters:
    input_array (numpy.ndarray): Input array of dimension M.
    new_dimension (int): Desired new dimension N for the reshaped array.

    Returns:
    numpy.ndarray: Reshaped array of dimension N.
    """

    # Calculate the size of each window
    original_length = len(input_array)
    window_size = original_length // new_dimension  # Calculate the size of each window
    reshaped_array = np.zeros(new_dimension)

    # Iterate over each segment/window and compute the average
    if original_length>new_dimension:
        # In case that we want to downgrade the dimension we can compute averages
        for i in range(new_dimension):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size

            # Compute the average of values in the current window
            if i == new_dimension - 1:  # Handle the last segment
                reshaped_array[i] = np.average(input_array[start_idx:])
            else:
                reshaped_array[i] = np.average(input_array[start_idx:end_idx])
    else:
        # In case that we need higher dimension we perform spline interpolation
        original_indices = np.linspace(0, original_length - 1, original_length)
        new_indices = np.linspace(0, original_length - 1, new_dimension)
        spline_interpolation = interp1d(original_indices, input_array, kind='cubic')
        reshaped_array = spline_interpolation(new_indices)

    return reshaped_array