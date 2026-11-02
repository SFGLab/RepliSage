"""Small file helpers."""
import os
import re

def natural_sort_key(s):
    # Splits string into parts of digits and non-digits
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

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

def save_parameters(filepath, **params):
    with open(filepath, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
