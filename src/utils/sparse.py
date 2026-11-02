"""Sparse matrix helpers (legacy utilities)."""
import numpy as np

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
