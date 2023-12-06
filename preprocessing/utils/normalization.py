import numpy as np

from custom_types import ComplexMatrix, RealMatrix


def normalize_complex_matrix(matrix: ComplexMatrix) -> ComplexMatrix:
    """
    Should normalize to max=1+0, min=-1+0, look up (x + iy) / |x + iy| in wolfram
    """
    return matrix / np.absolute(matrix)


def normalize_real_matrix(matrix: RealMatrix) -> RealMatrix:
    # for 2D matricies, normalizes per feature
    return (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0))


def normalize_real_tensor(tensor: RealMatrix) -> RealMatrix:
    # for 4D matricies, normalizes per channel
    return (tensor - np.min(tensor, axis=(0, 1, 2), keepdims=True)) / (
        np.max(tensor, axis=(0, 1, 2), keepdims=True)
        - np.min(tensor, axis=(0, 1, 2), keepdims=True)
    )
