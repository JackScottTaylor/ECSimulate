from numpy.typing import NDArray
from scipy.optimize import root_scalar
import numpy as np

from typing import Tuple

def chess_board_combine(*arrays: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Combines arrays of the same shape into a single array with interdigitated 
    values.

    Example:
    ```
    x = [1, 3,]
    y = [2, 4]
    z = chess_board_combine(x, y) -> [1, 2, 3, 4]
    ```

    :param arrays: The arrays to be combined
    :return: The interdigitated array.
    '''
    s = arrays[0].shape
    for A in arrays: assert A.shape == s, "Arrays must have same shape!"
    n_arrays = len(arrays)

    # Expand the array size correctly using dtype of first array
    C = np.zeros(tuple(n_arrays * np.array(s)), dtype=arrays[0].dtype)

    for array_idx, array in enumerate(arrays):
        for index, val in np.ndenumerate(array):
            # Compute offset properly
            combo_index = tuple(i * n_arrays + array_idx for i in index)
            C[combo_index] = val
    return C


def matrix_in_banded_format(
        M: NDArray[np.float64],
        lower: int,
        upper: int
    ) -> NDArray[np.float64]:
    '''
    This takes a banded 2D numpy array and returns the matrix in the form
    used by `scipy.linalg.solve_banded`.

    Example:
    ```
    x = np.array([
        [1, 2, 3, 0,],
        [4, 5, 6, 7,],
        [0, 8, 9, 1,],
        [0, 0, 2, 3,]
        ])
    y = matrix_in_banded_form(x, 2, 1) 
        ↳ np.array([[0, 0, 3, 7],
                    [0, 2, 6, 1],
                    [1, 5, 9, 3],
                    [4, 8, 2, 0]])
    ```


    :param M: 2D banded numpy array
    :param lower: Number of bands below the main diagonal
    :param upper: Number of bands above the main diagonal
    :return: M in banded form readable by `scipy.linalg.solve_banded`
    '''
    n = M.shape[0]
    banded = np.zeros((lower+upper+1, n), dtype=M.dtype)
    for i in range(n):
        for j in range(max(0, i - lower), min(n, i + upper + 1)):
            banded[upper + i - j, j] = M[i,j]
    return banded

def stretched_grid(
        dx_min: float,
        L: float, 
        N: int
        ) -> Tuple[NDArray, NDArray]:
    '''
    Creates an array of `N` positions between `0` and `L` with exponentially
    increasing separation starting with a separation of `dx_min`.

    Parameters:
        dx_min: minimum spacing near the left end in cm
        L: total domain length in cm
        N: number of points

    Returns:
        x: array of grid point positions (size N) in cm
        dx: array of step sizes (size N-1) in cm
    '''
    def objective(r):
        xi = np.linspace(0, 1, N)
        x = (np.exp(r * xi) - 1) / (np.exp(r) - 1)
        dx = np.diff(x) * L
        return dx[0] - dx_min  # try to match dx_min

    # Solve for r that matches dx_min
    res = root_scalar(objective, bracket=[1e-8, 100], method='bisect')

    r = res.root
    xi = np.linspace(0, 1, N)
    x = (np.exp(r * xi) - 1) / (np.exp(r) - 1) * L
    dx = np.diff(x)

    return x, dx