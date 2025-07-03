from numba import njit
import numpy as np

@njit(parallel=True)
def second_space_derivative_numba(
    conc: np.ndarray,
    dx: float,
) -> np.ndarray:
    '''
    Njit wrapper for Solute.second_space_derivative method.
    '''
    n = conc.shape[0]
    out = np.empty(n-2)
    for i in range(1, n-1):
        out[i-1] = (conc[i+1] - 2 * conc[i] + conc[i-1])
    return out / dx**2