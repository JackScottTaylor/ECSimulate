import numpy as np
from scipy.optimize import root_scalar

def stretched_grid(dx_min, L, N):
    '''
    Creates a 1D nonuniform grid from 0 to L with stretched spacing.

    Parameters:
        dx_min: minimum spacing near the left end
        dx_max: maximum spacing near the right end
        L: total domain length
        N: number of points

    Returns:
        x: array of grid point positions (size N)
        dx: array of step sizes (size N-1)
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