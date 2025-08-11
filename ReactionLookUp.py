import numpy as np
from numba import njit

@njit
def interpolate_rate_numba(vals, bins, rate_table):
    """
    Multilinear interpolation on regular grid.
    vals: tuple/list of coordinate values, length = dims
    bins: list of 1D arrays for each dimension, length = dims
    rate_table: n-dim array of rates
    """
    ndims = len(bins)
    indices = np.empty(ndims, dtype=np.int64)
    weights = np.empty(ndims)

    # Find lower indices and weights for interpolation
    for i in range(ndims):
        bin_arr = bins[i]
        val = vals[i]

        # Clamp value within bin range
        if val <= bin_arr[0]:
            indices[i] = 0
            weights[i] = 0.0
        elif val >= bin_arr[-1]:
            indices[i] = len(bin_arr) - 2
            weights[i] = 1.0
        else:
            # Binary search could speed this up if needed
            for j in range(len(bin_arr) - 1):
                if bin_arr[j] <= val <= bin_arr[j+1]:
                    indices[i] = j
                    weights[i] = (val - bin_arr[j]) / (bin_arr[j+1] - bin_arr[j])
                    break

    # Multilinear interpolation:
    # 2^ndims corners, sum weighted values
    result = 0.0
    for corner in range(1 << ndims):
        w = 1.0
        idxs = np.empty(ndims, dtype=np.int64)
        for d in range(ndims):
            if corner & (1 << d):
                w *= weights[d]
                idxs[d] = indices[d] + 1
            else:
                w *= 1.0 - weights[d]
                idxs[d] = indices[d]
        result += w * rate_table[tuple(idxs)]

    return result

class ReactionLookUp:
    def __init__(self, k, *max_concs, nbins=1024):
        self.k = k
        self.nbins = nbins

        self.bins = [
            np.linspace(0, max_conc, nbins) for max_conc in max_concs
        ]

        self._build_rate_table()

    def _build_rate_table(self):
        grids = np.meshgrid(*self.bins, indexing='ij')
        self.rate_table = self.k
        for g in grids:
            self.rate_table = self.rate_table * g

    def interpolate_rate(self, *vals):
        return interpolate_rate_numba(
            vals,
            self.bins,
            self.rate_table
        )
