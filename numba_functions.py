import numpy as np
import numba

from numpy.typing import NDArray


@numba.njit(inline="always")
def reduction_rate_constant_numba(k0, alpha, E0, f, E):
    return k0 * np.exp(-alpha * f * (E - E0))

@numba.njit(inline="always")
def oxidation_rate_constant_numba(k0, alpha, E0, f, E):
    return k0 * np.exp((1-alpha) * f * (E - E0))


@numba.njit
def _fill_M_et(
    M_et:           NDArray[np.float64],
    ox_indices:     NDArray[np.int64],
    red_indices:    NDArray[np.int64],
    k0s:            NDArray[np.float64],
    alphas:         NDArray[np.float64],
    E0s:            NDArray[np.float64],
    f:              float,
    E:              float,
    K_et:           float
    ) -> None:
    '''
    Calculates the correct electron-transfer matrices to act on the solute
    electrode-surface concentrations.

    :param M_et: Square matrix of zeroes to be populated
    :param ox_indices: The indices of the oxidised solute for each redox
    :param red_indices: The indices of the reduced solute for each redox
    :param k0s: The rate constants for each redox reaction in cm/s
    :param alphas: The transfer coefficient for each redox reaction
    :param E0s: The standard potential for each redox reaction in V
    :param f: Equal to F/RT in V⁻¹
    :param E: The applied electrode potential in V
    :param K_et: The appropriate electron-transfer constant in s cm⁻¹
    :return: `None`, modifies provided M_et
    '''
    nredox = len(k0s)
    for i in range(nredox):
        ox_i    = ox_indices[i]
        red_i   = red_indices[i]
        k0      = k0s[i]
        alpha   = alphas[i]
        E0      = E0s[i]

        k_red = reduction_rate_constant_numba(k0, alpha, E0, f, E)
        k_ox  = oxidation_rate_constant_numba(k0, alpha, E0, f, E)

        # Each redox holds two actual reactions so there are four separate
        # contributions to solute fluxes.
        M_et[red_i, red_i] += k_ox
        M_et[red_i, ox_i]  -= k_red
        M_et[ox_i, ox_i]   += k_red
        M_et[ox_i, red_i]  -= k_ox

    # The end matrix is dimensionless.
    M_et *= K_et


@numba.njit
def add_Aet_to_banded_A(
    banded_A:   NDArray[np.float64],
    A_et:       NDArray[np.float64],
    lower:      int,
    upper:      int
    ) -> NDArray[np.float64]:
    '''
    Adds the `A electron transfer matrix which is in full-form to `banded_A`
    stored in the banded format readable by `scipy.linalg.solve_banded` and
    returns the updated matrix again in banded format. Note that `A_et` only
    acts on the surface concentrations so is much smaller than the full `A`. 
    Therefore it is added to the top-left of A.

    :param banded_A: The A-matrix in banded format
    :param A_et: The electron-transfer contribution which is in full format
    :param lower: The number of bands below the main diagonal
    :param upper: The number of bands above the main diagonal.
    :return: Updated matrix in banded format.
    '''
    n = A_et.shape[0]
    for i in range(n):
        for j in range(n):
            if (i - j <= lower) and (j - i <= upper):
                banded_A[upper + i - j, j] += A_et[i,j]
    return banded_A
