import numpy as np
import numba

def thomas_decompose(A: np.ndarray):
    alphas = np.diagonal(A, offset=-1)
    betas  = np.diagonal(A, offset=0)
    gammas = np.diagonal(A, offset=1)
    return alphas, betas, gammas


def thomas_solve(alphas, betas, g_primes, d, N):
    '''
    Solves the system Ax = d using the Thomas algorithm
    '''
    d_primes = compute_delta_primes(alphas, betas, g_primes, d, N)
    return back_substitution(g_primes, d_primes, N)


def compute_gamma_primes(alphas, betas, gammas, N) -> np.ndarray:
    '''
    Compute modified upper diagonal (gamma primes)
    '''
    g_primes = np.zeros(N-1)
    g_primes[0] = gammas[0] / betas[0]
    for i in range(1, N-1):
        g_primes[i] = gammas[i] / (betas[i] - alphas[i-1] * g_primes[i-1])
    return g_primes


@numba.njit
def compute_delta_primes(alphas, betas, g_primes: np.ndarray, d: np.ndarray, N) -> np.ndarray:
    '''
    Compute modified right-hand side (delta primes)
    '''
    d_primes = np.zeros(N)
    d_primes[0] = d[0] / betas[0]
    for i in range(1, N):
        denom = betas[i] - alphas[i-1] * g_primes[i-1]
        d_primes[i] = (d[i] - alphas[i-1] * d_primes[i-1]) / denom
    return d_primes


@numba.njit
def back_substitution(g_primes, d_primes, N) -> np.ndarray:
    '''
    Back-substitution to recover solution vector x
    '''
    x = np.zeros(N)
    x[-1] = d_primes[-1]
    for i in range(N-2, -1, -1):
        x[i] = d_primes[i] - g_primes[i] * x[i + 1]
    return x

