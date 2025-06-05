import numpy as np
from scipy.stats import linregress
from .plt import plt

# Faraday constant in J/V
F = 96485.3329
# Temperature in K
T = 298.15
# Gas constant in J/K mol-1
R = 8.314462618

ln10 = np.log(10)  # Natural log of 10

def n_logK_analysis_approx(
        CO2_concs: np.ndarray,
        reduction_peaks: np.ndarray,
        max_CO2_conc = 0.139,
        ) -> tuple:
    '''
    Takes an array of CO2 concentration in % of maximum and an array of the
    corresponding reduction peaks in V.
    Uses the approximate modelling which a lot of papers use which follows
    (F/RT)∆E ≈ ln(K) + nln[CO₂]

    Returns a tuple corresponding to
    (n, n error, log10(K), log10(K) error, R^2)

    :param CO2_concs: CO2 concentrations in % of maximum
    :param reduction_peaks: Reduction peaks in V
    :param max_CO2_conc: Maximum CO2 concentration in M
    :return tuple: tuple of (n, n error, log10(K), log10(K) error, R^2)
    '''
    assert len(CO2_concs) == len(reduction_peaks), "len(CO2_concs) != len(reduction_peaks)"
    assert int(CO2_concs[0]) == 0, "CO2_concs[0] != 0"

    F_RT = F / (R * T) # F/(RT) useful constant
    CO2_concs = CO2_concs * max_CO2_conc / 100 # Convert to M
    red_shifts = reduction_peaks - reduction_peaks[0] # Calculate shifts

    # Remove first element from each array as cannot include 0% CO2.
    red_shifts, CO2_concs = red_shifts[1:], CO2_concs[1:]

    # Perform linear regression
    y = F_RT * red_shifts
    x = np.log(CO2_concs)
    result = linregress(x, y)

    n, intercept, r_value = result[:3]
    n_err, intercept_err = result.stderr, result.intercept_stderr

    # Calculate log10(K) from intercept
    logK = intercept / ln10
    logK_err = intercept_err / ln10
    return (n, n_err, logK, logK_err, r_value ** 2)

def n_logK_analysis_exact(
    CO2_concs: np.ndarray,
    reduction_peaks: np.ndarray,
    max_CO2_conc = 0.139,
    plot: bool = False
    ) -> tuple:
    '''
    Takes an array of CO2 concentration in % of maximum and an array of the
    corresponding reduction peaks in V.
    Does not use an approximation and instead follows
    ln(exp((F/RT)∆E)-1) = ln(K) + nln[CO₂]

    Returns a tuple corresponding to
    (n, n error, log10(K), log10(K) error, R^2)

    :param CO2_concs: CO2 concentrations in % of maximum
    :param reduction_peaks: Reduction peaks in V
    :param max_CO2_conc: Maximum CO2 concentration in M
    :param plot: If True, plot the data and the line of best fit
    :return tuple: tuple of (n, n error, log10(K), log10(K) error, R^2)
    '''
    assert len(CO2_concs) == len(reduction_peaks), "len(CO2_concs) != len(reduction_peaks)"
    assert int(CO2_concs[0]) == 0, "CO2_concs[0] != 0"

    F_RT = F / (R * T) # F/(RT) useful constant
    CO2_concs = CO2_concs * max_CO2_conc / 100 # Convert to M
    red_shifts = reduction_peaks - reduction_peaks[0] # Calculate shifts

    # Remove first element from each array as cannot include 0% CO2.
    red_shifts, CO2_concs = red_shifts[1:], CO2_concs[1:]

    # Perform linear regression
    y = np.log(np.exp(F_RT * red_shifts) - 1)
    x = np.log(CO2_concs)
    result = linregress(x, y)

    if plot: plt.scatter(x, y)

    n, intercept, r_value = result[:3]
    n_err, intercept_err = result.stderr, result.intercept_stderr

    if plot: plt.plot(x, n * x + intercept, linestyle='--')

    # Calculate log10(K) from intercept
    logK = intercept / ln10
    logK_err = intercept_err / ln10
    return (n, n_err, logK, logK_err, r_value ** 2)

def model_K1_K2(
        K1: float,
        K2: float,
        CO2_concs: np.ndarray,
        max_CO2_conc: float = 0.139,
        ) -> np.ndarray:
    '''
    Takes K1 and K2 values where K1 is the first equilibrium constant and K2 is 
    the second equilibrium constant:
    K1 = [Q(CO₂)] / [Q][CO₂]
    K2 = [Q(CO₂)₂] / [Q(CO₂)][CO₂]

    It is assumed that the CO₂ concentration is constant at every concentration
    such that the measured shift in redox potential is calculated via
    ΔE = (RT/F) * ln(1 + K1[CO₂] + K1K2[CO₂]²)
    where F is Faraday's constant, R is the gas constant and T is temperature.

    :param K1: First equilibrium constant
    :param K2: Second equilibrium constant
    :param CO2_concs: CO2 concentrations in % of maximum
    :param max_CO2_conc: Maximum CO2 concentration in M
    :return reduction_peaks: Array of calculated redox potentials
    '''
    assert CO2_concs[0] == 0, "CO2_concs[0] != 0"
    CO2_concs = CO2_concs * max_CO2_conc / 100 # Convert to M
    red_shifts = np.log(1 + K1 * CO2_concs + K1 * K2 * CO2_concs**2)
    red_shifts = red_shifts * (R * T) / F # Convert to V
    return red_shifts


def n_Kn_to_K1_K2(n: float, Kn: float, xmin: float = 0, xmax: float = 0.139
                  ) -> tuple:
    '''
    Takes experimental values of n and Kₙ and returns the corresponding values
    of K₁ and K₂

    |⟨x |x ⟩ ⟨x²|x ⟩| |K₁  | = |Kₙ⟨x |xⁿ⟩|
    |⟨x |x²⟩ ⟨x²|x²⟩| |K₁K₂| = |Kₙ⟨x²|xⁿ⟩|

    where x is the CO₂ concentration in M
    ⟨x |x ⟩ = ∫(x * x)dx from 0 to xmax

    :param n: Experimental value of n
    :param Kn: Experimental value of Kₙ
    :param xmax: Maximum CO₂ concentration in M
    :return tuple: tuple of (K1, K2)
    '''
    # Lambda function returns integral of x^m from 0 to xmax
    ip = lambda m: (1/(m+1)) * (xmax**(m+1) - xmin**(m+1))
    # Calculate the overlap matrix
    S = np.array([[ip(2), ip(3)],
                  [ip(3), ip(4)]])
    # Calculate the inverse of the overlap matrix
    S_inv = np.linalg.inv(S)
    # Calculate the non-overlap-corrected projection of Kₙxⁿ
    Kn_xn = np.array([ip(n+1), ip(n+2)]) * Kn
    # Apply inverse of overlap matrix to get K₁ and K₁K₂
    K1_K1K2 = np.dot(S_inv, Kn_xn)
    # Calculate K₁ and K₂
    K1 = K1_K1K2[0]
    K2 = K1_K1K2[1] / K1
    return (K1, K2)

def K1_K2_to_n_Kn(K1: float, K2: float, xmax: float = 0.139) -> tuple:
    '''
    Takes experimental values of K₁ and K₂ and returns the corresponding values
    of n and Kₙ

    |⟨x |x ⟩ ⟨x²|x ⟩| |K₁  | = |Kₙ⟨x |xⁿ⟩|
    |⟨x |x²⟩ ⟨x²|x²⟩| |K₁K₂| = |Kₙ⟨x²|xⁿ⟩|

    where x is the CO₂ concentration in M
    ⟨x |x ⟩ = ∫(x * x)dx from 0 to xmax
    '''
    # Lambda function returns integral of x^m from 0 to xmax
    ip = lambda m: (1/(m+1)) * (xmax**(m+1))
    # Calculate the overlap matrix
    S = np.array([[ip(2), ip(3)],
                  [ip(3), ip(4)]])
    # Calculate the RHS vector
    rhs = np.dot(S, np.array([K1, K1*K2]))

    # Note that ⟨x |xⁿ⟩ = xmax^(n+2)/(n+2) and ⟨x²|xⁿ⟩ = xmax^(n+3)/(n+3)
    # Therefore if we let the two components of rhs be a, b
    # then b/a = (n+2)/(n+3) * xmax
    # therefore let c = b xmax/a so c(n+3) = (n+2)
    # therefore n = (2-3c)/(c-1)
    c  = rhs[1] / (rhs[0] * xmax)
    n  = (2 - 3*c) / (c - 1)
    Kn = rhs[0] / (xmax**(n+2) / (n+2)) 
    return n, Kn