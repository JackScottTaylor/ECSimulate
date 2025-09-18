from numba_functions import reduction_rate_constant_numba,\
    oxidation_rate_constant_numba

from .Solute import Solute

class Redox:
    '''
    Class for defining a redox reaction.

    :param k0: The redox rate constant in cm s⁻¹
    :param alpha: The transfer coefficient, must be between 0 and 1
    :param E_standard: The standard potential of the redox reaction in V
    :param ox: The sollute corresponding to the oxidised species
    :param red: The solute corresponding to the reduced species
    '''
    def __init__(
            self,
            k0:         float,  # cm/s
            alpha:      float,  # Between 0 and 1
            E_standard: float,  # V
            ox:         Solute, # The oxidised species
            red:        Solute, # The reduced species
        ):
        self.k0, self.alpha, self.E_standard = k0, alpha, E_standard
        self.ox, self.red = ox, red
        self.n: int = ox.charge - red.charge

    def reduction_rate_constant(self, f: float, E: float):
        return reduction_rate_constant_numba(
            self.k0, self.alpha, self.E_standard, f, E
            )
    
    def oxidation_rate_constant(self, f: float, E: float):
        return oxidation_rate_constant_numba(
            self.k0, self.alpha, self.E_standard, f, E
        )