import numpy as np

from .Solute import Solute

# Important thermodynamic quantities
R = 8.31446261815324     # J mol⁻¹ K⁻¹
F = 96485.3321233100184  # C mol⁻¹

class Redox:
    '''
    Class used for defining a redox couple between two Solute species
    Ox + ne⁻ -> Red

    :param name: Name of the redox couple
    :param E0: Standard electrode potential in V
    :param n: Number of electrons transferred in the redox reaction
    :param reduced_species: The solute corresponding to the reduced species
    :param oxidised_species: The solute corresponding to the oxidised form
    '''
    def __init__(
            self,
            name:             str,
            E0:               float,
            n:                int,
            reduced_species:  Solute,
            oxidised_species: Solute,
        ) -> None:
        self.reduced_species  = reduced_species
        self.oxidised_species = oxidised_species
        self.name             = name
        self.E0               = E0
        self.n                = n
        self.nF_R             = n * F / R


    def electrode_potential(
            self,
            T: float = 298.15
        ) -> float:
        '''
        Calculates the electrode potential of the redox couple at the electrode
        which is assumed to be the first index in the concentration arrays.
        E = E0 - (RT/nF) * ln([Ox]/[Red])

        :keyword T: Temperature in Kelvin (default 298.15 K)
        :return: The electrode potential in V
        '''
        if not hasattr(self.reduced_species, 'conc') or \
           not hasattr(self.oxidised_species, 'conc'):
            raise ValueError("Concentrations of reduced and oxidised species " \
                             "must be initialised before calculating electrode"\
                             "potential")
        return self.E0 - (R * T / (self.n * F)) * \
                np.log(self.oxidised_species.conc[0] /
                        self.reduced_species.conc[0])
    

    def equilibrium_constant(
            self,
            electrode_potential: float,
            T: float = 298.15
        ) -> float:
        '''
        Uses the Nernst equation E = E0 - (RT/nF)ln([red]/[ox]) and returns the 
        value [red]/[ox] for the given electrode potential.

        :param electrode_potential: The electrode potential in V
        :param T: The temperature in K
        :return: The concentration constant
        '''
        K = np.exp((self.nF_R / T) * (self.E0 - electrode_potential))
        return K


