from .Solute import Solute

from typing import List

import numpy as np

class Reaction:
    '''
    Class for defining a reaction between solutes.

    If a reaction is A + A → B + C then the reactants will be [A, A] and the
    products will be [B, C].

    Note that rate constant should use units of Molar and seconds.
    
    :param label: The name of the reaction
    :param reactants: A list of solutes which are reactants in the reaction
    :param products: A list of solutes which are the products of the reaction
    :param rate_constant: The rate constant of the reaction
    '''
    def __init__(
            self,
            name:           str,
            reactants:      List[Solute],
            products:       List[Solute],
            rate_constant:  float
        ) -> None:
        self.name           = name
        self.reactants      = reactants
        self.products       = products
        self.rate_constant  = rate_constant


    def rate(
            self, n
        ) -> float:
        '''
        This function calculates the rate of the reaction at each point in the 
        simulation via rate = k * Π[reactant] such that rate has units of 
        mol dm⁻³ s⁻¹.

        :return: The rate of the reaction in mol dm⁻³ s⁻¹
        '''
        rate = np.full(n, self.rate_constant, dtype=float)
        for reactant in self.reactants:
            rate *= reactant.conc
        return rate

            
            