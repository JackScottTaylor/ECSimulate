import numpy as np
from .Solute import Solute

class Reaction:
    '''
    Class for holding information about a first order chemical reaction.
    A -> B
    '''
    def __init__(
            self,
            reactant: Solute,
            product: Solute,
            k: float
        ):
        self.reactant = reactant
        self.product = product
        self.k = k