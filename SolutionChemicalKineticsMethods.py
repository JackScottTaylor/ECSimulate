'''
This is a Mixin for the Solution class.
This contains some of the methods required for updating concentrations based
on reaction kinetics using a stiff-ODE solver.
'''

import numpy as np
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
from typing import List, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from .Solute import Solute
    from .Reaction import Reaction

def reaction_rates_func(
        t: float,
        solute_concs: np.ndarray,
        reactions: List["Reaction"],
        solute_to_index: Dict["Solute", int]
    ) -> np.ndarray:
    '''
    This function calculate the rate of change of concentration for different
    solutes based on the simulated chemical reactions given initial concs for 
    all the solutes.

    :param t: Required argument to run with scipy `solve_ivp`
    :param solute_concs: Array of the solute concentrations at the point being
        considered
    :param reactions: List of the included `Reaction`s
    :param solute_to_index: Precomputed dictionary to quickly go from `Solute` 
        to it's corresponding index in the solute_concs array.
    :return: Numpy array corresponding to the time derivative of concentration 
        for each of the `Solute`s.
    '''
    rates = np.zeros_like(solute_concs)
    for reaction in reactions:
        rate = reaction.rate_constant
        for reactant in reaction.reactants:
            rate *= solute_concs[solute_to_index[reactant]]
        for reactant in reaction.reactants:
            rates[solute_to_index[reactant]] -= rate
        for product in reaction.products:
            rates[solute_to_index[product]] += rate
    return rates


def integrate_point_kinetics(
        point: int,
        dt: float,
        solute_concs_at_point: np.ndarray,
        reactions: List["Reaction"],
        solute_to_index: Dict["Solute", int]
    ) -> np.ndarray:
    '''
    Uses scipy `solve_ivp` integrator to update the concentrations of all the 
    `Solute`s at the provided `point` over a time step of `dt` seconds.

    :param point: The index of the spatial grid point being updated
    :param dt: The time step in seconds
    :param solute_concs_at_point: Array of the concentrations of each `Solute`
        at the spatial grid point being considered.
    :param reactions: List of the `Reaction`s being simulated
    :param solute_to_index: Precomputed dictionary to quickly go from `Solute` 
        to it's corresponding index in the solute_concs array.
    :return: Array corresponding to the updated concentrations.
    '''
    sol = solve_ivp(
        fun=lambda t, y: reaction_rates_func(t, y, reactions, solute_to_index),
        t_span=(0, dt),
        y0=solute_concs_at_point,
        method='BDF',
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed at point {point}: {sol.message}")

    return point, np.maximum(sol.y[:, -1], 0.0)  # ensure non-negative


class MixinSolutionReactionKinetics:
    def save_solute_to_index(self) -> None:
        '''
        Precalculates the solute to index dictionary and stores it.
        '''
        self.solute_to_index: dict = {}
        for i, solute in enumerate(self.solutes):
            self.solute_to_index[solute] = i
    
    def integrate_chemical_kinetics(self, dt: float) -> None:
        '''
        Update the concentrations using scipy's stiff ODE solver in parallel.

        :param dt: The time step to update over in seconds.
        '''
        # Extract initial values
        reactions = self.reactions
        solute_to_index = self.solute_to_index

        # Collect initial concentrations per point
        solute_concs_per_point = [
            [solute.conc[point] for solute in self.solutes]
            for point in range(self.npoints)
        ]

        # Run in parallel
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(integrate_point_kinetics)(
                point,
                dt,
                solute_concs_per_point[point],
                reactions, solute_to_index
            ) for point in range(self.npoints)
        )

        # Apply results
        for point, updated_concs in results:
            for i, solute in enumerate(self.solutes):
                solute.conc[point] = updated_concs[i]