'''
This module is designed specifically for modelling CVs where any kinetics
involved are all first order such that we can easily use linear methods to try
and solve the equations efficiently.
'''

import numpy as np

from typing import List
from scipy.linalg import block_diag

class Solute:
    def __init__(self, name: str, D: float, bulk_conc: float, charge: int):
        self.name = name
        self.D = D
        self.bulk_conc = bulk_conc
        self.charge = charge

    def K(self, dt: float, dx_j: float, dx_j1: float) -> float:
        '''
        Calculates K = D*dt / (dx_j*dx_{j-1}*(dx_j+dx_{j-1}))

        This value is used in the construction of the matrices for modelling 
        diffusion via Crank-Nicholson.

        :param dt: The time step in seconds
        :param dx_j: The spatial separation between points j-1 and j in cm
        :param dx_j1: The spatial separation between points j and j+1 in cm
        :return: The calculated K value
        '''
        return self.D * dt / (dx_j*dx_j1*(dx_j+dx_j1))
    
    def A_diffusion_contribution(self, dxs: np.ndarray, dt: float) -> np.ndarray:
        '''
        Returns the A matrix contribution for modelling diffusion via CN.
        '''
        A = np.eye(len(dxs)+1)
        for i in range(1, len(dxs)):
            dx_j1, dx_j = dxs[i-1], dxs[i]
            K = self.K(dt, dx_j1, dx_j)
            A[i, i-1] = -K * dx_j1
            A[i, i] = 1 + K * (dx_j + dx_j1)
            A[i, i+1] = -K * dx_j
        return A
    
    def B_diffusion_contribution(self, dxs: np.ndarray, dt: float) -> np.ndarray:
        '''
        Returns the B matrix contribution for modelling diffusion via CN
        '''
        B = np.eye(len(dxs)+1)
        for i in range(1, len(dxs)):
            dx_j1, dx_j = dxs[i-1], dxs[i]
            K = self.K(dt, dx_j1, dx_j)
            B[i, i-1] = K * dx_j1
            B[i, i] = 1 - K * (dx_j + dx_j1)
            B[i, i+1] = K * dx_j
        return B
    
class Reaction:
    def __init__(self, name: str, reactant: Solute, product: Solute, k: float):
        self.reactant = reactant
        self.product = product
        self.k = k
        self.name = name

class Solution:
    def __init__(self):
        self.solutes = []
        self.redoxes = []
        self.reactions = []
        self.dxs = []
        self.electrode_potential = 0.0
        self.groups = []
        self.concs = []

    def add_solutes(self, *solutes: Solute):
        for solute in solutes: self.solutes.append(solute)

    def add_reactions(self, *reactions: Reaction):
        for reaction in reactions: self.reactions.append(reaction)

    def group_diffusion_matrix_contributions(self, group: List[int], A_diffusion, B_diffusion):
        '''
        Takes a group of reaction coupled solutes and creates a single large diffusion matrix
        for the whole group.
        '''
        n = A_diffusion[0].shape[0]
        group_matrix_size = len(group) * n
        As = [A_diffusion[i] for i in group]
        A_group = block_diag(*As)
        Bs = [B_diffusion[i] for i in group]
        B_group = block_diag(*Bs)
        return A_group, B_group
    
    def add_A_kinetic_terms(self, group_As, reactions, dt: float, nconcs: int):
        for A, group in zip(group_As, self.groups):
            group_solutes = [self.solutes[i] for i in group]
            solute_indices = {solute: idx for idx, solute in enumerate(group_solutes)}
            for r in reactions:
                if not (r.reactant in group_solutes and r.product in group_solutes):
                    continue
                # Add the reaction kinetics to the A matrix
                k_value = dt/2 * r.k
                for x in range(nconcs):
                    ids_increase = [solute_indices[r.reactant] * nconcs + x,
                                    solute_indices[r.product] * nconcs + x]
                    A[ids_increase[0], ids_increase[1]] -= k_value

                    ids_decrease = [solute_indices[r.product] * nconcs + x,
                                    solute_indices[r.reactant] * nconcs + x]
                    A[ids_decrease[0], ids_decrease[1]] += k_value
        return group_As
    
    def add_B_kinetic_terms(self, group_Bs, reactions, dt: float, nconcs: int):
        for B, group in zip(group_Bs, self.groups):
            group_solutes = [self.solutes[i] for i in group]
            solute_indices = {solute: idx for idx, solute in enumerate(group_solutes)}
            for r in reactions:
                if not (r.reactant in group_solutes and r.product in group_solutes):
                    continue
                # Add the reaction kinetics to the B matrix
                k_value = dt/2 * r.k
                for x in range(nconcs):
                    ids_increase = [solute_indices[r.reactant] * nconcs + x,
                                    solute_indices[r.product] * nconcs + x]
                    B[ids_increase[0], ids_increase[1]] -= k_value

                    ids_decrease = [solute_indices[r.product] * nconcs + x,
                                    solute_indices[r.reactant] * nconcs + x]
                    B[ids_decrease[0], ids_decrease[1]] += k_value
        return group_Bs
    
    def initialise_concs(self, nconcs):
        Cs = []
        for group in self.groups:
            C_individuals = [np.ones(nconcs) * self.solutes[i].bulk_conc for i in group]
            C = np.hstack(*C_individuals)
            Cs.append(C)
        self.concs = Cs

    def initialise(self, dxs, dt):
        A_diffusion = [s.A_diffusion_contribution(dxs, dt) for s in self.solutes]
        B_diffusion = [s.B_diffusion_contribution(dxs, dt) for s in self.solutes]

        group_As = [
            self.group_diffusion_matrix_contributions(
                group, A_diffusion, B_diffusion) for group in self.groups
                ]
        group_Bs = [
            self.group_diffusion_matrix_contributions(
                group, A_diffusion, B_diffusion) for group in self.groups
                ]
        
        # Now add the first_order reaction kinetics to the equation
        group_As = self.add_A_kinetic_terms(group_As, self.reactions, dt, len(dxs)+1)
        group_Bs = self.add_B_kinetic_terms(group_Bs, self.reactions, dt, len(dxs)+1)

        # Initialise the group concentrations
        self.initialise_concs(len(dxs)+1)
