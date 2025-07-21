from .Solute    import Solute
from .Redox     import Redox
from .Reaction  import Reaction
from .constants import FARADAY_CONSTANT

from .SolutionCVMethods import SolutionCVMethods

from typing               import List
from scipy.sparse.csgraph import connected_components
from scipy.sparse         import csr_matrix
from collections          import deque
from rich.progress        import Progress

import numpy as np

class Solution(
    SolutionCVMethods
    ):
    '''
    Class which holds all information pertaining to a 1D electrochemical
    solution with a single flat-electrode.
    '''
    def __init__(self) -> None:
        self.solutes:               List[Solute]    = []
        self.redoxes:               List[Redox]     = []
        self.reactions:             List[Reaction]  = []
        self.electrode_potential:   float           = 0.0


    def add_solutes(self, *solutes: Solute) -> None:
        for solute in solutes: self.solutes.append(solute)


    def add_redoxes(self, *redoxes: Redox) -> None:
        for redox in redoxes: self.redoxes.append(redox)


    def add_reactions(self, *reactions: Reaction) -> None:
        for reaction in reactions: self.reactions.append(reaction)

    
    def initialise(
            self,
            width: float,
            dx:    float,
            dt:    float,
            A:     float = np.pi * 1.5 ** 2
        ) -> None:
        '''
        Sets up parameters for the simulation including the width of the 
        simulation in cm and the spacing of the grid in cm.
        Each solute in the solution is set to their bulk concentrations

        :param width: The width of the simulation in cm
        :param dx: The spacing of the grid in cm
        :param dt: The time-step spacing which will be considered in seconds
        :param A: The area of the electrode in mm², the default is a circular
            electrode with radius 1.5 mm (A = π * 1.5²)
        '''
        self.dx = dx
        self.dt = dt
        self.A  = A * 1e-6  # Convert mm² to m²
        # Choose number of grid points to get close to desired width as possible
        npoints = int(width / dx)
        # For each solute, set up the concentrations and diffusion matrices
        for solute in self.solutes:
            solute.initialise_concentration(npoints)
            solute.save_diffusion_matrices(dx, dt)
        # Save number of grid points
        self.npoints = npoints
        # Save the redox connectiveness
        self.redox_connectivity = self.extract_connected_groups(
            self.redox_connectivity_matrix()
        )
        # Precompute scaling factor for current from flux
        self.FAdx = - FARADAY_CONSTANT * A / (dx * 1e-2)

    
    def redox_connectivity_matrix(self) -> np.ndarray:
        '''
        Constructs a matrix which shows the connectivity of the redox couples
        in the solution. The matrix is symmetric and has a 1 if there is a
        redox couple connecting two solutes, and 0 otherwise.
        
        :return: A square matrix showing the connectivity of redox couples
        '''
        n_solutes = len(self.solutes)
        M         = np.eye(n_solutes)
        ids       = {self.solutes[i]: i for i in range(n_solutes)}
        
        for r in self.redoxes:
            red, ox = ids[r.reduced_species], ids[r.oxidised_species]
            M[red, ox] = M[ox, red] = 1
        return M
    

    def extract_connected_groups(self, M: np.ndarray) -> List[List[int]]:
        '''
        Takes a square matrix M and returns a list of lists where each
        sublist contains the indices of the connected components in M.

        :param M: A square matrix representing connections
        :return: A list of lists, where each sublist contains indices of
            connected components
        '''
        # Create a binary adjacency matrix ignoring diagonal
        adjacency = (M != 0).astype(int)
        np.fill_diagonal(adjacency, 0)

        # Convert to sparse format for efficiency
        graph = csr_matrix(adjacency)

        # Find connected components
        n_components, labels = connected_components(
                                        csgraph=graph, directed=False
                                        )

        # Group indices by component label
        groups = [[] for _ in range(n_components)]
        for idx, label in enumerate(labels):
            groups[label].append(idx)
        
        return groups
    

    def diffuse(self) -> None:
        '''
        Diffuses all species in solution to diffuse over one time step
        '''
        for solute in self.solutes: solute.diffuse()

    
    def redox_conc_scaling_matrix(self) -> np.ndarray:
        '''
        Calculates the matrix corresponding to the relative concentrations
        required to achieve Nernstian equilibrium.
        '''
        M = np.eye(len(self.solutes))
        n_solutes = len(self.solutes)
        ids = {self.solutes[i]: i for i in range(n_solutes)}
        for r in self.redoxes:
            red, ox = ids[r.reduced_species], ids[r.oxidised_species]
            K = r.equilibrium_constant(self.electrode_potential)
            M[red, ox], M[ox, red] = 1/K, K
        return M


    def group_to_conc_scalings(
            self,
            group: List[int],
            scaling_matrix: np.ndarray
            ) -> List[float]:
        '''
        Takes a list of indices corresponding to Solutes which are linked by
        redox reactions. It also takes a scaling matrix which contains all of
        the scaling factors determined by the Redox reactions. The correct 
        relative concentration of all the species in the group is then
        calculated and returned.

        This function was written by ChatGPT.

        :param group: List of indices corresponding to coupled Solutes
        :param scaling_matrix: Square matrix which contains all the relative
            concentrations calculated from the Redox reactions
        :return: List of relative concentrations to achieve equilibrium.
        '''
        n = len(group)
        visited     = [False] * n
        rel_conc    = [None]  * n
        rel_conc[0] = 1.0  # Reference: first solute's concentration is 1
        visited[0]  = True

        # Map from global index to local index in group
        global_to_local = {g: i for i, g in enumerate(group)}

        # BFS to traverse the group and calculate relative concentrations
        queue = deque([0])

        while queue:
            i_local = queue.popleft()
            i_global = group[i_local]
            for j_local, j_global in enumerate(group):
                if visited[j_local]:
                    continue

                # If scaling exists from i to j
                scale = scaling_matrix[i_global, j_global]
                if scale > 0:
                    rel_conc[j_local] = rel_conc[i_local] * scale
                    visited[j_local] = True
                    queue.append(j_local)
                    continue

                # Try reverse
                scale = scaling_matrix[j_global, i_global]
                if scale > 0:
                    rel_conc[j_local] = rel_conc[i_local] / scale
                    visited[j_local] = True
                    queue.append(j_local)

        # Check if any solute remains unconnected (shouldn’t in connected group)
        if any(c is None for c in rel_conc):
            raise ValueError("Incomplete connection in redox scaling group.")

        return rel_conc
    

    def Nernstian_equilibrium(self) -> None:
        '''
        Goes through all of the groups connected by redox potentials and sets
        the Nernstian equilibrium
        '''
        scaling_matrix = self.redox_conc_scaling_matrix()
        for group in self.redox_connectivity:
            # If non-redox-active then do nothing
            if len(group) == 1: continue
            # Get the relatice concentrations for the group
            rel_conc = self.group_to_conc_scalings(group, scaling_matrix)
            # Calculate the total concentration of the group
            total_conc = sum(self.solutes[i].conc[0] for i in group)
            # Reference concentration is the first solute in the group
            ref_conc = self.solutes[group[0]].conc[0]
            # Set the concentrations of the solutes in the group
            for i, solute in enumerate(group):
                self.solutes[solute].conc[0] = ref_conc * rel_conc[i]
            # Set the total concentration to be the same as previously
            new_total_conc = sum(self.solutes[i].conc[0] for i in group)
            global_conc_scaling = total_conc / new_total_conc
            # Scale the concentrations to match the total concentration
            for i in group:
                self.solutes[i].conc[0] *= global_conc_scaling

    
    def total_solute_charge_at_electrode(self) -> float:
        '''
        Calculates the total charge in e at the electrode due to the solutes.
        
        :return: The total charge in e at the electrode
        '''
        total_charge = 0.0
        for solute in self.solutes:
            total_charge += solute.conc[0] * solute.charge
        return total_charge
    

    def run_chemical_reactions(self, time_split: int = 1) -> None:
        '''
        Runs all of the chemical reactions by iterating through the reactions, 
        calculating the rates of the reaction at every point in the simultaion
        and then updating the concentrations of the reactants and products.

        :param time_split: The number of time steps to split the reaction into
            to ensure stability of the simulation. Default is 1.
        '''
        for i in range(time_split):
            # Calculate the gradients
            solute_conc_gradients = self.reaction_concentration_time_gradients()
            # Now update the concentrations of the solutes
            for solute, gradient in zip(self.solutes, solute_conc_gradients):
                # Update the concentration at each point in the simulation
                solute.conc += gradient * self.dt / time_split
                # Ensure that concentrations do not go negative
                solute.conc = np.clip(solute.conc, 0, None)


    def reaction_concentration_time_gradients(self) -> List[np.ndarray]:
        '''
        Calculate the instantaneous concentration gradients for all of the 
        solutes

        :return: List of the instantaneous concentration time derivatives
            determined from the reaction kinetics.
        '''
        solute_to_index = {s: i for i, s in enumerate(self.solutes)}
        n = len(self.solutes[0].conc)
        solute_conc_gradients = [np.zeros(n) for _ in self.solutes]
        # Iterate through each reaction and update the conc gradients
        for reaction in self.reactions:
            rate = reaction.rate(self.npoints)
            # For each reactant, subtract the rate from the gradient
            for reactant in reaction.reactants:
                index = solute_to_index[reactant]
                solute_conc_gradients[index] -= rate
            # For each product, add the rate to the gradient
            for product in reaction.products:
                index = solute_to_index[product]
                solute_conc_gradients[index] += rate
        # Return the gradients
        return solute_conc_gradients
    

    def diffuse_coupled_kinetics(self) -> None:
        '''
        This causes all of the solutes to both chemically react and diffuse at
        the same time. The changes due to chemical reactions are calculated and
        used to create reaction matrices for each solute. Diffusion occurs via
        Crank-Nicholson.
        '''
        gradients = self.reaction_concentration_time_gradients()
        gradients = [g * self.dt for g in gradients]
        for solute, gradient in zip(self.solutes, gradients):
            gradient[0], gradient[-1] = 0, 0
            solute.diffuse_coupled_kinetics(gradient)

    
    def diffusion_Strang(self) -> None:
        '''
        This method allows chemical kinetics and diffusion to occur via
        Strang splitting. Reaction first happenn over dt/2 then diffusion
        over dt, then reaction again over dt/2.
        '''
        self.run_chemical_reactions(time_split=2)
        self.diffuse()
        self.run_chemical_reactions(time_split=2)

    
    def current_from_flux(
            self
            ) -> float:
        '''
        Calculates the current at the electrode from the flux of charged
        solutes. Sums the flux contributions from all solutes and converts to
        total current in mA.

        :return: The total current in mA at the electrode
        '''
        I = sum(s.current_contribution() for s in self.solutes)
        I *= self.FAdx
        return I * 1e-3