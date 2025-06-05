import numpy as np
from .plt import plt
from typing import List, Optional
from scipy.linalg import solve_banded
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import math
from collections import deque

R = 8.31446261815324  # J mol⁻¹ K⁻¹
F = 96485.3321233100184  # C mol⁻¹

class Solute:
    '''
    Class for defining a given solute

    :param label: The name of the solute
    :param D: The diffusion coefficient in cm²s⁻¹
    :param conc_bulk: The bulk concentration in mol dm⁻³
    '''
    def __init__(
            self, label: str, D: float, conc_bulk: float = 0.0,
            charge: int = 0):
        self.label     = label
        self.D         = D
        self.conc_bulk = conc_bulk
        self.charge    = charge

    def initialise_concentration(self, npoints: int):
        '''
        Sets up the concentration of the solute in the simulation

        :param npoints: The number of grid points in the simulation
        '''
        self.npoints = npoints
        self.conc    = np.ones(npoints) * self.conc_bulk

    def calculate_K(self, dx: float, dt: float) -> float:
        '''
        Calculates K = D*dt/2dx² (dimensionless)
        
        This value is used in consructing the matrices to be solved in modelling
        the diffusion of solutes.
        
        :return: The value of K
        '''
        K = self.D * 0.5 * dt / (dx ** 2)
        return K

    def construct_A(self) -> np.ndarray:
        '''
        Constructs the matrix A for the Crank-Nicholson method
        
        :return: The banded matrix A
        '''
        if not hasattr(self, 'K'):
            self.K = self.calculate_K()
        K = self.K
        A = np.zeros((self.npoints, self.npoints))
        for i in range(1, self.npoints-1):
            A[i, i-1] = -K
            A[i, i]   = 1 + 2*K
            A[i, i+1] = -K
        # Reflective Wall Boundary Conditions
        A[0, 0] = 1 + K
        A[0, 1] = -K
        # Dirichlet Boundary Condition
        A[-1, -1] = 1
        return A
    
    def construct_A_banded(self) -> np.ndarray:
        '''
        Converts the matrix A to a banded matrix format for efficient solving
        
        :param A: The matrix A to be converted
        :return: The banded matrix in the form of a 2D array
        '''
        if not hasattr(self, 'K'):
            self.K = self.calculate_K()
        K = self.K
        A_banded = np.zeros((3, self.npoints))
        A_banded[0, 1:  ] = -K
        A_banded[1, 1:-1] = 1 + 2*K
        A_banded[1, 0   ] = 1 + K
        A_banded[1, -1  ] = 1
        A_banded[2, :-2] = -K
        return A_banded

    def construct_B(self) -> np.ndarray:
        '''
        Constructs the matrix B for the Crank-Nicholson method
        
        :return: The banded matrix B
        '''
        if not hasattr(self, 'K'):
            self.K = self.calculate_K()
        K = self.K
        B = np.zeros((self.npoints, self.npoints))
        for i in range(1, self.npoints-1):
            B[i, i-1] = K
            B[i, i]   = 1 - 2*K
            B[i, i+1] = K
        # Reflective Wall Boundary Conditions
        B[0, 0] = 1 - K
        B[0, 1] = K
        # Dirichlet Boundary Condition
        B[-1, -1] = 1
        return B
    
    def initialise_diffusion_parameters(self, dx: float, dt: float):
        '''
        Initialises the required diffusion parameters for the solution.
        '''
        self.K = self.calculate_K(dx, dt)
        self.A_banded = self.construct_A_banded()
        self.B = self.construct_B()

    def diffuse(self):
        '''
        Diffuses the solute.
        '''
        self.conc = solve_banded((1, 1), self.A_banded, self.B @ self.conc)


class Redox:
    '''
    Class for defining a redox couple
    
    :param label: Name of the redox couple
    :param E0: Standard electrode potential in V
    :param n: Number of electrons transferred in the redox reaction
    :param reduced species: The solute corresponding to the reduced form
    :param oxidised_species: The solute corresponding to the oxidised form
    :param k0: Standard rate constant in cm s⁻¹ (default 1.0)
    :param alpha: Transfer coefficient (default 0.5)
    '''
    def __init__(self, label: str, E0: float, n: int,
                 reduced_species: Solute, oxidised_species: Solute,
                 k0: float = 1.0, alpha: float = 0.5):
        self.label            = label
        self.E0               = E0
        self.n                = n
        self.reduced_species  = reduced_species
        self.oxidised_species = oxidised_species
        self.k0               = k0
        self.alpha            = alpha

    def electrode_potential(self, T=298.15) -> float:
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
    
    def calculate_electrode_rate_constants(self, electrode_potential,
                                           T: float = 298.15) -> tuple:
        '''
        Calculates the forward and backwards rate constants where ox + e⁻ → red
        is the forwards reaction.

        :param electrode_potential: The potential of the electrode in V
        :param T: Temperature in K
        '''
        Delta_E = electrode_potential - self.E0
        kf = self.k0 * np.exp(-self.alpha * Delta_E * F / (R * T))
        kb = self.k0 * np.exp((1-self.alpha) * Delta_E * F / (R*T))
        return kf, kb
    
    def equilibrium_concentration_constant(self, electrode_potential,
                                           T: float = 298.15) -> float:
        '''
        Uses the Nernst equation E = E0 - (RT/nF)ln([red]/[ox]) and returns the 
        value [red]/[ox] for the given electrode potential.

        :param electrode_potential: The electrode potential in V
        :param T: The temperature in K
        :return: The concentration constant
        '''
        K = np.exp(((self.n * F) / (R * T))*(self.E0 - electrode_potential))
        return K
    

class Reaction: 
    '''
    Class for defining a reaction between solutes.

    If a reaction is A + A → B + C then the reactants will be [A, A] and the
    products will be [B, C].
    
    :param label: The name of the reaction
    :param reactants: A list of solutes which are reactants in the reaction
    :param products: A list of solutes which are the products of the reaction
    :param rate_constant: The rate constant of the reaction
    '''
    def __init__(self, label: str, reactants: List[Solute],
                 products: List[Solute], rate_constant: float):
        self.label = label
        self.reactants = reactants
        self.products = products
        self.rate_constant = rate_constant

    def rate(self, n) -> np.ndarray:
        '''
        This function takes the concentrations of the reactants everywhere and
        calculates the rate of the reaction at each point in the simulation.

        :param n: The number of grid points in the simulation
        :return: The rate of the reaction at each point in the simulation
        '''
        rate = self.rate_constant * np.ones(n)
        for reactant in self.reactants:
            rate *= reactant.conc
        return rate


class Solution:
    '''
    Class which contains all necessary detail
    '''
    def __init__(self):
        self.solutes:   List[Solute]    = []
        self.redoxes:   List[Redox]     = []
        self.reactions: List[Reaction]  = []
        self.electrode_potential        = 0.0

    def add_solutes(self, *solutes: Solute):
        for solute in solutes: self.solutes.append(solute)

    def add_redoxes(self, *redoxes: Redox):
        for redox in redoxes: self.redoxes.append(redox)

    def add_reactions(self, *reactions: Reaction):
        for reaction in reactions: self.reactions.append(reaction)

    def change_potential(self, Delta_E: float):
        '''
        Changes the electrode potential by Delta_E in V

        :param Delta_E: The desired change in electrode potential
        '''
        self.electrode_potential += Delta_E

    def initialise(self, width: float, dx: float,
                                  dt: float):
        '''
        Sets up parameters for the simulation including the width of the 
        simulation in cm and the spacing of the grid in cm.
        Each solute in the solution is set to their bulk concentrations

        :param width: The width of the simulation in cm
        :param dx: The spacing of the grid in cm
        :param dt: The time-step spacing which will be considered in seconds
        '''
        self.dx = dx
        self.dt = dt
        # Choose number of grid points to get close to desired width as possible
        npoints = int(width / dx)
        # For each solute, set up the concentrations
        for solute in self.solutes:
            solute.initialise_concentration(npoints)
            solute.initialise_diffusion_parameters(dx, dt)
        # Save the number of grid points
        self.npoints = npoints
        # Save the redox connectiveness
        self.redox_connectiveness = self.extract_connected_groups(
                                            self.redox_connectiveness_matrix())
    
    def diffuse(self):
        '''
        Diffuses all solutes in the solution
        '''
        # For each solute diffuse, set the right limit to be equal to the bulk.
        for solute in self.solutes:
            solute.diffuse()

    def electrode_kinetics_matrix(self) -> np.ndarray:
        '''
        Constructs the electrode kinetics matrix, K
        '''
        electrode_potential = self.electrode_potential
        n_solutes = len(self.solutes)
        ids = {self.solutes[i]: i for i in range(n_solutes)}
        K = np.zeros((n_solutes, n_solutes))
        for r in self.redoxes:
            red, ox = ids[r.reduced_species], ids[r.oxidised_species]
            kf,  kb = r.calculate_electrode_rate_constants(electrode_potential)
            # kf is rate constant of ox + e⁻ → red
            # kb is rate constant of red → ox + e⁻
            K[ox, ox ], K[red, red] = -kf, -kb
            K[ox, red], K[red, ox ] =  kb,  kf
        return K
    
    def electron_transfer(self):
        '''
        Allows all of the electron transfer reactions to happen at the electrode
        '''
        def kinetics_matrix(t, C):
            for conc, solute in zip(C, self.solutes):
                solute.conc[0] = conc
            return np.dot(self.electrode_kinetics_matrix(), C)

        # Use stiff integrator to calculate concentration change at the
        # electrode over the time period self.dt
        # Use BDF method and calculate over ten points in the dt segment
        C0 = [s.conc[0] for s in self.solutes]
        C1 = solve_ivp(kinetics_matrix, (0, self.dt),
                       C0, method='BDF').y[:,-1]
        for C, solute in zip(C1, self.solutes): solute.conc[0] = C


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
        n_components, labels = connected_components(csgraph=graph, directed=False)

        # Group indices by component label
        groups = [[] for _ in range(n_components)]
        for idx, label in enumerate(labels):
            groups[label].append(idx)
        
        return groups
    
    def redox_connectiveness_matrix(self) -> np.ndarray:
        '''
        Constructs a matrix which shows the connectivity of the redox couples
        in the solution. The matrix is symmetric and has a 1 if there is a
        redox couple connecting two solutes, and 0 otherwise.
        
        :return: A square matrix showing the connectivity of redox couples
        '''
        n_solutes = len(self.solutes)
        M = np.eye(n_solutes)
        ids = {self.solutes[i]: i for i in range(n_solutes)}
        
        for r in self.redoxes:
            red, ox = ids[r.reduced_species], ids[r.oxidised_species]
            M[red, ox] = M[ox, red] = 1
        return M
    
    def redox_concentration_scaling_matrix(self) -> np.ndarray:
        '''
        Calculates the matrix corresponding to the relative concentrations
        required to achieve Nernstian equilibrium.
        '''
        M = np.eye(len(self.solutes))
        n_solutes = len(self.solutes)
        ids = {self.solutes[i]: i for i in range(n_solutes)}
        for r in self.redoxes:
            red, ox = ids[r.reduced_species], ids[r.oxidised_species]
            K = r.equilibrium_concentration_constant(self.electrode_potential)
            M[red, ox], M[ox, red] = 1/K, K
        return M
    
    def group_to_conc_scalings(self, group, scaling_matrix):
        '''
        This function is courtesy of chatGPT.
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

        # Check if any solute remains unconnected (shouldn’t happen in connected group)
        if any(c is None for c in rel_conc):
            raise ValueError("Incomplete connection in redox scaling group.")

        return rel_conc
        
    def Nernstian_equilibrium(self):
        '''
        Goes through all of the groups connected by redox potentials and sets
        the Nernstian equilibrium
        '''
        scaling_matrix = self.redox_concentration_scaling_matrix()
        for group in self.redox_connectiveness:
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
    
    def run_chemical_reactions(self, time_split: int = 1):
        '''
        Runs all of the chemical reactions by iterating through the reactions, 
        calculating the rates of the reaction at every point in the simultaion
        and then updating the concentrations of the reactants and products.
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

    def reaction_concentration_time_gradients(self):
        '''
        Calculate the instantaneous concentration gradients for all of the 
        solutes
        '''
        solute_to_index = {s: i for i, s in enumerate(self.solutes)}
        n = len(self.solutes[0].conc)
        g = np.zeros(n)
        solute_conc_gradients = [g.copy() for _ in self.solutes]
        # Iterate through each reaction and update the conc gradients
        for reaction in self.reactions:
            rate = reaction.rate(n)
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


    def run(self):
        '''
        Runs the simultaion for one time step.
        '''
        # First come to Nernstian equilibrium
        self.Nernstian_equilibrium()
        # Then diffuse all solutes
        self.diffuse()


    def cyclic_voltammetry(self, E_min: float, E_max: float, scan_rate: float,
                           n_cycles: int = 1, T: float = 298.15,
                           start_positive_scan: bool = True
                           ) -> List[np.ndarray]:
        '''
        Models a cyclic voltammetry experiment and returns the voltage and the
        current data.

        :param E_min: The minimum potential in V
        :param E_max: The maximum potential in V
        :param scan_rate: The scan rate in V s⁻¹
        :param n_cycles: The number of cycles to run (default 1)
        :param T: The temperature in K (default 298.15)
        :param start_positive_scan: Whether to start the scan in the positive direction
            (default True)
        :return: A list of two lists, the first being the voltages and the second
            being the currents at each voltage.
        '''
        # Initialise the voltages and currents arrays
        potentials = []
        currents = []
        # Calculate what the step size is in V
        dE = scan_rate * self.dt
        # Record the initial potential
        E_start = self.electrode_potential
        # Calculate what the potentials should be for each cycle
        if start_positive_scan:
            cycle_potentials = np.arange(E_start, E_max + dE, dE).tolist() + \
                               np.arange(E_max, E_min - dE, -dE).tolist()  + \
                               np.arange(E_min, E_start + dE, dE).tolist()
        else:
            cycle_potentials = np.arange(E_start, E_min - dE, -dE).tolist() + \
                               np.arange(E_min, E_max + dE, dE).tolist() + \
                               np.arange(E_max, E_start + dE, -dE).tolist()
        # Run the cycles
        for cycle in range(n_cycles):
            for E in cycle_potentials:
                # Calculate the total charge at the electrode
                charge_before = self.total_solute_charge_at_electrode()
                # Set the electrode potential
                self.electrode_potential = E
                # Allow Nernstian equilibrium to occur
                self.Nernstian_equilibrium()
                # Calculate the total charge at the electrode after equilibrium
                charge_after = self.total_solute_charge_at_electrode()
                # Calculate the current as the change in charge over the time step
                current = (charge_after - charge_before) / self.dt
                # Append the potential and current to the lists
                potentials.append(E)
                # Allow the chemical reactions to occur
                self.run_chemical_reactions()
                # Append the current to the list
                currents.append(current)
                # Allow the solutes to diffuse
                self.diffuse()

        # Return the potentials and currents
        return potentials, currents












    



    