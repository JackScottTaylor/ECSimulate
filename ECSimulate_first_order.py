'''
This is a slight reworking of the ECSimulate package which is designed for optimising
modelling CVs where all kinetics can be modeled as first order.
'''

from typing import List
import numpy as np

from numba import njit
from scipy.linalg import solve_banded
from scipy.sparse.csgraph import connected_components
from scipy.sparse         import csr_matrix
from collections          import deque

import time


def banded_to_full(ab, l, u):
    """
    Convert a banded matrix in solve_banded format to its full 2D form.

    Parameters
    ----------
    ab : ndarray, shape (l+u+1, n)
        Banded matrix (solve_banded storage).
    l : int
        Number of sub-diagonals.
    u : int
        Number of super-diagonals.

    Returns
    -------
    A : ndarray, shape (n, n)
        Full square matrix.
    """
    n = ab.shape[1]
    A = np.zeros((n, n), dtype=ab.dtype)
    for j in range(n):
        for i in range(max(0, j - u), min(n, j + l + 1)):
            A[i, j] = ab[u + i - j, j]
    return A



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


class Reaction:
    def __init__(self, name: str, reactant: Solute, product: Solute, k: float):
        self.reactant = reactant
        self.product = product
        self.k = k
        self.name = name


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


class CVSimulator:
    def __init__(
            self,
            solutes: List[Solute],
            reactions: List[Reaction],
            redoxes: List[Redox],
            dxs: List[float],
            dt: float,
            electrode_area = np.pi * 1.5 ** 2
    ):
        self.electrode_area = electrode_area
        self.solutes = solutes
        self.reactions = reactions
        self.redoxes = redoxes
        self.dxs = dxs
        self.dt = dt

        self.solute_to_index = {solute.name: i for i, solute in enumerate(solutes)}
        self.nsolutes = len(solutes)

        self.nconcs = len(dxs) + 1
        self.total_nconcs = self.nconcs * self.nsolutes
        
        self.width = np.sum(dxs)

        self.C = np.zeros(self.nsolutes*self.nconcs)
        self.initialise_concentrations()

        # Initialise the banded matrices for iterating diffusion and reactions.
        nbands = 2 * self.nsolutes + 1
        self.A_banded = np.zeros((nbands, self.total_nconcs))
        self.B_banded = np.zeros((nbands, self.total_nconcs))
        self.apply_diffusion_contribution_to_A_banded()
        self.apply_diffusion_contribution_to_B_banded()
        self.apply_reaction_contribution_to_A_banded()
        self.apply_reaction_contribution_to_B_banded()
        
        self.construct_B_full()

        self.l_and_u = (self.nsolutes, self.nsolutes)

        self.redox_connectivity = self.extract_connected_groups(
            self.redox_connectivity_matrix()
        )

        self.electrode_potential = 0.0

        self.solute_charges = np.array([solute.charge for solute in self.solutes])
        self.solute_Ds = np.array([solute.D for solute in self.solutes])

        self.first_pos_idx = self.nsolutes
        self.second_pos_idx = self.nsolutes * 2

        self.current_scaling_factor = F * self.electrode_area / self.dxs[1] * 1e-2


    def initialise_concentrations(self):
        '''
        Concentration of all species is stored in the single vector `self.C`.
        The vector is ordered by position, not solute.
        Therefore order is A0,B0,C0,A1,B1,C1,...

        This sets the concentrations of all the solutes to their bulk
        concentrations initially.
        '''
        for x in range(self.nconcs):
            for i in range(self.nsolutes):
                idx = x * self.nsolutes + i
                self.C[idx] = self.solutes[i].bulk_conc


    def matrix_coords_to_banded(self, i, j):
        '''
        Converts matrix coordinates (i,j) to the coordinates in the banded
        matrix
        '''
        upper_diags = self.nsolutes
        return upper_diags + i - j, j
    

    def banded_coords_to_matrix(self, i, j):
        upper_diags = self.nsolutes
        return i - upper_diags + j, j
    

    def construct_B_full(self):
        B_full = np.zeros((self.total_nconcs, self.total_nconcs))
        for band in range(2*self.nsolutes+1):
            for j in range(self.nconcs):
                i_M, j_M = self.banded_coords_to_matrix(band, j)
                if i_M < 0 or i_M >= self.total_nconcs: continue
                B_full[i_M, j_M] = self.B_banded[band, j]
        self.B = B_full

    def apply_diffusion_contribution_to_A_banded(self):
        dxs = self.dxs
        dt = self.dt

        mid = self.nsolutes
        upp = 0
        low = -1


        dx_j1s, dx_js = dxs[0:-1], dxs[1:]

        for i, solute in enumerate(self.solutes):
            j = 1
            for dx_j1, dx_j in zip(dx_j1s, dx_js):
                K = solute.K(dt, dx_j, dx_j1)
                C_idx = self.nsolutes * j + i
                self.A_banded[upp, C_idx+self.nsolutes] = -K * dx_j1
                self.A_banded[mid, C_idx] = 1 + K * (dx_j1 + dx_j)
                self.A_banded[low, C_idx-self.nsolutes] = -K * dx_j
                j += 1

            # Apply boundary conditions
            self.A_banded[mid, i]   = 1
            self.A_banded[mid, -i-1]  = 1


    def apply_diffusion_contribution_to_B_banded(self):
        dxs = self.dxs
        dt = self.dt

        mid = self.nsolutes
        upp = 0
        low = -1

        dx_j1s, dx_js = dxs[0:-1], dxs[1:]

        for i, solute in enumerate(self.solutes):
            j = 1
            for dx_j1, dx_j in zip(dx_j1s, dx_js):
                K = solute.K(dt, dx_j, dx_j1)
                C_idx = self.nsolutes * j + i
                self.B_banded[upp, C_idx+self.nsolutes] = +K * dx_j1
                self.B_banded[mid, C_idx] = 1 - K * (dx_j1 + dx_j)
                self.B_banded[low, C_idx-self.nsolutes] = +K * dx_j
                j += 1

            # Apply boundary conditions
            self.B_banded[mid, i]   = 1
            self.B_banded[mid, -i-1]  = 1


    def apply_reaction_contribution_to_A_banded(self):
        s2i = self.solute_to_index
        mid = self.nsolutes
        for r in self.reactions:
            val = self.dt * r.k / 2
            react, prod = r.reactant, r.product
            for x in range(self.nconcs):
                r_idx, p_idx = x * self.nsolutes + s2i[react.name], x * self.nsolutes + s2i[prod.name]

                # First handle the reactant
                i, j = self.matrix_coords_to_banded(r_idx, r_idx)
                self.A_banded[i, j] += val

                # Then the product
                i, j = self.matrix_coords_to_banded(p_idx, r_idx)
                self.A_banded[i, j] -= val


    def apply_reaction_contribution_to_B_banded(self):
        s2i = self.solute_to_index
        mid = self.nsolutes
        for r in self.reactions:
            val = self.dt * r.k / 2
            react, prod = r.reactant, r.product
            for x in range(self.nconcs):
                r_idx, p_idx = x * self.nsolutes + s2i[react.name], x * self.nsolutes + s2i[prod.name]

                # First handle the reactant
                i, j = self.matrix_coords_to_banded(r_idx, r_idx)
                self.B_banded[i, j] -= val

                # Then the product
                i, j = self.matrix_coords_to_banded(p_idx, r_idx)
                self.B_banded[i, j] += val


    def BC(self):
        return self.B @ self.C
    

    def match_electrode_concs(self):
        for i in range(self.nsolutes):
            self.C[i] = self.C[i+self.nsolutes]


    def diffuse(self):
        BC = self.BC()
        self.C = solve_banded(self.l_and_u, self.A_banded, BC)
        self.match_electrode_concs()


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

    
    def redox_conc_scaling_matrix(self) -> np.ndarray:
        '''
        Calculates the matrix corresponding to the relative concentrations
        required to achieve Nernstian equilibrium.
        '''
        M = np.eye(self.nsolutes)
        ids = self.solute_to_index
        for r in self.redoxes:
            red, ox = ids[r.reduced_species.name], ids[r.oxidised_species.name]
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
            total_conc = sum(self.C[i] for i in group)
            # Reference concentration is the first solute in the group
            ref_conc = self.C[group[0]]
            # Set the concentrations of the solutes in the group
            for solute in group:
                self.C[solute] = ref_conc * rel_conc[solute]
            # Set the total concentration to be the same as previously
            new_total_conc = sum(self.C[i] for i in group)
            global_conc_scaling = total_conc / new_total_conc
            # Scale the concentrations to match the total concentration
            for solute in group:
                self.C[solute] *= global_conc_scaling


    def current_from_flux(self) -> float:
        current = 0.0
        for charge, D, i in zip(
                self.solute_charges,
                self.solute_Ds,
                range(self.nsolutes)
            ):
            if charge == 0.0: continue
            C = (self.C[self.second_pos_idx+i] - self.C[self.first_pos_idx+i])
            C *= charge * D
            current += C
        return current * self.current_scaling_factor


    def runCV(self,
              E_min, E_max, scan_rate, n_cycles, start_positive_direction=False):
        '''Runs the Cyclic Voltammetry Simulation'''

        # Calculate what the step size is in V
        dE = scan_rate * self.dt
        # Record the initial potential
        E_start = self.electrode_potential
        # Calculate what the potentials should be for each cycle depending on 
        # the initial scan direction.
        if start_positive_direction:
            cycle_potentials = np.concatenate([
                                        np.arange(E_start, E_max   + dE,  dE),
                                        np.arange(E_max,   E_min   - dE, -dE),
                                        np.arange(E_min,   E_start + dE,  dE)
                                        ])

        else:
            cycle_potentials = np.concatenate([
                                        np.arange(E_start, E_min   - dE, -dE),
                                        np.arange(E_min,   E_max   + dE,  dE),
                                        np.arange(E_max,   E_start + dE, -dE)
                                        ])
            
        # Initialise the voltages and currents arrays
        potentials  = np.tile(cycle_potentials, n_cycles)
        n_points    = len(potentials)
        currents    = np.zeros(n_points)

        start_time = time.time()
        hundredth_interval = n_points // 100
        for index, E in enumerate(potentials):
            self.electrode_potential = E
            self.Nernstian_equilibrium()
            self.diffuse()
            currents[index] = self.current_from_flux()
            print(max(self.C))
            if index % hundredth_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Progress: {index / n_points * 100:.0f}%, Time elapsed: {elapsed_time:.2f}s")

        print(f"Total time for CV: {time.time() - start_time:.2f}s")
        return potentials, currents
