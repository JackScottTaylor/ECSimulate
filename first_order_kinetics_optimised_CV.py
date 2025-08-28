from .Solute import Solute
from .Redox import Redox
from .constants import FARADAY_CONSTANT
from .plt import plt
from .Reaction import Reaction

from typing import List
from scipy.linalg import solve_banded
from scipy.sparse.csgraph import connected_components
from scipy.sparse         import csr_matrix
from collections          import deque
import numpy as np

from scipy.linalg import lu_factor, lu_solve
from scipy.linalg.lapack import dgbtrf, dgbtrs

import time as time

from numba import njit

def chess_board_combine(*arrays):
    s = arrays[0].shape
    for A in arrays:
        assert A.shape == s, "Arrays must have same shape!"
    n_arrays = len(arrays)

    # Expand the array size correctly
    C = np.zeros(tuple(n_arrays * np.array(s)), dtype=arrays[0].dtype)

    for array_idx, array in enumerate(arrays):
        for index, val in np.ndenumerate(array):
            # Compute offset properly
            combo_index = tuple(i * n_arrays + array_idx for i in index)
            C[combo_index] = val
    return C


def banded_to_full(ab, nsolutes):
    """
    Convert a banded matrix (as used by solve_banded) to a full square matrix.

    ab : array_like, shape (kl + ku + 1, n)
    kl : int, number of subdiagonals
    ku : int, number of superdiagonals
    """
    kl, ku = nsolutes, nsolutes
    n = ab.shape[1]
    full = np.zeros((n, n), dtype=ab.dtype)
    
    for j in range(n):
        # superdiagonals
        for i in range(max(0, j-ku), j):
            full[i, j] = ab[ku + i - j, j]
        # main diagonal
        full[j, j] = ab[ku, j]
        # subdiagonals
        for i in range(j+1, min(n, j+kl+1)):
            full[i, j] = ab[ku + i - j, j]
    return full


class CVSimulator:
    def __init__(
            self,
            dxs: List[float],
            dt: float,
            solutes: List[Solute],
            redoxes: List[Redox],
            reactions: List[Reaction],
            A: float = np.pi * 1.5 ** 2
            ):
        
        self.solutes = solutes
        self.redoxes = redoxes
        self.reactions = reactions

        self.dxs = dxs
        self.npoints = len(self.dxs)+1
        for s in self.solutes: s.initialise_concentration(self.npoints)
        self.C = chess_board_combine(*[s.conc for s in self.solutes])

        self.nsolutes = len(self.solutes)
        self.dt = dt
        self.l_and_u = (self.nsolutes, self.nsolutes)

        self.ids = {self.solutes[i]: i for i in range(self.nsolutes)}

        self.construct_A()
        self.construct_B()

        self.add_reactions_contributions_to_A()
        self.add_reactions_contributions_to_B()

        self.construct_sparse_B()

        self.FAdx = - FARADAY_CONSTANT * A / (dxs[1] * 1e-2)
        self.electrode_potential = 0.0

        self.redox_connectivity = self.extract_connected_groups(
            self.redox_connectivity_matrix()
        )

        self.charges = np.array([s.charge for s in self.solutes])
        self.Ds = np.array([s.D for s in self.solutes])

    def matrix_coords_to_banded(self, i, j):
        return self.nsolutes + i - j, j

    def construct_A(self):
        banded_As = [
            s.construct_A_banded_format_variable_dx(
                                                    self.dxs, self.dt
                                                    ) for s in self.solutes
                ]
        self.A_banded = np.zeros(
            (2*self.nsolutes + 1, self.npoints*self.nsolutes)
            )
        
        self.A_banded[0] = chess_board_combine(
            *[A[0] for A in banded_As]
        )
        self.A_banded[self.nsolutes] = chess_board_combine(
            *[A[1] for A in banded_As]
        )
        self.A_banded[-1] = chess_board_combine(
            *[A[2] for A in banded_As]
        )

    def construct_B(self):
        Bs = [
            s.construct_B_variable_dx(
                self.dxs, self.dt
            ) for s in self.solutes
        ]
        self.B = chess_board_combine(*Bs)


    def add_reactions_contributions_to_B(self):
        for r in self.reactions:
            react, prod = r.reactants[0], r.products[0]
            r_idx, p_idx = self.ids[react], self.ids[prod]
            val = r.rate_constant * self.dt / 2
            for x in range(self.npoints):
                self.B[
                    x*self.nsolutes+r_idx,
                    x*self.nsolutes+r_idx
                    ] -= val
                self.B[
                    x*self.nsolutes+p_idx,
                    x*self.nsolutes+r_idx
                ] += val

    def add_reactions_contributions_to_A(self):
        for r in self.reactions:
            react, prod = r.reactants[0], r.products[0]
            r_idx, p_idx = self.ids[react], self.ids[prod]
            val = r.rate_constant * self.dt / 2
            for x in range(self.npoints):
                i,j = self.matrix_coords_to_banded(
                    x*self.nsolutes+r_idx,
                    x*self.nsolutes+r_idx
                    )
                self.A_banded[
                    i,j
                    ] += val
                i,j = self.matrix_coords_to_banded(
                    x*self.nsolutes+p_idx,
                    x*self.nsolutes+r_idx
                    )
                self.A_banded[
                    i,j
                ] -= val

    def construct_sparse_B(self):
        self.B_csr = csr_matrix(self.B)
        

    def update(self):
        self.C = solve_banded(
            self.l_and_u, self.A_banded, self.B_csr.dot(self.C)
            )
        self.C[:self.nsolutes] = self.C[2*self.nsolutes:3*self.nsolutes]
        

    def redox_connectivity_matrix(self) -> np.ndarray:
        '''
        Constructs a matrix which shows the connectivity of the redox couples
        in the solution. The matrix is symmetric and has a 1 if there is a
        redox couple connecting two solutes, and 0 otherwise.
        
        :return: A square matrix showing the connectivity of redox couples
        '''
        M   = np.eye(self.nsolutes)
        ids = self.ids
        
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
        groups = [np.array(group) for group in groups]
        
        return groups

    
    def redox_conc_scaling_matrix(self) -> np.ndarray:
        '''
        Calculates the matrix corresponding to the relative concentrations
        required to achieve Nernstian equilibrium.
        '''
        M = np.eye(self.nsolutes)
        ids = self.ids
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

        return np.array(rel_conc)
    

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
            total_conc = np.sum(self.C[group])
            # Reference concentration is the first solute in the group
            ref_conc = self.C[group[0]]
            # Set the concentrations of the solutes in the group
            self.C[group] = ref_conc * rel_conc
            # Set the total concentration to be the same as previously
            new_total_conc = np.sum(self.C[group])
            global_conc_scaling = total_conc / new_total_conc
            # Scale the concentrations to match the total concentration
            if new_total_conc != 0:
                self.C[group] *= global_conc_scaling
    
    def current(self):
        grads = self.C[2*self.nsolutes:3*self.nsolutes] - \
            self.C[self.nsolutes:2*self.nsolutes]
        return np.sum(grads * self.charges * self.Ds) * 1e-4 * self.FAdx
    
    def runCV(
            self,
            E_min: float,
            E_max: float,
            scan_rate: float,
            n_cycles: int = 1,
            T: float = 298.15,
            start_positive_direction: bool = True,
        ):
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

        start = time.time()
        iteration_interval = int(n_points // 100)
        next_interval = iteration_interval
        for index, E in enumerate(potentials):
            self.electrode_potential = E
            self.Nernstian_equilibrium()
            self.update()
            currents[index] = self.current()

            if index == next_interval:
                print(f"{index*100 // n_points}% Completed, {time.time()-start} s")
                next_interval += iteration_interval

        print ("Time Taken: ", time.time() - start)
        return potentials, currents

    