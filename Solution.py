from .Solute    import Solute
from .Redox     import Redox
from .Reaction  import Reaction

from typing               import List
from scipy.sparse.csgraph import connected_components
from scipy.sparse         import csr_matrix
from collections          import deque
from rich.progress        import Progress

import numpy as np

class Solution:
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
            dt:    float
        ) -> None:
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
            solute.diffuse_coupled_kinetics(gradient)

    
    def cyclic_voltammetry(
            self,
            E_min: float,
            E_max: float,
            scan_rate: float,
            n_cycles: int = 1,
            T: float = 298.15,
            start_positive_direction: bool = True,
            kinetics_time_split: int = 1,
            save_concs: bool = False,
        ) -> List[np.ndarray]:
        '''
        Models a cyclic voltammetry experiment and returns the voltage and the
        current data.

        :param E_min: The minimum potential in V
        :param E_max: The maximum potential in V
        :param scan_rate: The scan rate in V s⁻¹
        :param n_cycles: The number of cycles to run (default 1)
        :param T: The temperature in K (default 298.15)
        :param start_positive_direction: Whether to start the scan in the
            positive direction (default True)
        :param kinetics_time_split: The number of times to run chemical
            reactions per time step
        :param save_concs: Whether to save and return the concentrations at 
            each potential (default False)
        :return: A list of two lists, the first being the voltages and the
            second being the currents at each voltage. If save_concs then a
            dictionary is returned with keys corresponding to solute labels
            and the values being an array of the concentrations at different
            times.
        '''
        # Initialise the voltages and currents arrays
        potentials  = []
        currents    = []
        # If saving concentrations, initialise the concentrations list
        if save_concs: concentrations = {s.label: [] for s in self.solutes}
        # Calculate what the step size is in V
        dE = scan_rate * self.dt
        # Record the initial potential
        E_start = self.electrode_potential
        # Calculate what the potentials should be for each cycle
        if start_positive_direction:
            cycle_potentials = np.arange(E_start, E_max   + dE,  dE).tolist() +\
                               np.arange(E_max,   E_min   - dE, -dE).tolist() +\
                               np.arange(E_min,   E_start + dE,  dE).tolist()
        else:
            cycle_potentials = np.arange(E_start, E_min   - dE, -dE).tolist() +\
                               np.arange(E_min,   E_max   + dE,  dE).tolist() +\
                               np.arange(E_max,   E_start + dE, -dE).tolist()
        # Run the cycles
        n_points = len(cycle_potentials) * n_cycles
        with Progress() as progress:
            task = progress.add_task("[cyan]Running Cyclic Voltammetry...",
                                     total=n_points)
            for cycle in range(n_cycles):
                for E in cycle_potentials:
                    # Calculate the total charge at the electrode
                    charge_before = self.total_solute_charge_at_electrode()
                    # Set the electrode potential
                    self.electrode_potential = E
                    # Allow Nernstian equilibrium to occur
                    self.Nernstian_equilibrium()
                    # Calculate total charge at the electrode after equilibrium
                    charge_after = self.total_solute_charge_at_electrode()
                    # Calculate  current as change in charge over the time step
                    current = (charge_after - charge_before) / self.dt
                    # Append the potential and current to the lists
                    potentials.append(E); currents.append(current)

                    # If saving concentrations, append the concentrations
                    if save_concs:
                        for solute in self.solutes:
                            concentrations[solute.label].append(
                                solute.conc.copy()
                                )
                            
                    # Allow the chemical reactions to occur
                    self.run_chemical_reactions(time_split=kinetics_time_split)
                    # Allow the solutes to diffuse
                    self.diffuse()
                    progress.update(task, advance=1)

        # Return the potentials and currents
        if save_concs:
            for s in self.solutes:
                concentrations[s.label] = np.array(concentrations[s.label])
            return np.array(potentials), np.array(currents), concentrations
        return np.array(potentials), np.array(currents)
    

    def cyclic_voltammetry_kinetics_coupled(
            self,
            E_min: float,
            E_max: float,
            scan_rate: float,
            n_cycles: int = 1,
            T: float = 298.15,
            start_positive_direction: bool = True,
            save_concs: bool = False,
        ) -> List[np.ndarray]:
        '''
        Models a cyclic voltammetry experiment and returns the voltage and the
        current data.

        :param E_min: The minimum potential in V
        :param E_max: The maximum potential in V
        :param scan_rate: The scan rate in V s⁻¹
        :param n_cycles: The number of cycles to run (default 1)
        :param T: The temperature in K (default 298.15)
        :param start_positive_direction: Whether to start the scan in the
            positive direction (default True)
        :param save_concs: Whether to save and return the concentrations at 
            each potential (default False)
        :return: A list of two lists, the first being the voltages and the
            second being the currents at each voltage. If save_concs then a
            dictionary is returned with keys corresponding to solute labels
            and the values being an array of the concentrations at different
            times.
        '''
        # Initialise the voltages and currents arrays
        potentials  = []
        currents    = []
        # If saving concentrations, initialise the concentrations list
        if save_concs: concentrations = {s.label: [] for s in self.solutes}
        # Calculate what the step size is in V
        dE = scan_rate * self.dt
        # Record the initial potential
        E_start = self.electrode_potential
        # Calculate what the potentials should be for each cycle
        if start_positive_direction:
            cycle_potentials = np.arange(E_start, E_max   + dE,  dE).tolist() +\
                               np.arange(E_max,   E_min   - dE, -dE).tolist() +\
                               np.arange(E_min,   E_start + dE,  dE).tolist()
        else:
            cycle_potentials = np.arange(E_start, E_min   - dE, -dE).tolist() +\
                               np.arange(E_min,   E_max   + dE,  dE).tolist() +\
                               np.arange(E_max,   E_start + dE, -dE).tolist()
        # Run the cycles
        n_points = len(cycle_potentials) * n_cycles
        with Progress() as progress:
            task = progress.add_task("[cyan]Running Cyclic Voltammetry...",
                                     total=n_points)
            for cycle in range(n_cycles):
                for E in cycle_potentials:
                    # Calculate the total charge at the electrode
                    charge_before = self.total_solute_charge_at_electrode()
                    # Set the electrode potential
                    self.electrode_potential = E
                    # Allow Nernstian equilibrium to occur
                    self.Nernstian_equilibrium()
                    # Calculate total charge at the electrode after equilibrium
                    charge_after = self.total_solute_charge_at_electrode()
                    # Calculate  current as change in charge over the time step
                    current = (charge_after - charge_before) / self.dt
                    # Append the potential and current to the lists
                    potentials.append(E); currents.append(current)

                    # If saving concentrations, append the concentrations
                    if save_concs:
                        for solute in self.solutes:
                            concentrations[solute.label].append(
                                solute.conc.copy()
                                )

                    # Have diffusion and chemical reactions occur at same time.        
                    self.diffuse_coupled_kinetics()
                    progress.update(task, advance=1)

        # Return the potentials and currents
        if save_concs:
            for s in self.solutes:
                concentrations[s.label] = np.array(concentrations[s.label])
            return np.array(potentials), np.array(currents), concentrations
        return np.array(potentials), np.array(currents)
    

    def cyclic_voltammetry_adaptive_time(self,
            E_min: float,
            E_max: float,
            scan_rate: float,
            n_cycles: int = 1,
            T: float = 298.15,
            start_positive_direction: bool = True,
            save_concs: bool = False,
            delta_t_min = 1e-4,
            delta_t_max = 1e-2,
            err_val = 0.01,
            assess_time_steps: bool = False
        ) -> List[np.ndarray]:
        '''
        Models a cyclic voltammetry experiment and returns the voltage and the
        current data.

        :param E_min: The minimum potential in V
        :param E_max: The maximum potential in V
        :param scan_rate: The scan rate in V s⁻¹
        :param n_cycles: The number of cycles to run (default 1)
        :param T: The temperature in K (default 298.15)
        :param start_positive_direction: Whether to start the scan in the
            positive direction (default True)
        :param save_concs: Whether to save and return the concentrations at 
            each potential (default False)
        :param delta_t_min: The minimum time step value
        :param delta_t_max: The maximum time step value
        :return: A list of two lists, the first being the voltages and the
            second being the currents at each voltage. If save_concs then a
            dictionary is returned with keys corresponding to solute labels
            and the values being an array of the concentrations at different
            times.
        '''
        # If saving concentrations, initialise the concentrations list
        if save_concs: concentrations = {s.label: [] for s in self.solutes}

        max_number_of_points = int((E_max - E_min)*n_cycles*1.1/ delta_t_min)
        # Initialise the voltages and currents arrays
        potentials  = np.zeros(max_number_of_points)
        currents    = np.zeros(max_number_of_points)

        potentials[0] = self.electrode_potential

        n_half_cycles = n_cycles * 2
        minimum_max_grad = err_val / delta_t_max
        maximum_max_grad = err_val / delta_t_min

        if assess_time_steps: delta_ts = [delta_t_min]

        scan_direction = + 1
        if not start_positive_direction: scan_direction = -1

        cycles_completed = 0
        index = 1
        while cycles_completed != n_half_cycles:
            # Calculate the diffusion time gradients
            max_grad = minimum_max_grad
            # Go through the solutes and find the maximum conc_change due to 
            # diffusion
            max_diff_grad = 0
            for solute in self.solutes:
                max_grad = max(
                            np.max(solute.diffusion_conc_gradient(self.dx)),
                            max_grad
                            )
                max_diff_grad = max(
                            np.max(solute.diffusion_conc_gradient(self.dx)),
                            max_diff_grad
                            )
            # Now find the maximum conc changes due to kinetics
            # We only save the greates change
            kinetic_gradients = self.reaction_concentration_time_gradients()
            for grad in kinetic_gradients:
                max_grad = max(
                    np.max(grad),
                    max_grad
                )
            max_grad = min(max_grad, maximum_max_grad)
            # Now calculate the time step determined by err_val / max_grad. 
            dt = err_val / max_grad

            # if dt has changed then need to update the diffusion matrices for
            # each solute
            if dt != self.dt:
                for solute in self.solutes:
                    solute.save_diffusion_matrices(self.dx, dt)
                self.dt = dt

            # Calculate the next potential
            E = self.electrode_potential + scan_rate * scan_direction * self.dt
            # Calculate the total charge at the electrode
            charge_before = self.total_solute_charge_at_electrode()
            # Set the electrode potential
            self.electrode_potential = E
            # Allow Nernstian equilibrium to occur
            self.Nernstian_equilibrium()
            # Calculate total charge at the electrode after equilibrium
            charge_after = self.total_solute_charge_at_electrode()
            # Calculate  current as change in charge over the time step
            current = (charge_after - charge_before) / self.dt
            # Append the potential and current to the lists
            potentials[index] = E; currents[index] = current
            index += 1

            # Check if potential limits reached
            if scan_direction == 1 and E >= E_max:
                scan_direction = -1
                cycles_completed += 1
            elif scan_direction == -1 and E <= E_min:
                scan_direction = +1
                cycles_completed += 1

            if assess_time_steps: delta_ts.append(self.dt)

            # If saving concentrations, append the concentrations
            if save_concs:
                for solute in self.solutes:
                    concentrations[solute.label].append(
                        solute.conc.copy()
                        )

            # Have diffusion and chemical reactions occur at same time.        
            self.diffuse_coupled_kinetics()

        # Return the potentials and currents
        if save_concs:
            for s in self.solutes:
                concentrations[s.label] = np.array(concentrations[s.label])
            return potentials[:index], currents[:index], concentrations
        if assess_time_steps:
            return potentials[:index], currents[:index], delta_ts
        return potentials[:index], currents[:index]

            



                
                

