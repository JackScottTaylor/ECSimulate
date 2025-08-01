'''
Contains the CV methods for the solution class
'''

from typing               import List, TYPE_CHECKING
from rich.progress        import Progress

import numpy as np

if TYPE_CHECKING:
    from .Solution import Solution

class SolutionCVMethods:
    def cyclic_voltammetry(
            self: "Solution",
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
            self: "Solution",
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
    

    def cyclic_voltammetry_adaptive_time(
            self: "Solution",
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
            # Set the electrode potential
            self.electrode_potential = E
            # Allow Nernstian equilibrium to occur
            self.Nernstian_equilibrium()
            # Have diffusion and chemical reactions occur at same time.        
            self.diffuse_coupled_kinetics()
            # Calculate the current using flux
            current = self.current_from_flux()
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

        # Return the potentials and currents
        if save_concs:
            for s in self.solutes:
                concentrations[s.label] = np.array(concentrations[s.label])
            return potentials[:index], currents[:index], concentrations
        if assess_time_steps:
            return potentials[:index], currents[:index], delta_ts
        return potentials[:index], currents[:index]
    

    def cyclic_voltammetry_kinetics_coupled_flux_current(
            self: "Solution",
            E_min: float,
            E_max: float,
            scan_rate: float,
            n_cycles: int = 1,
            T: float = 298.15,
            start_positive_direction: bool = True,
            save_concs: bool = False,
            target_n_points = 1000
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
        if save_concs: concentrations = {s.name: [] for s in self.solutes}
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
        max_counter = np.ceil(n_points // target_n_points)
        with Progress() as progress:
            task = progress.add_task("[cyan]Running Cyclic Voltammetry...",
                                     total=n_points)
            counter = 0
            for cycle in range(n_cycles):
                for E in cycle_potentials:
                    # Set the electrode potential
                    self.electrode_potential = E
                    # Allow Nernstian equilibrium to occur
                    self.Nernstian_equilibrium()
                    # Have diffusion and chemical reactions occur at same time.        
                    #self.diffusion_Strang()
                    self.diffuse_coupled_kinetics()
                    # Calculate current using the fluxes
                    current = self.current_from_flux()
                    # Append the potential and current to the lists if time to
                    if int(counter % target_n_points) == 0:
                        potentials.append(E); currents.append(current)

                    # If saving concentrations, append the concentrations
                    if save_concs:
                        for solute in self.solutes:
                            concentrations[solute.name].append(
                                solute.conc.copy()
                                )
                            
                    progress.update(task, advance=1)

        # Return the potentials and currents
        if save_concs:
            for s in self.solutes:
                concentrations[s.name] = np.array(concentrations[s.name])
            return np.array(potentials), np.array(currents), concentrations
        return np.array(potentials), np.array(currents)
    

    def cyclic_voltammetry_kinetics_coupled_flux_current_grad_logging(
            self: "Solution",
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
        if save_concs: concentrations = {s.name: [] for s in self.solutes}
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

        # Initialise lists to record maximum gradients
        diffusion_grads = []
        kinetic_grads = []

        with Progress() as progress:
            task = progress.add_task("[cyan]Running Cyclic Voltammetry...",
                                     total=n_points)
            for cycle in range(n_cycles):
                for E in cycle_potentials:
                    # Set the electrode potential
                    self.electrode_potential = E
                    # Allow Nernstian equilibrium to occur
                    self.Nernstian_equilibrium()

                    max_diff_grad = 0
                    max_kinetic_grad = 0

                    for solute in self.solutes:
                        max_diff_grad = max(
                                    np.max(solute.diffusion_conc_gradient(self.dx)),
                                    max_diff_grad
                                    )
                    diffusion_grads.append(max_diff_grad)

                    
                    kinetic_gradients = self.reaction_concentration_time_gradients()
                    for grad in kinetic_gradients:
                        max_kinetic_grad = max(
                            np.max(grad),
                            max_kinetic_grad
                        )
                    kinetic_grads.append(max_kinetic_grad)


                    # Have diffusion and chemical reactions occur at same time.for solute in self.solutes:  
                    self.diffuse_coupled_kinetics()
                    # Calculate current using the fluxes
                    current = self.current_from_flux()
                    # Append the potential and current to the lists
                    potentials.append(E); currents.append(current)

                    # If saving concentrations, append the concentrations
                    if save_concs:
                        for solute in self.solutes:
                            concentrations[solute.name].append(
                                solute.conc.copy()
                                )
                            
                    progress.update(task, advance=1)

        return np.array(potentials), np.array(currents), np.array(diffusion_grads), np.array(kinetic_grads)