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
            rich_progress_bar: bool = True,
            strang: bool = False
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
        :param rich_progress_bar: Whether to use a rich progress bar (default True)
        :return: A list of two `np.ndarrays`, the first being the voltages and the
            second being the currents at each voltage. 
        '''
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

        print(f"Cyclic Voltammetry Simulation")
        print(n_points)

        nernst_eq = self.Nernstian_equilibrium
        flux_to_current = self.current_from_flux
        if strang:
            self.precompute_back_eulers()
            conc_updater = self.diffusion_Strang_BackEuler
        else:
            conc_updater = self.diffuse_coupled_kinetics

        # If using rich progress bar, use it to show progress
        if rich_progress_bar:
            update_interval = max(1, n_points // 5000)
            with Progress() as progress:
                task = progress.add_task("[cyan]Running Cyclic Voltammetry...",
                                            total=n_points)
                for index, E in enumerate(potentials):
                    # Set the electrode potential
                    self.electrode_potential = E
                    # Allow Nernstian equilibrium to occur at electrode surface
                    nernst_eq()
                    # Have diffusion and chemical reactions occur at same time.
                    conc_updater()
                    # Calculate current using the fluxes
                    # Append the potential and current to the lists
                    currents[index] = flux_to_current()
                    if index % update_interval == 0:
                        progress.update(task, advance=update_interval)
        # Else, just loop through the potentials
        else:
            for index, E in enumerate(potentials):
                # Set the electrode potential
                self.electrode_potential = E
                # Allow Nernstian equilibrium to occur at electrode surface
                nernst_eq()
                # Have diffusion and chemical reactions occur at same time.
                conc_updater()
                # Calculate current using the fluxes
                # Append the potential and current to the lists
                currents[index] = flux_to_current()

        return potentials, currents