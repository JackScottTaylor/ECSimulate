'''
This package is designed to use the simulation capabilities of ECSimulate to try
and extract parameters from experimental data.
'''

import numpy as np
from typing import Tuple
from rich.progress import Progress

from .Solution import Solution

def interpolated_voltage_current(
        dt: float,
        exp_times: np.ndarray,
        exp_currents: np.ndarray,
        exp_voltages: np.ndarray, 
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    The current and voltage arrays are extrapolated linearly to find the
    expected experimental data evenly spaced.
    The time step in seconds, and the extracted current and voltage array are
    returned.

    :param exp_times: Experimental time data
    :param exp_currents: Experimental current data
    :param exp_voltages: Experimental voltage data

    :return: Tuple[times, voltages, currents]
    '''
    n_points = int((exp_times[-1] - exp_times[0]) / dt) + 1
    times = np.array([exp_times[0] + i * dt for i in range(n_points)])
    voltages = np.interp(times, exp_times, exp_voltages)
    currents = np.interp(times, exp_times, exp_currents)
    return times, voltages, currents


def RSS_error(
        exp_currents: np.ndarray,
        sim_currents: np.ndarray,
    ) -> float:
    '''
    Calculate the residual sum of squares error between the simulated and 
    interpolated experimental currents.

    :param exp_currents: Experimental current data
    :param sim_currents: Simulated current data
    :return: Residual sum of squares error
    '''
    return np.sum((exp_currents - sim_currents) ** 2)


def cyclic_voltammetry_specified_potentials(
        dt: float,
        potentials: np.ndarray,
        solution: Solution
    ) -> np.ndarray:
    
    currents = []
    with Progress() as progress:
        task = progress.add_task("[cyan]Simulating CV...", total=len(potentials))
        for E in potentials:
            solution.electrode_potential = E
            solution.Nernstian_equilibrium()
            solution.diffuse_coupled_kinetics()
            current = solution.current_from_flux()
            currents.append(current)
            progress.update(task, advance=1)

    currents = np.array(currents)
    return currents





