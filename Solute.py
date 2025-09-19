import numpy as np
from typing import List
from numpy.typing import NDArray

FARADAY_CONSTANT = 96485.3321233100184  # C/mol, Faraday's constant

from .useful_functions import stretched_grid

class Solute:
    '''
    Class for representing a solute in a solution and generating relevant values

    :param name: The name of the solute
    :param D: The diffusion coefficient in cm²s⁻¹
    :param conc_bulk: The bulk concentration of the solute in M
    :param charge: The charge of the solute in e
    '''
    def __init__(
            self,
            name:       str,
            D:          float, # cm²s⁻¹
            conc_bulk:  float, # mol dm⁻³
            charge:     int = 0
        ) -> None:
        self.name       = name
        self.D          = D
        self.conc_bulk  = conc_bulk * 1e-3 # mol cm⁻³
        self.charge     = charge


    def calculate_K(
            self,
            dt:     float,  # s
            dx_j:   float,  # cm
            dx_j1:  float   # cm
        ) -> float:         # cm⁻¹
        '''
                                 2 D Δt
        Calculates    K = ——————————————————————
                           ΔxⱼΔxⱼ₋₁(Δxⱼ + Δxⱼ₋₁)

        This value is used in the construction of the diffusion matrices.

        :param dt: The time step in seconds
        :param dx_j: The spatial separation between points j-1 and j in cm
        :param dx_j1: The spatial separation between points j and j+1 in cm
        :return: The calculated K value in cm⁻¹
        '''
        return 2 * self.D * dt / (dx_j*dx_j1*(dx_j+dx_j1))


    def A_Diffusion(
            self,
            dxs:    NDArray[np.float64],    # cm
            dt:     float,                  # s
            theta:  float                   # 0 ≤ θ ≤ 1
        ) -> NDArray[np.float64]:
        '''
        Calculates the diffusion A-matrix for the solute

        :param dxs: Array of dx values for the simulation in cm
        :param dt: The time step used in the simulation in s
        :param theta: The implicitness factor, must be between 0 and 1 inclusive
        :return: Square A-matrix for modelling diffusion of the solute.
        '''
        # There are one more points than there are differences
        npoints = len(dxs) + 1
        A_diff = np.zeros((npoints, npoints))

        # Calculate the interior values.
        for j in range(1, npoints-1):
            Kj = self.calculate_K(dt, dxs[j], dxs[j-1]) * theta # cm⁻¹
            # Multiplication by a dx in each case makes the matrix dimensionless
            A_diff[j,j-1]   = -Kj * dxs[j]
            A_diff[j,j]     =  Kj*(dxs[j] + dxs[j-1]) + 1
            A_diff[j,j+1]   = -Kj * dxs[j-1]

        # Set Dirichlet boundary condition on the right
        # i.e. it will always be the original bulk concentration value.
        A_diff[-1, -1] = 1

        # Set the zero-flux boundary condition on the left.
        # This is achieved by imagining a point at x = -dx[0] so that the
        # surface point can have second derivative calculated in the same way as
        # all the other points. For zero flux condition is must equal conc at
        # x = +dx[0]. Hence the 2K term off the main diagonal.
        K = self.calculate_K(dt, dxs[0], dxs[0]) * dxs[0] * theta
        A_diff[0,0]  = 1 + 2 * K
        A_diff[0, 1] = - 2 * K
        
        return A_diff
    

    def B_Diffusion(
            self,
            dxs:    NDArray[np.float64],    # cm
            dt:     float,                  # s
            theta:  float                   # 0 ≤ θ ≤ 1
        ) -> NDArray[np.float64]:
        '''
        Calculates the diffusion B-matrix for the solute

        :param dxs: Array of dx values for the simulation in cm
        :param dt: The time step used in the simulation in s
        :param theta: The implicitness factor, must be between 0 and 1 inclusive
        :return: Square A-matrix for modelling diffusion of the solute.
        '''
        # There are one more points than there are differences
        npoints = len(dxs) + 1
        B_diff = np.zeros((npoints, npoints))

        # Calculate the interior values.
        for j in range(1, npoints-1):
            Kj = self.calculate_K(dt, dxs[j], dxs[j-1]) * (1 - theta) # cm⁻¹
            # Multiplication by a dx in each case makes the matrix dimensionless
            B_diff[j,j-1]   =  Kj * dxs[j]
            B_diff[j,j]     = -Kj*(dxs[j] + dxs[j-1]) + 1
            B_diff[j,j+1]   =  Kj * dxs[j-1]

        # Set Dirichlet boundary condition on the right
        # i.e. it will always be the original bulk concentration value.
        B_diff[-1, -1] = 1

        # Set the zero-flux boundary condition on the left.
        # This is achieved by imagining a point at x = -dx[0] so that the
        # surface point can have second derivative calculated in the same way as
        # all the other points. For zero flux condition is must equal conc at
        # x = +dx[0]. Hence the 2K term off the main diagonal.
        K = self.calculate_K(dt, dxs[0], dxs[0]) * dxs[0] * theta
        B_diff[0,0]  = 1 - 2 * K
        B_diff[0, 1] = 2 * K
        
        return B_diff



