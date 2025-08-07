import numpy as np
from scipy.linalg import solve_banded
from typing import Annotated
from numpy.typing import NDArray

from ._SoluteNumbaFunctions import second_space_derivative_numba
from .ThomasAlgorithm import thomas_solve, compute_gamma_primes

class Solute:
    '''
    Represents a solute in a solution.

    :param name: The name of the solute
    :param D: The diffusion coefficient of the solute in cm²s⁻¹
    :param conc_bulk: The bulk concentration of the solute in mol dm⁻³
    :param charge: The charge of the solute in e
    '''
    def __init__(
            self,
            name:      str,
            D:         float,
            conc_bulk: float = 0.0,
            charge:    int   = 0,
        ) -> None:
        self.name       = name
        self.D          = D         # Diffusion coefficient in cm²s⁻¹
        self.conc_bulk  = conc_bulk # Bulk concentration in mol dm⁻³
        self.charge     = charge    # Charge in e


    def initialise_concentration(
            self,
            npoints: int
        ) -> None:
        '''
        Sets up the concentrations of the solute in the solution by creating an 
        array of length npoints, setting everywhere to the bulk concentration.

        :param npoints: The number of points in the concentration array
        '''
        self.conc = np.array(
            [self.conc_bulk] * npoints,
            dtype=float
            )
        self.npoints = npoints

    
    def calculate_K(
            self,
            dx:  float,
            dt:  float
        ) -> float:
        '''
        Calculates K = D*dt/2dx² (dimensionless)

        This value is used in constructing the matrices for modelling diffusion.
        Note that diffusion constant is in cm²s⁻¹, which is why dx is in cm.

        :param dx: The spatial step size in cm
        :param dt: The time step in seconds
        '''
        K = self.D * dt / (2 * dx**2)
        return K
    
    def calculate_K_variable_dx(
            self,
            dt: float,
            dx_j: float,
            dx_j1: float
        ) -> float:
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


    def construct_A_banded_format(
            self,
            dx: float,
            dt: float
        ) -> np.ndarray:
        '''
        Constructs the banded matrix A for modelling diffusion via the Crank-
        Nicholson method, with a reflective wall boundary condition on the left
        and a constant concentration boundary on the right.

        :return: The banded matrix A in the form accepted by the scipy linalg
            banded solver.
        '''
        K = self.calculate_K(dx, dt)
        if not hasattr(self, 'npoints'):
            raise ValueError("Concentration not initialised. " \
                             "Concentration must be initialised first.")
        
        # Initialise the banded matrix A
        A_banded          = np.zeros((3, self.npoints))
        # Upper diagonal all values equal to -K
        A_banded[0, 2:  ] = -K
        # Lower diagonal all but last value equal to -K
        # Last value is 0 because of the constant concentration boundary
        A_banded[2,  :-2] = -K
        # Main diagonal, 1 + 2K equal everywhere except first and last
        A_banded[1, 1:-1] = 1 + 2*K
        # First row, reflective wall boundary condition
        A_banded[1, 0   ] = 1
        # Last row, constant concentration boundary condition
        A_banded[1, -1  ] = 1
        return A_banded

    
    def construct_A_banded_format_variable_dx(
            self,
            dxs: np.ndarray,
            dt: float
        ) -> np.ndarray:
        '''
        Constructs the banded matrix A for modelling diffusion via the Crank-
        Nicholson formulation, using variable spatial grid spacing. Both left 
        and right boundary conditions are set to Dirchlet.

        :param dxs: A numpy array corresponding to the spatial grid spacings in
            cm
        :param dt: The time spacing in seconds
        '''
        # Firstly note that if there are n dxs then there are n+1 spatial points
        A_banded = np.zeros((3, self.npoints))
        assert len(dxs) == self.npoints - 1, "Number of spatial separations" \
                                             "does not match number of" \
                                             "concentration points"
        # Go through each point and calculate the relevant K and matrix values
        i = 1
        for dx_j1, dx_j in zip(dxs[0:-1], dxs[1:]):
            K = self.calculate_K_variable_dx(dt, dx_j, dx_j1)
            A_banded[0, i+1] = -K * dx_j1
            A_banded[1,i] = 1 + K * (dx_j + dx_j1)
            A_banded[2,i-1] = -K * dx_j
            i += 1

        # Assign the boundary condition values
        A_banded[1,0]   = 1
        A_banded[1,-1]  = 1

        return A_banded


    def construct_B(
            self,
            dx: float,
            dt: float
        ) -> np.ndarray:
        '''
        Constructs the matrix B for modelling diffusion via Crank-Nicholson
        method, with a reflective wall boundary condition on the left and a
        constant concentration boundary on the right. This is the matrix which
        is applied on the future time-step side of the equation.

        :return: The matrix B in full format.
        '''
        K = self.calculate_K(dx, dt)
        if not hasattr(self, 'npoints'):
            raise ValueError("Concentration not initialised. " \
                             "Concentration must be initialised first.")
        
        # Initialise the matrix B
        B = np.zeros((self.npoints, self.npoints))

        # Set the non-boundary values
        for i in range(1, self.npoints-1):
            B[i, i-1] = K
            B[i, i]   = 1 - 2*K
            B[i, i+1] = K

        # Reflective Wall Boundary Conditions
        B[0, 0] = 1

        # Dirichlet Boundary Condition
        B[-1, -1] = 1
        return B
    

    def construct_B_variable_dx(
            self,
            dxs: np.ndarray,
            dt: float
        ) -> np.ndarray:
        '''
        Constructs the banded matrix B for modelling diffusion via the Crank-
        Nicholson formulation, using variable spatial grid spacing. Both left 
        and right boundary conditions are set to Dirchlet.

        :param dxs: A numpy array corresponding to the spatial grid spacings in
            cm
        :param dt: The time spacing in seconds
        '''
        B = np.zeros((self.npoints, self.npoints))

        # Go through each point and calculate the relevant K and matrix values
        i = 1
        for dx_j1, dx_j in zip(dxs[0:-1], dxs[1:]):
            K = self.calculate_K_variable_dx(dt, dx_j, dx_j1)
            B[i, i+1] = K * dx_j1
            B[i, i  ] = 1 - K * (dx_j + dx_j1)
            B[i, i-1] = K * dx_j
            i += 1

        # Set the boundary conditions
        B[0,0], B[-1,-1] = 1, 1

        return B
    

    def save_diffusion_matrices(
            self,
            dx: float,
            dt: float
        ) -> None:
        '''
        Construct and save the reusable matrices relevant for modelling
        diffusion of the solute via the Crank-Nicholson technique.

        :param dx: The spatial step size in cm
        :param dt: The time step in seconds
        '''
        self.A_banded = self.construct_A_banded_format(dx, dt)

        self.alphas = self.A_banded[2, 0:-1]
        self.betas  = self.A_banded[1, :]

        gammas = self.A_banded[0, 1:]
        self.g_primes = compute_gamma_primes(
            self.alphas, self.betas, gammas, self.npoints
            )

        self.B        = self.construct_B(dx, dt)


    def save_diffusion_matrices_variable_dxs(
            self,
            dxs: np.ndarray,
            dt: float
        ) -> None:
        '''
        Construct and save the reusable matrices relevant for modelling
        diffusion of the solute via the Crank-Nicholson technique.

        :param dxs: The spatial step sizes in cm
        :param dt: The time step in seconds
        '''
        self.A_banded = self.construct_A_banded_format_variable_dx(dxs, dt)

        self.alphas = self.A_banded[2, 0:-1]
        self.betas  = self.A_banded[1, :]

        gammas = self.A_banded[0, 1:]
        self.g_primes = compute_gamma_primes(
            self.alphas, self.betas, gammas, self.npoints
            )

        self.B        = self.construct_B_variable_dx(dxs, dt)
        

    
    def diffuse(
            self
        ) -> None:
        '''
        Updates the concentration of the solute in the solution by allowing
        diffusion to occur over one time step using the Crank-Nicholson method.
        This method relies on the matrices A_banded and B being pre-computed
        and stored in the solute object.
        The time step and spatial step are set in the calculated matrices.

        Effectively solves the equation A * conc_new = B * conc_old to obtain
        the new concentrations.
        '''
        self.conc = solve_banded(
            (1,1),
            self.A_banded,
            self.B @ self.conc
            )
        # Set the ghost point concentration to the first point
        self.conc[0] = self.conc[1]
        
    def diffuse_coupled_kinetics(
            self,
            R: np.ndarray
        ) -> None:
        '''
        Updates the concentration of the solute in the solution by using the
        Crank-Nicholson approach. In this case changes due to reaction kinetics
        are also included and all changes treated together. The non-diffusion
        changes are passed in the 1D numpy array R. R is then made into a 
        diagonal matrix such that the concentration is updated by solving the 
        matrix equation A * conc_new = (B + R) * conc_old.

        :param R: 1D numpy array detailing the non-diffusion changes to
            concentration for the solute over the correct time-step.
        '''
        self.conc = solve_banded(
            (1,1),
            self.A_banded,
            self.B @ self.conc + R
            )
        # Set the ghost point concentration to the first point
        self.conc[0] = self.conc[1]

    
    def diffuse_coupled_kinetics_Thomas(
            self,
            R: np.ndarray
    ) -> None:
        '''
        Updates the concentration of the solute in the solution by using the
        Crank-Nicholson approach. In this case changes due to reaction kinetics
        are also included and all changes treated together. The non-diffusion
        changes are passed in the 1D numpy array R. R is then made into a 
        diagonal matrix such that the concentration is updated by solving the 
        matrix equation A * conc_new = (B + R) * conc_old.
        
        :param R: 1D numpy array detailing the non-diffusion changes to
            concentration for the solute over the correct time-step.
        '''
        d = self.B @ self.conc + R
        # Solve the system using the Thomas algorithm
        self.conc = thomas_solve(
            self.alphas, self.betas, self.g_primes, d, self.npoints
        )
        # Set the ghost point concentration to the first point
        self.conc[0] = self.conc[1]


    def diffuse_Thomas(self) -> None:
        '''
        Uses Thomas algorithm to model diffusion, without includeing reactions
        '''
        d = self.B @ self.conc
        # Solve the system using the Thomas algorithm
        self.conc = thomas_solve(
            self.alphas, self.betas, self.g_primes, d, self.npoints
        )
        # Set the ghost point concentration to the first point
        self.conc[0] = self.conc[1]
        

    
    def second_space_derivative(
            self,
            dx: float
        ) -> np.ndarray:
        '''
        This uses finite differences technique to compute the second spatial 
        derivative. 

        :param dx: The spatial grid spacing in cm
        '''
        return second_space_derivative_numba(
            self.conc,
            dx
        )
    
    def diffusion_conc_gradient(
            self,
            dx: float
        ) -> np.ndarray:
        '''
        Returns the second space derivative multiplied by the diffusion
        coefficient.

        This has horrific units so will convert to SI.
        Currently, mol/dm³ is used for concentration, so we need to convert
        to mol/cm³, so it will vibe with the diffusion coefficient and spatial
        grid spacing.

        :param dx: Th spatial grid spacing in cm
        '''
        return self.D * self.second_space_derivative(dx) * 1e-3
    
    def current_contribution(
            self
        ) -> float:
        '''
        Calculates and returns the contribution to current at the electrode
        from the solute. The returned value is to be converted by a Solution
        object into the correct units of A.

        We will convert all to SI units.

        The current contribution is given by the equation:
        I_i = z_i * D_i * (C_i(x=1, t) - C_i(x=0, t))

        
        '''
        return self.charge * self.D * 1e-4 * (self.conc[2] - self.conc[1]) * 1e3
        
            
