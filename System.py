import numpy as np
from typing import List
import scipy.sparse as sp
from scipy.linalg import solve_banded
from numpy.typing import NDArray

from .plt import plt, fig_w
from matplotlib.animation import FuncAnimation

FARADAY_CONSTANT = 96485.3321233100184  # C mol⁻¹
TEMPERATURE      = 298.15               # K
GAS_CONSTANT     = 8.31446261815324     # J K⁻¹ mol⁻¹
f_CONSTANT       = FARADAY_CONSTANT / (TEMPERATURE * GAS_CONSTANT) # V⁻¹

from .Solute import Solute
from .Redox import Redox
from .Reaction import Reaction

from .useful_functions import chess_board_combine, matrix_in_banded_format
from .numba_functions import _fill_M_et, add_Aet_to_banded_A

class System:
    '''
    Class for holding all information about the electrochemical system and 
    contains methods for updating and calculating current.

    :param solutes: List of solutes in the system
    :param redoxes: List of redox reactions which may take place
    :param dxs: The spatial spacings to be considered in the simulations in cm
    :param dt: The time-step to be considered in simulations in s
    :param theta: The implicitness factor, where 0 is fully explicit and 1 is
        fully implicit.
    :param A: The area of the electrode in cm²
    '''
    def __init__(
            self,
            solutes:    List[Solute],
            redoxes:    List[Redox],
            reactions:  List[Reaction],
            dxs:        List[float],
            dt:         float,
            theta:      float = 0.55,
            A:          float = np.pi * 0.15 ** 2
    ):
        self.solutes  = solutes
        self.nsolutes = len(solutes)
        self.redoxes  = redoxes
        self.reactions = reactions

        # Precompute a dictionary of solutes to their respective indices in 
        # self.solutes
        self.solute_to_index: dict = self._calc_solute_to_index()

        self.dt       = dt
        # Having the first two dxs be identical allows us to easily do a second
        # order approximation of the flux when calculating the current.
        self.dxs      = np.append(dxs[0], dxs)
        self.theta    = theta

        # Set to 0V initially
        self.electrode_potential          = 0.0
        self.electrode_potential_previous = 0.0

        # Initialise the system concentrations vector in mol cm⁻³
        self.C: NDArray[np.float64] = self._initialise_C()

        # Precalculate and save the diffusion matrices
        self.A_diff: NDArray[np.float64] = self._calc_A_diff()
        self.B_diff: NDArray[np.float64] = self._calc_B_diff()

        self.M_kinetics = self._calc_kinetic_Matrix()

        self.B_diffkin = self.B_diff + (1 - theta) * self.M_kinetics * self.dt
        self.A_diffkin = self.A_diff - theta       * self.M_kinetics * self.dt

        # Save Bdiff in csr format for quick multiplication
        self.B_diff_csr = sp.csr_matrix(self.B_diffkin)

        # Save A_diff in banded format for convenient solving
        self.A_diff_banded = matrix_in_banded_format(
            self.A_diffkin, self.nsolutes, self.nsolutes
            )

        # Precompute and save constants related to electron-transfer matrices.
        self.KetA: float = self._calc_KetA() # cm s⁻¹
        self.KetB: float = self._calc_KetB() # cm s⁻¹

        # Precompute arrays for speedy redox calculations
        self._ox_indices = np.array(
            [self.solute_to_index[r.ox] for r in redoxes],
            dtype=np.int64
            )
        self._red_indices = np.array(
            [self.solute_to_index[r.red] for r in redoxes],
            dtype=np.int64
        )
        self._k0 = np.array([r.k0 for r in redoxes], dtype=np.float64)
        self._alpha = np.array([r.alpha for r in redoxes], dtype=np.float64)
        self._E0 = np.array([r.E_standard for r in redoxes], dtype=np.float64)

        self._charges = np.array([s.charge for s in self.solutes])
        self._Ds = np.array([s.D for s in self.solutes])
        self._I_constant = - FARADAY_CONSTANT * A / (dxs[0] * 2) * 1e3
        self.A = A



    def initialise_electrode_potential(self, E: float):
        '''
        Initialises the system electrode potential to the provided value in V.

        :param E: Electrode potential in V
        '''
        self.electrode_potential            = E
        self.electrode_potential_previous   = E


    def _initialise_C(self) -> NDArray[np.float64]:
        '''
        Initialises the concentrations of all the solutes to their bulk cons
        across the simulation and then combines into single system concentration
        vector is chess-board style.

        :return: Initialised system concentration vector.
        '''
        npoints = len(self.dxs) + 1
        concs = [
            np.ones(npoints) * s.conc_bulk for s in self.solutes
        ]
        return chess_board_combine(*concs)


    def _calc_A_diff(self) -> NDArray[np.float64]:
        '''
        Calculates the diffusion contribution to the A-matrix (the future step
        matrix). Calls each solute to create their own matrix and then combines
        them in to a single matrix which can act on the system concentration 
        vector.

        :return: Diffusion contribution to A-matrix.
        '''
        A_diffs = [
            s.A_Diffusion(self.dxs, self.dt, self.theta) for s in self.solutes
            ]
        return chess_board_combine(*A_diffs)


    def _calc_B_diff(self):
        '''
        Calculates the diffusion contribution to the B-matrix (the current-step
        matrix). Calls each solute to create their own matrix and then combines
        them in to a single matrix which can act on the system concentration 
        vector.

        :return: Diffusion contribution to B-matrix.
        '''
        B_diffs = [
            s.B_Diffusion(self.dxs, self.dt, self.theta) for s in self.solutes
        ]
        return chess_board_combine(*B_diffs)


    def _calc_KetA(self) -> float:
        '''
        Calculates and returns the preconstant for the electron-transfer
        contribution to the A-matrix.
        Units of cm s⁻¹

        :return: Electron-transfer A-matrix preconstant, s cm⁻¹.
        '''
        return 2 * self.theta * self.dt / self.dxs[0]


    def _calc_KetB(self) -> float:
        '''
        Calculates and returns the preconstant for the electron-transfer
        contribution to the B-matrix.
        Units of cm s⁻¹

        :return: Electron-transfer B-matrix preconstant, s cm⁻¹.
        '''
        return - 2 * (1 - self.theta) * self.dt / self.dxs[0]


    def _calc_solute_to_index(self) -> dict:
        '''
        Calculates a useful dictionary which takes a solute as the key and
        returns its corresponding index in self.solutes.

        :return: Dictionary with solutes as keys and their respective indices in
            self.solutes
        '''
        solute_to_index = {}
        for i, solute in enumerate(self.solutes):
            solute_to_index[solute] = i
        return solute_to_index
    

    def A_et_small(
            self, E: float, f: float = f_CONSTANT,
        ) -> NDArray[np.float64]:
        '''
        Calculates and returns the electron-transfer contribution to the A
        matrix. This only acts on the surface concentrations so is stored in
        a `nsolutes × nsolutes` size matrix.

        :param E: The potential in V to evaluate the fluxes at.
        :param f: The preconstant required to make dimensionless, F/RT.
        :return: `nsolutes × nsolutes` dimensionless matrix.
        '''
        A_et = np.zeros((self.nsolutes, self.nsolutes))
        _fill_M_et(A_et, self._ox_indices, self._red_indices, self._k0,
                   self._alpha, self._E0, f, E, self.dxs[0])
        for row, D in zip(A_et, self._Ds):
            row /= D
        return A_et
    

    def B_et_small(
            self, E: float, f: float = f_CONSTANT,
        ) -> NDArray[np.float64]:
        '''
        Calculates and returns the electron-transfer contribution to the B
        matrix. This only acts on the surface concentrations so is stored in
        a `nsolutes × nsolutes` size matrix.

        :param E: The potential in V to evaluate the fluxes at.
        :param f: The preconstant required to make dimensionless, F/RT.
        :return: `nsolutes × nsolutes` dimensionless matrix.
        '''
        B_et = np.zeros((self.nsolutes, self.nsolutes))
        _fill_M_et(B_et, self._ox_indices, self._red_indices, self._k0,
                   self._alpha, self._E0, f, E, self.KetB)
        return B_et
    

    def _calc_kinetic_Matrix(self):
        '''
        Calculates the contribution to the A-matrix due to chemical kinetics
        '''
        M_kinetics = np.zeros_like(self.A_diff)
        s2i = self.solute_to_index
        nconc_points = len(self.dxs) + 1
        for rxn in self.reactions:
            for i in range(nconc_points):
                p, r, k = rxn.product, rxn.reactant, rxn.k
                istart = i * self.nsolutes
                pi, ri = s2i[p]+istart, s2i[r]+istart
                M_kinetics[ri,ri] -= k
                M_kinetics[pi,ri] += k
        return M_kinetics


            

    def BC(self) -> NDArray[np.float64]:
        '''
        Applies B = B_diff + B_et to the system concentration vector to produce
        another vector with units of mol cm⁻³.

        :return: (B_diff + B_et) @ C
        '''
        # Apply the diffusion part using fast csr matrix
        y = self.B_diff_csr @ self.C
        # Calculate the required electron-transfer matrix
        '''Bet = self.B_et_small(self.electrode_potential_previous)
        # Apply only to the surface concentrations.
        y[:self.nsolutes] += Bet @ self.C[:self.nsolutes]'''
        return y
    

    def update(self):
        '''
        Solves the matrix equation AC(t+Δt) = BC(t) to obtain the next time-step
        concentrations.
        '''
        # Calculate the RHS
        y = self.BC()
        # Create the correct A-matrix, in banded form.
        A = add_Aet_to_banded_A(
            self.A_diff_banded.copy(),
            self.A_et_small(self.electrode_potential),
            self.nsolutes,
            self.nsolutes
            )
        # Use scipy to solve the banded matrix equation
        self.C = solve_banded((self.nsolutes, self.nsolutes), A, y)
        # Set the previous electrode potential to the current one.
        self.electrode_potential_previous = self.electrode_potential


    def current(self):
        fluxes = 4*self.C[self.nsolutes:2*self.nsolutes] - \
                 3*self.C[:self.nsolutes] - \
                 self.C[2*self.nsolutes:3*self.nsolutes]
        return np.sum(fluxes * self._charges * self._Ds) * self._I_constant * 0.5 * 2



from useful_functions import stretched_grid

class CyclicVoltammogram(System):
    def __init__(
            self,
            solutes: List[Solute],
            redoxes: List[Redox],
            reactions: List[Reaction],
            dxs: List[float],
            dt: float,
            EMax: float,
            EMin: float,
            scan_rate: float,
            theta: float = 0.5,
            A: float = np.pi * 0.15 ** 2,
    ):
        super().__init__(solutes, redoxes, reactions, dxs, dt, theta=theta, A=A)
        self.dE = (scan_rate * 1e-3 * dt)
        E1 = np.arange(EMax, EMin, -self.dE)
        E2 = np.arange(EMin, EMax, self.dE)
        self.Es = np.append(E1, E2)

        self.electrode_potential = self.Es[0]
        self.electrode_potential_previous = self.Es[0]

    def run(self):
        Is = np.zeros_like(self.Es)
        for i, E in enumerate(self.Es):
            self.electrode_potential = E
            self.update()
            Is[i] = self.current()
        return self.Es, Is
    

class CyclicVoltammogramAnimation(CyclicVoltammogram):
    def __init__(
        self,
            solutes: List[Solute],
            redoxes: List[Redox],
            reactions: List[Reaction],
            dxs: List[float],
            dt: float,
            EMax: float,
            EMin: float,
            scan_rate: float,
            theta: float = 0.5,
            A: float = np.pi * 0.15 ** 2,
            fps: int = 30,
            animation_length: int = 100    
    ):
        super().__init__(solutes, redoxes, reactions, dxs, dt, EMax, EMin, scan_rate,
                         theta=theta, A=A)
        self.fps = fps
        self.animation_length = animation_length

        self.nframes = fps * animation_length
        self.iterations = len(self.Es)//self.nframes
        self.recorded_Es = []
        self.recorded_Is = []
        self.concs = []

    def run(self):
        self.Is = np.zeros_like(self.Es)
        for i, E in enumerate(self.Es):
            self.electrode_potential = E
            self.update()
            self.Is[i] = self.current()
            if i % self.iterations == 0:
                self.concs.append(self.C.copy())
                self.recorded_Es.append(self.Es[i])
                self.recorded_Is.append(self.Is[i])
        return self.Es, self.Is
    
    def animate(self):
        fig, axs = plt.subplots(1, 2)
        fig.set_figwidth(2*fig_w)
        conc_xs = np.append(0, np.cumsum(self.dxs))
        Es = self.Es
        Is = self.Is
        concs = self.concs
        rec_I, rec_E = self.recorded_Is, self.recorded_Es

        CV_line = axs[1].plot(Es, Is, color='black')[0]
        red_dot = axs[1].plot([], [], color='red', marker='o')[0]
        conc_lines = [axs[0].plot([], [], color='black')[0] for i in self.solutes]

        def init():
            axs[0].set_xlabel('Distance from Electrode / cm')
            axs[0].set_ylabel('Concentration / mol cm$^{-3}$')
            axs[0].set_xlim(0, np.cumsum(self.dxs)[-1])
            axs[0].set_ylim(0, max([s.conc_bulk for s in self.solutes])*1.2)
            axs[1].set_xlabel('Potential / V')
            axs[1].set_ylabel('Current / mA')
            return CV_line, red_dot, *conc_lines

        def update(frame):
            red_dot.set_data([rec_E[frame]], [rec_I[frame]])
            for i in range(self.nsolutes):
                conc_lines[i].set_data(conc_xs, concs[frame][i::self.nsolutes])
            return CV_line, red_dot, *conc_lines
        
        ani = FuncAnimation(fig, update, frames=[i for i in range(self.nframes)],
                    init_func=init, blit=True, interval=1000 / self.fps)
        
        plt.show()