import numpy as np
from typing import List, TYPE_CHECKING

from .plt import plt, fig_w
from matplotlib.animation import FuncAnimation

if TYPE_CHECKING: from matplotlib.axes import Axes

FARADAY_CONSTANT = 96485.3321233100184  # C mol⁻¹
TEMPERATURE      = 298.15               # K
GAS_CONSTANT     = 8.31446261815324     # J K⁻¹ mol⁻¹
f_CONSTANT       = FARADAY_CONSTANT / (TEMPERATURE * GAS_CONSTANT) # V⁻¹

from .Solute import Solute
from .Redox import Redox
from .Reaction import Reaction
from .System import System

GB = 1024 ** 3
MAXIMUM_ALLOWABLE_MEM_ALLOC = 1.0 * GB # Gigabytes

class CyclicVoltammogram(System):
    '''
    Class for simulating a cyclic voltammogram with associated saving and 
    exporting options.
    
    '''
    def __init__(
            self,
            solutes:    List[Solute],
            redoxes:    List[Redox],
            reactions:  List[Reaction],
            dxs:        List[float],
            dt:         float,
            EMax:       float,
            EMin:       float,
            scan_rate:  float,
            theta:      float = 0.5,
            A:          float = np.pi * 0.15 ** 2,
    ):
        super().__init__(solutes, redoxes, reactions, dxs, dt, theta=theta, A=A)
        self.dE = (scan_rate * 1e-3 * dt)

        # Need parameters to determine what mode we should run in.
        self.EI_low_mem:            bool = False
        self.EIconc_low_mem:        bool = False
        self.EI_update_iters:       int = 1
        self.EIconc_update_iters:   int = 1
        # Run check function to determine when low mem mode may be required
        self.check_memory_allocation(EMax, EMin, self.dE)

        self.EMax, self.EMin = EMax, EMin

        self.Es, self.Is, self.concs = [], [], [[]]
        

    def check_memory_allocation(self, EMax, Emin, dE):
        n_times = int(2 * abs(EMax - Emin) // dE)
        nsolutes = self.nsolutes
        n_concs = len(self.dxs) + 1
        n_total_concs = nsolutes * n_concs * n_times
        n_total_vals_to_store = 2 * n_times + n_total_concs
        memory_req_EI = 2 * n_times * np.dtype(np.float128).itemsize
        memory_req_max = n_total_vals_to_store * np.dtype(np.float128).itemsize

        print(
            'Estimated Memory Requirement for Storing E, I data:\n',
            f'{memory_req_EI / GB:.3f} GB',
        )
        print(
            'Estimated Memory Requirement for Storing E, I, concentration data:\n',
            f'{memory_req_max / GB:.3f} GB',
        )

        self.EI_low_mem = memory_req_EI > MAXIMUM_ALLOWABLE_MEM_ALLOC
        self.EIconc_low_mem = memory_req_max > MAXIMUM_ALLOWABLE_MEM_ALLOC

        if self.EI_low_mem:
            self.EI_update_iters = int(np.ceil(memory_req_EI / MAXIMUM_ALLOWABLE_MEM_ALLOC))
            print('If not saving concentrations, then data will be saved every',
                  f'{self.EI_update_iters} updates.')
            
        if self.EIconc_low_mem:
            self.EIconc_update_iters = int(np.ceil(memory_req_max / MAXIMUM_ALLOWABLE_MEM_ALLOC))
            print('If saving concentrations, then data will be saved every',
                  f'{self.EIconc_update_iters} updates.')


    def Es_Is_npoints(self, npoints: int = 1000):
        '''
        Returns the E and I data with approximately the desired number of points

        :param npoints: Number of desired points in the data
        '''
        full_len = len(self.Es)
        if full_len < npoints: npoints = full_len
        skip = int(full_len // npoints)
        return self.Es[::skip], self.Is[::skip]
    
    def Es_Is_concs_npoints(self, npoints: int = 1000):
        '''
        Returns the E and I data with approximately the desired number of points

        :param npoints: Number of desired points in the data
        '''
        full_len = len(self.Es)
        if full_len < npoints: npoints = full_len
        skip = int(full_len // npoints)
        return self.Es[::skip], self.Is[::skip], self.concs[::skip]


    def run(self, save_concs: bool = False):
        '''
        Runs the CV simulation and stores the current response in `self.Is`.
        '''
        if save_concs:
            if self.EIconc_low_mem: return self._run_low_mem_EIconc()
            else: return self._run_full_mem_EIconc()
        
        else:
            if self.EI_low_mem: return self._run_low_mem_EI()
            else: return self._run_full_mem_EI()
    

    def _run_full_mem_EI(self):
        E1 = np.arange(self.EMax, self.EMin, -self.dE)
        E2 = np.arange(self.EMin, self.EMax, self.dE)
        self.Es = np.append(E1, E2)
        Is = np.zeros_like(self.Es)

        for i, E in enumerate(self.Es):
            self.electrode_potential = E
            self.update()
            Is[i] = self.current()
        self.Is = Is
        return self.Es, Is
    

    def _run_full_mem_EIconc(self):
        E1 = np.arange(self.EMax, self.EMin, -self.dE)
        E2 = np.arange(self.EMin, self.EMax, self.dE)
        self.Es = np.append(E1, E2)
        Is = np.zeros_like(self.Es)
        concs = np.zeros((len(self.Es), len(self.C)))

        for i, E in enumerate(self.Es):
            self.electrode_potential = E
            self.update()
            Is[i] = self.current()
            concs[i] = self.C.copy()
        self.Is = Is
        self.concs = concs
        return self.Es, Is
    

    def _run_low_mem_EI(self):
        iters = self.EI_update_iters
        dE = self.dE
        dE_iter_list = [dE * j for j in range(iters)]

        dE_large = iters * dE
        E1 = np.arange(self.EMax, self.EMin, -dE_large)
        E2 = np.arange(self.EMin, self.EMax, dE_large)
        self.Es = np.append(E1, E2)
        Is = np.zeros_like(self.Es)

        for i, E_big_step in enumerate(E1):
            for dE_iteration in dE_iter_list:
                self.electrode_potential = E_big_step - dE_iteration
                self.update()
            Is[i] = self.current()

        i_start = len(E1)
        for j, E_big_step in enumerate(E2):
            for dE_iteration in dE_iter_list:
                self.electrode_potential = E_big_step + dE_iteration
                self.update()
            Is[i+j] = self.current()

        self.Is = Is
        return self.Es, Is
    

    def _run_low_mem_EIconc(self):
        iters = self.EI_update_iters
        dE = self.dE
        dE_iter_list = [dE * j for j in range(iters)]

        dE_large = iters * dE
        E1 = np.arange(self.EMax, self.EMin, -dE_large)
        E2 = np.arange(self.EMin, self.EMax, dE_large)
        self.Es = np.append(E1, E2)
        Is = np.zeros_like(self.Es)
        concs = np.zeros((len(self.Es), len(self.C)))

        for i, E_big_step in enumerate(E1):
            for dE_iteration in dE_iter_list:
                self.electrode_potential = E_big_step - dE_iteration
                self.update()
            Is[i] = self.current()
            concs[i] = self.C.copy()

        i_start = len(E1)
        for j, E_big_step in enumerate(E2):
            for dE_iteration in dE_iter_list:
                self.electrode_potential = E_big_step + dE_iteration
                self.update()
            Is[i+j] = self.current()
            concs[i+j] = self.C.copy()

        self.Is = Is
        self.concs = concs
        return self.Es, Is


    def plot(self, npoints: int = 1000, ax: "Axes" = None, **plot_kwargs):
        '''
        Plots the CV result
        '''
        if ax == None: ax = plt.gca()
        Es, Is = self.Es_Is_npoints(npoints)
        ax.plot(Es, Is, **plot_kwargs)
        ax.set_xlabel('Potential / V')
        ax.set_ylabel('Current / mA')
        

    def animate(
            self,
            title:          str         = '',
            nframes:        int         = 1000,
            fps:            int         = 30,
            CV_kwargs:      dict        = {},
            dot_kwargs:     dict        = {},
            conc_kwargs:    List[dict]  = []
            ) -> FuncAnimation:
        '''
        Creates an animation of the simulated CV where on the right is the
        the static CV with a det to show where in the curve the simulation is.
        On the left is the concentration profile of all the solutes over the 
        entire simulated range.

        :param nframes: The number of frames in the simulation
        :param fps: The frame rate of the simulation
        :CV_kwargs: Dictionary of plotting kwargs for the static CV figure
        :dot_kwargs: Dictionary of pltting kwargs for the dot
        :conc_kwargs: List of dictionaries of plotting kwargs for concentrations
            If not equal to number of solutes then empty dictionaries are added.
        :return: The `FUncAnimation` object
        '''
        fig, axs = plt.subplots(1, 2)
        fig.set_figwidth(2*fig_w)
        self.plot(npoints = nframes, ax = axs[1], **CV_kwargs)
        if title: fig.suptitle(title)

        nsolutes = self.nsolutes
        Es, Is, concs = self.Es_Is_concs_npoints(npoints=nframes)

        while len(conc_kwargs) < len(self.solutes): conc_kwargs.append({})
        for kwargs, name in zip(conc_kwargs, [s.name for s in self.solutes]):
            if 'label' not in kwargs.keys():
                kwargs['label'] = name

        if 'color' not in dot_kwargs.keys(): dot_kwargs['color'] = 'red'
        if 'marker' not in dot_kwargs.keys(): dot_kwargs['marker'] = 'o'
        red_dot = axs[1].plot([], [], **dot_kwargs)[0]
        conc_lines = [axs[0].plot([], [], **kw)[0] for i, kw in zip(self.solutes, conc_kwargs)]

        conc_xs = np.append(0, np.cumsum(self.dxs))

        def init():
            axs[0].set_xlabel('Distance from Electrode / cm')
            axs[0].set_ylabel('Concentration / mol cm$^{-3}$')
            axs[0].set_xlim(0, np.cumsum(self.dxs)[-1])
            axs[0].set_ylim(0, max([s.conc_bulk for s in self.solutes])*1.2)
            axs[0].legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol = len(self.solutes),
                frameon=False
            )
            return red_dot, *conc_lines

        def update(frame):
            red_dot.set_data([Es[frame]], [Is[frame]])
            for i in range(self.nsolutes):
                conc_lines[i].set_data(conc_xs, concs[frame][i::self.nsolutes])
            return red_dot, *conc_lines
        
        ani = FuncAnimation(fig, update, frames=[i for i in range(len(Es))],
                    init_func=init, blit=True, interval=1000 / fps)
        
        return ani
    
    def write_to_file(self, filename):
        '''
        Writes E, I to file
        '''
        with open(filename, 'w') as f:
            for E, I in zip(self.Es, self.Is):
                f.write(f'{E}\t{I}\n')