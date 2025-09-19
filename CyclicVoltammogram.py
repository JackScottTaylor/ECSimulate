import numpy as np
from typing import List

from .plt import plt, fig_w
from matplotlib.animation import FuncAnimation

FARADAY_CONSTANT = 96485.3321233100184  # C mol⁻¹
TEMPERATURE      = 298.15               # K
GAS_CONSTANT     = 8.31446261815324     # J K⁻¹ mol⁻¹
f_CONSTANT       = FARADAY_CONSTANT / (TEMPERATURE * GAS_CONSTANT) # V⁻¹

from .Solute import Solute
from .Redox import Redox
from .Reaction import Reaction
from .System import System


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
        E1 = np.arange(EMax, EMin, -self.dE)
        E2 = np.arange(EMin, EMax, self.dE)

        # Initialise arrays to store the potentials and currents
        self.Es = np.append(E1, E2)
        self.Is = np.zeros_like(self.Es)

        self.electrode_potential = self.Es[0]
        self.electrode_potential_previous = self.Es[0]

    def run(self):
        '''
        Runs the CV simulation and stores the current response in `self.Is`.
        '''
        Is = np.zeros_like(self.Es)
        for i, E in enumerate(self.Es):
            self.electrode_potential = E
            self.update()
            Is[i] = self.current()
        self.Is = Is
        return self.Es, Is
    
    def plot(self, npoints = 1000, ax = None, **plot_kwargs):
        '''
        Plots the CV result
        '''
        if ax == None: ax = plt.gca()
        if len(self.Es) < npoints: npoints = len(self.Es)
    

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