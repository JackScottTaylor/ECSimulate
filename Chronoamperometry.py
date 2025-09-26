'''
Author: Jack Taylor
'''

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

class PotentialStep(System):
    '''
    Class for simulating a Cottrell Experiment
    
    '''
    def __init__(
            self,
            solutes:    List[Solute],
            redoxes:    List[Redox],
            reactions:  List[Reaction],
            dxs:        List[float],
            dt:         float,
            t_total:    float,
            E_start:    float,
            E_set:          float,
            theta:      float = 0.5,
            A:          float = np.pi * 0.15 ** 2,
    ):
        super().__init__(solutes, redoxes, reactions, dxs, dt, theta=theta, A=A)

        # Calculate the time steps to be simulated
        ntimes = int(t_total // dt)
        self.ts = np.array([(n+1) * dt for n in range(ntimes)])

        self.t_total = self.ts[-1]

        # Initialise the array for storing current.
        self.Is = np.zeros_like(self.ts)

        self.electrode_potential = E_set
        self.E = E_set

    def run(self, save_concs: bool = False):
        '''
        Runs the simulation and stores the current response in `self.Is`
        If save_concs has been selected then saves the concentrations at each
        time point in `self.concs`
        '''
        if save_concs:  self._run_save_concs()
        else:           self._run_dont_save_concs()

    def _run_dont_save_concs(self):
        # Start the potential step immediately
        self.electorode_potential = self.E
        for i, t in enumerate(self.ts):
            self.update()
            self.Is[i] = self.current()

    def _run_save_concs(self):
        self.electrode_potential = self.E
        self.concs = np.zeros((len(self.ts), len(self.C)))
        for i, t in enumerate(self.ts):
            self.update()
            self.Is[i] = self.current()
            self.concs[i,:] = self.C[:]

    def plot(self, ax: "Axes" = None, **plot_kwargs):
        '''
        Plots the current response against time
        '''
        if ax == None: ax = plt.gca()
        ax.plot(self.ts, self.Is, **plot_kwargs)
        ax.set_xlabel('Time / s')
        ax.set_ylabel('Current / mA')

    def animate(
            self,
            title:          str         = '',
            nframes:        int         = 1000,
            fps:            int         = 30,
            plot_kwargs:    dict        = {},
            dot_kwargs:     dict        = {},
            conc_kwargs:    List[dict]  = []
            ) -> FuncAnimation:
        fig, axs = plt.subplots(1, 2)
        fig.set_figwidth(2*fig_w)
        self.plot(ax = axs[1], **plot_kwargs)
        if title: fig.suptitle(title)

        nsolutes = self.nsolutes
        ts, Is, concs = self.ts, self.Is, self.concs

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
            axs[0].set_xlim(0, np.cumsum(self.dxs)[-1]/100)
            axs[0].set_ylim(0, max([s.conc_bulk for s in self.solutes])*1.2)
            axs[0].legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol = len(self.solutes),
                frameon=False
            )
            return red_dot, *conc_lines

        def update(frame):
            red_dot.set_data([ts[frame]], [Is[frame]])
            for i in range(self.nsolutes):
                conc_lines[i].set_data(conc_xs, concs[frame][i::self.nsolutes])
            return red_dot, *conc_lines
        
        ani = FuncAnimation(fig, update, frames=[i for i in range(len(ts))],
                    init_func=init, blit=True, interval=1000 / fps)
        
        return ani


