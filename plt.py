import matplotlib.pyplot as plt

# Set figure sizes
fig_w, fig_h = 8.3, 6.225

# Set colour scheme
Wong = [
    '#000000', # black
    '#0072B2', # blue
    '#D55E00', # vermillion
    '#CC79A7', # reddish purple
    '#E69F00', # orange
    '#56B4E9', # sky blue
    '#009E73', # bluish green
    '#F0E442', # yellow
]
plt.rcParams['axes.prop_cycle'] 				= plt.cycler(color=Wong)


# Set some general parameters.
plt.rcParams['lines.linewidth'] 				= 3              # Linewidth
plt.rcParams['figure.figsize'] 					= [fig_w, fig_h] # Figure size
plt.rcParams['axes.linewidth'] 					= 1.2            # Axes linewidth
plt.rcParams['font.family'] 					= 'Arial'        # Font family
plt.rcParams['mathtext.fontset'] 				= 'dejavusans'   # Math font
plt.rcParams['font.size'] 						= 24             # Font size
plt.rcParams['savefig.dpi'] 					= 300            # Savefig dpi
plt.rcParams['savefig.format'] 					= 'pdf'          # Savefig format
plt.rcParams['figure.constrained_layout.h_pad'] = 0.1            # Padding in constrained layout
plt.rcParams['figure.constrained_layout.w_pad'] = 0.1            # Padding in constrained layout
plt.rcParams['figure.constrained_layout.use'] 	= True           # Use constrained layout
plt.rcParams['legend.labelspacing'] 			= 0.15           # Legend label spacing
plt.rcParams['legend.handletextpad']            = 0.3            # Legend handle text padding
plt.rcParams['axes.xmargin']					= 0.01           # X margin
plt.rcParams['axes.ymargin'] 					= 0.01           # Y margin
plt.rcParams['legend.frameon']                  = True           # Legend frame on
plt.rcParams['legend.fontsize']                 = 20             # Legend font size

