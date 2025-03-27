import matplotlib.pylab as plt
from utils.zscored_plots_utils import plot_average_trace_all_mice_high_low_cues
import matplotlib
import numpy as np


from utils.plotting_visuals import makes_plots_pretty

font = {'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']

fig, axs = plt.subplots(2, 2, figsize=[4, 2])

axs[0, 0].set_title('TS')
plot_average_trace_all_mice_high_low_cues(axs[0, :], 'tail', cmap=['#002F3A', '#76A8DA'])
axs[1, 0].set_title('VS')
plot_average_trace_all_mice_high_low_cues(axs[1, :], 'Nacc', cmap=['#E95F32', '#F9C0AF'])
makes_plots_pretty(np.ravel(axs))
plt.tight_layout()

plt.show()
