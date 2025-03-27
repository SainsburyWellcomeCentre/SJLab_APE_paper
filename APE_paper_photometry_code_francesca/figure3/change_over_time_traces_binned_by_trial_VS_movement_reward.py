from scipy.interpolate import interp1d
from utils.plotting import calculate_error_bars
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import os
from matplotlib.lines import Line2D
import matplotlib
from utils.plotting_visuals import makes_plots_pretty
from utils.change_over_time_plot_utils import *
font = {'size': 7}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']
bin_window = 200

# VS aligned to movement and reward

fig, axs = plt.subplots(2, 1, figsize=[2, 3.5])
axs[0].set_title('VS movement')
make_example_traces_plot('SNL_photo35', axs[0], window_for_binning=bin_window, legend=False, align_to='reward')
axs[1].set_title('VS reward')
make_example_traces_plot('SNL_photo35', axs[1], window_for_binning=bin_window, legend=False, align_to='reward')
makes_plots_pretty(axs)
plt.tight_layout()
figure_dir = r'T:\paper\revisions\cue movement reward comparisons VS TS'
plt.savefig(os.path.join(figure_dir, 'example_VS_movement_reward_over_trials.pdf'))
plt.show()