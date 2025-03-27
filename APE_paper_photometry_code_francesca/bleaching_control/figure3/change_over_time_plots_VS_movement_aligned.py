from set_global_params import change_over_time_mice
import matplotlib
import os
from utils.plotting_visuals import makes_plots_pretty
from utils.change_over_time_plot_utils import  *

# this makes the plot in the figure
font = {'size': 8, 'family':'sans-serif', 'sans-serif':['Arial']}

matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42

fig, ax = plt.subplots(1, 1, figsize=[2, 2])

nacc_mice = change_over_time_mice['Nacc']
make_change_over_time_plot(nacc_mice, ax, window_for_binning=50, colour='#E95F32', line='#E95F32', exp_type='movement_aligned')

makes_plots_pretty([ax])
plt.tight_layout()
figure_dir = r'T:\paper\revisions\cue movement reward comparisons VS TS'
plt.savefig(os.path.join(figure_dir, 'VS_movement_over_trials.pdf'))
plt.show()