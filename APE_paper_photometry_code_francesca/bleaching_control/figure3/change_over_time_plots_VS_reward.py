from set_global_params import change_over_time_mice
import matplotlib
from utils.plotting_visuals import makes_plots_pretty
from utils.change_over_time_plot_utils import  *


font = {'size': 7}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']

fig, ax = plt.subplots(2, 1, figsize=[2,4], constrained_layout=True)
tail_mice = change_over_time_mice['Nacc']
make_change_over_time_plot(tail_mice, ax[0], window_for_binning=50, colour='#E95F32', line='#E95F32', align_to='movement')

nacc_mice = change_over_time_mice['Nacc']
make_change_over_time_plot(nacc_mice, ax[1], window_for_binning=50, colour='#E95F32', line='#E95F32', align_to='reward')

makes_plots_pretty([ax[0], ax[1]])
plt.tight_layout()
figure_dir = r'T:\paper\revisions\cue movement reward comparisons VS TS'
#plt.savefig(os.path.join(figure_dir, 'VS_reward_and_movement_over_trials.pdf'))
plt.show()