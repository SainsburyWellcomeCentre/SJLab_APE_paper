import matplotlib.pylab as plt
from utils.zscored_plots_utils import get_all_mouse_data_for_site, plot_average_trace_all_mice
import matplotlib
from set_global_params import reproduce_figures_path, spreadsheet_path, processed_data_path
import os
from scipy.signal import decimate
import pandas as pd
from functools import partial
from utils.plotting_visuals import makes_plots_pretty

spreadsheet_folder = os.path.join(spreadsheet_path, 'ED_fig4')

font = {'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']
site='tail_ant'
fig = plt.figure(constrained_layout=True, figsize=[4, 2])
gs = fig.add_gridspec(nrows=1, ncols=2)


ant_tail_average_move_ax = fig.add_subplot(gs[0, 0])
ant_tail_average_move_ax.set_title('Choice')
ant_tail_average_outcome_ax = fig.add_subplot(gs[0, 1])
ant_tail_average_outcome_ax.set_title('Outcome')

if os.path.exists(os.path.join(reproduce_figures_path, 'ED_fig4')):
    print('loading cached group data')
    dir = os.path.join(reproduce_figures_path, 'ED_fig4')
else:
    dir = os.path.join(processed_data_path, 'for_figure')
ant_t_outcome_data, ant_t_move_data = plot_average_trace_all_mice(ant_tail_average_move_ax,  ant_tail_average_outcome_ax, dir, site, cmap=['#E95F32', '#F9C0AF'])

makes_plots_pretty([ant_tail_average_move_ax,  ant_tail_average_outcome_ax])
plt.tight_layout()


####################### write to csvs for nature
ds = partial(decimate, q=10)

ant_t_outcome_df = pd.DataFrame()
ant_t_outcome_df['time'] = ds(ant_t_outcome_data['time'])
for m in range(ant_t_outcome_data['data'][0].shape[0]):
    ant_t_outcome_df[f'reward_m{m}'] = ds(ant_t_outcome_data['data'][0][m])
    ant_t_outcome_df[f'no_reward_m{m}'] = ds(ant_t_outcome_data['data'][1][m])
ant_t_outcome_df.to_csv(os.path.join(spreadsheet_folder, 'ED_fig4O_ant_t_outcome_avg_traces.csv'))

ant_t_move_df = pd.DataFrame()
ant_t_move_df['time'] = ds(ant_t_move_data['time'])
for m in range(ant_t_move_data['data'][0].shape[0]):
    ant_t_move_df[f'contra_m{m}'] = ds(ant_t_move_data['data'][0][m])
    ant_t_move_df[f'ipsi_m{m}'] = ds(ant_t_move_data['data'][1][m])
ant_t_move_df.to_csv(os.path.join(spreadsheet_folder, 'ED_fig4N_ant_t_move_avg_traces.csv'))


plt.show()

