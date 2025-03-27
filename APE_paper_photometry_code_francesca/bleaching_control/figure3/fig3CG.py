from set_global_params import change_over_time_mice
import matplotlib
from utils.plotting_visuals import makes_plots_pretty
from utils.change_over_time_plot_utils import  *
from set_global_params import spreadsheet_path
import os
import pandas as pd

sh_path = os.path.join(spreadsheet_path, 'fig3')


# this makes the plot in the figure 3 C & G
font = {'size': 8}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(2, 1, figsize=[2, 4], constrained_layout=True)
tail_mice = change_over_time_mice['tail']
tail_ys = make_change_over_time_plot(tail_mice, ax[0], window_for_binning=50, colour='#002F3A', line='#002F3A')

nacc_mice = change_over_time_mice['Nacc']
nacc_ys = make_change_over_time_plot(nacc_mice, ax[1], window_for_binning=50, colour='#E95F32', line='#E95F32')

# save spreadsheet data
fn_nacc = os.path.join(spreadsheet_path,'fig3', 'fig3C_vs.csv')
nacc_df = pd.DataFrame(nacc_ys.T)
nacc_df = nacc_df.dropna(axis=0)
nacc_df.columns = [f'Mouse_{i}' for i in range(nacc_df.shape[1])]
nacc_df.to_csv(fn_nacc)
fn_tail = os.path.join(spreadsheet_path,'fig3','fig3G_ts.csv')
tail_df = pd.DataFrame(tail_ys.T)
tail_df = tail_df.dropna(axis=0)
tail_df.columns = [f'Mouse_{i}' for i in range(tail_df.shape[1])]
tail_df.to_csv(fn_tail)

makes_plots_pretty([ax[0], ax[1]])
plt.show()