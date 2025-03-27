# change over time traces binned by trial

import matplotlib
from utils.plotting_visuals import makes_plots_pretty
from utils.change_over_time_plot_utils import *
from save_to_excel import save_ax_data_to_excel
from set_global_params import spreadsheet_path
import os

sh_path = os.path.join(spreadsheet_path, 'fig3')
if not os.path.exists(sh_path):
    os.makedirs(sh_path)

font = {'size': 7}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['font.family']
bin_window = 200
fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=[2, 3.5])

make_example_traces_plot('SNL_photo17', axs[0], window_for_binning=bin_window, legend=False, align_to='movement')
make_example_traces_plot('SNL_photo35', axs[1], window_for_binning=bin_window, legend=False, align_to='cue')

sh_fn_nacc = 'fig3A_traces_per_session_example_mouse_nacc.xlsx'
sh_fn_tail = 'fig3E_traces_per_session_example_mouse_tail.xlsx'
save_ax_data_to_excel(axs[1], os.path.join(sh_path, sh_fn_nacc))
save_ax_data_to_excel(axs[0], os.path.join(sh_path, sh_fn_tail))
makes_plots_pretty(axs)
plt.tight_layout()
plt.show()
