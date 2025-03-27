import matplotlib.pylab as plt
from utils.zscored_plots_utils import get_example_data_for_recording_site, make_y_lims_same_heat_map, plot_all_heatmaps_same_scale, plot_average_trace_all_mice_cue_move_rew
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.colors
import numpy as np
import matplotlib
import pandas as pd
import cmocean as cmo
import os
from scipy.signal import decimate
from utils.plotting_visuals import makes_plots_pretty
from set_global_params import spreadsheet_path
from set_global_params import reproduce_figures_path
from utils.data_loading_utils import load_or_get_and_save_example_heatmap_aligned_to_keys

# location for saving/loading only the necessary data:
pickle_folder = os.path.join(reproduce_figures_path, 'ED_fig4')
# location for spreadsheets for nature:
spreadsheet_folder = os.path.join(spreadsheet_path, 'ED_fig4')

font = {'size': 8}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.sans-serif'] = 'Arial'


cmap_map = matplotlib.cm.get_cmap('cmo.solar') #('cet_fire')#cc.cm['linear_wyor_100_45_c55']) #cc.cm['linear_tritanopic_krw_5_95_c46']) cmo.solar
cmap_values = cmap_map(np.linspace(0, 1, 1200))

new_cmap_values = np.copy(cmap_values)
new_cmap_values[:, 2] = np.copy(cmap_values[:, 0])
new_cmap_values[:, 0] = np.copy(cmap_values[:, 2])

blue_cmap = ListedColormap(new_cmap_values)
red_cmap = ListedColormap(cmap_values)

# heatmaps
fig, axs = plt.subplots(1, 2, figsize=[4, 2])
ts_ax = axs[0]
ts_ax.set_title('cue TS')
vs_ax = axs[1]
vs_ax.set_title('cue VS')

ts_heat_map_axs = {'cue': [ts_ax]}
vs_heat_map_axs = {'cue': [vs_ax]}

colours = sns.color_palette("Set2")[:2]

ts_pickle_path = os.path.join(pickle_folder, 'ts_example_heatmap_data.pkl')
vs_pickle_path = os.path.join(pickle_folder, 'vs_example_heatmap_data.pkl')
time_window_size = 2


tail_hm_data = load_or_get_and_save_example_heatmap_aligned_to_keys(ts_pickle_path, 'TS', ['cue'],
                                                                    time_window_size=time_window_size)
vs_hm_data = load_or_get_and_save_example_heatmap_aligned_to_keys(vs_pickle_path, 'VS', ['cue'],
                                                                  time_window_size=time_window_size)

t_data, t_wd, t_flip_sort_order, t_y_mins, t_y_maxs = tail_hm_data
v_data, v_wd, v_flip_sort_order, v_y_mins, v_y_maxs = vs_hm_data

t_axs = [a[0] for a in ts_heat_map_axs.values()]
v_axs = [a[0] for a in vs_heat_map_axs.values()]
heat_map_t = plot_all_heatmaps_same_scale(t_axs, t_data, t_wd, t_flip_sort_order, [np.min(t_y_mins), np.max(t_y_maxs)], cmap=blue_cmap)
heat_map_v = plot_all_heatmaps_same_scale(v_axs, v_data, v_wd, v_flip_sort_order, [np.min(v_y_mins), np.max(v_y_maxs)], cmap=red_cmap)
areas = ['TS', 'VS']
heatmap_data = [hm.get_array() for hm in heat_map_t + heat_map_v]
# downsample because of nature's storage limit
heatmap_data = [decimate(hm, 50, axis=-1) for hm in heatmap_data]

for i, hm in enumerate(heatmap_data):
    fn = f'ED_fig4F_heatmap_data_{areas[i]}_cue_aligned.csv'
    fp = os.path.join(spreadsheet_folder, fn)
    hm_df = pd.DataFrame(hm.T)
    hm_df.columns = [f'Trial_{i}' for i in range(hm_df.shape[1])]
    hm_df.to_csv(fp)
    np.savetxt(fp, hm)
plt.tight_layout()
plt.show()

