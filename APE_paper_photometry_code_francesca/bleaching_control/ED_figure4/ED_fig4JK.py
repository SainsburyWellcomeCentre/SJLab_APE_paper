import os
from utils.plotting_visuals import set_plotting_defaults
import matplotlib.pyplot as plt
from utils.post_processing_utils import remove_exps_after_manipulations
from utils.plotting import calculate_error_bars
from utils.kernel_regression.return_to_centre_regression_utils import get_first_x_sessions_reg_rtc
import pandas as pd
from utils.tracking_analysis.camera_trigger_preprocessing_utils import *
from scipy.signal import decimate
from set_global_params import processed_data_path, experiment_record_path, mice_average_traces
from set_global_params import figure_directory, reproduce_figures_path, spreadsheet_path
import shutil


num_sessions = 3
site = 'tail'
mouse_ids = mice_average_traces[site]
timeframe = 300 # this is the window in which cosine similarity is calculated and the movement onset is detected
long_turns = False #False for paper
if long_turns:
    file_extend = '_long_turns'
    cutoff = 'no'
else:
    file_extend = ''
    cutoff = '1s'
# long turns is not for paper but was used for revision rebuttal
experiment_record = pd.read_csv(experiment_record_path)
experiment_record['date'] = experiment_record['date'].astype(str)
clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
all_experiments_to_process = clean_experiments[
    (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
    drop=True)
all_experiments_to_process = all_experiments_to_process[all_experiments_to_process['include return to centre'] != 'no'].reset_index(
    drop=True)
experiments_to_process = get_first_x_sessions_reg_rtc(all_experiments_to_process, x=num_sessions).reset_index(
    drop=True)
all_mice_contra_traces = np.zeros([len(mouse_ids), 100000])
all_mice_ipsi_traces = np.zeros([len(mouse_ids), 100000])

i = 0
time_points = np.linspace(-5, 5, 100000)
for mouse_num, mouse in enumerate(mouse_ids):
    experiments = experiments_to_process[experiments_to_process['mouse_id'] == mouse].reset_index(drop=True)
    contra_movement_traces = np.zeros([num_sessions, 100000])
    ipsi_movement_traces = np.zeros([num_sessions, 100000])
    for index, experiment in experiments.iterrows():
        mouse = experiment['mouse_id']
        date = experiment['date']
        save_dir = os.path.join(processed_data_path, 'return_to_centre', mouse)
        save_file = '{}_{}_return_to_centre_traces_aligned_to_movement_start_turn_ang_thresh_{}frame_window{}.npz'.format(mouse, date, timeframe, file_extend)
        # check if file exists in the reproducing figures folder - otherwise copy it over
        repro_load_path = os.path.join(reproduce_figures_path, 'ED_fig4', 'return_to_centre', mouse)
        if not os.path.exists(repro_load_path):
            os.makedirs(repro_load_path)
        if not os.path.exists(os.path.join(repro_load_path, save_file)):
            shutil.copy(os.path.join(save_dir, save_file), os.path.join(repro_load_path, save_file))
        traces = np.load(os.path.join(repro_load_path, save_file))
        print(traces['ipsi_movement'].shape, traces['contra_movement'].shape)
        if traces['ipsi_movement'].shape[1] >= 20 or traces['contra_movement'].shape[1] >= 20:
            contra_movement_traces[index, :] = np.mean(traces['contra_movement'], axis=1)
            ipsi_movement_traces[index, :] = np.mean(traces['ipsi_movement'], axis=1)
            i += 1
        else:
            contra_movement_traces[index, :] = np.nan
            ipsi_movement_traces[index, :] = np.nan
    all_mice_ipsi_traces[mouse_num, :] = np.nanmean(ipsi_movement_traces, axis=0)
    all_mice_contra_traces[mouse_num, :] = np.nanmean(contra_movement_traces, axis=0)

window_to_plot = [-0.5, 2]
inds_to_plot = np.where(np.logical_and(time_points>=window_to_plot[0], time_points<=window_to_plot[1]))
time_points_for_plot = decimate(time_points[inds_to_plot], 10)


set_plotting_defaults(font_size=8)
decimated_contra_traces = decimate(all_mice_contra_traces[:, inds_to_plot[0]], 10)
decimated_ipsi_traces = decimate(all_mice_ipsi_traces[:, inds_to_plot[0]], 10)
spreadsheet_file_path = os.path.join(spreadsheet_path, 'ED_fig4')
if not os.path.exists(os.path.join(spreadsheet_file_path, 'ED_fig4K_contra_traces.csv')):
    df_for_spreadsheet = pd.DataFrame(decimated_contra_traces.T)
    df_for_spreadsheet.insert(0, "Timepoints", time_points_for_plot)
    df_for_spreadsheet.to_csv(os.path.join(spreadsheet_file_path, 'ED_fig4K_contra_traces.csv'))
if not os.path.exists(os.path.join(spreadsheet_file_path, 'ED_fig4K_ipsi_traces.csv')):
    df_for_spreadsheet = pd.DataFrame(decimated_ipsi_traces.T)
    df_for_spreadsheet.insert(0, "Timepoints", time_points_for_plot)
    df_for_spreadsheet.to_csv(os.path.join(spreadsheet_file_path, 'ED_fig4K_ipsi_traces.csv'))

contra_mean_trace = decimate(np.nanmean(all_mice_contra_traces, axis=0)[inds_to_plot], 10)
ipsi_mean_trace = decimate(np.nanmean(all_mice_ipsi_traces, axis=0)[inds_to_plot], 10)
fig, axs = plt.subplots(1, 1, figsize=[2.5, 2])

axs.plot(time_points_for_plot, contra_mean_trace, color='#002F3A')
axs.plot(time_points_for_plot, ipsi_mean_trace, color='#76A8DA')

contra_error_bar_lower, contra_error_bar_upper = calculate_error_bars(contra_mean_trace,
                                                                decimated_contra_traces,
                                                                error_bar_method='sem')
axs.fill_between(time_points_for_plot, contra_error_bar_lower, contra_error_bar_upper, alpha=0.5,
                            facecolor='#002F3A', linewidth=0)

ipsi_error_bar_lower, ipsi_error_bar_upper = calculate_error_bars(ipsi_mean_trace,
                                                                decimated_ipsi_traces,
                                                                error_bar_method='sem')
axs.fill_between(time_points_for_plot, ipsi_error_bar_lower, ipsi_error_bar_upper, alpha=0.5,
                            facecolor='#76A8DA', linewidth=0)

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.yaxis.set_ticks_position('left')
axs.xaxis.set_ticks_position('bottom')

axs.axvline(0, color='k')
axs.set_ylabel('z-scored fluorescence')
axs.set_xlabel('time from movement start (s)')
plt.tight_layout()
axs.set_xlim(window_to_plot)
axs.set_ylim([-0.75, 0.6])
plt.tight_layout()
#plt.savefig(os.path.join(figure_directory, 'return_movements_{}_cutoff_average{}s.pdf'.format(cutoff, timeframe/30)))

# Example mouse

mouse = 'SNL_photo21'
num_sessions = 3
site = 'tail'

time_points = np.linspace(-5, 5, 100000)

experiments = experiments_to_process[experiments_to_process['mouse_id'] == mouse].reset_index(drop=True)
contra_movement_traces = np.zeros([num_sessions, 100000])
ipsi_movement_traces = np.zeros([num_sessions, 100000])
for index, experiment in experiments.iterrows():
    mouse = experiment['mouse_id']
    date = experiment['date']
    repro_load_path = os.path.join(reproduce_figures_path, 'ED_fig4', 'return_to_centre', mouse)
    save_file = '{}_{}_return_to_centre_traces_aligned_to_movement_start_turn_ang_thresh_{}frame_window{}.npz'.format(mouse, date, timeframe, file_extend)
    traces = np.load(os.path.join(repro_load_path, save_file))
    print(traces['ipsi_movement'].shape, traces['contra_movement'].shape)
    if traces['ipsi_movement'].shape[1] >= 20 or traces['contra_movement'].shape[1] >= 20:
        contra_movement_traces[index, :] = np.mean(traces['contra_movement'], axis=1)
        ipsi_movement_traces[index, :] = np.mean(traces['ipsi_movement'], axis=1)
    else:
        contra_movement_traces[index, :] = np.nan
        ipsi_movement_traces[index, :] = np.nan

fig, axs = plt.subplots(1, 1, figsize=[2.5, 2])

decimated_contra_traces_example = decimate(contra_movement_traces[:, inds_to_plot[0]], 10)
decimated_ipsi_traces_example = decimate(ipsi_movement_traces[:, inds_to_plot[0]], 10)
spreadsheet_file_path = os.path.join(spreadsheet_path, 'ED_fig4')
if not os.path.exists(os.path.join(spreadsheet_file_path, 'ED_fig4J_contra_traces.csv')):
    df_for_spreadsheet = pd.DataFrame(decimated_contra_traces_example.T)
    df_for_spreadsheet.insert(0, "Timepoints", time_points_for_plot)
    df_for_spreadsheet.to_csv(os.path.join(spreadsheet_file_path, 'ED_fig4J_contra_traces.csv'))
if not os.path.exists(os.path.join(spreadsheet_file_path, 'ED_fig4J_ipsi_traces.csv')):
    df_for_spreadsheet = pd.DataFrame(decimated_ipsi_traces_example.T)
    df_for_spreadsheet.insert(0, "Timepoints", time_points_for_plot)
    df_for_spreadsheet.to_csv(os.path.join(spreadsheet_file_path, 'ED_fig4J_ipsi_traces.csv'))


ipsi_mean_trace = decimate(np.nanmean(ipsi_movement_traces, axis=0)[inds_to_plot],10)
contra_mean_trace = decimate(np.nanmean(contra_movement_traces, axis=0)[inds_to_plot],10)
axs.plot(time_points_for_plot, contra_mean_trace, color='#002F3A')
axs.plot(time_points_for_plot, ipsi_mean_trace, color='#76A8DA')

contra_error_bar_lower, contra_error_bar_upper = calculate_error_bars(contra_mean_trace,
                                                                decimated_contra_traces_example ,
                                                                error_bar_method='sem')
axs.fill_between(time_points_for_plot, contra_error_bar_lower, contra_error_bar_upper, alpha=0.5,
                            facecolor='#002F3A', linewidth=0)

ipsi_error_bar_lower, ipsi_error_bar_upper = calculate_error_bars(ipsi_mean_trace,
                                                                decimated_ipsi_traces_example,
                                                                error_bar_method='sem')
axs.fill_between(time_points_for_plot, ipsi_error_bar_lower, ipsi_error_bar_upper, alpha=0.5,
                            facecolor='#76A8DA', linewidth=0)


axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.yaxis.set_ticks_position('left')
axs.xaxis.set_ticks_position('bottom')

axs.axvline(0, color='k')
axs.set_ylabel('z-scored fluorescence')
axs.set_xlabel('time from movement start (s)')
plt.tight_layout()
axs.set_xlim(window_to_plot)

plt.tight_layout()

#plt.savefig(os.path.join(figure_directory, 'return_movements_{}_cutoff_example{}s.pdf'.format(cutoff, timeframe/30)))

plt.show()


