import os
import matplotlib.pyplot as plt
import pandas as pd

from set_global_params import processed_data_path, figure_directory, experiment_record_path, post_processed_tracking_data_path, mice_average_traces, old_raw_tracking_path, raw_tracking_path
from utils.tracking_analysis.fede_load_tracking import prepare_tracking_data
from utils.tracking_analysis.dlc_processing_utils import get_raw_photometry_data
from utils.post_processing_utils import remove_exps_after_manipulations
from utils.kernel_regression.linear_regression_utils import rolling_zscore
from utils.kernel_regression.return_to_centre_regression_utils import get_first_x_sessions_reg_rtc
from utils.return_to_centre_utils import *


num_sessions = 3
site = 'tail'
mouse_ids = mice_average_traces[site]
use_old_tracking = False
short_turns = False # True for averages false for regression
timeframe = 300 # 300 for figures


old_mice = ['SNL_photo16', 'SNL_photo17', 'SNL_photo18', 'SNL_photo21', 'SNL_photo22', 'SNL_photo26']
experiment_record = pd.read_csv(experiment_record_path, dtype=str)
experiment_record['date'] = experiment_record['date'].astype(str)
clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
all_experiments_to_process = clean_experiments[
    (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
    drop=True)
all_experiments_to_process = all_experiments_to_process[all_experiments_to_process['include return to centre'] != 'no'].reset_index(
    drop=True)
experiments_to_process = get_first_x_sessions_reg_rtc(all_experiments_to_process, x=num_sessions).reset_index(
    drop=True)

# Initialize a list to store return duration data
data_list = []
not_included_data = []
for index, experiment in experiments_to_process.iterrows():
    plt.close('all')
    mouse = experiment['mouse_id']
    date = experiment['date']
    save_out_folder = os.path.join(post_processed_tracking_data_path, mouse)

    port_coords = {'side1': np.array([int(experiments_to_process['left side x'][0]), int(experiments_to_process['left side y'][0])]),
                  'centre': np.array([int(experiments_to_process['centre port x'][0]), int(experiments_to_process['centre port y'][0])]),
                  'side2': np.array([int(experiments_to_process['right side x'][0]), int(experiments_to_process['right side y'][0])])}

    if mouse in old_mice and use_old_tracking:
        file_path = os.path.join(old_raw_tracking_path, '{}_{}DLC_resnet50_two_acMay10shuffle1_600000.h5'.format(mouse, date))
        protocol = 'Two_Alternative_Choice'
        left_ang_thresh = 85
        right_ang_thresh = 280

    else:
        file_path = os.path.join(raw_tracking_path, mouse, date,'cameraDLC_resnet50_train_network_with_more_miceMar2shuffle1_800000.h5')
        protocol = 'Two_Alternative_Choice_CentrePortHold'
        left_ang_thresh = 100
        right_ang_thresh = 300
        if mouse in old_mice:
            protocol = 'Two_Alternative_Choice'
            left_ang_thresh = 85
            right_ang_thresh = 280

    body_parts = ('nose', 'left ear', 'right ear', 'tail base', 'tail tip')
    tracking_data = prepare_tracking_data(
        tracking_filepath=file_path,
        tracking=None,
        bodyparts=body_parts,
        likelihood_th=0.999,
        median_filter=True,
        filter_kwargs={},
        compute=True,
        smooth_dir_mvmt=True,
        interpolate_nans=True,
        verbose=False)

    camera_triggers, trial_start_stamps = get_camera_trigger_times(mouse, date, protocol) # currently requires the raw daq files
    trial_start_triggers = find_nearest_trials(trial_start_stamps, camera_triggers)
    dff_data, trial_data = get_raw_photometry_data(mouse, date)
    photometry_data = rolling_zscore(pd.Series(dff_data), window=10 * 10000)

    fiber_side = experiment['fiber_side']
    if fiber_side == 'left':
        ipsi_port = 'side1'
        contra_port = 'side2'
        ipsi_turn_ang_thresh = left_ang_thresh
        contra_turn_ang_thresh = right_ang_thresh
    else:
        ipsi_port = 'side2'
        contra_port = 'side1'
        ipsi_turn_ang_thresh = right_ang_thresh
        contra_turn_ang_thresh = left_ang_thresh

    fiber_options = np.array(['left', 'right'])
    fiber_side_numeric = (np.where(fiber_options == fiber_side)[0] + 1)[0]
    contra_fiber_side_numeric = (np.where(fiber_options != fiber_side)[0] + 1)[0]

    contra_movement_traces, contra_movement_onsets_time_stamps, contra_return_durations, contra_bad_durations = get_photometry_for_one_side_returns(fiber_side_numeric,
                                                                                                     camera_triggers,
                                                                                                     trial_data,
                                                                                                     tracking_data,
                                                                                                     ipsi_turn_ang_thresh,
                                                                                                     port_coords,
                                                                                                     photometry_data,
                                                                                                     side=ipsi_port,
                                                                                                     side_port=fiber_side,
                                                                                                     time_frame=timeframe,
                                                                                                     short_turns_only=short_turns)

    ipsi_movement_traces, ipsi_movement_onsets_time_stamps, ipsi_return_durations, ipsi_bad_durations = get_photometry_for_one_side_returns(contra_fiber_side_numeric,
                                                                                                     camera_triggers,
                                                                                                     trial_data,
                                                                                                     tracking_data,
                                                                                                     contra_turn_ang_thresh,
                                                                                                     port_coords,
                                                                                                     photometry_data,
                                                                                                     side=contra_port,
                                                                                                     side_port=fiber_options[np.where(fiber_options != fiber_side)],
                                                                                                     time_frame=timeframe,
                                                                                                     short_turns_only=short_turns)
    # Store data in the list
    for duration in ipsi_return_durations:
        data_list.append({'mouse': mouse, 'date': date, 'type': 'ipsi', 'duration': duration / 10000})
    for duration in contra_return_durations:
        data_list.append({'mouse': mouse, 'date': date, 'type': 'contra', 'duration': duration / 10000})

    for duration in ipsi_bad_durations:
        not_included_data.append({'mouse': mouse, 'date': date, 'type': 'ipsi', 'duration': duration / 10000})
    for duration in contra_bad_durations:
        not_included_data.append({'mouse': mouse, 'date': date, 'type': 'contra', 'duration': duration / 10000})

    save_dir = os.path.join(processed_data_path, 'return_to_centre', mouse)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if short_turns:
        save_file = '{}_{}_return_to_centre_traces_aligned_to_movement_start_turn_ang_thresh_{}frame_window.npz'.format(mouse, date, timeframe)
        time_stamp_save_file = '{}_{}_return_to_centre_movement_onset_times_{}frame_window.npz'.format(mouse, date, timeframe)
    else:
        save_file = '{}_{}_return_to_centre_traces_aligned_to_movement_start_turn_ang_thresh_{}frame_window_long_turns.npz'.format(
            mouse, date, timeframe)
        time_stamp_save_file = '{}_{}_return_to_centre_movement_onset_times_{}frame_window_long_turns.npz'.format(mouse, date, timeframe)
    np.savez(os.path.join(save_dir, save_file), contra_movement=contra_movement_traces, ipsi_movement=ipsi_movement_traces)
    np.savez(os.path.join(save_dir, time_stamp_save_file), contra_movement_return=contra_movement_onsets_time_stamps,
             ipsi_movement_return=ipsi_movement_onsets_time_stamps)

# Create a DataFrame from the collected data
duration_df = pd.DataFrame(data_list)

# Plot histograms for ipsi and contra return durations
bin_width = 0.1
bins = np.arange(0, duration_df['duration'].max() + bin_width, bin_width)

# Calculate mean and standard deviation for ipsi and contra durations
ipsi_mean = duration_df[duration_df['type'] == 'ipsi']['duration'].mean()
ipsi_std = duration_df[duration_df['type'] == 'ipsi']['duration'].std()
contra_mean = duration_df[duration_df['type'] == 'contra']['duration'].mean()
contra_std = duration_df[duration_df['type'] == 'contra']['duration'].std()


plt.close('all')
plt.figure(figsize=(6, 6))
plt.hist(duration_df[duration_df['type'] == 'ipsi']['duration'], bins=bins, alpha=0.5, label='Ipsi')
plt.hist(duration_df[duration_df['type'] == 'contra']['duration'], bins=bins, alpha=0.5, label='Contra')
plt.axvline(x=10, color='k', linestyle='--', linewidth=0.5)
plt.xlabel('Return Duration (s)')
plt.ylabel('Frequency')
plt.title('Histogram of Ipsi and Contra Return Durations')
# Add text for mean and std
plt.text(0.05, plt.ylim()[1] * 0.8, f'Ipsi Mean: {ipsi_mean:.2f}\nIpsi Std: {ipsi_std:.2f}', color='blue')
plt.text(0.05, plt.ylim()[1] * 0.6, f'Contra Mean: {contra_mean:.2f}\nContra Std: {contra_std:.2f}', color='orange')

plt.legend()
plt.savefig(os.path.join(figure_directory, 'return_movement_durations_timeframe_{}.pdf'.format(timeframe/30)))

not_included_df = pd.DataFrame(not_included_data)
num_included_trials = duration_df.shape[0]
inc_and_not_df = pd.concat([duration_df, not_included_df])
num_excluded_trials = not_included_df.shape[0]
print('proportion_data_included = {}'.format(num_included_trials/(num_excluded_trials + num_included_trials)))
plt.figure(figsize=(6, 6))
plt.hist(inc_and_not_df[inc_and_not_df['type'] == 'ipsi']['duration'], bins=bins, alpha=0.5, label='Ipsi')
plt.hist(inc_and_not_df[inc_and_not_df['type'] == 'contra']['duration'], bins=bins, alpha=0.5, label='Contra')
plt.axvline(x=10, color='k', linestyle='--', linewidth=0.5)
plt.xlabel('Return Duration (s)')
plt.ylabel('Frequency')
plt.title('Histogram of Ipsi and Contra Return Durations')
# Add text for mean and std
plt.text(0.05, plt.ylim()[1] * 0.8, f'Ipsi Mean: {ipsi_mean:.2f}\nIpsi Std: {ipsi_std:.2f}', color='blue')
plt.text(0.05, plt.ylim()[1] * 0.6, f'Contra Mean: {contra_mean:.2f}\nContra Std: {contra_std:.2f}', color='orange')

plt.legend()
plt.savefig(os.path.join(figure_directory, 'return_movement_durations_timeframe_{}_also_bad_cos_similarity.pdf'.format(timeframe/30)))