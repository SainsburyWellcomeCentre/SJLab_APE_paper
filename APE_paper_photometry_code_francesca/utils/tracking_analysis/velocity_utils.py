from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.signal import decimate
from utils.tracking_analysis.fede_geometry import *
from utils.tracking_analysis import dlc_processing_utils
from sklearn.metrics import mean_squared_error


def calculate_velocity(trajectory):
    dist = []
    for n, pos in enumerate(trajectory):
        # Get a pair of points
        if n == 0:  # get the position at time 0, velocity is 0
            p0 = pos
            dist.append(0)
        else:
            p1 = pos  # get position at current frame

            # Calc distance
            dist.append(np.abs(distance.euclidean(p0, p1)))

            # Prepare for next iteration, current position becomes the old one and repeat
            p0 = p1

    return np.array(dist)

def plot_trace_for_param_quantiles(param, traces, num_divisions=4):
    fig, ax = plt.subplots(1, num_divisions + 2, figsize=((num_divisions + 1) * 3, 3))
    clean_param_to_sort_by = np.asarray(param)
    vals, bins, q = ax[0].hist(clean_param_to_sort_by, bins=int(clean_param_to_sort_by.shape[0] / num_divisions),
                               color='k')
    colours = matplotlib.cm.viridis(np.linspace(0, 0.8, num_divisions))
    inds = []
    for q in range(0, num_divisions):
        chunk = 1 / num_divisions
        if q >= 1 and q < num_divisions - 1:
            lower_quantile = np.quantile(clean_param_to_sort_by, chunk * q)
            upper_quantile = np.quantile(clean_param_to_sort_by, chunk * q + chunk)
            trials_in_quantile = \
            np.where(np.logical_and(clean_param_to_sort_by > lower_quantile, clean_param_to_sort_by <= upper_quantile))[
                0]
            ax[0].axvline(lower_quantile, 0, max(vals), color='r')
            ax[0].axvline(upper_quantile, 0, max(vals), color='r')
        elif q == 0:
            quantile = np.quantile(clean_param_to_sort_by, chunk * q + chunk)
            trials_in_quantile = np.where(clean_param_to_sort_by < quantile)[0]
            ax[0].axvline(quantile, 0, max(vals), color='r')
        elif q == num_divisions - 1:
            quantile = np.quantile(clean_param_to_sort_by, chunk * q)
            trials_in_quantile = np.where(clean_param_to_sort_by >= quantile)[0]
            ax[0].axvline(quantile, 0, max(vals), color='r')
        mean_trace = np.mean(traces[trials_in_quantile, 60000:120000], axis=0)
        ax[num_divisions + 1].plot(mean_trace, color=colours[q], alpha=1, lw=2)
        ax[q + 1].plot(mean_trace, color=colours[q], alpha=1, lw=2)
        ax[q + 1].set_title('quantile: {}'.format(q))
        inds.append(trials_in_quantile)
    ax[num_divisions + 1].set_title('all quantiles')
    return inds


def format_tracking_data_and_photometry_old(head_coordinates, photometry_data, cot_triggers, choice_triggers):
    trial_nums = photometry_data.choice_data.contra_data.trial_nums
    contra_peaks = photometry_data.choice_data.contra_data.trial_peaks
    trace_midpoint = int(photometry_data.choice_data.contra_data.sorted_traces.shape[1]/2)
    photometry_traces = photometry_data.choice_data.contra_data.sorted_traces[:, trace_midpoint - 30000: trace_midpoint + 30000]
    velocity = calculate_velocity(zip(head_coordinates[0], head_coordinates[1]))
    max_velocities = []
    avg_velocities = []
    time_to_max_vels = []
    max_accels = []
    avg_accels = []
    time_to_max_accels = []
    velocities = []
    accelerations = []
    times_to_move = []
    traces = []
    photo_inds = []
    head_x = []
    head_y = []

    for i in trial_nums.astype(int):
        trial_velocity = velocity[cot_triggers[i]: choice_triggers[i]]
        acceleration = np.diff(trial_velocity)
        max_velocities.append(np.max(trial_velocity))
        avg_velocities.append(np.mean(trial_velocity))
        time_to_max_vels.append(np.where(trial_velocity == np.max(trial_velocity))[0][0])
        max_accels.append(np.max(acceleration))
        avg_accels.append(np.mean(acceleration))
        time_to_max_accels.append(np.where(acceleration == np.max(acceleration))[0][0])
        velocities.append(trial_velocity)
        accelerations.append(acceleration)
        photometry_ind = np.where(trial_nums == i)[0]
        photo_inds.append(photometry_ind)
        traces.append(photometry_traces[photometry_ind,:].T)
        head_x.append(head_coordinates[0][cot_triggers[i]: choice_triggers[i]])
        head_y.append(head_coordinates[1][cot_triggers[i]: choice_triggers[i]])
        if np.any(trial_velocity > 7):
            times_to_move.append(np.where(trial_velocity > 15)[0][0])
        else:
            times_to_move.append(np.nan)
    clean_peaks = []
    for c in contra_peaks:
        if c.size != 0:
            clean_peaks.append(c)
        else:
            clean_peaks.append(np.nan)
    data = {'max velocity': max_velocities, 'average velocity': avg_velocities, 'time to max vel': time_to_max_vels,
            'velocities': velocities
        , 'max acceleration': max_accels, 'average acceleration': avg_accels,
            'time to max acceleration': time_to_max_accels, 'accelerations': accelerations, 'APE peaks': clean_peaks,
            'time to move': times_to_move, 'trial numbers': trial_nums, 'traces': traces, 'head x': head_x, 'head y': head_y}
    data_df = pd.DataFrame(data)
    clean_df = data_df.drop(data_df.loc[data_df['max velocity'] > 35].index)
    non_nan_df = clean_df.drop(clean_df.loc[np.isnan(clean_df['APE peaks'])].index)
    non_nan_df = non_nan_df.drop(non_nan_df.loc[np.isnan(non_nan_df['time to move'])].index)
    ape_sorted_data = non_nan_df.sort_values(by='APE peaks', ascending=False).reset_index(drop=True)
    return ape_sorted_data


def format_tracking_data_and_photometry(tracking_data, photometry_data, cot_triggers, choice_triggers, trial_types, trial_numbers=None):
    trace_midpoint = int(photometry_data.choice_data.contra_data.sorted_traces.shape[1] / 2)
    if type(trial_numbers) == np.ndarray:
        contra_test_trial_nums, p_inds, inds = np.intersect1d(photometry_data.choice_data.contra_data.trial_nums, trial_numbers, return_indices=True)
        trial_nums = trial_numbers[inds]
        contra_peaks = np.array(photometry_data.choice_data.contra_data.trial_peaks)[p_inds]
        contra_reaction_times = np.array(photometry_data.choice_data.contra_data.reaction_times)[p_inds]
        contra_photometry_traces = photometry_data.choice_data.contra_data.sorted_traces[p_inds, trace_midpoint - 30000: trace_midpoint + 30000]
        ipsi_test_trial_nums, p_inds, inds = np.intersect1d(photometry_data.choice_data.ipsi_data.trial_nums, trial_numbers, return_indices=True)
        ipsi_trial_nums = trial_numbers[inds]
        ipsi_reaction_times = np.array(photometry_data.choice_data.ipsi_data.reaction_times)[p_inds]
        ipsi_peaks = np.array(photometry_data.choice_data.ipsi_data.trial_peaks)[p_inds]
        ipsi_photometry_traces = photometry_data.choice_data.ipsi_data.sorted_traces[p_inds, trace_midpoint - 30000: trace_midpoint + 30000]
    else:
        trial_nums = photometry_data.choice_data.contra_data.trial_nums
        print(len(trial_nums))
        contra_reaction_times = photometry_data.choice_data.contra_data.reaction_times
        contra_peaks = photometry_data.choice_data.contra_data.trial_peaks
        contra_photometry_traces = photometry_data.choice_data.contra_data.sorted_traces[:, trace_midpoint - 30000: trace_midpoint + 30000]
        ipsi_trial_nums = photometry_data.choice_data.ipsi_data.trial_nums
        ipsi_peaks = photometry_data.choice_data.ipsi_data.trial_peaks
        ipsi_reaction_times = photometry_data.choice_data.ipsi_data.reaction_times
        ipsi_photometry_traces = photometry_data.choice_data.ipsi_data.sorted_traces[:, trace_midpoint - 30000: trace_midpoint + 30000]

    trial_types_subset_contra = trial_types.iloc[trial_nums]['Trial type'].values
    trial_types_subset_ipsi = trial_types.iloc[ipsi_trial_nums]['Trial type'].values

    midpoints_y = (tracking_data['left ear']['y'] + tracking_data['right ear']['y']) / 2
    midpoints_x = (tracking_data['left ear']['x'] + tracking_data['right ear']['x']) / 2
    head_angles = calc_angle_between_vectors_of_points_2d(tracking_data['nose']['x'].values,
                                                          tracking_data['nose']['y'].values, midpoints_x, midpoints_y)
    head_angular_velocity = calc_ang_velocity(head_angles)
    head_ang_accel = derivative(head_angular_velocity)
    speed = tracking_data['nose']['speed'].values
    move_dir = tracking_data['nose']['direction_of_movement'].values
    acceleration = derivative(speed)


    fiber_options = np.array(['left', 'right'])
    fiber_side_numeric = (np.where(fiber_options == photometry_data.fiber_side)[0] + 1)[0]
    contra_fiber_side_numeric = (np.where(fiber_options != photometry_data.fiber_side)[0] + 1)[0]

    contra_formatted_data = format_movement_params_into_df(photometry_data, trial_nums, speed, cot_triggers, choice_triggers, acceleration, head_angles, head_angular_velocity, head_ang_accel, move_dir,
                                   tracking_data, contra_photometry_traces, contra_peaks, contra_reaction_times, trial_types_subset_contra, 'contra')
    contra_formatted_data['choice numeric'] = contra_fiber_side_numeric
    ipsi_formatted_data = format_movement_params_into_df(photometry_data, ipsi_trial_nums, speed, cot_triggers, choice_triggers, acceleration, head_angles, head_angular_velocity, head_ang_accel, move_dir,
                                   tracking_data, ipsi_photometry_traces, ipsi_peaks, ipsi_reaction_times, trial_types_subset_ipsi, 'ipsi')
    ipsi_formatted_data['choice numeric'] = fiber_side_numeric
    both_df = pd.concat([contra_formatted_data , ipsi_formatted_data]).reset_index(drop=True)
    ape_sorted_data = both_df.sort_values(by='APE peaks', ascending=False).reset_index(drop=True)
    return ape_sorted_data


def format_tracking_data_and_photometry_correct_incorrect(tracking_data, photometry_data, cot_triggers, choice_triggers, trial_types, trial_numbers=None):
    trace_midpoint = int(photometry_data.choice_data.contra_correct_data.sorted_traces.shape[1] / 2)
    if type(trial_numbers) == np.ndarray:
        correct_test_trial_nums, p_inds, inds = np.intersect1d(photometry_data.choice_data.contra_correct_data.trial_nums, trial_numbers, return_indices=True)
        trial_nums = trial_numbers[inds]
        correct_peaks = np.array(photometry_data.choice_data.contra_correct_data.trial_peaks)[p_inds]
        correct_reaction_times = np.array(photometry_data.choice_data.contra_correct_data.reaction_times)[p_inds]
        correct_photometry_traces = photometry_data.choice_data.contra_correct_data.sorted_traces[p_inds, trace_midpoint - 30000: trace_midpoint + 30000]


        incorrect_test_trial_nums, p_inds, inds = np.intersect1d(photometry_data.choice_data.contra_incorrect_data.trial_nums, trial_numbers, return_indices=True)
        incorrect_trial_nums = trial_numbers[inds]
        incorrect_reaction_times = np.array(photometry_data.choice_data.contra_incorrect_data.reaction_times)[p_inds]
        incorrect_peaks = np.array(photometry_data.choice_data.contra_incorrect_data.trial_peaks)[p_inds]
        incorrect_photometry_traces = photometry_data.choice_data.contra_incorrect_data.sorted_traces[p_inds, trace_midpoint - 30000: trace_midpoint + 30000]
    else:
        trial_nums = photometry_data.choice_data.contra_correct_data.trial_nums
        print(len(trial_nums))
        correct_reaction_times = photometry_data.choice_data.contra_correct_data.reaction_times
        correct_peaks = photometry_data.choice_data.contra_correct_data.trial_peaks
        correct_photometry_traces = photometry_data.choice_data.contra_correct_data.sorted_traces[:, trace_midpoint - 30000: trace_midpoint + 30000]
        incorrect_trial_nums = photometry_data.choice_data.contra_incorrect_data.trial_nums
        incorrect_peaks = photometry_data.choice_data.contra_incorrect_data.trial_peaks
        incorrect_reaction_times = photometry_data.choice_data.contra_incorrect_data.reaction_times
        incorrect_photometry_traces = photometry_data.choice_data.contra_incorrect_data.sorted_traces[:, trace_midpoint - 30000: trace_midpoint + 30000]

    trial_types_subset_contra = trial_types.iloc[trial_nums]['Trial type'].values
    trial_types_subset_ipsi = trial_types.iloc[incorrect_trial_nums]['Trial type'].values

    midpoints_y = (tracking_data['left ear']['y'] + tracking_data['right ear']['y']) / 2
    midpoints_x = (tracking_data['left ear']['x'] + tracking_data['right ear']['x']) / 2
    head_angles = calc_angle_between_vectors_of_points_2d(tracking_data['nose']['x'].values,
                                                          tracking_data['nose']['y'].values, midpoints_x, midpoints_y)
    head_angular_velocity = calc_ang_velocity(head_angles)
    head_ang_accel = derivative(head_angular_velocity)
    speed = tracking_data['nose']['speed'].values
    move_dir = tracking_data['nose']['direction_of_movement'].values
    acceleration = derivative(speed)


    fiber_options = np.array(['left', 'right'])
    fiber_side_numeric = (np.where(fiber_options == photometry_data.fiber_side)[0] + 1)[0]
    contra_fiber_side_numeric = (np.where(fiber_options != photometry_data.fiber_side)[0] + 1)[0]

    correct_formatted_data = format_movement_params_into_df(photometry_data, trial_nums, speed, cot_triggers, choice_triggers, acceleration, head_angles, head_angular_velocity, head_ang_accel, move_dir,
                                   tracking_data, correct_photometry_traces, correct_peaks, correct_reaction_times, trial_types_subset_contra, 'correct')
    correct_formatted_data['choice numeric'] = contra_fiber_side_numeric
    incorrect_formatted_data = format_movement_params_into_df(photometry_data, incorrect_trial_nums, speed, cot_triggers, choice_triggers, acceleration, head_angles, head_angular_velocity, head_ang_accel, move_dir,
                                   tracking_data, incorrect_photometry_traces, incorrect_peaks, incorrect_reaction_times, trial_types_subset_ipsi, 'incorrect')
    incorrect_formatted_data['choice numeric'] = contra_fiber_side_numeric
    both_df = pd.concat([correct_formatted_data , incorrect_formatted_data]).reset_index(drop=True)
    ape_sorted_data = both_df.sort_values(by='APE peaks', ascending=False).reset_index(drop=True)
    return ape_sorted_data


def format_movement_params_into_df(photometry_data, trial_nums, speed, cot_triggers, choice_triggers, acceleration, head_angles, head_angular_velocity, head_ang_accel, move_dir,
                                   tracking_data, photometry_traces, peaks, photometry_reaction_times, trial_types_subset, side):
    """
        Extracts, computes, and formats trial-aligned behavioral and photometry metrics into a DataFrame for a given trial set.

        This function processes trial-by-trial photometry and movement data, calculates kinematic variables (e.g., speed,
        acceleration, head angle dynamics), fits sigmoids to angular velocity integrals (for turning behavior), and compiles
        all variables into a structured DataFrame.

        Args:
            photometry_data (PhotometryData): Object containing trial-based photometry data and fiber side info.
            trial_nums (np.ndarray): Array of trial indices to process.
            speed (np.ndarray): Per-frame nose speed values.
            cot_triggers (np.ndarray): Cue-onset trigger frame indices for each trial.
            choice_triggers (np.ndarray): Choice event trigger frame indices for each trial.
            acceleration (np.ndarray): Per-frame acceleration values.
            head_angles (np.ndarray): Per-frame head angles.
            head_angular_velocity (np.ndarray): Per-frame angular velocity of head angle.
            head_ang_accel (np.ndarray): Per-frame angular acceleration.
            move_dir (np.ndarray): Per-frame movement direction (angle or directional vector).
            tracking_data (dict): Dictionary of tracking coordinates for body parts ('nose', 'left ear', 'right ear'),
                                  each containing x and y position time series.
            photometry_traces (np.ndarray): Aligned photometry traces per trial.
            peaks (list): List of APE peak amplitudes for each trial.
            photometry_reaction_times (list): Reaction times for each trial.
            trial_types_subset (np.ndarray): Trial type labels for this subset of trials.
            side (str): Either 'ipsi' or 'contra' to indicate trial side.

        Returns:
            pd.DataFrame: A DataFrame where each row represents a trial, and each column is a behavioral or photometry feature, including:
                - Speed, acceleration, time to move
                - Head kinematics (angle, angular velocity, angular acceleration)
                - Sigmoid fit params on cumulative angular velocity (e.g., slope, MSE, AUC)
                - Photometry trace and APE peaks
                - Trajectory (x/y) and distance traveled
                - Trial metadata (trial type, number, side, reaction time, etc.)

        Notes:
            - The direction of turning behavior is accounted for by flipping metrics based on the fiber side.
            - Sigmoid fitting is used to characterize turn onset and slope based on integrated angular velocity.
            - Trials with extreme outliers or missing data (e.g., peak not found) are removed from the final DataFrame.
            - If `max speed > 35`, or APE peak / time-to-move is NaN, the trial is excluded.

        Important Calculations:
            -  fitted max cumsum ang vel : the plateau of the fitted sigmoid to cumsum ang vel (the abs of this is used as turn angle in the paper)
            -  avg speed : average speed (pixels/frame) during the choice movement
            - `max initial turn`: cumulative angular velocity within first few frames post cue.
            - `turn slope`: from slope of sigmoid fit of cumulative angular velocity.
            - `time to move`: time after cue where speed exceeds threshold (7).
            - `area under sigmoid`: area under fitted sigmoid of cumsum ang velocity

        """
    if photometry_data.fiber_side == 'left':
        max_function = np.min
        arg_max_function = np.argmin
    else:
        max_function = np.max
        arg_max_function = np.argmax

    max_speeds = []
    avg_speeds = []
    time_to_max_speeds = []
    max_accels = []
    avg_accels = []
    time_to_max_accels = []
    speeds = []
    accelerations = []
    times_to_move = []
    traces = []
    photo_inds = []
    head_x = []
    head_y = []
    head_angs = []
    head_ang_vs = []
    max_ang_v = []
    time_to_max_ang_v = []
    cumsum_ang_v = []
    max_cum_sum_ang_v = []
    time_to_max_cumsum_ang_v = []
    max_initial_turn = []
    angular_accels = []
    turn_starts = []
    turn_slopes = []
    sigmoid_xs = []
    sigmoid_ys = []
    sig_mse = []
    move_dirs = []
    distance_travelled = []
    last_10_frames_x = []
    last_10_frames_y = []
    fitted_max_cumsum = []
    aucs = []
    reaction_times = []
    for i in trial_nums.astype(int):
        trial_speed = speed[cot_triggers[i]: choice_triggers[i]]
        trial_acceleration = acceleration[cot_triggers[i]: choice_triggers[i]]
        trial_head_angs = head_angles[cot_triggers[i]: choice_triggers[i]]
        head_angs.append(trial_head_angs)
        max_initial_turn.append(np.cumsum(head_angular_velocity[cot_triggers[i]: choice_triggers[i]])[6])
        head_ang_vs.append(head_angular_velocity[cot_triggers[i]: choice_triggers[i]])
        angular_accels.append(head_ang_accel[cot_triggers[i]: choice_triggers[i]])
        max_ang_v.append(max_function(head_angular_velocity[cot_triggers[i]: choice_triggers[i]]))
        time_to_max_ang_v.append(arg_max_function(head_angular_velocity[cot_triggers[i]: choice_triggers[i]]))
        trial_cumsum_ang_v = np.cumsum(head_angular_velocity[cot_triggers[i]: choice_triggers[i]])

        move_dirs.append(move_dir[cot_triggers[i]: choice_triggers[i]])
        distance_travelled.append(np.sum(trial_speed))
        try:

            popt, pcov = dlc_processing_utils.fit_sigmoid(np.arange(0, trial_cumsum_ang_v.shape[0]),
                                                          (trial_cumsum_ang_v / max_function(trial_cumsum_ang_v)))

            s = (dlc_processing_utils.sigmoid(np.arange(0, trial_cumsum_ang_v.shape[0]), popt[0], popt[1], popt[2],
                                              popt[3])) * max_function(trial_cumsum_ang_v)

            start = np.where(np.abs(np.diff(s)) > 1)[0][0]
            turn_starts.append(start)
            turn_slopes.append(popt[2])
            sig_mse.append(mean_squared_error(trial_cumsum_ang_v, s, squared=False))
            sigmoid_xs.append(np.arange(0, trial_cumsum_ang_v.shape[0] - start))
            sigmoid_ys.append(s[start:] - s[start])
            cumsum_ang_v.append(trial_cumsum_ang_v[start:] - trial_cumsum_ang_v[start])
            aucs.append(np.trapz(y=s[start:] - s[start], x=np.arange(0, trial_cumsum_ang_v.shape[0] - start)))
            fitted_max_cumsum.append(s[-1])
        except(RuntimeError, IndexError, ValueError):
            turn_starts.append(np.nan)
            turn_slopes.append(np.nan)
            empty_arr = np.empty(trial_cumsum_ang_v.shape)
            empty_arr[:] = np.nan
            sigmoid_xs.append(empty_arr)
            sigmoid_ys.append(empty_arr)
            sig_mse.append(np.nan)
            aucs.append(np.nan)
            cumsum_ang_v.append(empty_arr)
            fitted_max_cumsum.append(np.nan)

        # max_cum_sum_ang_v.append(np.cumsum(head_angular_velocity[cot_triggers[i]: choice_triggers[i]])[-1])
        max_cum_sum_ang_v.append(trial_cumsum_ang_v[-1])
        time_to_max_cumsum_ang_v.append(
            arg_max_function(np.cumsum(head_angular_velocity[cot_triggers[i]: choice_triggers[i]])))
        max_speeds.append(np.max(trial_speed))
        avg_speeds.append(np.mean(trial_speed))
        time_to_max_speeds.append(np.where(trial_speed == np.max(trial_speed))[0][0])
        max_accels.append(np.max(trial_acceleration))
        avg_accels.append(np.mean(trial_acceleration))
        time_to_max_accels.append(np.where(trial_acceleration == np.max(trial_acceleration))[0][0])
        speeds.append(trial_speed)
        accelerations.append(trial_acceleration)
        photometry_ind = np.where(trial_nums == i)[0]
        photo_inds.append(photometry_ind)
        traces.append(np.squeeze(photometry_traces[photometry_ind, :].T))
        reaction_times.append(photometry_reaction_times[photometry_ind][0])

        x = (tracking_data['nose']['x'][cot_triggers[i]: choice_triggers[i]].values)
        y = (tracking_data['nose']['y'][cot_triggers[i]: choice_triggers[i]].values)
        head_x.append(x)
        head_y.append(y)
        last_10_frames_x.append(tracking_data['nose']['x'][cot_triggers[i] - 10: cot_triggers[i]].values)
        last_10_frames_y.append(tracking_data['nose']['y'][cot_triggers[i] - 10: cot_triggers[i]].values)
        if np.any(trial_speed > 7):
            times_to_move.append(np.where(trial_speed > 7)[0][0])
        else:
            times_to_move.append(np.nan)

    clean_peaks = []
    for c in peaks:
        if c.size != 0:
            clean_peaks.append(c)
        else:
            clean_peaks.append(np.nan)
    data = {'max speed': max_speeds, 'average speed': avg_speeds, 'time to max speed': time_to_max_speeds,
            'speeds': speeds
        , 'max acceleration': max_accels, 'average acceleration': avg_accels,
            'time to max acceleration': time_to_max_accels, 'accelerations': accelerations, 'APE peaks': clean_peaks,
            'time to move': times_to_move, 'trial numbers': trial_nums, 'traces': traces, 'head x': head_x,
            'head y': head_y, 'head angles': head_angs,
            'angular velocity': head_ang_vs, 'max angular velocity': max_ang_v,
            'time to max angular vel': time_to_max_ang_v, 'cumsum ang vel': cumsum_ang_v,
            'max cumsum ang vel': max_cum_sum_ang_v, 'time to max cum sum ang vel': time_to_max_cumsum_ang_v,
            'max initial turn': max_initial_turn, 'angular acceleration': angular_accels,
            'turn slopes': turn_slopes, 'turn starts': turn_starts, 'sig x': sigmoid_xs, 'sig y': sigmoid_ys,
            'sig mse': sig_mse, 'movement dir': move_dirs, 'distance travelled': distance_travelled,
            'last 10 x': last_10_frames_x, 'last 10 y': last_10_frames_y,
            'fitted max cumsum ang vel': fitted_max_cumsum, 'area under sigmoid': aucs,
            'trial type': trial_types_subset, 'reaction times': reaction_times}
    data_df = pd.DataFrame(data)
    data_df['side'] = side
    clean_df = data_df.drop(data_df.loc[data_df['max speed'] > 35].index) # tracking error - this is too fast for a mouse
    non_nan_df = clean_df.drop(clean_df.loc[np.isnan(clean_df['APE peaks'])].index)
    non_nan_df = non_nan_df.drop(non_nan_df.loc[np.isnan(non_nan_df['time to move'])].index)
    return data_df #used to be non_nan_df


def format_only_photometry(photometry_data, trial_types, trial_numbers=None, align_to='choice'):

    if align_to == 'choice':
        data = photometry_data.choice_data
        trace_midpoint = int(photometry_data.choice_data.contra_data.sorted_traces.shape[1] / 2)
    elif align_to == 'cue':
        data = photometry_data.cue_data
        trace_midpoint = int(photometry_data.cue_data.contra_data.sorted_traces.shape[1] / 2)
    elif align_to == 'reward':
        data = photometry_data.outcome_data
        trace_midpoint = int(photometry_data.outcome_data.reward_data.sorted_traces.shape[1] / 2)
    else:
        print('invalid alignment argument')
    if align_to != 'reward':
        if type(trial_numbers) == np.ndarray:
            contra_test_trial_nums, p_inds, inds = np.intersect1d(data.contra_data.trial_nums, trial_numbers, return_indices=True)
            trial_nums = trial_numbers[inds]
            contra_peaks = np.array(data.contra_data.trial_peaks)[p_inds]
            contra_traces = data.contra_data.sorted_traces[p_inds]
            contra_reaction_times = data.contra_data.reaction_times[p_inds]
            ipsi_test_trial_nums, p_inds, inds = np.intersect1d(data.ipsi_data.trial_nums, trial_numbers, return_indices=True)
            ipsi_trial_nums = trial_numbers[inds]
            ipsi_peaks = np.array(data.ipsi_data.trial_peaks)[p_inds]
            ipsi_traces = data.ipsi_data.sorted_traces[p_inds]
            ipsi_reaction_times = data.ipsi_data.reaction_times[p_inds]
        else:
            trial_nums = data.contra_data.trial_nums
            contra_peaks = data.contra_data.trial_peaks
            contra_traces = data.contra_data.sorted_traces
            contra_reaction_times = data.contra_data.reaction_times
            ipsi_trial_nums = data.ipsi_data.trial_nums
            ipsi_peaks = data.ipsi_data.trial_peaks
            ipsi_traces = data.ipsi_data.sorted_traces
            ipsi_reaction_times = data.ipsi_data.reaction_times
        contra_traces = decimate(contra_traces, 10)
        ipsi_traces = decimate(ipsi_traces, 10)
        trial_types_subset_contra = trial_types.iloc[trial_nums]['Trial type'].values
        trial_types_subset_ipsi = trial_types.iloc[ipsi_trial_nums]['Trial type'].values
    else:
        if type(trial_numbers) == np.ndarray:
            contra_test_trial_nums, p_inds, inds = np.intersect1d(data.reward_data.trial_nums, trial_numbers, return_indices=True)
            trial_nums = trial_numbers[inds]
            contra_peaks = np.array(data.reward_data.trial_peaks)[p_inds]
            contra_traces = data.reward_data.sorted_traces[p_inds]
            ipsi_test_trial_nums, p_inds, inds = np.intersect1d(data.no_reward_data.trial_nums, trial_numbers, return_indices=True)
            ipsi_trial_nums = trial_numbers[inds]
            ipsi_peaks = np.array(data.no_reward_data.trial_peaks)[p_inds]
            ipsi_traces = data.no_reward_data.sorted_traces[p_inds]
        else:
            trial_nums = data.reward_data.trial_nums
            contra_peaks = data.reward_data.trial_peaks
            contra_traces = data.reward_data.sorted_traces
            ipsi_trial_nums = data.no_reward_data.trial_nums
            ipsi_peaks = data.no_reward_data.trial_peaks
            ipsi_traces = data.no_reward_data.sorted_traces
        contra_traces = decimate(contra_traces, 10)
        ipsi_traces = decimate(ipsi_traces, 10)
        trial_types_subset_contra = trial_types.iloc[trial_nums]['Trial type'].values
        trial_types_subset_ipsi = trial_types.iloc[ipsi_trial_nums]['Trial type'].values
    clean_peaks = []
    clean_ipsi_peaks =[]
    for i, c in enumerate(contra_peaks):
        if c.size != 0:
            clean_peaks.append(c)
        else:
            clean_peaks.append(np.nan)
    for i, c in enumerate(ipsi_peaks):
        if c.size != 0:
            clean_ipsi_peaks.append(c)
        else:
            clean_ipsi_peaks.append(np.nan)
    contra_data = {'APE peaks': clean_peaks, 'trial type': trial_types_subset_contra, 'trial numbers': trial_nums}
    contra_data_df = pd.DataFrame(contra_data)
    list_contra_traces = [row.tolist() for row in contra_traces]
    contra_data_df['traces'] = list_contra_traces
    contra_data_df['side'] = 'contra'
    ipsi_data = {'APE peaks': clean_ipsi_peaks, 'trial type': trial_types_subset_ipsi, 'trial numbers': ipsi_trial_nums}
    ipsi_data_df = pd.DataFrame(ipsi_data)
    list_ipsi_traces = [row.tolist() for row in ipsi_traces]
    ipsi_data_df['traces'] = list_ipsi_traces
    ipsi_data_df['side'] = 'ipsi'
    both_df = pd.concat([contra_data_df, ipsi_data_df]).reset_index(drop=True)
    non_nan_df = both_df.drop(both_df.loc[np.isnan(both_df['APE peaks'])].index)
    ape_sorted_data = non_nan_df.sort_values(by='APE peaks', ascending=False).reset_index(drop=True)
    return ape_sorted_data
