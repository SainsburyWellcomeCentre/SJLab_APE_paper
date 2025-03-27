from utils.tracking_analysis.fede_geometry import calc_angle_between_vectors_of_points_2d, calc_ang_velocity
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib
from utils.individual_trial_analysis_utils import get_photometry_around_event
from utils.tracking_analysis.velocity_utils import calculate_velocity
from utils.tracking_analysis.camera_trigger_preprocessing_utils import *
from numpy.linalg import norm
import pandas as pd


def get_return_to_centre_window(trials, all_trials, all_camera_triggers):
    choice = (trials[(trials['State name'] == 'WaitForResponse') & (
            trials['Instance in state'] == trials['Max times in state'])]['Time end'].values * 10000)
    choice = choice.astype(int)
    choice_triggers = find_nearest_trials(choice, all_camera_triggers)
    trial_nums = trials['Trial num'].values
    next_trial_nums = trial_nums + 1
    max_trial_num = np.max(all_trials['Trial num'].values)
    valid_indices = np.where(next_trial_nums <= max_trial_num)
    next_trial_nums = next_trial_nums[valid_indices]
    choice_triggers = choice_triggers[valid_indices]
    next_trials = all_trials[all_trials['Trial num'].isin(next_trial_nums)]
    next_trial_starts = (next_trials[(next_trials['State name'] == 'WaitForPoke') & (
            next_trials['Instance in state'] == 1)]['Time end'].values * 10000)
    next_trial_starts = next_trial_starts.astype(int)
    next_trial_start_triggers = find_nearest_trials(next_trial_starts, all_camera_triggers)
    return choice_triggers, next_trial_start_triggers, choice, next_trial_starts


def get_initial_heading_vector(tracking_data, onset_times, offset_times, time_frame=15):
    colourmap = matplotlib.cm.viridis
    colours = colourmap(np.linspace(0, 0.8, onset_times.shape[0]))
    heading_vectors = []
    body_parts = ['nose', 'left ear', 'right ear']
    fig, axs = plt.subplots(1, 2)
    for trial_num, start_time in enumerate(onset_times):
        initial_tracking = {}
        for body_part in body_parts:
            trial_tracking_x = tracking_data[body_part]['x'][start_time: offset_times[trial_num]]
            trial_tracking_y = tracking_data[body_part]['y'][start_time: offset_times[trial_num]]
            initial_tracking_body_part = pd.DataFrame({'x': trial_tracking_x[:time_frame].values, 'y': trial_tracking_y[:time_frame].values})
            initial_tracking[body_part] = initial_tracking_body_part
        vectors_in_time = np.vstack([np.diff(initial_tracking['nose']['x']), np.diff(initial_tracking['nose']['y'])])
        initial_vector = np.sum(vectors_in_time, axis=1)
        heading_vectors.append(initial_vector)

        #axs[0].plot(initial_tracking['nose']['x'], initial_tracking['nose']['y'], color=colours[trial_num])
        #axs[1].plot([0, initial_vector[0]], [0, initial_vector[1]], color=colours[trial_num])

    return heading_vectors, axs


def get_turn_start_times(tracking_data, onset_times, offset_times, turn_ang_thresh,  time_frame=40, side_port='left', short_turns_only=True):
    colourmap = matplotlib.cm.viridis
    colours = colourmap(np.linspace(0, 0.8, onset_times.shape[0]))
    heading_vectors = []
    body_parts = ['nose', 'left ear', 'right ear']
    #fig, axs = plt.subplots(1, 4)
    #fig1, axs1 = plt.subplots(1, 4)
    cumsum_ang_vs = []
    turn_onsets = []
    for trial_num, start_time in enumerate(onset_times):
        initial_tracking = {}
        for body_part in body_parts:
            trial_tracking_x = tracking_data[body_part]['x'][start_time: offset_times[trial_num]]
            trial_tracking_y = tracking_data[body_part]['y'][start_time: offset_times[trial_num]]
            initial_tracking_body_part = pd.DataFrame(
                {'x': trial_tracking_x[:time_frame].values, 'y': trial_tracking_y[:time_frame].values})
            initial_tracking[body_part] = initial_tracking_body_part
        midpoints_y = (initial_tracking['left ear']['y'] + initial_tracking['right ear']['y']) / 2
        midpoints_x = (initial_tracking['left ear']['x'] + initial_tracking['right ear']['x']) / 2
        head_angles = calc_angle_between_vectors_of_points_2d(initial_tracking['nose']['x'].values,
                                                              initial_tracking['nose']['y'].values, midpoints_x,
                                                              midpoints_y)
        head_angular_velocity = calc_ang_velocity(head_angles)
        cumsum_ang_v = np.cumsum(head_angular_velocity)

        #axs1[0].plot(cumsum_ang_v, color=colours[trial_num])
        #axs1[1].plot(head_angular_velocity, color=colours[trial_num])
        #axs1[2].plot(head_angles)
        #axs1[0].axhline(turn_ang_thresh)


        if side_port == 'left':
            #turn_ang_thresh = 85
            if np.any(cumsum_ang_v <= turn_ang_thresh):
                turn_onset_ind = np.where(cumsum_ang_v <= turn_ang_thresh)[0][0]
                makes_full_turn = np.any(cumsum_ang_v[turn_onset_ind: turn_onset_ind + 50] <= turn_ang_thresh - 90)
                crosses_threshold = np.all(cumsum_ang_v[turn_onset_ind:] <= turn_ang_thresh)
                if short_turns_only:
                    short_duration = cumsum_ang_v[turn_onset_ind:].shape[0] <= 30
                else:
                    short_duration = True
                if makes_full_turn & crosses_threshold & short_duration:
                    turn_onsets.append(turn_onset_ind * 1 / 30)
                    #axs[0].plot(cumsum_ang_v[turn_onset_ind:], color=colours[trial_num])
                    #axs[1].plot(head_angular_velocity[turn_onset_ind:], color=colours[trial_num])
                    #axs[2].plot(head_angles[turn_onset_ind:])
                else:
                    turn_onsets.append(False)
            else:
                turn_onsets.append(False)
        else:
            #turn_ang_thresh = 280
            if np.any(cumsum_ang_v >= turn_ang_thresh):
                turn_onset_ind = np.where(cumsum_ang_v >= turn_ang_thresh)[0][0]
                makes_full_turn = np.any(cumsum_ang_v[turn_onset_ind: turn_onset_ind+50]>=turn_ang_thresh + 90)
                crosses_threshold = np.all(cumsum_ang_v[turn_onset_ind:]>=turn_ang_thresh)
                if short_turns_only:
                    short_duration = cumsum_ang_v[turn_onset_ind:].shape[0] <= 30
                else:
                    short_duration = True
                if makes_full_turn & crosses_threshold & short_duration:
                    turn_onsets.append(turn_onset_ind * 1 / 30)
                    #axs[0].plot(cumsum_ang_v[turn_onset_ind:], color=colours[trial_num])
                    #axs[1].plot(head_angular_velocity[turn_onset_ind:], color=colours[trial_num])
                    #axs[2].plot(head_angles[turn_onset_ind:])
                else:
                    turn_onsets.append(False)
            else:
                turn_onsets.append(False)
    # turn_onsets are in seconds
    return cumsum_ang_vs, turn_onsets


def get_movement_start_times(tracking_data, onset_times, offset_times, body_part='nose'):
    fig, axs = plt.subplots(1, 3)
    colourmap = matplotlib.cm.viridis
    colours = colourmap(np.linspace(0, 0.8, len(onset_times)))
    velocities = []
    accelerations = []
    movement_onsets = []
    for trial_num, start_time in enumerate(onset_times):
        trial_tracking_x = tracking_data[body_part]['x'][start_time: offset_times[trial_num]]
        trial_tracking_y = tracking_data[body_part]['y'][start_time: offset_times[trial_num]]
        velocity = calculate_velocity(zip(trial_tracking_x, trial_tracking_y))
        acceleration = np.diff(velocity)
        velocities.append(velocity)
        accelerations.append(accelerations)
        axs[1].plot(acceleration, color=colours[trial_num], alpha=0.5)
        axs[0].plot(velocity, color=colours[trial_num], alpha=0.5)
        axs[2].plot(np.cumsum(velocity), color=colours[trial_num], alpha=0.5)
        axs[2].axhline(40)
        #movement_onsets.append(np.where(np.cumsum(velocity) >= 40)[0][0] * 1/30)
        if np.any(np.cumsum(acceleration) >= 15):
            movement_onsets.append(np.where(np.cumsum(acceleration) >= 15)[0][0] * 1 / 30)
        else:
            movement_onsets.append(False)
    return movement_onsets


def get_photometery_return_to_centre_certain_trials(trial_indices, all_trials, photometry_data):
    event_times = all_trials['Time end'].values[trial_indices]
    traces = get_photometry_around_event(event_times, photometry_data)
    norm_traces = stats.zscore(traces.T, axis=0)
    plt.figure()
    plt.plot(np.mean(norm_traces, axis=1))
    return norm_traces


def get_photometery_return_to_centre_certain_trials_aligned_to_movement(trial_indices, all_trials, movement_onset_delays, photometry_data):
    valid_trials = np.where(np.array(movement_onset_delays) != False)[0]
    trial_indices = trial_indices[valid_trials]
    movement_onset_delays = np.array(movement_onset_delays)[valid_trials]
    event_times = all_trials['Time end'].values[trial_indices] + movement_onset_delays
    traces = get_photometry_around_event(event_times, photometry_data)
    norm_traces = traces.T #stats.zscore(traces.T, axis=0)
    plt.figure()
    plt.plot(np.mean(norm_traces, axis=1))
    return norm_traces


def calculate_cosine_similarity_to_goal_vector(port_coords, heading_vectors, axs, side='side1'):
    similarities = []
    goal_vector = port_coords['centre'] - port_coords[side]
    for heading_vector in heading_vectors:
        similarities.append(calculate_cosine_similarity(heading_vector, goal_vector))
    plt.plot([0, goal_vector[0]], [0, goal_vector[1]], color='k', lw=4)
    axs[0].scatter([port_coords[side][0], port_coords['centre'][0]], [port_coords[side][1], port_coords['centre'][1]], color='k')
    return similarities, goal_vector


def calculate_cosine_similarity(a, b):
    cos_sim = np.dot(a, b)/(norm(a)*norm(b))
    return cos_sim


def plot_direct_trials(similarities, heading_vectors, goal_vector, threshold=0.6):
    fig, axs = plt.subplots(1,1)
    axs.hist(similarities, bins=20)
    axs.axvline(threshold)
    trial_indices = np.where(np.array(similarities) >= threshold)[0]
    direct_vectors = np.array(heading_vectors)[trial_indices]
    colourmap = matplotlib.cm.viridis
    colours = colourmap(np.linspace(0, 0.8, len(direct_vectors)))
    fig, axs = plt.subplots(1, 2)
    for trial_num, vector in enumerate(direct_vectors):
        axs[0].plot([0, vector[0]], [0, vector[1]], color=colours[trial_num])
    axs[0].plot([0, goal_vector[0]], [0, goal_vector[1]], color='k', lw=4)

    trial_indices = np.where(np.array(similarities) < threshold)[0]
    direct_vectors = np.array(heading_vectors)[trial_indices]
    colourmap = matplotlib.cm.inferno
    colours = colourmap(np.linspace(0, 0.8, len(direct_vectors)))
    for trial_num, vector in enumerate(direct_vectors):
        axs[1].plot([0, vector[0]], [0, vector[1]], color=colours[trial_num])
    axs[1].plot([0, goal_vector[0]], [0, goal_vector[1]], color='k', lw=4)


def get_photometry_for_one_side_returns(fiber_side_numeric, camera_triggers, trial_data, tracking_data, turn_ang_thresh, port_coords, photometry_data, side='side1', side_port='left', time_frame=300,  short_turns_only=True):
    trials = trial_data[(trial_data['State name'] == 'WaitForResponse') & (
            trial_data['Instance in state'] == trial_data['Max times in state']) & (
                                         trial_data['Response'] == fiber_side_numeric)]

    choice_triggers, next_trial_start_triggers, choice_times, next_trial_start_times = get_return_to_centre_window(trials, trial_data, camera_triggers)

    heading_vectors, axs = get_initial_heading_vector(tracking_data, choice_triggers,
                                                      next_trial_start_triggers, time_frame=time_frame)

    cosine_similarities, goal_vector = calculate_cosine_similarity_to_goal_vector(port_coords, heading_vectors, axs,
                                                                                  side=side)
    plot_direct_trials(cosine_similarities, heading_vectors, goal_vector)
    trial_indices = np.where(np.array(cosine_similarities) >= 0.9)[0]
    other_trial_inds = np.where(np.array(cosine_similarities) < 0.9)[0]

    # find movement onsets for direct trials
    cum_sum_ang_vs, turn_onsets = get_turn_start_times(tracking_data, choice_triggers[trial_indices],
                                                                 next_trial_start_triggers[trial_indices],
                                                                 turn_ang_thresh, side_port=side_port,
                                                                 time_frame=time_frame, short_turns_only=short_turns_only)
    contra_movement_inds = [i for i, t in enumerate(turn_onsets) if t]
    valid_choices = choice_times[trial_indices]
    return_durations_good_cos_similarity = next_trial_start_times[trial_indices] - choice_times[trial_indices]
    return_durations_bad_cos_similarity = next_trial_start_times[other_trial_inds] - choice_times[other_trial_inds]
    contra_movement_onsets_time_stamps = np.array(valid_choices)[contra_movement_inds] / 10000 + \
                                         np.array(turn_onsets)[
                                             contra_movement_inds]  # these times are in seconds

    contra_movement_traces = get_photometery_return_to_centre_certain_trials_aligned_to_movement(trial_indices,
                                                                                                 trials,
                                                                                                 turn_onsets,
                                                                                                 photometry_data)
    return contra_movement_traces, contra_movement_onsets_time_stamps, return_durations_good_cos_similarity, return_durations_bad_cos_similarity
