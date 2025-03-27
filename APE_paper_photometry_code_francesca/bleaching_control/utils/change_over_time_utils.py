import pickle
import numpy as np
import peakutils
import os
from utils.reaction_time_utils import get_bpod_trial_nums_per_session
from scipy.signal import decimate
from set_global_params import processed_data_path


def get_valid_traces(mouse, dates, window_around_mean=0.2, recording_site='tail', side='contra', window_size=40, align_to='movement'):
    """
    Takes traces from trials where the reaction time is +- window_around_mean seconds
    from the mean reaction time of all trials.
    Then finds peaks of traces after behavioural event and bins these in groups of window_size.
    Args:
        mouse (str): mouse name
        dates (list): dates in YYYYMMDD format
        window_around_mean (float): window in seconds around mean reaction time to consider
        recording_site (str): Nacc or tail
        side (str): ipsi or contra choices
        window_size (int): number of trials in a bin

    Returns:
        rolling_mean_x (list): trial number means for each bin
        rolling_mean_peak (list): peak of mean trace for each bin
        all_peak_trace_inds (list): index of mean trace where peak is found for each bin
        rolling_mean_trace (list): mean trace for each bin
    """
    session_starts = get_bpod_trial_nums_per_session(mouse, dates)
    saving_folder = os.path.join(processed_data_path, mouse)
    all_bins = []
    all_reaction_times = []
    all_trial_numbers = []
    all_actual_trial_numbers = []
    all_traces = []

    for date_num, date in enumerate(dates):
        aligned_filename = os.path.join(saving_folder, mouse + '_' + date + '_' + 'aligned_traces.p')

        with open(aligned_filename, 'rb') as f:
            data = pickle.load(f)
        if align_to == 'movement':
            if side == 'contra':
                recording_site_data = data.choice_data.contra_data
            elif side == 'ipsi':
                recording_site_data = data.choice_data.ipsi_data
            else:
                print('invalid side')
            trial_nums = recording_site_data.trial_nums
            reaction_times = recording_site_data.reaction_times
            sorted_traces = recording_site_data.sorted_traces
        elif align_to == 'cue':
            if side == 'contra':
                recording_site_data = data.cue_data.contra_data
            elif side == 'ipsi':
                recording_site_data = data.cue_data.ipsi_data
            else:
                print('invalid side')
            trial_nums = recording_site_data.trial_nums
            reaction_times = recording_site_data.reaction_times
            sorted_traces = recording_site_data.sorted_traces

        elif align_to == 'reward':
            contra_recording_site_data = data.reward_data.contra_data
            ipsi_recording_site_data = data.reward_data.contra_data
            unsorted_trial_nums = np.concatenate([contra_recording_site_data.trial_nums, ipsi_recording_site_data.trial_nums])
            indices = np.argsort(unsorted_trial_nums)
            reaction_times = np.concatenate([contra_recording_site_data.sorted_next_poke, ipsi_recording_site_data.sorted_next_poke])[indices]
            sorted_traces = np.concatenate([contra_recording_site_data.sorted_traces, ipsi_recording_site_data.sorted_traces])[indices, :]
            trial_nums = unsorted_trial_nums[indices]

        print(date, len(reaction_times))
        all_reaction_times.append(reaction_times)
        actual_trial_numbers = trial_nums + session_starts[date_num]
        all_traces.append(decimate(sorted_traces, 10))
        all_actual_trial_numbers.append(actual_trial_numbers)
        all_trial_numbers.append(len(reaction_times))
        all_bins.append(np.arange(start=min(reaction_times),
                                  stop=max(reaction_times) + 0.1, step=0.1))

    flattened_actual_trial_nums = [item for sublist in all_actual_trial_numbers for item in sublist]
    flattened_reaction_times = [item for sublist in all_reaction_times for item in sublist]
    trials_per_session = [i.shape[0] for i in all_traces]
    num_trials = np.sum(trials_per_session)
    trials_per_session.insert(0, 0)
    flattened_traces = np.zeros([num_trials, all_traces[0].shape[1]])
    cum_sum = np.cumsum(trials_per_session)
    for i, trace in enumerate(all_traces):
        flattened_traces[cum_sum[i]:cum_sum[i + 1], :] = trace
    median_reaction_time = np.median(flattened_reaction_times)
    valid_trials = np.where(
        np.logical_and(np.greater_equal(flattened_reaction_times, median_reaction_time - window_around_mean),
                       np.less_equal(flattened_reaction_times, median_reaction_time + window_around_mean)))
    valid_flattened_traces = flattened_traces[valid_trials[0], :]
    valid_trial_nums = np.array(flattened_actual_trial_nums)[valid_trials]
    rolling_mean_trace = []
    rolling_mean_x = []
    rolling_mean_peak = []
    all_peak_trace_inds = []
    num_bins = int(np.shape(valid_flattened_traces)[0]/window_size)
    print(valid_flattened_traces.shape)
    for window_num in range(num_bins):
        rolling_mean_trace.append(np.nanmean(valid_flattened_traces[window_num * window_size: (window_num + 1) * window_size, :], axis=0))
        rolling_mean_x.append(np.nanmean(valid_trial_nums[window_num * window_size: (window_num + 1) * window_size]))
        half_way = int(valid_flattened_traces.shape[1]/2)
        trace_from_event = rolling_mean_trace[window_num][half_way:half_way + int(1000*(median_reaction_time))]
        trial_peak_inds = peakutils.indexes(trace_from_event.flatten('F'))
        if trial_peak_inds.shape[0] > 0 or len(trial_peak_inds > 1):
            trial_peak_inds = trial_peak_inds[0]
            trial_peaks = trace_from_event.flatten('F')[trial_peak_inds]
        else:
            trial_peak_inds = np.argmax(trace_from_event)
            trial_peaks = np.max(trace_from_event)
            print(window_num)
        rolling_mean_peak.append(trial_peaks)
        peak_trace_inds = trial_peak_inds + half_way
        all_peak_trace_inds.append(peak_trace_inds)
    return rolling_mean_x, rolling_mean_peak, all_peak_trace_inds, rolling_mean_trace


def get_valid_traces_movement_aligned(mouse, dates, window_around_mean=0.2, side='contra', window_size=40):
    """
    Takes traces from trials where the reaction time is +- window_around_mean seconds
    from the mean reaction time of all trials.
    Then finds peaks of traces after behavioural event and bins these in groups of window_size.
    This is only for movement aligned dopamine (not used for original preprint)
    Args:
        mouse (str): mouse name
        dates (list): dates in YYYYMMDD format
        window_around_mean (float): window in seconds around mean reaction time to consider
        side (str): ipsi or contra choices
        window_size (int): number of trials in a bin

    Returns:
        rolling_mean_x (list): trial number means for each bin
        rolling_mean_peak (list): peak of mean trace for each bin
        all_peak_trace_inds (list): index of mean trace where peak is found for each bin
        rolling_mean_trace (list): mean trace for each bin
    """
    session_starts = get_bpod_trial_nums_per_session(mouse, dates)
    saving_folder = processed_data_path + mouse + '\\'
    data_root = processed_data_path + 'peak_analysis'
    all_bins = []
    all_reaction_times =[]
    all_trial_numbers = []
    all_actual_trial_numbers = []
    all_traces = []

    for date_num, date in enumerate(dates):
        peaks_saving_folder = os.path.join(data_root, mouse)
        aligned_filename = saving_folder + mouse + '_' + date + '_' + 'aligned_traces.p'

        with open(aligned_filename, 'rb') as f:
            data = pickle.load(f)
        if side == 'contra':
            recording_site_data =data.choice_data.contra_data
        elif side=='ipsi':
            recording_site_data = data.choice_data.ipsi_data
        else:
            print('invalid side')
        all_reaction_times.append(recording_site_data.reaction_times)


        print(date, len(recording_site_data.reaction_times))
        actual_trial_numbers = recording_site_data.trial_nums + session_starts[date_num]
        all_traces.append(decimate(recording_site_data.sorted_traces, 10))
        all_actual_trial_numbers.append(actual_trial_numbers)
        all_trial_numbers.append(len(recording_site_data.reaction_times))
        all_bins.append(np.arange(start=min(recording_site_data.reaction_times),
                                  stop=max(recording_site_data.reaction_times) + 0.1, step=0.1))

    flattened_actual_trial_nums = [item for sublist in all_actual_trial_numbers for item in sublist]
    flattened_reaction_times = [item for sublist in all_reaction_times for item in sublist]
    trials_per_session = [i.shape[0] for i in all_traces]
    num_trials = np.sum(trials_per_session)
    trials_per_session.insert(0, 0)
    flattened_traces = np.zeros([num_trials, all_traces[0].shape[1]])
    cum_sum = np.cumsum(trials_per_session)
    for i, trace in enumerate(all_traces):
        flattened_traces[cum_sum[i]:cum_sum[i + 1], :] = trace
    median_reaction_time = np.median(flattened_reaction_times)
    valid_trials = np.where(
        np.logical_and(np.greater_equal(flattened_reaction_times, median_reaction_time - window_around_mean),
                       np.less_equal(flattened_reaction_times, median_reaction_time + window_around_mean)))
    valid_flattened_traces = flattened_traces[valid_trials[0], :]
    valid_trial_nums = np.array(flattened_actual_trial_nums)[valid_trials]
    rolling_mean_trace = []
    rolling_mean_x = []
    rolling_mean_peak = []
    all_peak_trace_inds = []
    num_bins = int(np.shape(valid_flattened_traces)[0]/window_size)
    for window_num in range(num_bins):
        rolling_mean_trace.append(np.nanmean(valid_flattened_traces[window_num * window_size: (window_num + 1) * window_size, :], axis=0))
        rolling_mean_x.append(np.nanmean(valid_trial_nums[window_num * window_size: (window_num + 1) * window_size]))
        half_way = int(valid_flattened_traces.shape[1]/2)
        trace_from_event = rolling_mean_trace[window_num][half_way:half_way + int(1000*(median_reaction_time))]
        trial_peak_inds = peakutils.indexes(trace_from_event.flatten('F'))
        if trial_peak_inds.shape[0] > 0 or len(trial_peak_inds > 1):
            trial_peak_inds = trial_peak_inds[0]
            trial_peaks = trace_from_event.flatten('F')[trial_peak_inds]
        else:
            trial_peak_inds = np.argmax(trace_from_event)
            trial_peaks = np.max(trace_from_event)
            print(window_num)
        rolling_mean_peak.append(trial_peaks)
        peak_trace_inds = trial_peak_inds + half_way
        all_peak_trace_inds.append(peak_trace_inds)
    return rolling_mean_x, rolling_mean_peak, all_peak_trace_inds, rolling_mean_trace


