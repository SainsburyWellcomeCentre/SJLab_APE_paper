import peakutils
import pickle
from tqdm import tqdm
import numpy as np
import os
from set_global_params import processed_data_path
import matplotlib.pyplot as plt


def get_peak_times(traces, time_stamps):
    zero_ind = (np.abs(time_stamps)).argmin()
    peak_times = []
    for trial_num in range(0, traces.shape[0]):
        trial_trace = traces[trial_num, :]
        trial_peak_inds = peakutils.indexes(trial_trace[zero_ind:].flatten('F'))
        if trial_peak_inds.shape[0] > 0 or len(trial_peak_inds > 1):
            trial_peak_ind = trial_peak_inds[0] + zero_ind
            trial_peak_time = time_stamps[trial_peak_ind]
            peak_times.append(trial_peak_time)
    return peak_times


def get_standard_dev_of_baseline_slopes():
    dir = os.path.join(processed_data_path, 'for_figure')
    file_name = 'thresholds_for_rise_time.npz'
    trace_slope_sds = np.load(os.path.join(dir, file_name))
    all_trace_slope_sds = np.concatenate([trace_slope_sds['VS'], trace_slope_sds['TS']])
    threshold = np.mean(all_trace_slope_sds)
    return threshold


def get_DA_peak_times_and_slopes_from_cue(experiments_to_process, time_range=(-1.5, 1.5)):
    """
    This takes the peak dopamine response after the cue (before choice is made) for each trial. It finds the time to
    the peak and the slope of the rise to it.
    Args:
        experiments_to_process (pd.dataframe): experimental records for all the mice you want to average the traces for
        time_range (tuple): time window seconds before and after event to get traces for

    Returns:

    """
    thresholds = []
    mouse_peak_times = []
    mouse_slopes = []
    for mouse in tqdm(experiments_to_process['mouse_id'].unique(), desc='Mouse: '):
        df = experiments_to_process[experiments_to_process.mouse_id == mouse]
        data_dir = os.path.join(processed_data_path, 'for_figure' ,mouse)
        date_peak_times = []
        date_slopes = []
        for date in df['date']:
            filename = mouse + '_' + date + '_' + 'aligned_traces_for_fig.p'
            peak_times = []
            trial_slopes = []
            with open(os.path.join(data_dir, filename), 'rb') as f:
                session_data = pickle.load(f)
            contra_cues = session_data.cue_data.contra_data.sorted_traces
            time_stamps = session_data.choice_data.contra_data.time_points
            contra_choice_times = session_data.cue_data.contra_data.outcome_times
            peak_times, trial_slopes, thresholds = calc_peak_rise_time_one_side(contra_cues, time_stamps, contra_choice_times, peak_times, trial_slopes, thresholds)

            ipsi_cues = session_data.cue_data.ipsi_data.sorted_traces
            time_stamps = session_data.choice_data.contra_data.time_points
            ipsi_choice_times = session_data.cue_data.ipsi_data.outcome_times
            peak_times, trial_slopes, thresholds = calc_peak_rise_time_one_side(ipsi_cues, time_stamps, ipsi_choice_times, peak_times, trial_slopes,
                                         thresholds)

            date_peak_times.append(np.median(peak_times))
            date_slopes.append(np.nanmedian(trial_slopes))
        mouse_peak_times.append(np.median(date_peak_times))
        mouse_slopes.append(np.median(date_slopes))
    return mouse_peak_times, mouse_slopes, thresholds


def calc_peak_rise_time_one_side(cues, time_stamps, choice_times, peak_times, trial_slopes, thresholds):
    window_start_ind = (np.abs(time_stamps)).argmin()
    window_end_inds = find_nearest_time_stamp_to_event(choice_times, time_stamps)
    for trace_num in range(0, cues.shape[0]):
        time_window = [window_start_ind, window_end_inds[trace_num]]
        trace = cues[trace_num, :][time_window[0]: time_window[1]]
        trial_peak_inds = peakutils.indexes(trace.flatten('F'))
        if trial_peak_inds.shape[0] > 0 or len(trial_peak_inds > 1):
            trial_peak_ind = trial_peak_inds[0] + time_window[0]
            trial_peak_time = time_stamps[trial_peak_ind]
            trial_slope, threshold = find_slope_thresh_crossing_time(cues[trace_num, :], time_stamps,
                                                                     time_window, trial_peak_ind)
            peak_times.append(trial_peak_time)
            trial_slopes.append(trial_slope)
            thresholds.append(threshold)
    return peak_times, trial_slopes, thresholds


def find_nearest_time_stamp_to_event(event_times, time_stamps):
    event_inds = []
    for event_time in event_times:
        event_inds.append(np.abs(time_stamps - event_time).argmin())
    return event_inds


def find_slope_to_peak(trace, time_stamps, time_window, trial_peak_inds, deviation_threshold=0.5, plot=False):
    time = time_stamps[time_window[0]: time_window[1]][:trial_peak_inds[0]]
    data = trace[time_window[0]: time_window[1]][:trial_peak_inds[0]]



    # Find the index where the data deviates from zero
    deviation_index = np.argmax(data - data[0] > deviation_threshold)

    slope, intercept = np.polyfit(time, data, 1)
    if plot:
        # Calculate the fitted values using the slope and intercept
        fitted_values = slope * time_stamps + intercept
        # Plot the original data and the fitted slope
        plt.figure(figsize=(8, 6))
        plt.scatter(time_stamps, trace, label='Original Data')
        plt.plot(time_stamps, fitted_values, color='red', label='Fitted Slope')
        plt.axvline(x=time[deviation_index], color='green', linestyle='--', label='Deviation Point')
        plt.xlabel('Time')
        plt.ylabel('Data')
        plt.title('Time Series Data with Fitted Slope')
        plt.legend()
        plt.grid(True)
        plt.show()
    return time[deviation_index]


def find_slope_thresh_crossing_time(trace, time_stamps, time_window, trial_peak_inds, deviation_threshold=1, plot=False):
    # Calculate the derivative of the time series
    derivative = np.gradient(trace, time_stamps)

    # Define the baseline period for calculating the standard deviation
    baseline_start = 0
    baseline_end = time_window[0]

    # Calculate the standard deviation of the baseline period
    baseline_std = np.std(derivative[baseline_start:baseline_end])
    baseline_mean = np.mean(derivative[baseline_start:baseline_end])
    # Define the threshold as 2 times the baseline standard deviation
    threshold = deviation_threshold * baseline_std

    # Find indices where the derivative crosses the threshold
    crossings = np.where(derivative > baseline_mean + threshold)[0]
    valid_crossings = crossings[(crossings >= time_window[0]) & (crossings < trial_peak_inds)]

    if valid_crossings.shape[0] > 0:
        if plot:
            # Plot the derivative and the threshold crossings
            plt.figure(figsize=(8, 6))
            plt.plot(time_stamps, derivative, label='Derivative')
            plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
            plt.scatter(time_stamps[valid_crossings], derivative[valid_crossings], color='green',
                        label='Threshold Crossings')
            plt.xlabel('Time')
            plt.ylabel('Derivative')
            plt.title('Derivative and Threshold Crossings')
            plt.legend()
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(8, 6))
            plt.plot(time_stamps, trace, label='Derivative')
            plt.scatter(time_stamps[valid_crossings], trace[valid_crossings], color='green',
                        label='Threshold Crossings')
            plt.xlabel('Time')
            plt.ylabel('Trace')
            plt.title('Trace and Threshold Crossings')
            plt.legend()
            plt.grid(True)
            plt.show()

            print(f"Baseline Standard Deviation: {baseline_std}")
            print(f"Threshold: {threshold}")
            print(f"Threshold Crossings at Time Points: {time_stamps[valid_crossings][0]}")
        return time_stamps[valid_crossings][0], threshold
    else:
        return np.nan, threshold