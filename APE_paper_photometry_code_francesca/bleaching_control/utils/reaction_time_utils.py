import pickle
import datetime
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np
import math
import pandas as pd
import os
from set_global_params import processed_data_path, behavioural_data_path


def plot_reaction_times(mouse, dates):
    saving_folder = os.path.join(processed_data_path, mouse)
    fig, axs = plt.subplots(math.ceil(len(dates)/2), ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    axs = axs.T.flatten()
    num_types = len(dates)
    colours = cm.viridis(np.linspace(0, 0.8, num_types))
    fig.suptitle(mouse + ' reaction')
    for date_num, date in enumerate(dates):
        mean_and_sem_filename = os.path.join(saving_folder, mouse + '_' + date + '_' + 'peaks_correct_data.p')
        data = pickle.load( open(mean_and_sem_filename, "rb" ))
        reaction_times = data.contra_reaction_times
        bins = np.arange(start=min(reaction_times), stop=max(reaction_times)+0.1, step=0.1)
        axs[date_num].hist(np.reshape(reaction_times, [len(reaction_times), 1]), bins=bins, density=True, color=colours[date_num])
        axs[date_num].axvline(np.mean(reaction_times), color='k')
        axs[date_num].axvline(np.mean(reaction_times)+ np.std(reaction_times), color='grey')
    plt.xlim(0, 6)
    plt.show()


def plot_reaction_times_overlayed(mouse, dates):
    saving_folder = os.path.join(processed_data_path, mouse)
    fig, ax = plt.subplots(1, ncols=1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.2)
    num_types = len(dates)
    colours = cm.viridis(np.linspace(0, 0.8, num_types))
    fig.suptitle(mouse + ' reaction')
    for date_num, date in enumerate(dates):
        mean_and_sem_filename = os.path.join(saving_folder,  mouse + '_' + date + '_' + 'aligned_traces.p')
        data = pickle.load(open(mean_and_sem_filename, "rb" ))
        reaction_times = data.choice_data.contra_data.reaction_times
        bins = np.arange(start=min(reaction_times), stop=max(reaction_times)+0.1, step=0.1)
        ax.hist(np.reshape(reaction_times, [len(reaction_times), 1]),bins=bins, density=True, color=colours[date_num], alpha=0.2)

    plt.show()


def get_valid_trials(mouse, dates, window_around_mean=0.2, recording_site='tail', side='contra'):
    session_starts = get_bpod_trial_nums_per_session(mouse, dates)
    data_root = processed_data_path + 'peak_analysis'
    all_peaks = []
    all_bins = []
    all_reaction_times =[]
    all_trial_numbers = []
    all_actual_trial_numbers = []
    # if I can get the trial numbers ever that things belong to, then we are in business
    for date_num, date in enumerate(dates):
        print(date)
        peaks_saving_folder = os.path.join(data_root, mouse)
        #aligned_filename = saving_folder +  mouse + '_' + date + '_' + 'aligned_traces.p'
        filename = mouse + '_' + date + '_' + 'peaks.p'
        aligned_filename = os.path.join(peaks_saving_folder, filename)
        #mean_and_sem_filename = saving_folder + mouse + '_' + date + '_' + 'peaks_' + type_of_session + '_data.p'
        with open(aligned_filename, 'rb') as f:
            data = pickle.load(f)
        if recording_site == 'tail':
            recording_site_data = data.choice_data.contra_data
            actual_trial_numbers = recording_site_data.trial_nums + session_starts[date_num]
            all_actual_trial_numbers.append(actual_trial_numbers)
            all_trial_numbers.append(len(recording_site_data.reaction_times))
            all_reaction_times.append(recording_site_data.reaction_times)
            all_peaks.append(data.choice_data.contra_data.trial_peaks)
            all_bins.append(np.arange(start=min(recording_site_data.reaction_times),
                                      stop=max(recording_site_data.reaction_times) + 0.1, step=0.1))
        if recording_site == 'Nacc':
            if side == 'high':
                recording_site_data = data.cue_data.high_cue_data
                actual_trial_numbers = recording_site_data.trial_nums + session_starts[date_num]
                all_actual_trial_numbers.append(actual_trial_numbers)
                all_trial_numbers.append(len(recording_site_data.reaction_times))
                all_reaction_times.append(recording_site_data.reaction_times)
                all_peaks.append(recording_site_data.trial_peaks)
                all_bins.append(np.arange(start=min(recording_site_data.reaction_times),
                                          stop=max(recording_site_data.reaction_times) + 0.1, step=0.1))
            if side == 'low':
                recording_site_data = data.cue_data.low_cue_data
                actual_trial_numbers = recording_site_data.trial_nums + session_starts[date_num]
                all_actual_trial_numbers.append(actual_trial_numbers)
                all_trial_numbers.append(len(recording_site_data.reaction_times))
                all_reaction_times.append(recording_site_data.reaction_times)
                all_peaks.append(recording_site_data.trial_peaks)
                all_bins.append(np.arange(start=min(recording_site_data.reaction_times),
                                          stop=max(recording_site_data.reaction_times) + 0.1, step=0.1))
            if side == 'contra':
                recording_site_data = data.cue_data.contra_data
                actual_trial_numbers = recording_site_data.trial_nums + session_starts[date_num]
                all_actual_trial_numbers.append(actual_trial_numbers)
                all_trial_numbers.append(len(recording_site_data.reaction_times))
                all_reaction_times.append(recording_site_data.reaction_times)
                all_peaks.append(recording_site_data.trial_peaks)
                all_bins.append(np.arange(start=min(recording_site_data.reaction_times),
                                          stop=max(recording_site_data.reaction_times) + 0.1, step=0.1))

            elif side == 'ipsi':
                recording_site_data = data.cue_data.ipsi_data
                actual_trial_numbers = recording_site_data.trial_nums + session_starts[date_num]
                all_actual_trial_numbers.append(actual_trial_numbers)
                all_trial_numbers.append(len(recording_site_data.reaction_times))
                all_reaction_times.append(recording_site_data.reaction_times)
                all_peaks.append(data.cue_data.ipsi_data.trial_peaks)
                all_bins.append(np.arange(start=min(recording_site_data.reaction_times),
                                          stop=max(recording_site_data.reaction_times) + 0.1, step=0.1))

            else:
                contra_data = data.cue_data.contra_data
                ipsi_data = data.cue_data.ipsi_data
                contra_trial_numbers = contra_data.trial_nums + session_starts[date_num]
                ipsi_trial_numbers = ipsi_data.trial_nums + session_starts[date_num]
                all_actual_trial_numbers.append(np.concatenate((contra_trial_numbers, ipsi_trial_numbers)))
                all_trial_numbers.append(len(contra_data.reaction_times) + len(ipsi_data.trial_nums))
                all_reaction_times.append(np.concatenate((contra_data.reaction_times, ipsi_data.reaction_times)))
                all_peaks.append(np.concatenate((data.cue_data.contra_data.trial_peaks, data.cue_data.ipsi_data.trial_peaks)))
                all_bins.append(np.arange(start=min(all_reaction_times[-1]), stop=max(all_reaction_times[-1])+0.1, step=0.1))


    flattened_actual_trial_nums = [item for sublist in all_actual_trial_numbers for item in sublist]
    flattened_reaction_times = [item for sublist in all_reaction_times for item in sublist]
    all_trial_nums = np.arange(1, np.sum(all_trial_numbers) + 1, step=1)
    median_reaction_time = np.median(flattened_reaction_times)
    valid_trials = np.where(np.logical_and(np.greater_equal(flattened_reaction_times, median_reaction_time - window_around_mean),
                                           np.less_equal(flattened_reaction_times, median_reaction_time + window_around_mean)))
    flattened_peaks = [item for sublist in all_peaks for item in sublist]
    for i, val in enumerate(flattened_peaks):
        if type(val) != float:
            if val.size == 0:
                flattened_peaks[i] = np.nan
    valid_peaks = np.array(flattened_peaks)[valid_trials]
    valid_trial_nums = np.array(flattened_actual_trial_nums)[valid_trials]
    valid_reaction_times = np.array(flattened_reaction_times)[valid_trials]
    return(session_starts, valid_trials, valid_reaction_times, valid_peaks, valid_trial_nums)


def get_bpod_trial_nums_per_session(mouse, dates):
    """
    Finds the number of trials in behavioural training sessions
    Args:
        mouse (str): mouse name
        dates (list): dates of training sessions in YYYYMMDD format

    Returns:
        session_first_trials (list): trial number of start of sessions in context of all trials ever done
    """
    BpodProtocol = 'Two_Alternative_Choice'
    GeneralDirectory = behavioural_data_path
    DFfile = os.path.join(GeneralDirectory, mouse, BpodProtocol, 'Data_Analysis', mouse + '_dataframe.pkl')
    behavioural_stats = pd.read_pickle(DFfile)
    sessions = MakeDatesPretty(dates)
    session_first_trials = []
    for session in sessions:
        session_first_trial = behavioural_stats[behavioural_stats['SessionTime'].str.contains(session)].index[0]
        session_first_trials.append(session_first_trial)
    return session_first_trials


def MakeDatesPretty(inputDates):
    """
    Converts YYYYMMDD strings to datetime
    Args:
        inputDates (list): dates in YYYYMMDD format

    Returns:
        outputDates (list): dates as datetime
    """
    outputDates = []
    for date in inputDates:
            x = datetime.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]))
            outputDates.append(x.strftime("%b%d"))
    return outputDates

