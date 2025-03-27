import sys
import numpy as np
import pandas as pd
from utils.individual_trial_analysis_utils import ZScoredTraces, SessionData, HeatMapParams
from set_global_params import experiment_record_path, processed_data_path
import pickle
import os
from set_global_params import experiment_record_path, processed_data_path, reproduce_figures_path
import shutil


def get_all_experimental_records():
    """
    Reads in experimental record
    Returns:
        experiment_record (pd.dataframe): experimental record
    """
    experiment_record = pd.read_csv(experiment_record_path, dtype=str)
    experiment_record['date'] = experiment_record['date'].astype(str)
    return experiment_record


def find_manipulation_days(experiment_records, mice, exemption_list=('psychometric', 'state change medium cloud', 'value blocks', 'state change white noise',
                      'omissions and large rewards', 'contingency switch', 'ph3', 'saturation', 'value switch', 'omissions and large rewards')):
    """
    Looks for experiments where there were behavioural manipulations
    Args:
        experiment_records (pd.dataframe): all experiment records
        mice (list): mice to be included
        exemption_list (tuple): types of experiment to be found

    Returns:
        mouse_dates (pd.dataframe): experiments with behavioural manipulations (mouse_id and date)
    """
    # I have removed centre port hold as an exemption - this can be added back in if needed
    experiments = experiment_records[experiment_records['mouse_id'].isin(mice)]
    exemptions = '|'.join(exemption_list)
    index_to_remove = experiments[experiments['experiment_notes'].str.contains(exemptions,na=False)].index
    mouse_dates = experiments.loc[index_to_remove][['mouse_id', 'date']].reset_index(drop=True)
    reformatted_dates = pd.to_datetime(mouse_dates['date'])
    mouse_dates['date'] = reformatted_dates
    return mouse_dates


def remove_exps_after_manipulations(experiments, mice):
    """
    Removes the experiments after behavioural manipulations have taken place including psychometric experiments
    Args:
        experiments (pd.dataframe): all experiment records
        mice (list): mice to be included

    Returns:
        experiments (pd.dataframe): cleaned experimental records
    """
    manipulation_days = find_manipulation_days(experiments, mice)
    for mouse in manipulation_days['mouse_id'].unique():
        all_manipulation_days = manipulation_days[manipulation_days['mouse_id'] == mouse]
        earliest_manipulation_day = all_manipulation_days.min()

        index_to_remove = experiments[np.logical_and((experiments['mouse_id'] == mouse),
                                                     (pd.to_datetime(experiments['date']) >= earliest_manipulation_day['date']))].index
        if index_to_remove.shape[0] > 0:
            dates_being_removed = experiments['date'][index_to_remove].unique()
            experiments = experiments.drop(index=index_to_remove)
            print('removing {}: {}'.format(mouse, dates_being_removed))
    return experiments


def remove_exps_after_manipulations_not_including_psychometric(experiments, mice):
    """
    Removes the experiments after behavioural manipulations have taken place apart from psychometric sessions
    (which are not really manipulations as neither task rules nor value changes)
    Args:
        experiments (pd.dataframe): all experiment records
        mice (list): mice to be included

    Returns:
        experiments (pd.dataframe): cleaned experimental records
    """
    exemption_list = ['state change medium cloud', 'value blocks', 'state change white noise',
                      'omissions and large rewards', 'contingency switch', 'ph3', 'saturation', 'value switch',
                      'omissions and large rewards']
    manipulation_days = find_manipulation_days(experiments, mice, exemption_list=exemption_list)
    for mouse in manipulation_days['mouse_id'].unique():
        all_manipulation_days = manipulation_days[manipulation_days['mouse_id'] == mouse]
        earliest_manipulation_day = all_manipulation_days.min()

        index_to_remove = experiments[np.logical_and((experiments['mouse_id'] == mouse),
                                                     (pd.to_datetime(experiments['date']) >= earliest_manipulation_day['date']))].index
        if index_to_remove.shape[0] > 0:
            dates_being_removed = experiments['date'][index_to_remove].unique()
            experiments = experiments.drop(index=index_to_remove)
            experiments = remove_manipulation_days(experiments)
    return experiments


def remove_manipulation_days(experiments):
    """
    Removes days where there is a behavioural manipulation (value change, state change etc...) from the experimental record
    Args:
        experiments (pd.dataframe): all experiment records

    Returns:
        cleaned_experiments (pd.dataframe): experiment records without manipulation days
    """
    exemption_list = ['psychometric', 'state change medium cloud', 'value blocks', 'state change white noise', 'omissions and large rewards']
    exemptions = '|'.join(exemption_list)
    index_to_remove = experiments[np.logical_xor(experiments['include'] == 'no', experiments['experiment_notes'].str.contains(exemptions, na=False))].index
    cleaned_experiments = experiments.drop(index=index_to_remove)
    return cleaned_experiments


def remove_unsuitable_recordings(experiments):
    """
    Removes specific recordings from certain analyses
    (due to mostly poor numbers of trials with clean beahviour
     - clean behaviour means the mice did not go back and forth many times
    between the ports as this becomes hard to align to)
    Most commonly used when running regressions for sessions early in training.
    The regression needs decent numbers of behavioural events for each regressor,
    otherwise there are no events to predict the signal from
    This is not not always possible early in training as the mice have very unstereotyped behaviour,
    so many trials have unclean behavioural events that do not follow the cue-> choice -> reward pattern.
    The same sessions are still included for plotting average traces as this is less dependent on number of events.)
    Args:
        experiments (pd.dataframe): all experiment records

    Returns:
        cleaned_experiments (pd.dataframe): experiment records without the bad recordings
    """
    index_to_remove = experiments[experiments['include'] == 'no'].index
    cleaned_experiments = experiments.drop(index=index_to_remove)
    return cleaned_experiments


def open_experiment(experiment_to_add):
    """
    Opens an experiment (dataframe) This now does something very similar to open_one_experiment
    Args:
        experiment_to_add (pd.dataframe):

    Returns:
        session_traces (np.array): demodulated photometry signal
        trial_data (pd.dataframe): reformatted behavioural data
    """
    for index, experiment in experiment_to_add.iterrows():
        saving_folder = os.path.join(processed_data_path, experiment['mouse_id'])
        restructured_data_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(os.path.join(saving_folder, restructured_data_filename))
        session_traces = SessionData(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
    return session_traces, trial_data


def open_one_experiment(experiment):
    """
    Loads a single experiment (row of a dataframe)
    Args:
        experiment (pd.dataframe): row of dataframe with one experiment record

    Returns:
        trial_data (pd.dataframe): reformatted behavioural data
        session_traces (np.array): demodulated photometry signal
    """
    ## takes a row of a dataframe as input
    saving_folder = os.path.join(processed_data_path, experiment['mouse_id'] )
    restructured_data_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(os.path.join(saving_folder, restructured_data_filename))
    session_traces = SessionData(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
    return trial_data, session_traces

def open_experiment_just_behaviour(experiment_to_add, root_dir=processed_data_path):
    """
    Opens only the behaviour file for an experiment (dataframe)
    Args:
        experiment_to_add (pd.dataframe):

    Returns:
        trial_data (pd.dataframe): reformatted behavioural data
    """
    for index, experiment in experiment_to_add.iterrows():
        saving_folder = os.path.join(root_dir, experiment['mouse_id'])
        restructured_data_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(os.path.join(saving_folder, restructured_data_filename))
    return trial_data

def copy_behaviour_to_folder_mouse_name(experiment_to_add, source_dir=processed_data_path, target_dir=reproduce_figures_path):
    """
    Copies the behavioural file over to a target directory (and makes a subfolder for the mouse there)
    Args:
        experiment_to_add (pd.dataframe):

    Returns:
    """
    for index, experiment in experiment_to_add.iterrows():
        origin_folder = os.path.join(source_dir, experiment['mouse_id'])
        saving_folder = os.path.join(target_dir, experiment['mouse_id'])
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)
        restructured_data_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'restructured_data.pkl'
        shutil.copy(os.path.join(origin_folder, restructured_data_filename),
                    os.path.join(saving_folder, restructured_data_filename))
    return


def get_first_x_sessions(experiment_record, mouse_ids, site, x=3):
    """
    Finds first x sessions for each mouse
    Args:
        experiment_record (pd.dataframe): dataframe of experiments (can be mulitple mice)
        mouse_ids (list): mice to be included
        site (str): recording site
        x (int): number of sessions for each mouse

    Returns:
        exps (pd.dataframe): the dataframe only containing the first x sessions per mouse
    """
    experiment_record['date'] = experiment_record['date'].astype(str)
    clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
    all_experiments_to_process = clean_experiments[
        (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
        drop=True)
    i = []
    inds = []
    for mouse in np.unique(all_experiments_to_process['mouse_id']):
        i.append(all_experiments_to_process[all_experiments_to_process['mouse_id'] == mouse][0:x].index)
        inds += range(0, x)
    flattened_i = [val for sublist in i for val in sublist]
    exps = all_experiments_to_process.loc[flattened_i].reset_index(drop=True)
    exps['session number'] = inds
    return exps


def add_experiment_to_aligned_data(experiments_to_add, for_heat_map_figure=False, cue=True, choice=True, reward=True, outcome=True):
    """
    Aligns photometry data to behavioural events for multiple experiments and saves out formatted data

    Args:
        experiments_to_add (pd.dataframe): the experiments that will be processed and saved out as reformatted objects
        for_heat_map_figure (bool): is this to preprocess data prior to making heatmaps for figure 3?
        cue (bool): align data to cue?
        choice (bool): align data to choice?
        reward (bool): align data to reward? This is only rewarded trials (ipsi vs contra)
        outcome (bool): align data to outcome? This is correct vs incorrect

    Returns:

    """
    if for_heat_map_figure:
        data_root = os.path.join(processed_data_path, 'for_figure')
        file_name_tag = 'aligned_traces_for_fig.p'
    else:
        data_root = processed_data_path
        file_name_tag = 'aligned_traces.p'
    for index, experiment in experiments_to_add.iterrows():
        print(experiment['mouse_id'],' ', experiment['date'])
        saving_folder = os.path.join(data_root, experiment['mouse_id'])
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)

        session_traces = SessionData(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
        if choice:
            session_traces.get_choice_responses()
        if cue:
            session_traces.get_cue_responses()
        if reward:
            session_traces.get_reward_responses()
        if outcome:
            session_traces.get_outcome_responses()
        aligned_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + file_name_tag
        save_filename = os.path.join(saving_folder, aligned_filename)
        pickle.dump(session_traces, open(save_filename, "wb"))


class CustomAlignedData(object):
    def __init__(self, session_data, params, peak_quantification=True):
        """
        Creates formatted data object with aligned traces for a given set of parameters
        Args:
            session_data (pd.dataframe): experiment record for the session to process
            params (HeatMapParams): the parameters for how to align the photometry data
            peak_quantification (bool): whether to get the peaks of the photometry data after the aligned to events
        """
        saving_folder = os.path.join(processed_data_path, session_data.mouse)

        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(os.path.join(saving_folder, restructured_data_filename))
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(os.path.join(saving_folder, dff_trace_filename))

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        self.ipsi_data = ZScoredTraces(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.contra_data = ZScoredTraces(trial_data, dff,params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        if peak_quantification:
            self.contra_data.get_peaks()
            self.ipsi_data.get_peaks()
