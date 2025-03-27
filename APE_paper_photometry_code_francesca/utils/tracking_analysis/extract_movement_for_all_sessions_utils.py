import os
from utils.tracking_analysis.fede_load_tracking import prepare_tracking_data
from set_global_params import processed_data_path, raw_tracking_path
from utils.tracking_analysis.camera_trigger_preprocessing_utils import *
from utils.tracking_analysis.tracking_plotting import *
from utils.tracking_analysis.dlc_processing_utils import get_photometry_data, get_photometry_data_correct_incorrect_normal_task
from utils.tracking_analysis.velocity_utils import format_tracking_data_and_photometry, format_tracking_data_and_photometry_correct_incorrect

from utils.reaction_time_utils import get_bpod_trial_nums_per_session
from utils.post_processing_utils import get_all_experimental_records
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings, remove_manipulation_days


def get_actual_trial_numbers(per_session_trial_nums, date, mouse, recording_site='tail'):
    all_experiments = get_all_experimental_records()
    all_experiments = remove_exps_after_manipulations(all_experiments, [mouse])
    all_experiments = remove_manipulation_days(all_experiments)
    all_experiments = remove_unsuitable_recordings(all_experiments)
    experiments_to_process = all_experiments[
        (all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == recording_site)]
    dates = experiments_to_process['date'].values
    session_starts = get_bpod_trial_nums_per_session(mouse, dates)
    date_num = np.where(dates == date)[0][0]
    actual_trial_numbers = per_session_trial_nums + session_starts[date_num]
    return actual_trial_numbers


def get_movement_properties_for_session(mouse, date, protocol='Two_Alternative_Choice', multi_session=True):
    file_path = os.path.join(raw_tracking_path, mouse, date, 'cameraDLC_resnet50_train_network_with_more_miceMar2shuffle1_800000.h5')
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

    camera_triggers, trial_start_stamps = get_camera_trigger_times(mouse, date, protocol)
    trial_start_triggers = find_nearest_trials(trial_start_stamps, camera_triggers)
    photometry_data = get_photometry_data(mouse, date)

    saving_folder = os.path.join(processed_data_path, mouse)
    restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(saving_folder + restructured_data_filename)

    first_cot = (trial_data[(trial_data['State name'] == 'CueDelay') & (
            trial_data['Instance in state'] == trial_data['Max times in state'])]['Time start'].values * 10000)
    first_cot = first_cot.astype(int)

    first_cot_triggers = find_nearest_trials(first_cot, camera_triggers)


    choice = (trial_data[(trial_data['State name'] == 'WaitForResponse') & (
            trial_data['Instance in state'] == trial_data['Max times in state'])]['Time end'].values * 10000)
    choice = choice.astype(int)
    choice_triggers = find_nearest_trials(choice, camera_triggers)
    choice_starts = (trial_data[(trial_data['State name'] == 'WaitForResponse') & (
            trial_data['Instance in state'] == trial_data['Max times in state'])]['Time start'].values * 10000)
    choice_starts = choice_starts.astype(int)
    choice_start_triggers = find_nearest_trials(choice_starts, camera_triggers)
    trial_types = trial_data[['Trial num', 'Trial type']]
    trial_types_per_trial = trial_types.drop_duplicates(subset='Trial num').set_index('Trial num')
    formatted_data = format_tracking_data_and_photometry(tracking_data, photometry_data, first_cot_triggers,
                                                         choice_triggers, trial_types_per_trial)
    previous_trial_nums = formatted_data['trial numbers'] - 1
    next_trial_nums = formatted_data['trial numbers'] + 1
    if multi_session:
        actual_trial_numbers = get_actual_trial_numbers(formatted_data['trial numbers'].values, date, mouse)
        formatted_data['actual trial numbers'] = actual_trial_numbers
    if photometry_data.fiber_side == 'left':
        sides = ['ipsi', 'contra']
    else:
        sides = ['contra', 'ipsi']
    previous_trial_confidence = []
    last_outcomes = []
    last_choices = []
    for t in previous_trial_nums:
        if t >= 0:
            last_trial_type = trial_data[trial_data['Trial num'] == t]['Trial type'].values[0]
            last_outcome = trial_data[trial_data['Trial num'] == t]['Trial outcome'].values[0]
            last_response = trial_data[trial_data['Trial num'] == t]['Response'].values[0]
            if np.isnan(last_response):
                last_choices.append(np.nan)
            else:
                last_choice = sides[int(last_response - 1)]
                last_choices.append(last_choice)
            previous_trial_confidence.append(last_trial_type)
            last_outcomes.append(last_outcome)

        else:
            previous_trial_confidence.append(np.nan)
            last_outcomes.append(np.nan)
            last_choices.append(np.nan)
    formatted_data['last trial type'] = previous_trial_confidence
    formatted_data['last outcome'] = last_outcomes
    formatted_data['last choice'] = last_choices

    next_outcomes = []
    next_choices = []
    next_trial_types = []
    next_choices_numeric = []
    for t1 in next_trial_nums.values:
        if t1 < np.max(trial_data['Trial num']):
            next_trial_type = trial_data[trial_data['Trial num'] == t1]['Trial type'].values[0]
            next_outcome = trial_data[trial_data['Trial num'] == t1]['Trial outcome'].values[0]
            next_response = trial_data[trial_data['Trial num'] == t1]['Response'].values[0]

            if np.isnan(next_response):
                next_choices.append(np.nan)
                next_choices_numeric.append(np.nan)
            else:
                next_choice = sides[int(next_response - 1)]
                next_choices.append(next_choice)
                next_choices_numeric.append(next_response)
            next_trial_types.append(next_trial_type)
            next_outcomes.append(next_outcome)

        else:
            next_trial_types.append(np.nan)
            next_outcomes.append(np.nan)
            next_choices.append(np.nan)
            next_choices_numeric.append(np.nan)
    formatted_data['next trial type'] = next_trial_types
    formatted_data['next outcome'] = next_outcomes
    formatted_data['next choice'] = next_choices
    formatted_data['next choice numeric'] = next_choices_numeric
    stay_trials = trial_data[(trial_data['Last response'] == trial_data['Response'])]['Trial num'].unique()
    switch_trials = trial_data[(trial_data['Last response'] != trial_data['Response'])]['Trial num'].unique()
    switch_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, switch_trials)].index.values
    stay_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, stay_trials)].index.values
    correct_trials = trial_data[(trial_data['First choice correct'] == 1)]['Trial num'].unique() #trial_data[(trial_data['Trial outcome'] == 1)]['Trial num'].unique()
    incorrect_trials = trial_data[(trial_data['First choice correct'] == 0)]['Trial num'].unique() #trial_data[(trial_data['Trial outcome'] == 0)]['Trial num'].unique()
    correct_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, correct_trials)].index.values
    incorrect_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, incorrect_trials)].index.values
    formatted_data['stay or switch'] = np.empty(formatted_data.shape[0])
    temp = np.empty(formatted_data.shape[0])
    temp[:] = np.nan
    formatted_data['outcome'] = temp
    formatted_data.loc[correct_inds, 'outcome'] = 1
    formatted_data.loc[incorrect_inds, 'outcome'] = 0
    formatted_data['stay or switch'] = np.empty(formatted_data.shape[0])
    formatted_data.loc[switch_inds, 'stay or switch'] = 'swiatch'
    formatted_data.loc[stay_inds, 'stay or switch'] = 'stay'
    formatted_data['APE quantile'] = pd.qcut(formatted_data['APE peaks'], q=4)
    return formatted_data, trial_data


def get_movement_properties_for_session_correct_incorrect(mouse, date, protocol='Two_Alternative_Choice', multi_session=True):
    file_path = os.path.join(raw_tracking_path, mouse, date,'cameraDLC_resnet50_train_network_with_more_miceMar2shuffle1_800000.h5')
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

    camera_triggers, trial_start_stamps = get_camera_trigger_times(mouse, date, protocol)
    trial_start_triggers = find_nearest_trials(trial_start_stamps, camera_triggers)
    photometry_data = get_photometry_data_correct_incorrect_normal_task(mouse, date)

    saving_folder = os.path.join(processed_data_path, mouse)
    restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(os.path.join(saving_folder, restructured_data_filename))

    first_cot = (trial_data[(trial_data['State name'] == 'CueDelay') & (
            trial_data['Instance in state'] == trial_data['Max times in state'])]['Time start'].values * 10000)
    first_cot = first_cot.astype(int)

    first_cot_triggers = find_nearest_trials(first_cot, camera_triggers)


    choice = (trial_data[(trial_data['State name'] == 'WaitForResponse') & (
            trial_data['Instance in state'] == trial_data['Max times in state'])]['Time end'].values * 10000)
    choice = choice.astype(int)
    choice_triggers = find_nearest_trials(choice, camera_triggers)
    choice_starts = (trial_data[(trial_data['State name'] == 'WaitForResponse') & (
            trial_data['Instance in state'] == trial_data['Max times in state'])]['Time start'].values * 10000)
    choice_starts = choice_starts.astype(int)
    choice_start_triggers = find_nearest_trials(choice_starts, camera_triggers)
    trial_types = trial_data[['Trial num', 'Trial type']]
    trial_types_per_trial = trial_types.drop_duplicates(subset='Trial num').set_index('Trial num')
    formatted_data = format_tracking_data_and_photometry_correct_incorrect(tracking_data, photometry_data, first_cot_triggers,
                                                         choice_triggers, trial_types_per_trial)
    previous_trial_nums = formatted_data['trial numbers'] - 1
    next_trial_nums = formatted_data['trial numbers'] + 1
    if multi_session:
        actual_trial_numbers = get_actual_trial_numbers(formatted_data['trial numbers'].values, date, mouse)
        formatted_data['actual trial numbers'] = actual_trial_numbers
    sides = ['correct', 'incorrect']
    previous_trial_confidence = []
    last_outcomes = []
    last_choices = []
    for t in previous_trial_nums:
        if t >= 0:
            last_trial_type = trial_data[trial_data['Trial num'] == t]['Trial type'].values[0]
            last_outcome = trial_data[trial_data['Trial num'] == t]['Trial outcome'].values[0]
            last_response = trial_data[trial_data['Trial num'] == t]['Response'].values[0]
            if np.isnan(last_response):
                last_choices.append(np.nan)
            else:
                last_choice = sides[int(last_response - 1)]
                last_choices.append(last_choice)
            previous_trial_confidence.append(last_trial_type)
            last_outcomes.append(last_outcome)

        else:
            previous_trial_confidence.append(np.nan)
            last_outcomes.append(np.nan)
            last_choices.append(np.nan)
    formatted_data['last trial type'] = previous_trial_confidence
    formatted_data['last outcome'] = last_outcomes
    formatted_data['last choice'] = last_choices

    next_outcomes = []
    next_choices = []
    next_trial_types = []
    next_choices_numeric = []
    for t1 in next_trial_nums.values:
        if t1 < np.max(trial_data['Trial num']):
            next_trial_type = trial_data[trial_data['Trial num'] == t1]['Trial type'].values[0]
            next_outcome = trial_data[trial_data['Trial num'] == t1]['Trial outcome'].values[0]
            next_response = trial_data[trial_data['Trial num'] == t1]['Response'].values[0]

            if np.isnan(next_response):
                next_choices.append(np.nan)
                next_choices_numeric.append(np.nan)
            else:
                next_choice = sides[int(next_response - 1)]
                next_choices.append(next_choice)
                next_choices_numeric.append(next_response)
            next_trial_types.append(next_trial_type)
            next_outcomes.append(next_outcome)

        else:
            next_trial_types.append(np.nan)
            next_outcomes.append(np.nan)
            next_choices.append(np.nan)
            next_choices_numeric.append(np.nan)
    formatted_data['next trial type'] = next_trial_types
    formatted_data['next outcome'] = next_outcomes
    formatted_data['next choice'] = next_choices
    formatted_data['next choice numeric'] = next_choices_numeric
    stay_trials = trial_data[(trial_data['Last response'] == trial_data['Response'])]['Trial num'].unique()
    switch_trials = trial_data[(trial_data['Last response'] != trial_data['Response'])]['Trial num'].unique()
    switch_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, switch_trials)].index.values
    stay_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, stay_trials)].index.values
    correct_trials = trial_data[(trial_data['First choice correct'] == 1)]['Trial num'].unique() #trial_data[(trial_data['Trial outcome'] == 1)]['Trial num'].unique()
    incorrect_trials = trial_data[(trial_data['First choice correct'] == 0)]['Trial num'].unique() #trial_data[(trial_data['Trial outcome'] == 0)]['Trial num'].unique()
    correct_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, correct_trials)].index.values
    incorrect_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, incorrect_trials)].index.values
    formatted_data['stay or switch'] = np.empty(formatted_data.shape[0])
    temp = np.empty(formatted_data.shape[0])
    temp[:] = np.nan
    formatted_data['outcome'] = temp
    formatted_data.loc[correct_inds, 'outcome'] = 1
    formatted_data.loc[incorrect_inds, 'outcome'] = 0
    formatted_data['stay or switch'] = np.empty(formatted_data.shape[0])
    formatted_data.loc[switch_inds, 'stay or switch'] = 'swiatch'
    formatted_data.loc[stay_inds, 'stay or switch'] = 'stay'
    formatted_data['APE quantile'] = pd.qcut(formatted_data['APE peaks'], q=4)
    return formatted_data, trial_data
