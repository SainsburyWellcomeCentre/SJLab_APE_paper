from scipy.signal import filtfilt
from utils.tracking_analysis.velocity_utils import format_tracking_data_and_photometry, format_only_photometry
from scipy.optimize import curve_fit
import pickle
import pandas as pd
from utils.tracking_analysis.fede_load_tracking import prepare_tracking_data
from utils.tracking_analysis.camera_trigger_preprocessing_utils import *
from utils.tracking_analysis.tracking_plotting import *
from set_global_params import processed_data_path, old_raw_tracking_path
import os


def get_photometry_data_correct_incorrect_normal_task(mouse, date):
    saving_folder = processed_data_path + '\\for_figure\\' + mouse + '\\'
    aligned_filename = saving_folder + mouse + '_' + date + '_' + 'aligned_traces_correct_incorrect.p'
    with open(aligned_filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_x_y_data(data, scorer, bodypart):
    # get x_y_data
    print('bodypart is: ', bodypart)
    bodypart_data = (data.loc[(scorer, bodypart)])

    bodypart_data_x = bodypart_data.loc[('x')]
    bodypart_data_y = bodypart_data.loc[('y')]

    return bodypart_data_x, bodypart_data_y


def get_x_y_data_cleanup(data, likelihood):
    # sets any value below a particular point to value 0 in x and y, this 0 value can then be used by a later
    # interpolation algorithm

    bodypart_data = data

    x_coords = []
    y_coords = []

    for index in bodypart_data:
        if bodypart_data.loc['likelihood'][index] > likelihood:
            x_coords.append(bodypart_data.loc['x'][index])
            y_coords.append(bodypart_data.loc['y'][index])
        else:
            x_coords.append(0)
            y_coords.append(0)

    return x_coords, y_coords


def start_value_cleanup(coords):
    # This is for when the starting value of the coords == 0; interpolation will not work on these coords until the first 0
    # is changed. The 0 value is changed to the first non-zero value in the coords lists
    for index, value in enumerate(coords):
        if value > 0:
            start_value = value
            start_index = index
            print(start_index)
            break

    for x in range(start_index):
        coords[x] = start_value


def interp_0_coords(coords_list):
    #coords_list is one if the outputs of the get_x_y_data = a list of co-ordinate points
    for index, value in enumerate(coords_list):
        if value == 0:
            if coords_list[index-1] > 0:
                value_before = coords_list[index-1]
                interp_start_index = index-1
                #print('interp_start_index: ', interp_start_index)
                #print('interp_start_value: ', value_before)
                #print('')

        if index < len(coords_list)-1:
            if value ==0:
                if coords_list[index+1] > 0:
                    interp_end_index = index+1
                    value_after = coords_list[index+1]
                    #print('interp_end_index: ', interp_end_index)
                    #print('interp_end_value: ', value_after)
                    #print('')

                    #now code to interpolate over the values
                    try:
                        interp_diff_index = interp_end_index - interp_start_index
                    except UnboundLocalError:
                        print('the first value in list is 0, use the function start_value_cleanup to fix')
                        break
                    #print('interp_diff_index is:', interp_diff_index)

                    new_values = np.linspace(value_before, value_after, interp_diff_index)
                    #print(new_values)

                    interp_index = interp_start_index+1
                    for x in range(interp_diff_index):
                        #print('interp_index is:', interp_index)
                        #print('new_value should be:', new_values[x])
                        coords_list[interp_index] = new_values[x]
                        interp_index +=1
        if index == len(coords_list)-1:
            if value ==0:
                for x in range(30):
                    coords_list[index-x] = coords_list[index-30]
                    #print('')
    print('function exiting')
    return(coords_list)


def get_data_for_body_part(dlc_trajectories, start_frame, end_frame, body_part):
    scorer = dlc_trajectories.index[0][0]
    body_part_data = dlc_trajectories.loc[(scorer, body_part), start_frame:end_frame]
    mousedata_body_part_x0s, mousedata_body_part_y0s = get_x_y_data_cleanup(body_part_data, 0.999)
    start_value_cleanup(mousedata_body_part_x0s)
    start_value_cleanup(mousedata_body_part_y0s)
    body_part_interpolated_x = interp_0_coords(mousedata_body_part_x0s)
    body_part_interpolated_y = interp_0_coords(mousedata_body_part_y0s)
    n = 2  # the larger n is, the smoother curve will be
    nom = [1.0 / n] * n
    denom = 1
    body_part_interpolated_lfilt_x = filtfilt(nom, denom, body_part_interpolated_x)
    body_part_interpolated_lfilt_y = filtfilt(nom, denom, body_part_interpolated_y)
    return (body_part_interpolated_lfilt_x, body_part_interpolated_lfilt_y)


def get_photometry_data(mouse, date):
    saving_folder = os.path.join(processed_data_path, mouse)
    aligned_filename = os.path.join(saving_folder, mouse + '_' + date + '_' + 'aligned_traces.p')
    with open(aligned_filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_photometry_data_correct_incorrect(mouse, date):
    saving_folder = os.path.join(processed_data_path, 'for_psychometric', mouse)
    aligned_filename = os.path.join(saving_folder, mouse + '_' + date + '_' + 'aligned_traces_for_psychometric.p')
    with open(aligned_filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_photometry_for_trial_numbers(test_trial_numbers, all_trial_numbers,  photometry_peaks):
    contra_test_trial_nums, inds, _ = np.intersect1d(all_trial_numbers, test_trial_numbers, return_indices=True)
    inds = inds.astype(int)
    test_peaks = np.array(photometry_peaks)[inds]
    return contra_test_trial_nums, test_peaks


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return (y)


def fit_sigmoid(x_data, y_data):
    p0 = [max(y_data), np.median(x_data), 1, min(y_data)]  # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, x_data, y_data, p0, method='dogbox')
    return popt, pcov


def get_raw_photometry_data(mouse, date):
    saving_folder = os.path.join(processed_data_path, mouse)
    restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(os.path.join(saving_folder, restructured_data_filename))
    with open(os.path.join(saving_folder, restructured_data_filename), "rb") as fh:
        trial_data = pickle.load(fh)
    dff_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
    dff = np.load(os.path.join(saving_folder, dff_trace_filename))
    return dff, trial_data


def get_movement_properties_for_session(mouse, date):
    file_path = os.path.join(old_raw_tracking_path, '{}_{}DLC_resnet50_two_acMay10shuffle1_600000.h5'.format(
        mouse, date))
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

    camera_triggers, trial_start_stamps = get_camera_trigger_times(mouse, date, 'Two_Alternative_Choice')
    trial_start_triggers = find_nearest_trials(trial_start_stamps, camera_triggers)
    photometry_data = get_photometry_data(mouse, date)

    saving_folder = os.path.join(processed_data_path, mouse)
    restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(os.path.join(saving_folder, restructured_data_filename))

    first_cot = (trial_data[(trial_data['State name'] == 'CueDelay') & (
            trial_data['Instance in state'] == trial_data['Max times in state'])]['Time start'].values * 10000)
    first_cot = first_cot.astype(int)

    first_cot_triggers = find_nearest_trials(first_cot, camera_triggers)

    contra_peaks = photometry_data.choice_data.contra_data.trial_peaks
    clean_peaks = [c for c in contra_peaks if c.size != 0]
    trial_nums = photometry_data.choice_data.contra_data.trial_nums

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
    correct_trials = trial_data[(trial_data['Trial outcome'] == 1)]['Trial num'].unique()
    incorrect_trials = trial_data[(trial_data['Trial outcome'] == 0)]['Trial num'].unique()
    correct_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, correct_trials)].index.values
    incorrect_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, incorrect_trials)].index.values
    formatted_data['stay or switch'] = np.empty(formatted_data.shape[0])
    temp = np.empty(formatted_data.shape[0])
    temp[:] = np.nan
    formatted_data['outcome'] = temp
    formatted_data.loc[correct_inds, 'outcome'] = 1
    formatted_data.loc[incorrect_inds, 'outcome'] = 0
    formatted_data['stay or switch'] = np.empty(formatted_data.shape[0])
    formatted_data.loc[switch_inds, 'stay or switch'] = 'switch'
    formatted_data.loc[stay_inds, 'stay or switch'] = 'stay'
    formatted_data['APE quantile'] = pd.qcut(formatted_data['APE peaks'], q=4)
    return formatted_data, trial_data


def get_peaks_and_trial_types(mouse, date, align_to='choice'):
    saving_folder = os.path.join(processed_data_path, mouse)
    restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(os.path.join(saving_folder, restructured_data_filename))
    if align_to == 'choice':
        photometry_data = get_photometry_data(mouse, date)
    elif align_to == 'cue':
        photometry_data = get_photometry_data(mouse, date)
    elif align_to == 'reward':
        photometry_data = get_photometry_data_correct_incorrect(mouse, date)
    else:
        print('invalid alignment')
    trial_types = trial_data[['Trial num', 'Trial type']]
    trial_types_per_trial = trial_types.drop_duplicates(subset='Trial num').set_index('Trial num')
    formatted_data = format_only_photometry(photometry_data, trial_types_per_trial, align_to=align_to)
    previous_trial_nums = formatted_data['trial numbers'] - 1
    next_trial_nums = formatted_data['trial numbers'] + 1
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
    for t1 in next_trial_nums:
        if t1 < np.max(trial_data['Trial num']):
            next_trial_type = trial_data[trial_data['Trial num'] == t1]['Trial type'].values[0]
            next_outcome = trial_data[trial_data['Trial num'] == t1]['Trial outcome'].values[0]
            next_response = trial_data[trial_data['Trial num'] == t1]['Response'].values[0]
            if np.isnan(next_response):
                next_choices.append(np.nan)
            else:
                next_choice = sides[int(next_response - 1)]
                next_choices.append(next_choice)
            next_trial_types.append(next_trial_type)
            next_outcomes.append(next_outcome)

        else:
            next_trial_types.append(np.nan)
            next_outcomes.append(np.nan)
            next_choices.append(np.nan)
    formatted_data['next trial type'] = next_trial_types
    formatted_data['next outcome'] = next_outcomes
    formatted_data['next choice'] = next_choices
    if align_to == 'reward':
        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == photometry_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != photometry_data.fiber_side)[0] + 1)[0]
        ipsi_trials = trial_data[trial_data['Response'] == fiber_side_numeric]['Trial num'].unique()
        contra_trials = trial_data[trial_data['Response'] == contra_fiber_side_numeric]['Trial num'].unique()
        ipsi_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, ipsi_trials)].index.values
        contra_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, contra_trials)].index.values
        formatted_data.loc[ipsi_inds, 'side'] = 'ipsi'
        formatted_data.loc[contra_inds, 'side'] = 'contra'

    stay_trials = trial_data[(trial_data['Last response'] == trial_data['Response'])]['Trial num'].unique()
    switch_trials = trial_data[(trial_data['Last response'] != trial_data['Response'])]['Trial num'].unique()
    switch_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, switch_trials)].index.values
    stay_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, stay_trials)].index.values
    correct_trials = trial_data[(trial_data['Trial outcome'] == 1)]['Trial num'].unique()
    incorrect_trials = trial_data[(trial_data['Trial outcome'] == 0)]['Trial num'].unique()
    correct_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, correct_trials)].index.values
    incorrect_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, incorrect_trials)].index.values
    formatted_data['stay or switch'] = np.empty(formatted_data.shape[0])
    temp = np.empty(formatted_data.shape[0])
    temp[:] = np.nan
    formatted_data['outcome'] = temp
    formatted_data.loc[correct_inds, 'outcome'] = 1
    formatted_data.loc[incorrect_inds, 'outcome'] = 0
    formatted_data.loc[switch_inds, 'stay or switch'] = 'switch'
    formatted_data.loc[stay_inds, 'stay or switch'] = 'stay'
    formatted_data['APE quantile'] = pd.qcut(formatted_data['APE peaks'], q=4)
    return formatted_data, trial_data


def get_trial_type_data(trial_data, formatted_data, trial_type_num, key='cumsum ang vel'):
    trial_type_7 = trial_data[(trial_data['Trial type'] == trial_type_num)]['Trial num'].unique()
    type7_inds = formatted_data[np.isin(formatted_data['trial numbers'].values, trial_type_7)].index.values
    type_7_df = formatted_data.iloc[type7_inds]

    all_xs, all_ys, _ = plot_quantiles_formatted_data(type_7_df, key, sort_by='APE peaks',
                                                      plot_means=False, num_divisions=1)
    return all_xs, all_ys