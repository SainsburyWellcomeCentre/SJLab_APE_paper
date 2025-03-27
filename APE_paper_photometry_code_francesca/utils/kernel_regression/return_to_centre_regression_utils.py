import sklearn
from utils.kernel_regression.linear_regression_utils import *
import os
from scipy.signal import decimate
from set_global_params import processed_data_path


def get_first_x_sessions_reg_rtc(sorted_experiment_record, x=3):
    i = []
    inds = []
    for mouse in np.unique(sorted_experiment_record['mouse_id']):
        if mouse == 'SNL_photo57':
            y = 2
        else:
            y = x
        i.append(sorted_experiment_record[sorted_experiment_record['mouse_id'] == mouse][0:y].index)
        inds += range(0, y)
    flattened_i = [val for sublist in i for val in sublist]
    exps = sorted_experiment_record.loc[flattened_i].reset_index(drop=True)
    exps['session number'] = inds
    return exps


def run_regression_return_to_centre_one_mouse_one_session(mouse, date, duration_list, within_2sd_durations, all_trial_durations, sample_rate=10000, decimate_factor=100, window_size_seconds = 10, reg_type='_return_to_centre'):
    print('proccessing' + mouse + date)
    dlc_save_dir = os.path.join(processed_data_path, 'return_to_centre', mouse)
    if reg_type == '_return_to_centre' or reg_type == '_return_to_centre_trimmed_traces':
        time_stamp_save_file = '{}_{}_return_to_centre_movement_onset_times.npz'.format(mouse, date)
    elif reg_type == '_return_to_centre_300frames' or '_return_to_centre_300frames_long_turns':
        time_stamp_save_file = '{}_{}_return_to_centre_movement_onset_times_300frame_window_long_turns.npz'.format(mouse, date)
        return_to_centre_timestamps = np.load(os.path.join(dlc_save_dir, time_stamp_save_file))

    saving_folder = os.path.join(processed_data_path, mouse)
    events_folder = os.path.join(processed_data_path, mouse, 'linear_regression')
    dff_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
    dff = np.load(os.path.join(saving_folder, dff_trace_filename))

    rolling_zscored_dff = rolling_zscore(pd.Series(dff), window=window_size_seconds * sample_rate)
    downsampled_zscored_dff = decimate(
        decimate(rolling_zscored_dff[window_size_seconds * sample_rate:], int(decimate_factor / 10)),
        int(decimate_factor / 10))

    num_samples = downsampled_zscored_dff.shape[0]
    aligned_filename = mouse + '_' + date + '_' + 'behavioural_events_with_no_rewards_all_cues_matched_trials.p'
    save_filename = os.path.join(events_folder, aligned_filename)
    example_session_data = pickle.load(open(save_filename, "rb"))

    ipsi_choices = convert_behavioural_timestamps_into_samples(example_session_data.choice_data.ipsi_data.event_times,
                                                               window_size_seconds)
    contra_choices = convert_behavioural_timestamps_into_samples(
        example_session_data.choice_data.contra_data.event_times, window_size_seconds)
    high_cues = convert_behavioural_timestamps_into_samples(example_session_data.cue_data.high_cue_data.event_times,
                                                            window_size_seconds)
    low_cues = convert_behavioural_timestamps_into_samples(example_session_data.cue_data.low_cue_data.event_times,
                                                           window_size_seconds)
    rewards = convert_behavioural_timestamps_into_samples(example_session_data.reward_data.reward_data.event_times,
                                                          window_size_seconds)
    no_rewards = convert_behavioural_timestamps_into_samples(
        example_session_data.reward_data.no_reward_data.event_times, window_size_seconds)
    contra_reaction_times = example_session_data.choice_data.contra_data.reaction_times
    ipsi_reaction_times = example_session_data.choice_data.ipsi_data.reaction_times
    # Store data in the list
    for duration in ipsi_reaction_times:
        duration_list.append({'mouse': mouse, 'date': date, 'type': 'ipsi', 'duration': duration})
    for duration in contra_reaction_times:
        duration_list.append({'mouse': mouse, 'date': date, 'type': 'contra', 'duration': duration})

    if reg_type == '_return_to_centre' or reg_type == 'return_to_centre_trimmed_traces' or reg_type == '_return_to_centre_300frames' or reg_type == '_return_to_centre_300frames_long_turns':
        contra_returns = convert_behavioural_timestamps_into_samples(
            return_to_centre_timestamps['contra_movement_return'],
            window_size_seconds)
        ipsi_returns = convert_behavioural_timestamps_into_samples(return_to_centre_timestamps['ipsi_movement_return'],
                                                                   window_size_seconds)
        parameters = turn_timestamps_into_continuous(num_samples, high_cues, low_cues, ipsi_choices, contra_choices,
                                                    rewards, no_rewards, contra_returns, ipsi_returns)
    else:
        parameters = turn_timestamps_into_continuous(num_samples, high_cues, low_cues, ipsi_choices, contra_choices,
                                                    rewards, no_rewards)
    all_trial_starts = np.unique(np.concatenate([example_session_data.cue_data.high_cue_data.trial_starts,
                                                 example_session_data.cue_data.low_cue_data.trial_starts,
                                                 example_session_data.choice_data.contra_data.trial_starts,
                                                 example_session_data.choice_data.ipsi_data.trial_starts,
                                                 example_session_data.reward_data.no_reward_data.trial_starts,
                                                 example_session_data.reward_data.reward_data.trial_starts]))
    all_trial_ends = np.unique(np.concatenate(
        [example_session_data.cue_data.high_cue_data.trial_ends, example_session_data.cue_data.low_cue_data.trial_ends,
         example_session_data.choice_data.contra_data.trial_ends, example_session_data.choice_data.ipsi_data.trial_ends,
         example_session_data.reward_data.no_reward_data.trial_ends,
         example_session_data.reward_data.reward_data.trial_ends]))
    trial_durations = all_trial_ends - all_trial_starts
    trial_starts_samps = np.squeeze(convert_behavioural_timestamps_into_samples(all_trial_starts, window_size_seconds))
    trial_ends_samps = np.squeeze(convert_behavioural_timestamps_into_samples(all_trial_ends, window_size_seconds))

    trials_to_include = pd.DataFrame(
        {'trial starts': trial_starts_samps, 'trial ends': trial_ends_samps, 'durations': trial_durations})
    trials_to_remove = trials_to_include[
        trials_to_include['durations'] > np.mean(trial_durations) + 2 * np.std(trial_durations)].reset_index(drop=True)
    updated_durations = trials_to_include[
        trials_to_include['durations'] <= np.mean(trial_durations) + 2 * np.std(trial_durations)].reset_index(drop=True)['durations'].values
    all_trial_durations.append(trials_to_include['durations'].values.tolist())
    inds_to_go = []
    for ind, trial in trials_to_remove.iterrows():
        inds_to_go.append(slice(int(trial['trial starts']), int(trial['trial ends'])))
    ind = np.indices(downsampled_zscored_dff.shape)[0]
    rm = np.hstack([ind[i] for i in inds_to_go])
    trace_for_reg = np.take(downsampled_zscored_dff, sorted(set(ind) - set(rm)))
    params_for_reg = []
    for param in parameters:
        param_new = np.take(param, sorted(set(ind) - set(rm)))
        params_for_reg.append(param_new)
    if reg_type == '_return_to_centre' or reg_type == '_return_to_centre_trimmed_traces' or reg_type == '_return_to_centre_300frames' or reg_type == '_return_to_centre_300frames_long_turns':
        param_names = ['high cues', 'low cues', 'ipsi choices', 'contra choices', 'rewards', 'no rewards', 'contra returns',
                      'ipsi returns']
    else:
        param_names = ['high cues', 'low cues', 'ipsi choices', 'contra choices', 'rewards', 'no rewards']
    shifts, windows = make_shifts_for_params(param_names)
    param_inds, X = make_design_matrix_different_shifts(params_for_reg, shifts, windows)
    model = LinearRegression()
    results = model.fit(X, trace_for_reg)
    var_exp = results.score(X, trace_for_reg)
    print(var_exp)

    save_filename = mouse + '_' + date + '_'
    save_kernels_different_shifts_diff_reg_types(saving_folder + save_filename, param_names, params_for_reg, results, trace_for_reg,
                                  X.astype(int), shifts, windows, reg_type=reg_type)

    per_trial_exp_vars = get_exp_var_only_trials(trials_to_include, parameters, shifts, windows, param_names,
                                                 model, downsampled_zscored_dff, high_cues, low_cues, ipsi_choices, contra_choices, rewards, no_rewards, contra_returns, ipsi_returns)
    mean_per_trial_exp_var = np.mean(per_trial_exp_vars)
    print(mean_per_trial_exp_var)
    within_2sd_durations.append(updated_durations.tolist())
    return var_exp, duration_list, within_2sd_durations, all_trial_durations


def get_exp_var_only_trials(trials_to_include, parameters, shifts, windows, param_names, model, downsampled_zscored_dff, *args):
    param_inds_new, X_new = make_design_matrix_different_shifts(parameters, shifts, windows)
    exp_vars = []
    included_event_types = []
    pred_ys = []
    actual_traces = []
    for i in range(trials_to_include.shape[0]):
        start_ind = trials_to_include['trial starts'].loc[i]
        end_ind = trials_to_include['trial ends'].loc[i]
        event_types = [arg for arg in args]
        trial_events = []
        per_trial_event_names = []
        for ii, event_type_stamps in enumerate(event_types):
            events = event_type_stamps[np.where((event_type_stamps > start_ind) & (event_type_stamps <= end_ind))]
            if ((event_type_stamps > start_ind) & (event_type_stamps <= end_ind)).any():
                trial_events.append(events.tolist())
                per_trial_event_names.append(param_names[ii])

        sorted_events = np.sort([item for sublist in trial_events for item in sublist])
        if sorted_events.shape[0] >= 2:
            first_event = int(sorted_events[0] - 0.25 * 10000 / 100)
            last_event = int(sorted_events[-1] + 0.5 * 10000 / 100)
            test_X = X_new[first_event: last_event, :]
            pred_y = model.predict(test_X)
            pred_ys.append(pred_y)
            actual_traces.append(downsampled_zscored_dff[first_event: last_event])
            trial_exp_var = sklearn.metrics.explained_variance_score(downsampled_zscored_dff[first_event: last_event],
                                                                     pred_y)
            exp_vars.append(trial_exp_var)
            included_event_types.append(per_trial_event_names)
    return exp_vars


def run_regression_return_to_centre_one_mouse_one_session_trimmed_traces(mouse, date, sample_rate=10000, decimate_factor=100, window_size_seconds = 10, reg_type='_return_to_centre_trimmed_traces'):
    print('proccessing' + mouse + date)
    dlc_save_dir = os.path.join(processed_data_path, 'return_to_centre\\{}'.format(mouse))

    if reg_type == '_return_to_centre_trimmed_traces' or reg_type == '_return_to_centre':
        time_stamp_save_file = '{}_{}_return_to_centre_movement_onset_times.npz'.format(mouse, date)
        return_to_centre_timestamps = np.load(os.path.join(dlc_save_dir, time_stamp_save_file))
    elif reg_type == '_return_to_centre_trimmed_traces_300frames_long_turns':
        time_stamp_save_file = '{}_{}_return_to_centre_movement_onset_times_300frame_window_long_turns.npz'.format(mouse, date)
        return_to_centre_timestamps = np.load(os.path.join(dlc_save_dir, time_stamp_save_file))

    saving_folder = os.path.join(processed_data_path, mouse)
    events_folder = os.path.join(processed_data_path, mouse, 'linear_regression')
    dff_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
    dff = np.load(os.path.join(saving_folder, dff_trace_filename))

    rolling_zscored_dff = rolling_zscore(pd.Series(dff), window=window_size_seconds * sample_rate)
    downsampled_zscored_dff = decimate(
        decimate(rolling_zscored_dff[window_size_seconds * sample_rate:], int(decimate_factor / 10)),
        int(decimate_factor / 10))

    num_samples = downsampled_zscored_dff.shape[0]
    aligned_filename = mouse + '_' + date + '_' + 'behavioural_events_with_no_rewards_all_cues_matched_trials.p'
    save_filename = os.path.join(events_folder, aligned_filename)
    example_session_data = pickle.load(open(save_filename, "rb"))

    ipsi_choices = convert_behavioural_timestamps_into_samples(example_session_data.choice_data.ipsi_data.event_times,
                                                               window_size_seconds)
    contra_choices = convert_behavioural_timestamps_into_samples(
        example_session_data.choice_data.contra_data.event_times, window_size_seconds)
    high_cues = convert_behavioural_timestamps_into_samples(example_session_data.cue_data.high_cue_data.event_times,
                                                            window_size_seconds)
    low_cues = convert_behavioural_timestamps_into_samples(example_session_data.cue_data.low_cue_data.event_times,
                                                           window_size_seconds)
    rewards = convert_behavioural_timestamps_into_samples(example_session_data.reward_data.reward_data.event_times,
                                                          window_size_seconds)
    no_rewards = convert_behavioural_timestamps_into_samples(
        example_session_data.reward_data.no_reward_data.event_times, window_size_seconds)

    if reg_type == '_return_to_centre' or reg_type == '_return_to_centre_trimmed_traces_300frames_long_turns':
        contra_returns = convert_behavioural_timestamps_into_samples(
            return_to_centre_timestamps['contra_movement_return'],
            window_size_seconds)
        ipsi_returns = convert_behavioural_timestamps_into_samples(return_to_centre_timestamps['ipsi_movement_return'],
                                                                   window_size_seconds)

        parameters = turn_timestamps_into_continuous(num_samples, high_cues, low_cues, ipsi_choices, contra_choices,
                                                    rewards, no_rewards, contra_returns, ipsi_returns)
    else:
        parameters = turn_timestamps_into_continuous(num_samples, high_cues, low_cues, ipsi_choices, contra_choices,
                                                    rewards, no_rewards)
    all_trial_starts = np.unique(np.concatenate([example_session_data.cue_data.high_cue_data.trial_starts,
                                                 example_session_data.cue_data.low_cue_data.trial_starts,
                                                 example_session_data.choice_data.contra_data.trial_starts,
                                                 example_session_data.choice_data.ipsi_data.trial_starts,
                                                 example_session_data.reward_data.no_reward_data.trial_starts,
                                                 example_session_data.reward_data.reward_data.trial_starts]))
    all_trial_ends = np.unique(np.concatenate(
        [example_session_data.cue_data.high_cue_data.trial_ends, example_session_data.cue_data.low_cue_data.trial_ends,
         example_session_data.choice_data.contra_data.trial_ends, example_session_data.choice_data.ipsi_data.trial_ends,
         example_session_data.reward_data.no_reward_data.trial_ends,
         example_session_data.reward_data.reward_data.trial_ends]))
    trial_durations = all_trial_ends - all_trial_starts
    trial_starts_samps = np.squeeze(convert_behavioural_timestamps_into_samples(all_trial_starts, window_size_seconds))
    trial_ends_samps = np.squeeze(convert_behavioural_timestamps_into_samples(all_trial_ends, window_size_seconds))

    trials_to_include = pd.DataFrame(
        {'trial starts': trial_starts_samps, 'trial ends': trial_ends_samps, 'durations': trial_durations})
    trials_to_remove = trials_to_include[
        trials_to_include['durations'] > np.mean(trial_durations) + 2 * np.std(trial_durations)].reset_index(drop=True)
    inds_to_go = []
    for ind, trial in trials_to_remove.iterrows():
        inds_to_go.append(slice(int(trial['trial starts']), int(trial['trial ends'])))
    ind = np.indices(downsampled_zscored_dff.shape)[0]
    rm = np.hstack([ind[i] for i in inds_to_go])
    trace_for_reg = np.take(downsampled_zscored_dff, sorted(set(ind) - set(rm)))
    params_for_reg = []
    for param in parameters:
        param_new = np.take(param, sorted(set(ind) - set(rm)))
        params_for_reg.append(param_new)
    if reg_type == '_return_to_centre' or reg_type == '_return_to_centre_trimmed_traces' or reg_type == '_return_to_centre_trimmed_traces_300frames_long_turns':
        param_names = ['high cues', 'low cues', 'ipsi choices', 'contra choices', 'rewards', 'no rewards', 'contra returns',
                      'ipsi returns']
    else:
        param_names = ['high cues', 'low cues', 'ipsi choices', 'contra choices', 'rewards', 'no rewards']
    shifts, windows = make_shifts_for_params(param_names)
    param_inds, X = make_design_matrix_different_shifts(params_for_reg, shifts, windows)
    valid_rows = ~np.all(X == 0, axis=1)
    trimmed_X = X[valid_rows]
    trimmed_trace = trace_for_reg[valid_rows]
    model = LinearRegression()
    results = model.fit(trimmed_X, trimmed_trace)

    var_exp = results.score(trimmed_X, trimmed_trace)
    print(var_exp)

    save_filename = mouse + '_' + date + '_'
    save_kernels_different_shifts_diff_reg_types(saving_folder + save_filename, param_names, params_for_reg, results, trimmed_trace,
                                  trimmed_X.astype(int), shifts, windows, reg_type=reg_type)

    return var_exp


def run_regression_one_mouse_one_session_no_return_no_trim(experiment):
    mouse = experiment['mouse_id']
    date = experiment['date']
    print('proccessing' + mouse + date)
    saving_folder = os.path.join(processed_data_path, mouse)
    events_folder = os.path.join(processed_data_path, mouse, 'linear_regression')
    dff_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
    dff = np.load(os.path.join(saving_folder, dff_trace_filename))

    window_size_seconds = 10
    sample_rate = 10000
    decimate_factor = 100

    rolling_zscored_dff = rolling_zscore(pd.Series(dff), window=window_size_seconds * sample_rate)
    downsampled_zscored_dff = decimate(
        decimate(rolling_zscored_dff[window_size_seconds * sample_rate:], int(decimate_factor / 10)),
        int(decimate_factor / 10))

    num_samples = downsampled_zscored_dff.shape[0]
    aligned_filename = mouse + '_' + date + '_' + 'behavioural_events_with_no_rewards_all_cues_matched_trials.p'#'behavioural_events_with_no_rewards_added_matched_trials.py' #'behavioural_events_with_no_rewards_added_not_cleaned.py' #'behavioural_events_with_no_rewards_added.py'
    save_filename = os.path.join(events_folder, aligned_filename)
    example_session_data = pickle.load(open(save_filename, "rb"))

    ipsi_choices = convert_behavioural_timestamps_into_samples(example_session_data.choice_data.ipsi_data.event_times,
                                                               window_size_seconds)
    contra_choices = convert_behavioural_timestamps_into_samples(
        example_session_data.choice_data.contra_data.event_times, window_size_seconds)
    high_cues = convert_behavioural_timestamps_into_samples(example_session_data.cue_data.high_cue_data.event_times,
                                                            window_size_seconds)
    low_cues = convert_behavioural_timestamps_into_samples(example_session_data.cue_data.low_cue_data.event_times,
                                                           window_size_seconds)
    rewards = convert_behavioural_timestamps_into_samples(example_session_data.reward_data.reward_data.event_times,
                                                          window_size_seconds)
    no_rewards = convert_behavioural_timestamps_into_samples(
        example_session_data.reward_data.no_reward_data.event_times, window_size_seconds)

    parameters = turn_timestamps_into_continuous(num_samples, high_cues, low_cues, ipsi_choices, contra_choices,
                                                 rewards, no_rewards)

    all_trial_starts = np.unique(np.concatenate([example_session_data.cue_data.high_cue_data.trial_starts,
                                                 example_session_data.cue_data.low_cue_data.trial_starts,
                                                 example_session_data.choice_data.contra_data.trial_starts,
                                                 example_session_data.choice_data.ipsi_data.trial_starts,
                                                 example_session_data.reward_data.no_reward_data.trial_starts,
                                                 example_session_data.reward_data.reward_data.trial_starts]))
    all_trial_ends = np.unique(np.concatenate(
        [example_session_data.cue_data.high_cue_data.trial_ends, example_session_data.cue_data.low_cue_data.trial_ends,
         example_session_data.choice_data.contra_data.trial_ends, example_session_data.choice_data.ipsi_data.trial_ends,
         example_session_data.reward_data.no_reward_data.trial_ends,
         example_session_data.reward_data.reward_data.trial_ends]))
    trial_durations = all_trial_ends - all_trial_starts
    trial_starts_samps = np.squeeze(convert_behavioural_timestamps_into_samples(all_trial_starts, window_size_seconds))
    trial_ends_samps = np.squeeze(convert_behavioural_timestamps_into_samples(all_trial_ends, window_size_seconds))

    trials_to_include = pd.DataFrame({'trial starts': trial_starts_samps, 'trial ends': trial_ends_samps, 'durations': trial_durations})
    trials_to_remove = trials_to_include[
        trials_to_include['durations'] > np.mean(trial_durations) + 2 * np.std(trial_durations)].reset_index(drop=True)
    inds_to_go = []
    for ind, trial in trials_to_remove.iterrows():
        inds_to_go.append(slice(int(trial['trial starts']), int(trial['trial ends'])))
    ind = np.indices(downsampled_zscored_dff.shape)[0]
    rm = np.hstack([ind[i] for i in inds_to_go])
    trace_for_reg = np.take(downsampled_zscored_dff, sorted(set(ind) - set(rm)))
    params_for_reg = []
    for param in parameters:
        param_new = np.take(param, sorted(set(ind) - set(rm)))
        params_for_reg.append(param_new)
    param_names = ['high cues', 'low cues', 'ipsi choices', 'contra choices', 'rewards', 'no rewards']
    shifts, windows = make_shifts_for_params(param_names)
    param_inds, X = make_design_matrix_different_shifts(params_for_reg, shifts, windows)
    results = LinearRegression().fit(X, trace_for_reg)
    score = results.score(X, trace_for_reg)
    print('without trimming ', score)

    shifts, windows = make_shifts_for_params(param_names)
    param_inds, X = make_design_matrix_different_shifts(params_for_reg, shifts, windows)
    valid_rows = ~np.all(X == 0, axis=1)
    trimmed_X = X[valid_rows]
    trimmed_trace = trace_for_reg[valid_rows]
    model = LinearRegression()
    results = model.fit(trimmed_X, trimmed_trace)

    var_exp = results.score(trimmed_X, trimmed_trace)
    print('with trimming ', var_exp)
    save_filename = mouse + '_' + date + '_only_trials_'
    save_kernels_different_shifts_diff_reg_types(saving_folder + save_filename, param_names, parameters, results, trimmed_trace,
                                  trimmed_X.astype(int), shifts, windows, reg_type='_trimmed_no_return_to_centre')
    return score, var_exp


def calculate_prop_data_covered_by_design_matrix(experiment):
    mouse = experiment['mouse_id']
    date = experiment['date']
    print('proccessing' + mouse + date)
    saving_folder = processed_data_path + mouse + '\\'
    events_folder = processed_data_path + mouse + '\\linear_regression\\'
    dff_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
    dff = np.load(saving_folder + dff_trace_filename)

    window_size_seconds = 10
    sample_rate = 10000
    decimate_factor = 100

    rolling_zscored_dff = rolling_zscore(pd.Series(dff), window=window_size_seconds * sample_rate)
    downsampled_zscored_dff = decimate(
        decimate(rolling_zscored_dff[window_size_seconds * sample_rate:], int(decimate_factor / 10)),
        int(decimate_factor / 10))

    num_samples = downsampled_zscored_dff.shape[0]
    aligned_filename = mouse + '_' + date + '_' + 'behavioural_events_with_no_rewards_all_cues_matched_trials.p'#'behavioural_events_with_no_rewards_added_matched_trials.py' #'behavioural_events_with_no_rewards_added_not_cleaned.py' #'behavioural_events_with_no_rewards_added.py'
    save_filename = events_folder + aligned_filename
    example_session_data = pickle.load(open(save_filename, "rb"))

    ipsi_choices = convert_behavioural_timestamps_into_samples(example_session_data.choice_data.ipsi_data.event_times,
                                                               window_size_seconds)
    contra_choices = convert_behavioural_timestamps_into_samples(
        example_session_data.choice_data.contra_data.event_times, window_size_seconds)
    high_cues = convert_behavioural_timestamps_into_samples(example_session_data.cue_data.high_cue_data.event_times,
                                                            window_size_seconds)
    low_cues = convert_behavioural_timestamps_into_samples(example_session_data.cue_data.low_cue_data.event_times,
                                                           window_size_seconds)
    rewards = convert_behavioural_timestamps_into_samples(example_session_data.reward_data.reward_data.event_times,
                                                          window_size_seconds)
    no_rewards = convert_behavioural_timestamps_into_samples(
        example_session_data.reward_data.no_reward_data.event_times, window_size_seconds)

    parameters = turn_timestamps_into_continuous(num_samples, high_cues, low_cues, ipsi_choices, contra_choices,
                                                 rewards, no_rewards)

    all_trial_starts = np.unique(np.concatenate([example_session_data.cue_data.high_cue_data.trial_starts,
                                                 example_session_data.cue_data.low_cue_data.trial_starts,
                                                 example_session_data.choice_data.contra_data.trial_starts,
                                                 example_session_data.choice_data.ipsi_data.trial_starts,
                                                 example_session_data.reward_data.no_reward_data.trial_starts,
                                                 example_session_data.reward_data.reward_data.trial_starts]))
    all_trial_ends = np.unique(np.concatenate(
        [example_session_data.cue_data.high_cue_data.trial_ends, example_session_data.cue_data.low_cue_data.trial_ends,
         example_session_data.choice_data.contra_data.trial_ends, example_session_data.choice_data.ipsi_data.trial_ends,
         example_session_data.reward_data.no_reward_data.trial_ends,
         example_session_data.reward_data.reward_data.trial_ends]))
    trial_durations = all_trial_ends - all_trial_starts
    trial_starts_samps = np.squeeze(convert_behavioural_timestamps_into_samples(all_trial_starts, window_size_seconds))
    trial_ends_samps = np.squeeze(convert_behavioural_timestamps_into_samples(all_trial_ends, window_size_seconds))

    trials_to_include = pd.DataFrame({'trial starts': trial_starts_samps, 'trial ends': trial_ends_samps, 'durations': trial_durations})
    trials_to_remove = trials_to_include[
        trials_to_include['durations'] > np.mean(trial_durations) + 2 * np.std(trial_durations)].reset_index(drop=True)
    inds_to_go = []
    for ind, trial in trials_to_remove.iterrows():
        inds_to_go.append(slice(int(trial['trial starts']), int(trial['trial ends'])))
    ind = np.indices(downsampled_zscored_dff.shape)[0]
    rm = np.hstack([ind[i] for i in inds_to_go])
    trace_for_reg = np.take(downsampled_zscored_dff, sorted(set(ind) - set(rm)))
    params_for_reg = []
    for param in parameters:
        param_new = np.take(param, sorted(set(ind) - set(rm)))
        params_for_reg.append(param_new)
    param_names = ['high cues', 'low cues', 'ipsi choices', 'contra choices', 'rewards', 'no rewards']


    shifts, windows = make_shifts_for_params(param_names)
    param_inds, X = make_design_matrix_different_shifts(params_for_reg, shifts, windows)
    valid_rows = ~np.all(X == 0, axis=1)
    trimmed_X = X[valid_rows]
    perc_covered = trimmed_X.shape[0]/X.shape[0] * 100
    return perc_covered