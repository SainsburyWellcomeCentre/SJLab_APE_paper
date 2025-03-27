from utils.kernel_regression.linear_regression_utils import *
import gc
import os
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings
from set_global_params import experiment_record_path, processed_data_path, mice_average_traces
from scipy.signal import decimate

site = 'Nacc'
mouse_ids = mice_average_traces[site]

experiment_record = pd.read_csv(experiment_record_path)
experiment_record['date'] = experiment_record['date'].astype(str)
good_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
clean_experiments = remove_unsuitable_recordings(good_experiments)
all_experiments_to_process = clean_experiments[(clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(drop=True)
experiments_to_process = get_first_x_sessions(all_experiments_to_process)
for index, experiment in experiments_to_process.iterrows():
    mouse = experiment['mouse_id']
    date = experiment['date']
    print('proccessing' + mouse + date)
    saving_folder = os.path.join(processed_data_path, mouse)
    events_folder = os.path.join(processed_data_path, mouse, 'linear_regression')
    restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
    trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
    # loads demodulated and smoothed photometry (produced by data_preprocessing/preprocessing.py, but these files are provided)
    dff_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
    dff = np.load(os.path.join(saving_folder,dff_trace_filename))

    window_size_seconds = 10
    sample_rate = 10000
    decimate_factor = 100
    window_min = -0.5 * 10000 / 100
    window_max = 1.5 * 10000 / 100

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
    # we remove trials where the mice have lost interest (the behaviour is self paced so this does happen a bit early in training)
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
    print(results.score(X, trace_for_reg))

    save_filename = mouse + '_' + date + '_'

    save_kernels_different_shifts(os.path.join(saving_folder, save_filename), param_names, params_for_reg, results, trace_for_reg, X.astype(int), shifts, windows)
    gc.collect()