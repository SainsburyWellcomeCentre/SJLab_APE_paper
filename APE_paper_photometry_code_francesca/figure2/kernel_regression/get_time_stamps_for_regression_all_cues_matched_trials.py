import numpy as np
import pickle
from utils.individual_trial_analysis_regression_all_cues import SessionEvents
import pandas as pd
import os
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings
from utils.kernel_regression.linear_regression_utils import get_first_x_sessions

from set_global_params import experiment_record_path, processed_data_path, mice_average_traces


def get_all_experimental_records():
    experiment_record = pd.read_csv(experiment_record_path)
    experiment_record['date'] = experiment_record['date'].astype(str)
    return experiment_record


def add_timestamps_to_aligned_data(experiments_to_add):
    for index, experiment in experiments_to_add.iterrows():
        data_folder = processed_data_path + experiment['mouse_id'] + '\\'
        saving_folder = processed_data_path + experiment['mouse_id'] + '\\linear_regression\\'
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)

        restructured_data_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'restructured_data.pkl'
        #trial_data = pd.read_pickle(data_folder + restructured_data_filename)
        dff_trace_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'smoothed_signal.npy'
        #dff = np.load(data_folder + dff_trace_filename)
        print(experiment['mouse_id'], experiment['date'])
        session_events = SessionEvents(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
        session_events.get_choice_events()
        session_events.get_cue_events()
        session_events.get_reward_events()

        choice_trial_nums = np.sort(np.concatenate(
            [session_events.choice_data.contra_data.trial_nums, session_events.choice_data.ipsi_data.trial_nums]))
        cue_trial_nums = np.sort(np.concatenate(
            [session_events.cue_data.low_cue_data.trial_nums, session_events.cue_data.high_cue_data.trial_nums]))
        common_trial_nums = np.intersect1d(cue_trial_nums, choice_trial_nums)

        session_events.choice_data.contra_data.filter_by_trial_nums(common_trial_nums)
        session_events.choice_data.ipsi_data.filter_by_trial_nums(common_trial_nums)
        session_events.cue_data.high_cue_data.filter_by_trial_nums(common_trial_nums)
        session_events.cue_data.low_cue_data.filter_by_trial_nums(common_trial_nums)

        aligned_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'behavioural_events_with_no_rewards_all_cues_matched_trials.p'
        save_filename = os.path.join(saving_folder, aligned_filename)
        pickle.dump(session_events, open(save_filename, "wb"))


def remove_manipulation_days(experiments):
    exemption_list = ['psychometric', 'state change medium cloud', 'value blocks', 'state change white noise', 'omissions and large rewards']
    exemptions = '|'.join(exemption_list)
    index_to_remove = experiments[np.logical_xor(experiments['include'] == 'no', experiments['experiment_notes'].str.contains(exemptions, na=False))].index
    cleaned_experiments = experiments.drop(index=index_to_remove)
    return cleaned_experiments


if __name__ == '__main__':
    site = 'Nacc'
    mouse_ids = mice_average_traces[site]
    experiment_record = pd.read_csv(experiment_record_path)
    experiment_record['date'] = experiment_record['date'].astype(str)
    good_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
    clean_experiments = remove_unsuitable_recordings(good_experiments)
    all_experiments_to_process = clean_experiments[
        (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
        drop=True)
    experiments_to_process = get_first_x_sessions(all_experiments_to_process)
    add_timestamps_to_aligned_data(experiments_to_process)

