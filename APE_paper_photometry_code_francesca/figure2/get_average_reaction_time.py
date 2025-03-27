import pandas as pd
import numpy as np
from utils.reaction_time_utils import get_bpod_trial_nums_per_session
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings, get_first_x_sessions, get_all_experimental_records, add_experiment_to_aligned_data
from set_global_params import experiment_record_path, mice_average_traces, processed_data_path


def get_first_3_session_reaction_times_for_experiments(site='tail'):
    mouse_ids = mice_average_traces[site]
    experiment_record = pd.read_csv(experiment_record_path)

    experiments_to_process = get_first_x_sessions(experiment_record, mouse_ids, site).reset_index(drop=True)
    reaction_times = []
    for index, experiment in experiments_to_process.iterrows():
        mouse = experiment['mouse_id']
        date = experiment.date
        saving_folder = processed_data_path + mouse + '\\'
        restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'

        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        red_trial_data = trial_data[trial_data['State type'] == 5]
        reaction_times.append(np.mean(red_trial_data['Time end'] - red_trial_data['Time start']))

    experiments_to_process['average reaction time'] = reaction_times
    return experiments_to_process


tail_performance_df = get_first_3_session_reaction_times_for_experiments(site='tail')
nacc_performance_df = get_first_3_session_reaction_times_for_experiments(site='Nacc')
all_experiments = pd.concat([tail_performance_df, nacc_performance_df])
mean_min_per_mouse = np.mean(all_experiments.groupby(['mouse_id'])['average reaction time'].apply(np.min).reset_index())
mean_max_per_mouse = np.mean(all_experiments.groupby(['mouse_id'])['average reaction time'].apply(np.max).reset_index())
mean_reaction_time = np.mean(all_experiments.groupby(['mouse_id'])['average reaction time'].apply(np.mean).reset_index())
print('min: {}, max: {}, avg: {}'.format(mean_min_per_mouse[0], mean_max_per_mouse[0], mean_reaction_time[0]))

mean_min_per_mouse = np.mean(tail_performance_df.groupby(['mouse_id'])['average reaction time'].apply(np.min).reset_index())
mean_max_per_mouse = np.mean(tail_performance_df.groupby(['mouse_id'])['average reaction time'].apply(np.max).reset_index())
mean_reaction_time = np.mean(tail_performance_df.groupby(['mouse_id'])['average reaction time'].apply(np.mean).reset_index())
print('tail only: min: {}, max: {}, avg: {}'.format(mean_min_per_mouse[0], mean_max_per_mouse[0], mean_reaction_time[0]))

mean_min_per_mouse = np.mean(nacc_performance_df.groupby(['mouse_id'])['average reaction time'].apply(np.min).reset_index())
mean_max_per_mouse = np.mean(nacc_performance_df.groupby(['mouse_id'])['average reaction time'].apply(np.max).reset_index())
mean_reaction_time = np.mean(nacc_performance_df.groupby(['mouse_id'])['average reaction time'].apply(np.mean).reset_index())
print('nacc only: min: {}, max: {}, avg: {}'.format(mean_min_per_mouse[0], mean_max_per_mouse[0], mean_reaction_time[0]))
