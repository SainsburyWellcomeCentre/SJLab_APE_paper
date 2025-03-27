import pickle
import os
from utils.individual_trial_analysis_utils import SessionDataPsychometric
import pandas as pd
from utils.post_processing_utils import remove_exps_after_manipulations, get_all_experimental_records
from set_global_params import experiment_record_path
from set_global_params import processed_data_path, bias_analysis_mice


def add_experiment_to_aligned_data(experiments_to_add):
    data_root = os.path.join(processed_data_path, 'for_psychometric')
    for index, experiment in experiments_to_add.iterrows():
        print(experiment['mouse_id'],' ', experiment['date'])
        saving_folder = os.path.join(data_root, experiment['mouse_id'])
        if not os.path.exists(saving_folder):
            os.makedirs(saving_folder)

        session_traces = SessionDataPsychometric(experiment['fiber_side'], experiment['recording_site'], experiment['mouse_id'], experiment['date'])
        session_traces.get_choice_responses()
        session_traces.get_cue_responses()
        session_traces.get_reward_responses()
        session_traces.get_outcome_responses()
        aligned_filename = experiment['mouse_id'] + '_' + experiment['date'] + '_' + 'aligned_traces_for_psychometric.p'
        save_filename = os.path.join(saving_folder, aligned_filename)
        pickle.dump(session_traces, open(save_filename, "wb"))


if __name__ == '__main__':
    site = 'Nacc'
    mouse_ids = bias_analysis_mice[site] # just examples
    experiment_record = pd.read_csv(experiment_record_path)
    experiment_record['date'] = experiment_record['date'].astype(str)
    clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
    all_experiments_to_process = clean_experiments[
        (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
        drop=True)

    date = '20201214' #Change this to the date of the psychometric experiment you want to process
    for mouse_id in mouse_ids:
        all_experiments = get_all_experimental_records()

        if (mouse_id =='all') & (date == 'all'):
            experiments_to_process = all_experiments
        elif (mouse_id == 'all') & (date != 'all'):
            experiments_to_process = all_experiments[all_experiments['date'] == date]
        elif (mouse_id != 'all') & (date == 'all'):
            experiments_to_process = all_experiments[all_experiments['mouse_id'] == mouse_id]
        elif (mouse_id != 'all') & (date != 'all'):
            experiments_to_process = all_experiments[(all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse_id)]
        add_experiment_to_aligned_data(experiments_to_process)

