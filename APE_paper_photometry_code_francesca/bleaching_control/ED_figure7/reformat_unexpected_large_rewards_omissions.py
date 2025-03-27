import os
import pandas as pd
from set_global_params import processed_data_path, large_reward_omission_mice
from utils.post_processing_utils import open_experiment, CustomAlignedData, get_all_experimental_records
from utils.large_reward_omission_utils import get_traces_and_reward_types

site = 'tail' # or 'Nacc'
exp_name = 'large_rewards_omissions'
mice = large_reward_omission_mice[site]
processed_data_dir = os.path.join(processed_data_path, 'large_rewards_omissions_data')
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)
all_experiments = get_all_experimental_records()
processed_data_file = os.path.join(processed_data_dir, 'all_{}_reward_change_data.pkl'.format(site))

for mouse_num, mouse_id in enumerate(mice):
    sessions = all_experiments[
        (all_experiments['mouse_id'] == mouse_id) & (all_experiments['experiment_notes'] == 'omissions and large rewards')][
        'date'].values
    for session_idx, date in enumerate(sessions):
        experiment_to_process = all_experiments[(all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse_id)]
        session_data, behavioural_data = open_experiment(experiment_to_process)
        params = {'state_type_of_interest': 5,
                  'outcome': 2,
                  'last_outcome': 0,  # NOT USED CURRENTLY
                  'no_repeats': 0,
                  'last_response': 0,
                  'align_to': 'Time end',
                  'instance': -1,
                  'plot_range': [-6, 6],
                  'first_choice_correct': 0,
                  'cue': 'None'}
        data = CustomAlignedData(session_data, params, peak_quantification=False)
        session_reward_type_data = get_traces_and_reward_types(data, behavioural_data)
        session_reward_type_data['mouse'] = mouse_id
        session_reward_type_data['session'] = date
        if (session_idx > 0) or (mouse_num > 0):
            all_reward_type_data = pd.concat([all_reward_type_data, session_reward_type_data], ignore_index=True)
        else:
            all_reward_type_data = session_reward_type_data

all_reward_type_data.to_pickle(processed_data_file)

