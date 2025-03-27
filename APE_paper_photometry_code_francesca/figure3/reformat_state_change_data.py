from utils.post_processing_utils import *
from set_global_params import processed_data_path, state_change_mice
import os

site = 'Nacc' # or 'Nacc'
processed_data_dir = os.path.join(processed_data_path, 'state_change_data')
state_change_data_file = os.path.join(processed_data_dir, 'state_change_data_{}_mice_only_correct_py36.p'.format(site))
trial_num_of_switch = 150
mice = state_change_mice[site]
if site == 'tail':
    state_to_align_to = 5
elif site == 'Nacc':
    state_to_align_to = 3
else:
    print('recording site is not tail or Nacc')
for mouse_num, mouse_id in enumerate(mice):
    state_change_data = {}
    exp_type = 'state change white noise'
    all_experiments = get_all_experimental_records()
    all_experiments = remove_unsuitable_recordings(all_experiments)
    experiment_to_process = all_experiments[(all_experiments['experiment_notes'] == exp_type) & (all_experiments['mouse_id'] == mouse_id)]
    session_data = open_experiment(experiment_to_process)[0]

    params = {'state_type_of_interest': state_to_align_to, # 3 for nacc, 5 for tail
        'outcome': 1,
        'last_outcome': 0,  # NOT USED CURRENTLY
        'no_repeats' : 0,
        'last_response': 0,
        'align_to' : 'Time start',
        'instance': -1,
        'plot_range': [-6, 6],
        'first_choice_correct': 0,
         'cue': 'None'}
    aligned_data = CustomAlignedData(session_data, params)

    state_change_data['trial number'] = aligned_data.contra_data.trial_nums
    clean_peaks = []
    for peak in aligned_data.contra_data.trial_peaks:
        if type(peak) == np.ndarray:
            clean_peaks.append(np.nan)
        else:
            clean_peaks.append(peak)
    state_change_data['peak size'] = clean_peaks
    state_change_dataFrame = pd.DataFrame(state_change_data)
    list_traces = [aligned_data.contra_data.sorted_traces[i,:] for i in range(aligned_data.contra_data.sorted_traces.shape[0])]
    state_change_dataFrame['traces'] = pd.Series(list_traces, index=state_change_dataFrame.index)

    state_change_dataFrame['trial type'] = np.where(state_change_dataFrame['trial number'] < trial_num_of_switch, 'pre', 'post')
    state_change_dataFrame['mouse'] = mouse_id
    if mouse_num > 0:
        all_state_change_data = pd.concat([all_state_change_data, state_change_dataFrame], ignore_index=True)
    else:
        all_state_change_data = state_change_dataFrame

all_state_change_data.to_pickle(state_change_data_file)


