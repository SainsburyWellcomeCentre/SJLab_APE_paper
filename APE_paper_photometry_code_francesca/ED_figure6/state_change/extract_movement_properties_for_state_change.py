import os
from utils.tracking_analysis.extract_movement_for_all_sessions_utils import *
from utils.post_processing_utils import get_all_experimental_records
from utils.post_processing_utils import remove_unsuitable_recordings
from set_global_params import state_change_mice, raw_tracking_path



# for state change
recording_site = 'tail'
mice = state_change_mice[recording_site]
protocol = 'State_Change_Two_Alternative_Choice'
exp_name = 'state change white noise'


load_saved = False
for mouse in mice:
    all_experiments = get_all_experimental_records()
    all_experiments = remove_unsuitable_recordings(all_experiments)
    experiments_to_process = all_experiments[
        (all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == recording_site) & (all_experiments['experiment_notes'] == exp_name)] 
    dates = experiments_to_process['date'].values[-4:]

    for date in dates:
        save_out_folder = os.path.join(raw_tracking_path, mouse, date)
        if not os.path.exists(save_out_folder):
            os.makedirs(save_out_folder)
        movement_param_file = os.path.join(save_out_folder, 'APE_tracking{}_{}.pkl'.format(mouse, date))
        if not os.path.isfile(movement_param_file) & (load_saved):
            quantile_data, trial_data = get_movement_properties_for_session(mouse, date, protocol=protocol, multi_session=False)
            quantile_data.to_pickle(movement_param_file)