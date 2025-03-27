import os
from utils.tracking_analysis.extract_movement_for_all_sessions_utils import *
from utils.post_processing_utils import get_all_experimental_records
from utils.post_processing_utils import remove_unsuitable_recordings
from set_global_params import change_over_time_mice, raw_tracking_path


recording_site = 'tail'
mice = change_over_time_mice[recording_site]
load_saved = False
for mouse in mice:
    all_experiments = get_all_experimental_records()
    all_experiments = remove_exps_after_manipulations(all_experiments, [mouse])
    all_experiments = remove_manipulation_days(all_experiments)
    all_experiments = remove_unsuitable_recordings(all_experiments)
    experiments_to_process = all_experiments[
        (all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == recording_site)]
    dates = experiments_to_process['date'].values[-4:]

    for date in dates:
        save_out_folder = os.path.join(raw_tracking_path, mouse, date)
        if not os.path.exists(save_out_folder):
            os.makedirs(save_out_folder)
        movement_param_file = os.path.join(save_out_folder, 'APE_tracking{}_{}.pkl'.format(mouse, date))
        if not os.path.isfile(movement_param_file) & (load_saved):
            quantile_data, trial_data = get_movement_properties_for_session(mouse, date)
            quantile_data.to_pickle(movement_param_file)