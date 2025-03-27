from utils.post_processing_utils import get_all_experimental_records
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings, remove_manipulation_days
from utils.change_over_time_utils import get_valid_traces_movement_aligned
import os
import numpy as np
from set_global_params import processed_data_path, change_over_time_mice


# Saves out the files needed to plot change over time
data_root = processed_data_path + 'peak_analysis'
recording_site = 'Nacc'
mice = change_over_time_mice[recording_site]
side ='contra'
window_for_binning = 50 #50 is default for scatter, 200 for trace
mice = change_over_time_mice['Nacc']
for mouse_num, mouse in enumerate(mice):
    all_experiments = get_all_experimental_records()
    all_experiments = remove_exps_after_manipulations(all_experiments, [mouse])
    all = remove_manipulation_days(all_experiments)
    all_experiments = remove_unsuitable_recordings(all_experiments)
    experiments_to_process = all_experiments[
        (all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == recording_site)]
    dates = experiments_to_process['date'].values
    rolling_mean_x, rolling_mean_peaks, peak_trace_inds, rolling_mean_traces = get_valid_traces_movement_aligned(mouse, dates, window_around_mean=0.2, side=side, window_size=window_for_binning)
    saving_folder = os.path.join(data_root, mouse)
    filename = mouse + '_binned_' + str(window_for_binning) + '_average_then_peaks_peaks_movement_aligned_{}.npz'.format(side)
    save_filename = os.path.join(saving_folder, filename)
    np.savez(save_filename, rolling_mean_x=rolling_mean_x, rolling_mean_peaks=rolling_mean_peaks, rolling_mean_trace=rolling_mean_traces, peak_trace_inds=peak_trace_inds)



