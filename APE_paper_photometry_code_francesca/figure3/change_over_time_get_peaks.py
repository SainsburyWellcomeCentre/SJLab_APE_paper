from utils.post_processing_utils import get_all_experimental_records
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings, remove_manipulation_days
from utils.change_over_time_utils import get_valid_traces
import os
import numpy as np
from set_global_params import processed_data_path, change_over_time_mice


# Saves out the files needed to plot change over time
save_path = os.path.join(processed_data_path, 'peak_analysis')
recording_site = 'tail'
mice = change_over_time_mice[recording_site]
side = 'contra'
align_to = 'movement'  # movement for tail in fig3, cue for Nacc in fig 3
window_for_binning = 50  # 50 is default for scatter, 200 for trace
for mouse_num, mouse in enumerate(mice):
    all_experiments = get_all_experimental_records()
    all_experiments = remove_exps_after_manipulations(all_experiments, [mouse])
    all = remove_manipulation_days(all_experiments)
    all_experiments = remove_unsuitable_recordings(all_experiments)
    experiments_to_process = all_experiments[
        (all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == recording_site)]
    dates = experiments_to_process['date'].values
    rolling_mean_x, rolling_mean_peaks, peak_trace_inds, rolling_mean_traces = get_valid_traces(mouse, dates, window_around_mean=0.2, recording_site=recording_site, side=side, window_size=window_for_binning, align_to=align_to)
    saving_folder = os.path.join(save_path, mouse)
    if (recording_site == 'tail') and (align_to == 'movement') and (side == 'contra'): # tail defaults for fig 3
        filename = mouse + '_binned_' + str(window_for_binning) + '_average_then_peaks_peaks.npz'
    elif (recording_site == 'Nacc') and (align_to == 'cue') and (side == 'contra'): # nacc defaults for fig 3
        filename = mouse + '_binned_' + str(window_for_binning) + '_average_then_peaks_peaks.npz'
    else: # more flexible for if you want to align to different things (e.g. change_over_time_plots_VS_reward.py)
        filename = mouse + '_binned_' + str(window_for_binning) + '_average_then_peaks_peaks_{}_aligned_to_{}.npz'.format(side, align_to)
    save_filename = os.path.join(saving_folder, filename)
    np.savez(save_filename, rolling_mean_x=rolling_mean_x, rolling_mean_peaks=rolling_mean_peaks, rolling_mean_trace=rolling_mean_traces, peak_trace_inds=peak_trace_inds)



