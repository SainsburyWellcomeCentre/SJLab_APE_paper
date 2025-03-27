from utils.large_reward_omission_utils import make_example_traces_plot, get_unexpected_reward_change_data_for_site, compare_peaks_across_trial_types
from utils.plotting_visuals import set_plotting_defaults
from set_global_params import reproduce_figures_path, large_reward_omission_example_mice
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import decimate
import peakutils
import pandas as pd
import os

"""
reformat_unexpected_large_rewards_omissions.py saves much too big files for easy distribution
- it's both too high res and more time points than we'll ever need
so here we reduce to what is necessary for the figures
"""
sites = ['Nacc', 'tail']

for site in sites:
    # downsample traces and save out downsampled traces from time of reward/ omission only (to make files smaller)
    site_data = get_unexpected_reward_change_data_for_site(site)
    avg_traces = site_data.groupby(['mouse', 'reward'])['traces'].apply(np.mean)
    # takes from time of reward/ omisison only
    decimated = [decimate(trace[int(len(trace)/2):], 10) for trace in avg_traces]
    avg_traces = avg_traces.reset_index()
    avg_traces['decimated'] = pd.Series([_ for _ in decimated])
    first_peak_ids = [peakutils.indexes(i)[0] for i in avg_traces['decimated']]
    avg_traces['peakidx'] = first_peak_ids
    # we get DA reponses slightly differently here than for cue/ movement
    # cue/ movement responses are normally peaks (increases in signal from the previous timepoints)
    # Because we have omissions here there are dips (you see it in the average traces)
    # so it's not appropriate to use peakutils in the way we use it for cue/ movement responses
    # We went for the easiest option of just finding the mean of the initial DA response
    peaks = [np.mean(trace[:600]) for idx, trace in zip(first_peak_ids, avg_traces['decimated'])]
    avg_traces['peak'] = peaks
    avg_traces.set_index(['mouse', 'reward'])
    repro_path = os.path.join(reproduce_figures_path, 'ED_fig7', 'omissions_large_rewards')
    if not os.path.exists(repro_path):
        os.makedirs(repro_path)
    repro_file = os.path.join(repro_path, f'omissions_large_rewards_downsampled_traces_peaks_{site}.pkl')
    avg_decimated_traces = avg_traces.drop(columns='traces')
    avg_decimated_traces.to_pickle(repro_file)

    # downsample and save out example mouse
    mouse_name = large_reward_omission_example_mice[site]
    time_points = site_data['time points'].reset_index(drop=True)[0]
    all_trials = site_data[(site_data['mouse'] == mouse_name)]
    all_trials_clipped = all_trials.copy()
    # we don't need -8:8 seconds. let's try [-2:2)
    time_window_size = 2
    bool_idx = (time_points < time_window_size) & (time_points >= -time_window_size)
    all_trials_clipped['time points'] = all_trials['time points'].map(np.array).map(lambda x: decimate(x[bool_idx], q=60)) # used to be q=10 but need to downsample for csvs
    all_trials_clipped['traces'] = all_trials['traces'].map(np.array).map(lambda x: decimate(x[bool_idx], q=60)) # used to be q=10 but need to downsample for csvs
    repro_example_file = os.path.join(repro_path, f'omissions_large_rewards_downsampled_traces_example_{site}_{mouse_name}.pkl')
    all_trials_clipped.to_pickle(repro_example_file)

