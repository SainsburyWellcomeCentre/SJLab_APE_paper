from utils.large_reward_omission_utils import make_example_traces_plot, get_unexpected_reward_change_data_for_site, compare_peaks_across_trial_types
from utils.plotting_visuals import set_plotting_defaults
from set_global_params import reproduce_figures_path, value_change_example_mice, processed_data_path
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import decimate
import peakutils
import pandas as pd
import os

"""
reformat_value_change.py saves much too big files for easy distribution 
- it's both too high res and more time points than we'll ever need
so here we reduce to what is necessary for the figures
"""
sites = ['Nacc', 'tail']
num_trials_to_look_at = 50
min_num_trials = 70
for site in sites:
    exp_name = 'value_change'
    # load in output of reformat_value_change.py
    processed_data_dir = os.path.join(processed_data_path, 'value_change_data')
    block_data_file = os.path.join(processed_data_dir, exp_name + '_' + site + '.p')
    all_reward_block_data = pd.read_pickle(block_data_file)

    # for group data just save out the mean trace per trial type per mouse
    sorted_data = all_reward_block_data.sort_values(['mouse', 'session', 'trial number']).reset_index(drop=True)
    sorted_data['block switches'] = sorted_data['block number'] - sorted_data['block number'].shift()
    sorted_data['new sessions'] = sorted_data['session'].ne(sorted_data['session'].shift().bfill()).astype(int)
    sorted_data.iloc[0, sorted_data.columns.get_loc('new sessions')] = 1
    sorted_data.loc[sorted_data['new sessions'] == 1, 'block switches'] = 1
    block_switch_inds = sorted_data.loc[sorted_data['block switches'] != 0].reset_index(drop=True)
    traces = []
    peaks = []
    trial_nums = []
    block_inds = []
    rel_reward_amounts = []
    reward_amounts = []
    mouse_ids = []
    session_ids = []
    for block_num, block in block_switch_inds.iterrows():
        mouse = block['mouse']
        session = block['session']
        block_id = block['block number']
        all_session_trials = sorted_data[(sorted_data['mouse'] == mouse) & (sorted_data['session'] == session)]
        all_block_trials = all_session_trials[all_session_trials['block number'] == block_id]
        if block_id == 0 or all_block_trials.shape[0] >= min_num_trials:
            last_trials_of_block = all_block_trials[-num_trials_to_look_at:]
            avg_trace = last_trials_of_block.groupby(['mouse', 'contra reward amount'])['traces'].apply(np.mean).values[0]
            decimated = decimate(avg_trace[int(len(avg_trace)/2):], 10)
            peak_idx = peakutils.indexes(decimated)[0]
            peak = decimated[peak_idx]
            traces.append(decimated)
            peaks.append(peak)
            trial_nums.append(last_trials_of_block['trial number'].values)
            rel_reward_amounts.append(last_trials_of_block['relative reward amount'].values[0])
            reward_amounts.append(last_trials_of_block['contra reward amount'].values[0])
            block_inds.append(last_trials_of_block.index.values[0])
            mouse_ids.append(mouse)
            session_ids.append(session)

    avg_block_data = {}
    avg_block_data['block id'] = block_inds
    avg_block_data['peaks'] = peaks
    avg_block_data['relative reward amount'] = rel_reward_amounts
    avg_block_data['contra reward amount'] = reward_amounts
    avg_block_data['mouse'] = mouse_ids
    avg_block_data['session'] = session_ids
    avg_block_dataf = pd.DataFrame(avg_block_data)
    avg_block_dataf['avg traces'] = pd.Series(traces, index=avg_block_dataf.index)
    df_for_plot = avg_block_dataf.groupby(['mouse', 'session', 'relative reward amount', 'contra reward amount'])['peaks'].apply(np.mean)
    df_for_plot = df_for_plot.reset_index()

    # save out downsampled data and peaks
    repro_path = os.path.join(reproduce_figures_path, 'ED_fig7', 'value_change')
    if not os.path.exists(repro_path):
        os.makedirs(repro_path)
    repro_file = os.path.join(repro_path, f'value_change_downsampled_traces_peaks_{site}.pkl')
    df_for_plot.to_pickle(repro_file)

    # downsample and save out example mouse
    mouse_name = value_change_example_mice[site]
    time_points = all_reward_block_data['time points'].iloc[0]
    all_trials = all_reward_block_data[(all_reward_block_data['mouse'] == mouse_name)]
    all_trials_clipped = all_trials.copy()
    # we don't need -8:8 seconds. let's try [-2:2)
    time_window_size = 2
    bool_idx = (time_points < time_window_size) & (time_points >= -time_window_size)
    all_trials_clipped['time points'] = all_trials['time points'].map(np.array).map(lambda x: decimate(x[bool_idx], q=60)) # used to be q=10 but need to downsample for csvs
    all_trials_clipped['traces'] = all_trials['traces'].map(np.array).map(lambda x: decimate(x[bool_idx], q=60)) # used to be q=10 but need to downsample for csvs
    repro_example_file = os.path.join(repro_path, f'value_change_downsampled_traces_example_{site}_{mouse_name}.pkl')
    all_trials_clipped.to_pickle(repro_example_file)

