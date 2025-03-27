import peakutils
from matplotlib import colors, cm
from scipy.signal import decimate
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns
from scipy import stats
from utils.post_processing_utils import open_one_experiment
from set_global_params import processed_data_path, value_change_example_mice, reproduce_figures_path, spreadsheet_path
from utils.value_change_utils import CustomAlignedDataRewardBlocks, get_all_experimental_records, plot_mean_trace_for_condition, get_block_change_info, add_traces_and_peaks, one_session_get_block_changes
from utils.plotting import output_significance_stars_from_pval
from utils.stats import cohen_d_paired

def get_site_data_all_mice(site):
    """
    Loads the reformatted data for all mice for a given recording site
    Args:
        site (str): Nacc or tail

    Returns:
        all_reward_block_data (pd.dataframe): behavioural and photometry data for value change experiment
        all_time_points (np.array): time points for plotting trace (x-axis)
    """
    exp_name = 'value_change'
    processed_data_dir = os.path.join(processed_data_path, 'value_change_data')
    block_data_file = os.path.join(processed_data_dir, exp_name + '_' + site + '.p')
    all_reward_block_data = pd.read_pickle(block_data_file)
    all_time_points = all_reward_block_data['time points'].iloc[0]
    return all_reward_block_data, all_time_points


def make_example_plot(site):
    """
    Makes example traces plot for value block change experiment
    Args:
        site (str): recording site

    Returns:

    """
    mouse_name = value_change_example_mice[site]
    repro_path = os.path.join(reproduce_figures_path, 'ED_fig7', 'value_change')
    repro_example_file = os.path.join(repro_path, f'value_change_downsampled_traces_example_{site}_{mouse_name}.pkl')
    data = pd.read_pickle(repro_example_file)
    time_points = data['time points'].reset_index(drop=True)[0]
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2))
    plot_mean_trace_for_condition(ax, site, data, time_points,
                                  'relative reward amount', error_bar_method='sem', save_location=None)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc='lower left', bbox_to_anchor=(0.7, 0.8),
                borderaxespad=0, frameon=False, prop={'size': 6 })
    plt.tight_layout()


def make_group_data_plot(all_reward_block_data, colour):
    """
    Makes the group data plot for different value blocks of trials showing the change in response size
    Args:
        all_reward_block_data (pd.dataframe): behavioural and photometry data for value change experiment

    Returns:

    """
    peak_size_data = find_peaks_for_value_blocks(all_reward_block_data)
    plotted_points = plot_group_data(peak_size_data, colour)
    return plotted_points


def find_peaks_for_value_blocks(df_for_plot):
    """
    Identifies the response change for different value blocks for each mouse
    Args:
        df_for_plot (pd.dataframe): behavioural and photometry data for value change experiment at mean per block level

    Returns:
        df_for_plot1 (pd.dataframe): peak change per mouse for different relative value blocks
    """
    mice = []
    reward_changes = []
    cue_changes = []
    session_ids = []

    first_blocks = df_for_plot[df_for_plot['relative reward amount'] == 0]
    for first_block_ind, first_block in first_blocks.iterrows():
        mouse = first_block['mouse']
        session = first_block['session']
        session_blocks = df_for_plot[np.logical_and(df_for_plot['mouse'] == mouse, df_for_plot['session'] == session)]
        other_block = session_blocks.loc[session_blocks['relative reward amount'] != 0]
        if other_block.shape[0] > 0:
            mice.append(mouse)
            reward_changes.append(other_block['relative reward amount'].values[0])
            cue_change = (other_block['peaks'] - first_block['peaks']).values[0]
            cue_changes.append(cue_change)
            session_ids.append(session)
    diff_block_data = {}
    diff_block_data['mouse'] = mice
    diff_block_data['reward size change'] = reward_changes
    diff_block_data['cue size change'] = cue_changes
    diff_block_data['session'] = session_ids
    diff_block_dataf = pd.DataFrame(diff_block_data)

    df_for_plot1 = diff_block_dataf.groupby(['mouse', 'reward size change'])['cue size change'].apply(np.mean)
    df_for_plot1 = df_for_plot1.reset_index()
    return df_for_plot1


def plot_group_data(df_for_plot, colour):
    """
    Plots change in photometry response for the different value blocks
    Args:
        df_for_plot (pd.dataframe): peak change per mouse for different relative value blocks

    Returns:
        df1 (pd.dataframe): plotted data (mean across sessions - data as in plots)
    """
    small_data = df_for_plot[(df_for_plot['reward size change'] == -4)]['cue size change'].values
    big_data = df_for_plot[(df_for_plot['reward size change'] == 4)]['cue size change'].values
    stat, pval = stats.ttest_rel(small_data, big_data)
    cohensd = cohen_d_paired(big_data, small_data)
    print(pval)

    df1 = df_for_plot.pivot(index='reward size change', columns='mouse', values='cue size change').sort_values('reward size change', ascending=False)

    fg = sns.FacetGrid(data=df_for_plot, hue='mouse', aspect=0.8, height=3)
    fg.map(plt.scatter, 'reward size change', 'cue size change', marker='o', s=30, facecolor=colour, alpha=0.3)
    ax = fg.axes[0][0]
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks([-4, 4])
    ax.set_xlabel(r'relative value ($\mu$l)')
    ax.set_xlim([-6, 6])
    plt.ylabel(r'$\Delta$ z-score')

    significance_stars = output_significance_stars_from_pval(pval)
    y = df1.to_numpy().max() + .2
    h = .05
    plt.plot([-4, -4, 4, 4], [y, y+h, y+h, y], c='k', lw=1)
    ax.text(0, y+h, significance_stars, ha='center', fontsize=10)
    plt.tight_layout()
    return  df1