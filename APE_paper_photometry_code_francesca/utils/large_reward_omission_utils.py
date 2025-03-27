import os
import peakutils
from scipy.signal import decimate
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from utils.plotting import calculate_error_bars
from set_global_params import processed_data_path, plotting_colours, large_reward_omission_example_mice, daq_sample_rate, reproduce_figures_path, spreadsheet_path
from utils.plotting_visuals import makes_plots_pretty
from utils.plotting import multi_conditions_plot, output_significance_stars_from_pval
from utils.stats import cohen_d_paired


def get_traces_and_reward_types(photometry_data, trial_data):
    """
    Reformats photometry data and behavioural data into a dataframe together for unexpected reward change experiment
    Args:
        photometry_data (np.array): photometry data
        trial_data (pd.dataframe): behavioural data

    Returns:
        all_reward_type_data (pd.dataframe): behavioural data and photometry combined
    """
    omission_trials = trial_data.loc[trial_data['State type'] == 10]['Trial num'].values
    left_large_reward_trials = trial_data.loc[trial_data['State type'] == 12]['Trial num'].values
    right_large_reward_trials = trial_data.loc[trial_data['State type'] == 13]['Trial num'].values
    all_large_reward_trials = np.concatenate([left_large_reward_trials, right_large_reward_trials])
    normal_left_reward_trials = trial_data.loc[trial_data['State type'] == 6]['Trial num'].values
    normal_right_reward_trials = trial_data.loc[trial_data['State type'] == 7]['Trial num'].values
    normal_all_reward_trials = np.concatenate([normal_left_reward_trials, normal_right_reward_trials])
    pre_trials = normal_all_reward_trials
    post_trials = all_large_reward_trials
    contra_data = photometry_data.contra_data
    ipsi_data = photometry_data.ipsi_data
    pre_contra_trial_nums, pre_contra_trials, _ = np.intersect1d(contra_data.trial_nums, pre_trials,
                                                                 return_indices=True)
    post_contra_trial_nums, post_contra_trials, _ = np.intersect1d(contra_data.trial_nums, post_trials,
                                                                   return_indices=True)
    omission_contra_trial_nums, omission_contra_trials, _ = np.intersect1d(contra_data.trial_nums, omission_trials,
                                                                           return_indices=True)
    pre_ipsi_trial_nums, pre_ipsi_trials, _ = np.intersect1d(ipsi_data.trial_nums, pre_trials, return_indices=True)
    post_ipsi_trial_nums, post_ipsi_trials, _ = np.intersect1d(ipsi_data.trial_nums, post_trials, return_indices=True)
    omission_ipsi_trial_nums, omission_ipsi_trials, _ = np.intersect1d(ipsi_data.trial_nums, omission_trials,
                                                                       return_indices=True)
    pre_contra_traces = contra_data.sorted_traces[pre_contra_trials, :]
    post_contra_traces = contra_data.sorted_traces[post_contra_trials, :]
    omission_contra_traces = contra_data.sorted_traces[omission_contra_trials, :]
    pre_ipsi_traces = ipsi_data.sorted_traces[pre_ipsi_trials, :]
    post_ipsi_traces = ipsi_data.sorted_traces[post_ipsi_trials, :]
    omission_ipsi_traces = ipsi_data.sorted_traces[omission_ipsi_trials, :]
    all_contra_trial_nums = np.concatenate([pre_contra_trial_nums, post_contra_trial_nums])
    all_ipsi_trial_nums = np.concatenate([pre_ipsi_trial_nums, post_ipsi_trial_nums])
    all_post_trial_nums = np.concatenate([post_contra_trial_nums, post_ipsi_trial_nums])
    all_trial_nums = np.concatenate([all_contra_trial_nums, all_ipsi_trial_nums])
    all_contra_traces = np.concatenate([pre_contra_traces, post_contra_traces])
    all_ipsi_traces = np.concatenate([pre_ipsi_traces, post_ipsi_traces])
    all_traces = np.concatenate([all_contra_traces, all_ipsi_traces])
    all_omission_traces = np.concatenate([omission_contra_traces, omission_ipsi_traces])
    all_omission_trial_nums = np.concatenate([omission_contra_trial_nums, omission_ipsi_trial_nums])
    list_traces = [all_traces[i, :] for i in range(all_traces.shape[0])]
    list_omission_traces = [all_omission_traces[i, :] for i in range(all_omission_traces.shape[0])]
    omission_data = {}
    omission_data['trial number'] = all_omission_trial_nums
    omission_data['side'] = np.where(np.isin(all_omission_trial_nums, omission_ipsi_trial_nums), 'ipsi', 'contra')
    omission_dataf = pd.DataFrame(omission_data)
    omission_dataf['traces'] = pd.Series(list_omission_traces, index=omission_dataf.index)
    omission_dataf['reward'] = 'omission'
    one_session_data = {}
    ipsi_contra_labels = np.where(np.isin(all_trial_nums, all_ipsi_trial_nums), 'ipsi', 'contra')
    label = 'large reward'
    reward_size_labels = np.where(np.isin(all_trial_nums, all_post_trial_nums), label, 'normal')
    one_session_data['reward'] = reward_size_labels
    one_session_data['trial number'] = all_trial_nums
    one_session_data['side'] = ipsi_contra_labels
    one_session_dataf = pd.DataFrame(one_session_data)
    one_session_dataf['traces'] = pd.Series(list_traces, index=one_session_dataf.index)
    all_reward_type_data = pd.concat([one_session_dataf, omission_dataf])
    all_reward_type_data['time points'] = pd.Series([photometry_data.contra_data.time_points] *
                                                    (len(list_traces) + len(all_omission_traces)),
                                                    index=all_reward_type_data.index)
    return all_reward_type_data


def plot_mean_trace_for_condition(ax, site, trial_type_info, time_points, key, error_bar_method=None, save_location=None, colourmap=None):
    """
    Plots the average trace across trials for a single mouse for different values of a certain condition (e.g. reward amounts)
    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): axes for plot
        trial_type_info (pd.Dataframe):
        time_points (np.array): time points for x-axis
        key (str): column in trial_type_info to separate data by
        error_bar_method (str): 'SEM', 'ci' or None
        save_location (str): path where to save error bars or None

    Returns:

    """
    mouse = trial_type_info['mouse'].iloc[0]
    if key == 'reward ipsi':
        condition = 'reward'
    elif key == 'side':
        condition = 'side'
    elif key == 'reward contra':
        condition = 'reward'
    elif key == 'reward':
        condition = 'reward'
    else:
        raise ValueError('Condition not recognised')
    trial_types = np.sort(trial_type_info[condition].unique())

    if not colourmap:
        colours = colourmap(np.linspace(0, 0.8, trial_types.shape[0]))
    else:
        colours = colourmap

    for trial_type_indx, trial_type in enumerate(trial_types):
        rows = trial_type_info[(trial_type_info[condition] == trial_type)]
        traces = rows['traces'].values
        flat_traces = np.zeros([traces.shape[0], traces[0].shape[0]])
        for idx, trace in enumerate(traces):
            flat_traces[idx, :] = trace
        subfig = 'C' if site == 'tail' else 'E'
        csv_file = os.path.join(spreadsheet_path, 'ED_fig7', f'EDfig7{subfig}_{trial_type}_traces_{site}.csv')
        if not os.path.exists(csv_file):
            df_for_spreadsheet = pd.DataFrame(flat_traces.T)
            df_for_spreadsheet.insert(0, "Timepoints", time_points)
            df_for_spreadsheet.to_csv(csv_file)
        mean_trace = np.mean(flat_traces, axis=0)
        ax.plot(time_points, mean_trace, lw=1.5, color=colours[trial_type_indx], label=trial_type)
        if error_bar_method is not None:
            # bootstrapping takes a long time. calculate once and save:
            filename = 'errors_clipped_short_{}_{}_{}.npz'.format(mouse, key, trial_type)
            if save_location is None:
                error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                        flat_traces,
                                                                        error_bar_method=error_bar_method)
            else:
                if not os.path.isfile(os.path.join(save_location, filename)):
                    error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                            flat_traces,
                                                                            error_bar_method=error_bar_method)
                    np.savez(os.path.join(save_location, filename), error_bar_lower=error_bar_lower,
                             error_bar_upper=error_bar_upper)
                else:
                    print('loading')
                    error_info = np.load(os.path.join(save_location, filename))
                    error_bar_lower = error_info['error_bar_lower']
                    error_bar_upper = error_info['error_bar_upper']
            ax.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                             facecolor=colours[trial_type_indx], linewidth=0)

    ax.axvline(0, color='k')
    ax.set_xlim([-2, 2])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('z-scored fluorescence')



def get_processed_data_for_example_mouse(mouse_name, site_data):
    """
    Gets example mouse data
    Args:
        mouse_name (str): example mouse
        site_data (pd.dataframe): behavioural and photometry data for all mice recorded at a given site

    Returns:
        all_trials (pd.dataframe): photometry and behavioural data for an example mouse
        time_points (np.array): time_points for traces for example trace plot
    """
    time_points = site_data['time points'].reset_index(drop=True)[0]
    all_trials = site_data[(site_data['mouse'] == mouse_name)]
    return all_trials, time_points


def make_example_traces_plot(site):
    """
    Maks plot with traces for different reward amounts aligned to outcome for a recording site
    Args:
        site (str): Recording site (Nacc or tail)
    Returns:

    """
    mouse_name = large_reward_omission_example_mice[site]
    repro_path = os.path.join(reproduce_figures_path, 'ED_fig7', 'omissions_large_rewards')
    repro_example_file = os.path.join(repro_path,
                                      f'omissions_large_rewards_downsampled_traces_example_{site}_{mouse_name}.pkl')
    data = pd.read_pickle(repro_example_file)
    all_trials, time_points = get_processed_data_for_example_mouse(mouse_name, data)

    fig, ax = plt.subplots(1, 1, figsize=[2.2, 2])
    plot_mean_trace_for_condition(ax, site, all_trials, time_points,
                                  'reward', error_bar_method='sem', save_location=None,
                                  colourmap=plotting_colours[site])
    lg1 = ax.legend(loc='lower left', bbox_to_anchor=(0.6, 0.8), borderaxespad=0, frameon=False, prop={'size': 6})
    ax.set_ylim([-1.5, 4.1])
    makes_plots_pretty(ax)
    plt.tight_layout()


def get_unexpected_reward_change_data_for_site(site):
    """
    Gets behavioural and photometry data for unexpected reward change experiment
    Args:
        site (str): recording site (Nacc or tail)

    Returns:
        all_reward_block_data (pd.dataframe): photometry and behavioural data
    """
    processed_data_dir = os.path.join(processed_data_path, 'large_rewards_omissions_data')
    block_data_file = os.path.join(processed_data_dir, 'all_{}_reward_change_data.pkl'.format(site))
    all_reward_block_data = pd.read_pickle(block_data_file)
    return all_reward_block_data


def get_unexpected_reward_change_data_for_site_for_plotting(site):
    """
    Gets behavioural and photometry data for unexpected reward change experiment
    Args:
        site (str): recording site (Nacc or tail)

    Returns:
        all_reward_block_data (pd.dataframe): photometry and behavioural data
        (much reduced in size due to downsampling traces)
    """
    repro_path = os.path.join(reproduce_figures_path, 'ED_fig7', 'omissions_large_rewards')
    repro_file = os.path.join(repro_path, f'omissions_large_rewards_downsampled_traces_peaks_{site}.pkl')
    all_reward_block_data = pd.read_pickle(repro_file)
    return all_reward_block_data


def compare_peaks_across_trial_types(avg_traces, colour='gray'):
    """
    Compares peak response size for different reward amounts
    Args:
        avg_traces (pd.Dataframe): behavioural and trace data and extracted peak (or dip)
            responses avg across trials of different reward amounts, for multiple mice,
            all recorded at the same site
        colour (str): colour for lines in plot

    Returns:

    """
    normal_peak = avg_traces[avg_traces['reward'] == 'normal']['peak']
    large_reward_peak = avg_traces[avg_traces['reward'] =='large reward']['peak']
    omission_peak = avg_traces[avg_traces['reward'] =='omission']['peak']
    stat1, pval1 = stats.ttest_rel(normal_peak, large_reward_peak)
    stat2, pval2 = stats.ttest_rel(normal_peak, omission_peak)
    cohend1 = cohen_d_paired(large_reward_peak, normal_peak)
    cohend2 = cohen_d_paired(normal_peak, omission_peak)

    # Repeated measures anova to check for a main effect of reward.
    # Subsequently, we want to do pairwise testing between the three reward conditions. Need to correct for multiple comparisons

    reject, corrected_pvals, corrected_alpha_sidak, corrected_bonf = multipletests([pval1, pval2], method='bonferroni')

    print(corrected_pvals)

    df1 = avg_traces
    df_for_plot = df1.pivot(index='reward', columns='mouse', values='peak').sort_values('reward', ascending=False)

    fig, ax = plt.subplots(figsize=[1.5, 2])
    multi_conditions_plot(ax, df_for_plot, mean_linewidth=0, show_err_bar=False, colour=colour)
    plt.xticks([0, 1, 2], ['omission', 'normal\nreward', '3 x normal\nreward'], fontsize=7)
    plt.ylabel('Z-scored fluorescence', fontsize=7)
    ax.set_xlabel(' ')
    # show significance stars
    # for first comparison
    y = df_for_plot.T['large reward'].max() + .2
    h = .1
    plt.plot([0, 0, 1, 1], [y, y+h, y+h, y],c='k',lw=1)
    significance_stars1 = output_significance_stars_from_pval(corrected_pvals[0])
    ax.text(.5, y+h, significance_stars1, ha='center', fontsize=8)
    # for second comparison
    l = .2
    plt.plot([1, 1, 2, 2], [y+l, y+h+l, y+h+l, y+l],c='k', linewidth=1)
    significance_stars2 = output_significance_stars_from_pval(corrected_pvals[1])
    ax.text(1.5, y+h+l, significance_stars2, ha='center', fontsize=8)
    ax.set_ylim([-1, 3.4])
    plt.tight_layout()
    return df1
