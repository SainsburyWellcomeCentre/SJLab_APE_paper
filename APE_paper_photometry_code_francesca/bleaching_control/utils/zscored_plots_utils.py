import os
import pickle
import numpy as np
import pandas as pd

from utils.individual_trial_analysis_utils import SessionData, ZScoredTraces
import matplotlib.pyplot as plt
from scipy.signal import decimate
import seaborn as sns
from matplotlib.colors import ListedColormap
from utils.plotting import calculate_error_bars
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from set_global_params import processed_data_path, reproduce_figures_path, spreadsheet_path
from scipy import stats
from scipy.stats import shapiro
import matplotlib.patches as mpatches
import matplotlib


class ZScoredTracesCuePlotOnly(object):
    def __init__(self, sorted_traces, outcome_times, reaction_times, sorted_next_poke, time_points):
        self.sorted_traces = sorted_traces
        self.outcome_times = outcome_times
        self.reaction_times = reaction_times
        self.sorted_next_poke = sorted_next_poke
        self.time_points = time_points


def get_example_data_for_figure(recording_site):
    """
    Loads in data for an example mouse for the heatmaps for the heatmaps.

    For VS we choose SNL_photo35 and for TS we choose SNL_photo26 as representative examples.

    Args:
        recording_site (str): 'VS' or 'TS'

    Returns:
        example_session_data (SessionData): the object containing aligned traces to various behavioural events
    """
    if recording_site == 'VS':
        example_mouse = 'SNL_photo35'
        example_date = '20201119'
    elif recording_site == 'TS':
        example_mouse = 'SNL_photo26'
        example_date = '20200812'

    saving_folder = os.path.join(processed_data_path, 'for_figure', example_mouse)
    aligned_filename = example_mouse + '_' + example_date + '_' + 'aligned_traces_for_fig.p'
    save_filename = os.path.join(saving_folder, aligned_filename)
    example_session_data = pickle.load(open(save_filename, "rb"))
    return example_session_data


def combine_ipsi_and_contra_cues(contra_data, ipsi_data):
    all_trial_numbers = np.concatenate([contra_data.trial_nums, ipsi_data.trial_nums])
    indices = np.argsort(all_trial_numbers)
    unsorted_traces = np.concatenate([contra_data.sorted_traces, ipsi_data.sorted_traces])[indices, :]

    unsorted_reaction_times = np.concatenate([contra_data.reaction_times, ipsi_data.reaction_times])[indices]
    unsorted_time_points = contra_data.time_points
    unsorted_outcome_times = np.concatenate([contra_data.outcome_times, ipsi_data.outcome_times])[indices]
    unsorted_next_poke = np.concatenate([contra_data.sorted_next_poke, ipsi_data.sorted_next_poke])[indices]
    return ZScoredTracesCuePlotOnly(unsorted_traces, unsorted_outcome_times, unsorted_reaction_times, unsorted_next_poke, unsorted_time_points)


def get_correct_data_for_plot(session_data, plot_type):
    """
    Gets traces aligned to correct behavioural events for plot
    Args:
        session_data (SessionData): Object with photometry aligned to all sorts of behavioural events
        plot_type (str): what is the plot aligned to? (ipsi, contra, rewarded, unrewarded)

    Returns:
        output_data (ZScoredTraces): Photometry data relevant to behavioural event needed for plot
    """
    if plot_type == 'ipsi':
        output_data = session_data.choice_data.ipsi_data, 'event end'
    elif plot_type == 'contra':
        output_data = session_data.choice_data.contra_data, 'event end'
    elif plot_type == 'rewarded':
        output_data = session_data.outcome_data.reward_data, 'event start' #'next trial'
    elif plot_type == 'unrewarded':
        output_data = session_data.outcome_data.no_reward_data, 'event start' #'next trial'
    elif plot_type == 'cue':
        contra_cue = session_data.cue_data.contra_data
        ipsi_cue = session_data.cue_data.ipsi_data
        all_cues = combine_ipsi_and_contra_cues(contra_cue, ipsi_cue)
        output_data = all_cues, 'event start'

    else:
        raise ValueError('Unknown type of plot specified.')
    return output_data


def get_example_data_for_recording_site(recording_site, keys):
    """
    Gets example mouse data for heatmaps for a given recording site (TS or VS)
    Args:
        recording_site (str): 'VS' or 'TS'
        keys (list): list of keys, e.g. ['ipsi', 'contra']

    Returns:
        axes (list): all plot types for a recording site
        all_data (list): list of ZScoredTraces objects containing relevant to the plot types in axes
        all_white_dot_points (list): list of lists containing reaction times for heatmap white dots
        all_flip_sort_order (list): list of bools (should fastest or slowest reaction time be top?)
        ymins (list): min heatmap value per plot
        ymaxs (list): max heatmap value per plot
    """
    aligned_session_data = get_example_data_for_figure(recording_site)
    all_data = []
    all_white_dot_points = []
    all_flip_sort_order = []
    ymins = []
    ymaxs = []
    for key in keys:
        data, sort_by = get_correct_data_for_plot(aligned_session_data, key)
        if sort_by == 'event end':
            white_dot_point = data.reaction_times
            flip_sort_order = True
        elif sort_by == 'event start':
            white_dot_point = data.reaction_times
            data.reaction_times = data.reaction_times
            flip_sort_order = False
        else:
            raise ValueError('Unknown method of sorting trials')
        all_data.append(data)
        all_white_dot_points.append(white_dot_point)
        all_flip_sort_order.append(flip_sort_order)
        ymin, ymax = get_min_and_max(data)
        ymins.append(ymin)
        ymaxs.append(ymax)
    return all_data, all_white_dot_points, all_flip_sort_order, ymins, ymaxs


def plot_all_heatmaps_same_scale(axes, all_data, all_white_dot_points, all_flip_sort_order, cb_range, cmap='viridis'):
    """
    Plots the heatmaps for traces aligned to different behavioural events using the same colorbar
    Args:
        axes (list): list of axis label/ axis pairs
        all_data (list): list of ZScoredTraces objects to be plotted
        all_white_dot_points (list): list of arrays with the white dot for plots
        all_flip_sort_order (list): list of bools whether to sort by shortest or longest reaction time
        cb_range (list): list of min and max values for each heatmap
        cmap (ListedColormap): colormap to use for heatmap

    Returns:
        heat_map (AxesImage):
    """
    heat_maps = []
    for ax_num, ax_id in enumerate(axes):
        heat_map = plot_heat_map(ax_id, all_data[ax_num], all_white_dot_points[ax_num], all_flip_sort_order[ax_num], dff_range=cb_range, cmap=cmap)
        ax_id.set_xlim([-1.5, 1.5])
        divider = make_axes_locatable(ax_id)
        cax = divider.append_axes("right", size="5%", pad=0.02)
        cb = plt.colorbar(heat_map, cax=cax)
        cb.ax.set_title('z-score', fontsize=8, pad=0.02)
        heat_maps.append(heat_map)
    return heat_maps


def get_min_and_max(data):
    """
    Finds min and max of array
    Args:
        data (np.array): data to analyse

    Returns:
        ymin (float): min of array
        ymax (float): max of array
    """
    ymin = np.min(data.sorted_traces)
    ymax = np.max(data.sorted_traces)
    return ymax, ymin

def plot_average_trace(ax, data, error_bar_method='sem', colour='navy'):
    mean_trace = decimate(data.mean_trace, 10)
    time_points = decimate(data.time_points, 10)
    traces = decimate(data.sorted_traces, 10)
    ax.plot(time_points, mean_trace, lw=1, color=colour)# color='navy')

    if error_bar_method is not None:
        error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                traces,
                                                                error_bar_method=error_bar_method)
        ax.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                            facecolor=colour, linewidth=0)


    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('z-score')


def get_all_mouse_data_for_site(site, dir=os.path.join(processed_data_path, 'for_figure') , file_ext='_new_mice_added_with_cues.npz'):
    file_name = 'group_data_avg_across_sessions_' + site + file_ext
    data = np.load(os.path.join(dir, file_name))
    return data


def plot_average_trace_all_mice(move_ax, outcome_ax, dir, site, error_bar_method='sem', cmap=sns.color_palette("Set2"), x_range=[-1.5, 1.5]):
    """
    Plots the average trace across mice for movement (contra, ipsi) and outcome (reward, no reward) aligned data
    Args:
        move_ax (matplotlib.axes._subplots.AxesSubplot): axes to plot ipsi/ contra movement aligned traces
        outcome_ax (matplotlib.axes._subplots.AxesSubplot): axes to plot correct/ incorrect outcome aligned data
        site (str): 'VS' or 'TS'
        error_bar_method (str): sem or ci or None
        cmap (list): colours for the two types of data (ipsi vs contra, correct vs incorrect)
        x_range (list): time windon around behavioural event on x-axis

    Returns:

    """
    all_data = get_all_mouse_data_for_site(site, dir=dir, file_ext='_new_mice_added_with_cues.npz')
    time_stamps = all_data['time_stamps']
    data = dict(all_data)
    del data['time_stamps'], data['cue']
    move_data = {'data': [data['contra_choice'], data['ipsi_choice']], 'axis': move_ax, 'time': time_stamps}
    outcome_data = {'data': [data['reward'], data['no_reward']], 'axis': outcome_ax, 'time': time_stamps}
    axs = {'contra_choice': move_ax, 'ipsi_choice': move_ax, 'reward': outcome_ax, 'no_reward': outcome_ax}
    colours = {'contra_choice': cmap[0], 'ipsi_choice': cmap[1], 'reward': cmap[0], 'no_reward': cmap[1]}
    for trace_type, traces in data.items():
        mean_trace = decimate(np.mean(traces, axis=0), 10)
        time_points = decimate(time_stamps, 10)
        traces = decimate(traces, 10)
        mean_trace = mean_trace[int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
            x_range[1] * 1000)]
        time_points = time_points[int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
            x_range[1] * 1000)]
        traces = traces[:, int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
            x_range[1] * 1000)]
        axs[trace_type].plot(time_points, mean_trace, lw=1, color=colours[trace_type])# color='navy')

        if error_bar_method is not None:
            error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                    traces,
                                                                    error_bar_method=error_bar_method)
            axs[trace_type].fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                                 facecolor=colours[trace_type], linewidth=0)


        axs[trace_type].axvline(0, color='k', linewidth=0.8)
        axs[trace_type].set_xlabel('Time (s)')
        axs[trace_type].set_ylabel('z-score')
        axs[trace_type].set_xlim([-1.5, 1.5])

    plot_significance_patches(move_data)
    plot_significance_patches(outcome_data)
    return outcome_data, move_data


def plot_average_trace_all_mice_high_low_cues(ax, site, error_bar_method='sem', cmap=sns.color_palette("Set2"), x_range=[-1.5, 1.5]):
    """
    Plots the average trace across mice for movement (contra, ipsi) and outcome (reward, no reward) aligned data
    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): axes to plot high/low cue aligned traces
        site (str): 'VS' or 'TS'
        error_bar_method (str): sem or ci or None
        cmap (list): colours for the two types of data (ipsi vs contra, correct vs incorrect)
        x_range (list): time windon around behavioural event on x-axis

    Returns:

    """
    if os.path.exists(os.path.join(reproduce_figures_path, 'ED_fig4')):
        dir = os.path.join(reproduce_figures_path, 'ED_fig4')
    else:
        dir = os.path.join(processed_data_path, 'for_figure')
    all_data = get_all_mouse_data_for_site(site, dir=dir, file_ext='_new_mice_added_high_low_cues_ipsi_contra.npz')
    time_stamps = all_data['time_stamps']
    data = dict(all_data)
    del data['time_stamps']
    axs = {'contra_high_cues': ax[0], 'ipsi_low_cues': ax[0], 'contra_low_cues': ax[1], 'ipsi_high_cues': ax[1]}
    colours = {'contra_high_cues': cmap[0], 'ipsi_low_cues': cmap[1], 'contra_low_cues': cmap[0], 'ipsi_high_cues': cmap[1]}
    labels = {'contra_high_cues': 'contra high Hz', 'ipsi_low_cues': 'ipsi low Hz', 'contra_low_cues': 'contra low Hz', 'ipsi_high_cues': 'ipsi high Hz'}
    for trace_type, traces in data.items():
        mean_trace = decimate(np.mean(traces, axis=0), 10)
        time_points = decimate(time_stamps, 10)
        traces = decimate(traces, 10)
        mean_trace = mean_trace[int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
            x_range[1] * 1000)]
        time_points = time_points[int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
            x_range[1] * 1000)]
        traces = traces[:, int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
            x_range[1] * 1000)]
        axs[trace_type].plot(time_points, mean_trace, lw=1, color=colours[trace_type], label=labels[trace_type])# color='navy')

        if error_bar_method is not None:
            error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                    traces,
                                                                    error_bar_method=error_bar_method)
            axs[trace_type].fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                                 facecolor=colours[trace_type], linewidth=0)

        subfig_num = 'H' if site == 'tail' else 'I'
        spreadsheet_file = os.path.join(spreadsheet_path, 'ED_fig4',
                                        f'ED_fig4{subfig_num}_{site}_{trace_type}_traces.csv')
        if not os.path.exists(spreadsheet_file):
            df_for_spreadsheet = pd.DataFrame(traces.T)
            df_for_spreadsheet.insert(0, "Timepoints", time_points)
            df_for_spreadsheet.to_csv(spreadsheet_file)

        axs[trace_type].axvline(0, color='k', linewidth=0.8)
        axs[trace_type].set_xlabel('Time (s)')
        axs[trace_type].set_ylabel('z-score')
        axs[trace_type].set_xlim([-1.5, 1.5])
        axs[trace_type].legend(frameon=False)


def plot_average_trace_all_mice_cue_move_rew(cue_ax, move_ax, outcome_ax, error_bar_method='sem', cmap=sns.color_palette("Set2"), x_range=[-1.5, 1.5]):
    """
    Plots the average trace across mice for movement (contra, ipsi) and outcome (reward, no reward) aligned data
    Args:
        cue_ax (matplotlib.axes._subplots.AxesSubplot): axes to plot contra cue aligned traces
        move_ax (matplotlib.axes._subplots.AxesSubplot): axes to plot contra movement aligned traces
        outcome_ax (matplotlib.axes._subplots.AxesSubplot): axes to plot correct outcome aligned data
        site (str): 'VS' or 'TS'
        error_bar_method (str): sem or ci or None
        cmap (list): colours for the two types of data (TS vs VS)
        x_range (list): time window around behavioural event on x-axis

    Returns:
    """
    sites = ['tail', 'Nacc']
    for i, site in enumerate(sites):
        if os.path.exists(os.path.join(reproduce_figures_path, 'ED_fig4')):
            dir = os.path.join(reproduce_figures_path, 'ED_fig4')
        else:
            dir = os.path.join(processed_data_path, 'for_figure')
        all_data = get_all_mouse_data_for_site(site, dir=dir, file_ext='_new_mice_added_with_cues.npz')
        time_stamps = all_data['time_stamps']
        data = dict(all_data)
        del data['time_stamps'], data['ipsi_choice'], data['no_reward']
        axs = {'cue': cue_ax, 'contra_choice': move_ax, 'reward': outcome_ax}
        colours = {'contra_choice': cmap[i], 'cue': cmap[i], 'reward': cmap[i]}
        for trace_type, traces in data.items():
            mean_trace = decimate(np.mean(traces, axis=0), 10)
            time_points = decimate(time_stamps, 10)
            traces = decimate(traces, 10)
            mean_trace = mean_trace[int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
                x_range[1] * 1000)]
            time_points = time_points[int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
                x_range[1] * 1000)]
            traces = traces[:, int(traces.shape[1] / 2) + int(x_range[0] * 1000): int(traces.shape[1] / 2) + int(
                x_range[1] * 1000)]
            axs[trace_type].plot(time_points, mean_trace, lw=1, color=colours[trace_type])# color='navy')

            if error_bar_method is not None:
                error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                        traces,
                                                                        error_bar_method=error_bar_method)
                axs[trace_type].fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                                     facecolor=colours[trace_type], linewidth=0)
            subfig_nums = {'cue': 'C', 'contra_choice': 'D', 'reward': 'E'}
            spreadsheet_file = os.path.join(spreadsheet_path, 'ED_fig4', f'ED_fig4{subfig_nums[trace_type]}_{site}_{trace_type}_traces.csv')
            if not os.path.exists(spreadsheet_file):
                df_for_spreadsheet = pd.DataFrame(traces.T)
                df_for_spreadsheet.insert(0, "Timepoints", time_points)
                df_for_spreadsheet.to_csv(spreadsheet_file)

            axs[trace_type].axvline(0, color='k', linewidth=0.8)
            axs[trace_type].set_xlabel('Time (s)')
            axs[trace_type].set_ylabel('z-score')
            axs[trace_type].set_xlim([-1.5, 1.5])







def plot_heat_map(ax, data, white_dot_point, flip_sort_order, dff_range=None, x_range=[-1.5, 1.5], cmap='viridis'):
    """
    Plots a single heatmap on a single axis
    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): axes to plot on
        data (ZScoredTraces): photometry traces to plot in heatmap
        white_dot_point (np.array): timestamps (relative to aligned to event) of white dot -normally represents reaction time
        flip_sort_order (bool): shortest or longest reaction time top?
        dff_range (tuple): min and max for colorbar
        x_range (list): time range for x-axis of heatmap (seconds from behavioural event)
        cmap (ListedColormap): colormap for heatmap

    Returns:
        heat_im (AxesImage): the heatmap
    """
    data.sorted_next_poke[-1] = np.nan
    if flip_sort_order:
        arr1inds = (-1 *white_dot_point).argsort()[::-1]
    else:
        arr1inds = white_dot_point.argsort()[::-1]
    data.reaction_times = data.reaction_times[arr1inds]
    data.outcome_times = data.outcome_times[arr1inds]
    data.sorted_traces = data.sorted_traces[arr1inds]
    data.sorted_next_poke = data.sorted_next_poke[arr1inds]

    traces = data.sorted_traces
    clipped_traces = traces[:, int(traces.shape[1] / 2) + int(x_range[0] * 10000): int(traces.shape[1] / 2) + int(x_range[1] * 10000)]
    heat_im = ax.imshow(clipped_traces, aspect='auto',
                        extent=[x_range[0], x_range[1], clipped_traces.shape[0], 0], cmap=cmap)

    ax.axvline(0, color='w', linewidth=1)

    ax.scatter(data.reaction_times,
               np.arange(data.reaction_times.shape[0]) + 0.5, color='w', s=0.01)
    ax.scatter(data.sorted_next_poke,
               np.arange(data.sorted_next_poke.shape[0]) + 0.5, color='k', s=0.5)
    ax.tick_params(labelsize=8)
    ax.set_ylim([data.sorted_traces.shape[0], 0])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial (sorted)')
    if dff_range:
        vmin = dff_range[0]
        vmax = dff_range[1]
        edge = max(abs(vmin), abs(vmax))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        heat_im.set_norm(norm)
    return heat_im



def make_y_lims_same_heat_map(ymins, ymaxs):
    ylim_min = min(ymins)
    ylim_max = max(ymaxs)
    return ylim_min, ylim_max

def get_significance_between_traces(kernel1, kernel2, time_stamps, bin_size=0.1, alpha=0.01):
    bin_numbers = np.digitize(time_stamps,
                              np.arange(time_stamps[0], time_stamps[-1], bin_size))
    downsampled_kernel1 = np.array(
        [kernel1[:, bin_numbers == i].mean(axis=1) for i in np.unique(bin_numbers)])
    downsampled_kernel2 = np.array(
        [kernel2[:, bin_numbers == i].mean(axis=1) for i in np.unique(bin_numbers)])
    decimated_timestamps = np.array(
        [time_stamps[bin_numbers == i].mean() for i in np.unique(bin_numbers)])
    p_vals = []
    for i in range(0, downsampled_kernel2.shape[0]):
        differences = downsampled_kernel1[i, :] - downsampled_kernel2[i, :]
        print(shapiro(differences))
        _, p = stats.mannwhitneyu(downsampled_kernel1[i, :], downsampled_kernel2[i, :])
        p_vals.append(p)
    significant_time_stamps = decimated_timestamps[np.where(np.array(p_vals) < alpha)[0]]
    return significant_time_stamps


def plot_significance_patches(data):
    axs = data['axis']
    traces = data['data']
    timestamps = data['time']
    sig_times = get_significance_between_traces(traces[0], traces[1], timestamps)

    if len(sig_times) > 0:
        min_y, max_y = axs.get_ylim()

        # Find gaps between consecutive significant timestamps
        gaps_between_significant_time_stamps = np.diff(sig_times)

        # Initialize the start of the first window
        window_starts = [sig_times[0] - 0.05]
        window_ends = []

        # Iterate through gaps to identify windows of significance
        for i, gap in enumerate(gaps_between_significant_time_stamps):
            if gap > 0.11:
                # End the current window and start a new one
                window_ends.append(sig_times[i] + 0.05)
                window_starts.append(sig_times[i + 1] - 0.05)

        # Add the end of the last window
        window_ends.append(sig_times[-1] + 0.05)

        # Create and add patches for each window
        for start, end in zip(window_starts, window_ends):
            rect = mpatches.Rectangle((start, min_y),
                                      end - start,
                                      max_y - min_y,
                                      fill=True,
                                      color="grey",
                                      alpha=0.2,
                                      linewidth=0)
            axs.add_patch(rect)

        axs.set_ylim(min_y, max_y)
