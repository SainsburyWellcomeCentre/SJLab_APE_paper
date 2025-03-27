from scipy.interpolate import interp1d
from utils.plotting import calculate_error_bars
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import os
from matplotlib.lines import Line2D
import matplotlib
from set_global_params import processed_data_path, reproduce_figures_path
from utils.plotting_visuals import makes_plots_pretty
import shutil


def make_change_over_time_plot(mice, ax, window_for_binning=40, colour ='#1b5583', line='k', align_to=None, **file_name_extras):
    """
    Makes change over time plot ith mean line across mice and error bars showing SEM
    Args:
        mice (list): mice to add into the plot
        ax (matplotlib.axes._subplots.AxesSubplot): axes for plot
        window_for_binning (int): number of trials in each bin
        colour (str): error bar colour
        line (str): mean line colour

    Returns:

    """

    if file_name_extras:
        exp_type = file_name_extras['exp_type']
        file_name_suffix ='_binned_' + str(window_for_binning) + '_average_then_peaks_peaks_{}_contra.npz'.format(exp_type)
    else:
        file_name_suffix = '_binned_' + str(window_for_binning) + '_average_then_peaks_peaks.npz'
    if align_to:
        file_name_suffix = '_binned_' + str(window_for_binning) + '_average_then_peaks_peaks_contra_aligned_to_{}.npz'.format(align_to)
    data_root = processed_data_path + 'peak_analysis'

    interp_x = []
    interp_y = []
    font = {'size': 8}
    matplotlib.rc('font', **font)
    fig, axs = plt.subplots(1, 1, figsize=[2, 2], constrained_layout=True)
    for mouse_num, mouse in enumerate(mice):
        saving_folder = os.path.join(data_root, mouse)
        filename = mouse + file_name_suffix
        save_filename = os.path.join(saving_folder, filename)
        fig_path_for_reproducing = os.path.join(reproduce_figures_path, 'fig3', 'peak_analysis')
        if not os.path.exists(fig_path_for_reproducing):
            os.makedirs(fig_path_for_reproducing)
        repro_filename = os.path.join(fig_path_for_reproducing, filename)
        if not os.path.exists(os.path.join(fig_path_for_reproducing, filename)):
            shutil.copy(save_filename, repro_filename)
        rolling_mean_data = np.load(save_filename)
        rolling_mean_x = rolling_mean_data['rolling_mean_x']
        rolling_mean_peaks = rolling_mean_data['rolling_mean_peaks']
        f = interp1d(rolling_mean_x, rolling_mean_peaks)
        xnew = np.arange(int(np.min(rolling_mean_x)) + 1, int(np.max(rolling_mean_x)) - 1)
        ynew = f(xnew)
        interp_x.append(xnew)
        interp_y.append(ynew)

    max_x = max([np.max(i) for i in interp_x])
    plot_cutoff = min([np.max(i) for i in interp_x])
    size_of_ys = max_x + 1
    all_ys = np.empty((len(interp_y), size_of_ys))
    all_ys[:] = np.NaN
    for mouse_num, mouse_data in enumerate(interp_y):
        xs = interp_x[mouse_num]
        all_ys[mouse_num, xs] = mouse_data
        plot_cut_off_ind = np.where(interp_x[mouse_num] >= plot_cutoff)[0][0]
        axs.plot(interp_x[mouse_num][:plot_cut_off_ind], interp_y[mouse_num][:plot_cut_off_ind])
    mean_y = np.mean(all_ys, axis=0)
    error_bar_lower, error_bar_upper = calculate_error_bars(mean_y, all_ys, error_bar_method='sem')
    ax.plot(np.arange(0, size_of_ys), mean_y, color=line, lw=1)
    ax.fill_between(np.arange(0, size_of_ys), error_bar_lower, error_bar_upper, alpha=0.2,
                    facecolor=colour, linewidth=0)
    ax.set_ylabel('Peak size (z-score)')
    ax.set_xlabel('Trial number')

    axs.set_ylabel('Peak size (z-score)')
    axs.set_xlabel('Trial number')
    axs.sharex(ax)
    makes_plots_pretty([axs])
    return all_ys


def example_scatter_change_over_time(mouse, ax, window_for_binning=40, colour='#7FB5B5'):
    """
    Makes example scatter change over time plot showing one mouse
    Args:
        mouse (str): mouse name
        ax (matplotlib.axes._subplots.AxesSubplot): axes for plot
        window_for_binning (int): number of trials in each bin
        colour (str): scatter colour

    Returns:
        keys (matplotlib.axes._subplots.AxesSubplot): axes
    """
    data_root = os.path.join(processed_data_path, 'peak_analysis')
    saving_folder = os.path.join(data_root, mouse)
    filename = mouse + '_binned_' + str(window_for_binning) + '_average_then_peaks_peaks.npz'
    save_filename = os.path.join(saving_folder, filename)
    fig_path_for_reproducing = os.path.join(reproduce_figures_path, 'fig3', 'peak_analysis')
    if not os.path.exists(fig_path_for_reproducing):
        os.makedirs(fig_path_for_reproducing)
    repro_filename = os.path.join(fig_path_for_reproducing, filename)
    if not os.path.exists(os.path.join(fig_path_for_reproducing, filename)):
        shutil.copy(save_filename, repro_filename)

    rolling_mean_data = np.load(repro_filename)
    rolling_mean_x = rolling_mean_data['rolling_mean_x']
    rolling_mean_peaks = rolling_mean_data['rolling_mean_peaks']
    sns.scatterplot(x=rolling_mean_x, y=rolling_mean_peaks, color=colour, ax=ax, s=7)
    ax.set_ylabel('Peak size (z-score)')
    ax.set_xlabel('Trial number')
    plt.tight_layout()
    return ax


def make_example_traces_plot(mouse, ax, window_for_binning=50, side='contra', legend=True, align_to='movement'):
    """
    Makes plot showing example mouse change over time showing the mean traces where each trace is made by averaging
    window_for_binning number of trials
    Args:
        mouse (str): mouse name
        ax (matplotlib.axes._subplots.AxesSubplot): axes for plot
        window_for_binning (int): number of trials to be averaged for each trace
        side (str): ipsi or contra trials
        legend (bool): show legend?

    Returns:

    """
    data_root = os.path.join(processed_data_path, 'peak_analysis')
    saving_folder = os.path.join(data_root, mouse)
    filename = mouse + '_binned_' + str(window_for_binning) + '_average_then_peaks_peaks_{}_aligned_to_{}.npz'.format(side, align_to)
    save_filename = os.path.join(saving_folder, filename)
    rolling_mean_data = np.load(save_filename)
    rolling_mean_traces = rolling_mean_data['rolling_mean_trace']
    num_bins = len(rolling_mean_traces)
    colours = cm.viridis(np.linspace(0, 0.8, num_bins))
    for trace_num, trace in enumerate(rolling_mean_traces):
        x_vals = np.linspace(-8, 8, np.shape(trace)[0])
        ax.plot(x_vals, trace, color=colours[trace_num], alpha=0.8, lw=2)
    ax.set_xlim([-0.5, 1.2])
    custom_lines = [Line2D([0], [0], color=colours[0], lw=4),
                    Line2D([0], [0], color=colours[int(num_bins / 2)], lw=4),
                    Line2D([0], [0], color=colours[-1], lw=4)]

    if legend:
        ax.legend(custom_lines, ['Early', 'Middle', 'Late'], frameon=False, prop={'size': 6}, loc='upper left', bbox_to_anchor=(0.8, 0.9))
    ax.set_ylabel('Peak size (z-score)')
    ax.set_xlabel('Time (s)')





