import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from tqdm import tqdm
import matplotlib
from scipy.signal import decimate
from set_global_params import daq_sample_rate

import seaborn as sns
from utils.individual_trial_analysis_utils import SessionData, ChoiceAlignedData, ZScoredTraces

class HeatMapParams(object):
    def __init__(self, state_type_of_interest, response, first_choice, last_response, outcome, last_outcome,first_choice_correct, align_to, instance, no_repeats, plot_range):

        self.state = state_type_of_interest
        self.outcome = outcome
        #self.last_outcome = last_outcome
        self.response = response
        self.last_response = last_response
        self.align_to = align_to
        self.other_time_point = np.array(['Time start', 'Time end'])[np.where(np.array(['Time start', 'Time end']) != align_to)]
        self.instance = instance
        self.plot_range = plot_range
        self.no_repeats = no_repeats
        self.first_choice_correct = first_choice_correct
        self.first_choice = first_choice


def get_photometry_around_event(all_trial_event_times, demodulated_trace, pre_window=5, post_window=5, sample_rate=daq_sample_rate):
    num_events = len(all_trial_event_times)
    event_photo_traces = np.zeros((num_events, sample_rate*(pre_window + post_window)))
    for event_num, event_time in enumerate(all_trial_event_times):
        plot_start = int(round(event_time*sample_rate)) - pre_window*sample_rate
        plot_end = int(round(event_time*sample_rate)) + post_window*sample_rate
        if plot_end - plot_start != sample_rate*(pre_window + post_window):
            print(event_time)
            plot_start = plot_start + 1
            print(plot_end - plot_start)
        event_photo_traces[event_num, :] = demodulated_trace[plot_start:plot_end]
        #except:
        #   event_photo_traces = event_photo_traces[:event_num,:] 
    print(event_photo_traces.shape)
    return event_photo_traces


def get_next_centre_poke(trial_data, events_of_int):
    trial_numbers = events_of_int['Trial num'].values
    next_centre_poke_times = []
    for event_trial_num in range(len(trial_numbers)-1):
        trial_num = trial_numbers[event_trial_num]
        next_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num + 1)]
        wait_for_pokes = next_trial_events.loc[(next_trial_events['State type'] == 2)]
        next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)]
        next_centre_poke_times.append(next_wait_for_poke['Time end'].values[0])
    return next_centre_poke_times
    
    
def get_mean_and_sem(trial_data, demod_signal, params, norm_window=8, sort=False, error_bar_method='sem'):     
    response_names = ['both left and right','left', 'right']
    outcome_names = ['incorrect', 'correct']

    if  params.state == 10:
        omission_events = trial_data.loc[(trial_data['State type'] == params.state)]
        trials_of_int = omission_events['Trial num'].values
        omission_trials_all_states = trial_data.loc[(trial_data['Trial num'].isin(trials_of_int))]
        events_of_int = omission_trials_all_states.loc[(omission_trials_all_states['State type'] == 4)]
    else:
        events_of_int = trial_data.loc[(trial_data['State type'] == params.state)]
    if params.response != 0:
        events_of_int = events_of_int.loc[events_of_int['Response'] == params.response]
    if params.first_choice != 0:
        events_of_int = events_of_int.loc[events_of_int['First response'] == params.first_choice]
    if params.last_response != 0:
        events_of_int = events_of_int.loc[events_of_int['Last response'] == params.last_response]
        title = ' last response: ' + response_names[params.last_response]
    else:
        title = response_names[params.response]
    if not params.outcome == 3:
        events_of_int = events_of_int .loc[events_of_int['Trial outcome'] == params.outcome]
    #events_of_int = events_of_int.loc[events_of_int['Last outcome'] == 0]
    
    if params.state ==10 or params.outcome == 3:
        title = title +' ' + 'omission'
    else:
        title = title +' ' + outcome_names[params.outcome]
        
    if params.instance == -1:
        events_of_int = events_of_int.loc[
            (events_of_int['Instance in state'] / events_of_int['Max times in state'] == 1)]
    elif params.instance == 1:
        events_of_int = events_of_int.loc[(events_of_int['Instance in state'] == 1)]
        if params.no_repeats == 1:
            events_of_int = events_of_int.loc[events_of_int['Max times in state'] == 1]
    if params.first_choice_correct:
        events_of_int = events_of_int.loc[
            (events_of_int['First choice correct'] == 1)]
        
    event_times = events_of_int[params.align_to].values
    state_name = events_of_int['State name'].values[0]
    last_event = np.asarray(
        np.squeeze(events_of_int[params.other_time_point].values) - np.squeeze(events_of_int[params.align_to].values))
    next_centre_poke = get_next_centre_poke(trial_data, events_of_int)
    next_centre_poke.append(event_times[-1])
    next_centre_poke_norm = next_centre_poke - event_times
    event_photo_traces = get_photometry_around_event(event_times, demod_signal,  pre_window=norm_window, post_window=norm_window)
    norm_traces = stats.zscore(event_photo_traces.T, axis=0)
    
    if len(last_event) != norm_traces.shape[1]:
        last_event = last_event[:norm_traces.shape[1]]
    print(last_event.shape, event_times.shape)
    if sort:
        arr1inds =  last_event.argsort()
        sorted_last_event = last_event [arr1inds[::-1]]
        sorted_traces = norm_traces.T [arr1inds[::-1]]
        sorted_next_poke = next_centre_poke_norm [arr1inds[::-1]]
    else:
        sorted_last_event = last_event 
        sorted_traces = norm_traces.T
        sorted_next_poke = next_centre_poke_norm

    x_vals = np.linspace(-norm_window, norm_window, norm_traces.shape[0], endpoint=True, retstep=False, dtype=None, axis=0)
    y_vals = np.mean(sorted_traces, axis=0)
    if error_bar_method == 'ci':
        sem = bootstrap(sorted_traces, n_boot=1000, ci=95)
    elif error_bar_method == 'sem':
        sem = np.std(sorted_traces, axis=0)
    print(np.mean(next_centre_poke_norm), np.std(next_centre_poke_norm))
    print(sorted_last_event[-1])
    return x_vals, y_vals, sem, sorted_traces, sorted_last_event, state_name, title, sorted_next_poke


def heat_map_and_mean(aligned_session_data, *mean_data, error_bar_method='sem', sort=False, mean_across_mice=False, xlims=[-2, 2], white_dot='default'):
    if mean_across_mice:
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(7.5, 4))
        fig.tight_layout(pad=1.3)
    else:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(5.5, 5.5))
        fig.tight_layout(pad=2.1)

    font = {'size': 10}
    matplotlib.rc('font', **font)

    min_dff_ipsi = np.min(aligned_session_data.ipsi_data.sorted_traces)
    max_dff_ipsi = np.max(aligned_session_data.ipsi_data.sorted_traces)
    min_dff_contra = np.min(aligned_session_data.contra_data.sorted_traces)
    max_dff_contra = np.max(aligned_session_data.contra_data.sorted_traces)
    heatmap_min, heatmap_max = make_y_lims_same((min_dff_ipsi, max_dff_ipsi), (min_dff_contra, max_dff_contra))
    dff_range = (heatmap_min, heatmap_max)
    ipsi_heatmap = plot_one_side(aligned_session_data.ipsi_data, fig, axs[1, 0], axs[1, 1], dff_range, error_bar_method=error_bar_method, sort=sort, white_dot=white_dot)
    contra_heatmap = plot_one_side(aligned_session_data.contra_data, fig, axs[0, 0], axs[0, 1], dff_range, error_bar_method=error_bar_method, sort=sort, white_dot=white_dot)
    ylim_ipsi = axs[1, 0].get_ylim()
    ylim_contra = axs[0, 0].get_ylim()
    ylim_min, ylim_max = make_y_lims_same(ylim_ipsi, ylim_contra)
    axs[0, 0].set_ylim([ylim_min, ylim_max])
    axs[1, 0].set_ylim([ylim_min, ylim_max])
    axs[0, 0].set_xlim(xlims)
    axs[1, 0].set_xlim(xlims)
    axs[1, 1].set_xlim(xlims)
    axs[0, 1].set_xlim(xlims)
    axs[0,0].set_ylabel('z-score')
    axs[1, 0].set_ylabel('z-score')


    cb_ipsi = fig.colorbar(ipsi_heatmap, ax=axs[1, 1], orientation='vertical', fraction=.1)
    cb_contra = fig.colorbar(contra_heatmap, ax=axs[0, 1], orientation='vertical', fraction=.1)
    cb_ipsi.ax.set_title('z-score', fontsize=9, pad=2)
    cb_contra.ax.set_title('z-score', fontsize=9, pad=2)

    if mean_across_mice:
        x_range = axs[0, 0].get_xlim()
        ipsi_data = mean_data[0]
        contra_data = mean_data[1]
        line_plot_dff(aligned_session_data.ipsi_data.time_points, ipsi_data, axs[1, 2], x_range)
        line_plot_dff(aligned_session_data.ipsi_data.time_points, contra_data, axs[0, 2], x_range)
        ylim_ipsi = axs[1, 2].get_ylim()
        ylim_contra = axs[0, 2].get_ylim()
        ylim_min, ylim_max = make_y_lims_same(ylim_ipsi, ylim_contra)
        axs[0, 2].set_ylim([ylim_min, ylim_max])
        axs[1, 2].set_ylim([ylim_min, ylim_max])

        for ax in [axs[0, 0], axs[1, 0]]:
            adjust_label_distances(ax, x_space=0.2, y_space=0.12)
        for ax in [axs[0, 1], axs[1, 1],  axs[0, 2],  axs[1, 2]]:
            adjust_label_distances(ax, x_space=0.2, y_space=0.2)

    return fig


def adjust_label_distances(ax, x_space, y_space):
    ax.yaxis.set_label_coords(-y_space, 0.5)
    ax.xaxis.set_label_coords(0.5, -x_space)

def make_y_lims_same(ylim_ipsi, ylim_contra):
    ylim_min = min(ylim_ipsi[0], ylim_contra[0])
    ylim_max = max(ylim_ipsi[1], ylim_contra[1])
    return ylim_min, ylim_max


def line_plot_dff(x_vals, y_vals, ax, x_range):
    ax.plot(x_vals, y_vals, color='#3F888F', lw=2)
    ax.axvline(0, color='k', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('z-score')
    ax.set_xlim(x_range)


def plot_one_side(one_side_data, fig,  ax1, ax2, dff_range=None, error_bar_method='sem', sort=False, white_dot='default'):
    mean_trace = decimate(one_side_data.mean_trace, 10)
    time_points = decimate(one_side_data.time_points, 10)
    traces = decimate(one_side_data.sorted_traces, 10)
    ax1.plot(time_points, mean_trace, lw=1.5, color='#3F888F')

    if error_bar_method is not None:
        error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                traces,
                                                                error_bar_method=error_bar_method)
        ax1.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                            facecolor='#7FB5B5', linewidth=0)


    ax1.axvline(0, color='k', linewidth=1)
    ax1.set_xlim(one_side_data.params.plot_range)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('z-score')

    if white_dot == 'reward':
        white_dot_point = one_side_data.outcome_times
    else:
        white_dot_point = one_side_data.reaction_times
    if sort:
        arr1inds = white_dot_point.argsort()
        one_side_data.reaction_times = one_side_data.reaction_times[arr1inds[::-1]]
        one_side_data.outcome_times = one_side_data.outcome_times[arr1inds[::-1]]
        one_side_data.sorted_traces = one_side_data.sorted_traces[arr1inds[::-1]]
        one_side_data.sorted_next_poke = one_side_data.sorted_next_poke[arr1inds[::-1]]

    heat_im = ax2.imshow(one_side_data.sorted_traces, aspect='auto',
                            extent=[-10, 10, one_side_data.sorted_traces.shape[0], 0], cmap='viridis')

    ax2.axvline(0, color='w', linewidth=1)
    if white_dot == 'reward':
        ax2.scatter(one_side_data.outcome_times,
                       np.arange(one_side_data.reaction_times.shape[0]) + 0.5, color='w', s=1)
    else:
        ax2.scatter(one_side_data.reaction_times,
                    np.arange(one_side_data.reaction_times.shape[0]) + 0.5, color='w', s=1)
    ax2.scatter(one_side_data.sorted_next_poke,
                   np.arange(one_side_data.sorted_next_poke.shape[0]) + 0.5, color='k', s=1)
    ax2.tick_params(labelsize=10)
    ax2.set_xlim(one_side_data.params.plot_range)
    ax2.set_ylim([one_side_data.sorted_traces.shape[0], 0])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Trial (sorted)')
    if dff_range:
        vmin = dff_range[0]
        vmax = dff_range[1]
        edge = max(abs(vmin), abs(vmax))
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        heat_im.set_norm(norm)
    return heat_im


def calculate_error_bars(mean_trace, data, error_bar_method='sem'):
    """
    Calculates error bars for trace
    Args:
        mean_trace (np.array): mean trace (mean of data)
        data (np.array): all traces
        error_bar_method (str): ci or sem

    Returns:
        lower_bound (np.array): lower error bar
        upper_bound (np.array): upper error bar
    """
    if error_bar_method == 'sem':
        sem = stats.sem(data, axis=0)
        lower_bound = mean_trace - sem
        upper_bound = mean_trace + sem
    elif error_bar_method == 'ci':
        lower_bound, upper_bound = bootstrap(data, n_boot=1000, ci=68)
    return lower_bound, upper_bound


def bootstrap(data, n_boot=10000, ci=68):
    """
    Helper function for lineplot_boot. Bootstraps confidence intervals for plotting time series.

    Args:
        data (np.array): data (2D)
        n_boot (int): number of bootstraps
        ci (float): confidence interval

    Returns:
        s1 (np.array): lower bound
        s2 (np.array): upper bound
    """
    boot_dist = []
    for i in tqdm(range(int(n_boot)), desc='Bootstrapping...'):
        resampler = np.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(np.mean(sample, axis=0))
    b = np.array(boot_dist)
    s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50. - ci / 2.)
    s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50. + ci / 2.)
    return s1, s2


def multi_conditions_plot(ax, data, show_err_bar=False, mean_linewidth=4, mean_line_color='blue', colour='grey'):
    """
    Produces a line plot comparing different conditions for multiple animals, showing significance stars
    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): axes for plot
        data (pd.dataframe): data to plot
        show_err_bar (bool): produce error bars on mean line?
        mean_linewidth (float): width of mean line - if no mean line is desired = 0
        mean_line_color (str): colour for mean line
        colour (str): colour for individual subject lines

    Returns:

    """
    data.plot(ax=ax, color=colour, legend=False)
    data.mean(1).plot(ax=ax, linewidth=mean_linewidth, color=mean_line_color)

    if show_err_bar:
        yerr = data.std(axis=1)

        plt.errorbar(np.array([0, 1]), data.mean(1), yerr, color=mean_line_color, linewidth=4)

    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def output_significance_stars_from_pval(pval):
    if pval >= 0.05:
        return ' '
    elif (pval < 0.05) & (pval >= 0.01):
        return '*'
    elif (pval < 0.01) & (pval >= 0.001):
        return '**'
    elif pval < 0.001:
        return '***'


