# Add modules to the path
from matplotlib import colors, cm
import numpy as np
from scipy.signal import decimate
import pandas as pd
from utils.plotting import HeatMapParams
from utils.plotting import get_photometry_around_event
from scipy import stats
from utils.individual_trial_analysis_utils import SessionData, get_photometry_around_event, get_next_centre_poke, get_peak_each_trial, get_peak_each_trial_with_nans, get_next_reward_time, HeatMapParams
from utils.plotting import calculate_error_bars
import os
from set_global_params import experiment_record_path, processed_data_path, spreadsheet_path
from utils.post_processing_utils import open_experiment, open_one_experiment


class CustomAlignedDataRewardBlocks(object):
    def __init__(self, session_data, params, reward_block):
        saving_folder = os.path.join(processed_data_path, session_data.mouse)
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(os.path.join(saving_folder, dff_trace_filename))

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]
        self.contra_fiber_side_numeric = contra_fiber_side_numeric
        self.contra_data = ZScoredTracesRewardBlocks(trial_data, dff,params, contra_fiber_side_numeric, contra_fiber_side_numeric,reward_block)
        self.contra_data.get_peaks()


def get_all_experimental_records():
    experiment_record = pd.read_csv(experiment_record_path)
    experiment_record['date'] = experiment_record['date'].astype(str)
    return experiment_record


class ZScoredTracesRewardBlocks(object):
    def __init__(self,  trial_data, dff, params, response, first_choice, reward_block):
        self.trial_peaks = None
        self.params = HeatMapParams(params, response, first_choice)
        self.reward_block = reward_block
        self.time_points, self.mean_trace, self.sorted_traces, self.reaction_times, self.state_name, title, self.sorted_next_poke, self.trial_nums, self.event_times, self.outcome_times = find_and_z_score_traces_blocks(
            trial_data, dff, self.params, self.reward_block)

    def get_peaks(self):
        self.trial_peaks = get_peak_each_trial_with_nans(self.sorted_traces, self.time_points, self.outcome_times)


def find_and_z_score_traces_blocks(trial_data, demod_signal, params, reward_block, norm_window=8, sort=False, get_photometry_data=True):
    response_names = ['both left and right', 'left', 'right']
    outcome_names = ['incorrect', 'correct', 'both correct and incorrect']
    events_of_int = trial_data.loc[(trial_data['State type'] == params.state)]
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
    if not params.outcome == 2: # if you don't care about the reward or not
        events_of_int = events_of_int.loc[events_of_int['Trial outcome'] == params.outcome]
    #events_of_int = events_of_int.loc[events_of_int['Last outcome'] == 0]

    if params.cue == 'high':
        events_of_int = events_of_int.loc[events_of_int['Trial type'] == 7]
    elif params.cue == 'low':
        events_of_int = events_of_int.loc[events_of_int['Trial type'] == 1]
    if reward_block == 'all':
        pass
    else:
        events_of_int = events_of_int.loc[events_of_int['Reward block'] == reward_block]



    if params.state == 10:
        title = title + ' ' + 'omission'
    else:
        title = title + ' ' + outcome_names[params.outcome]

    if params.instance == -1:
        events_of_int = events_of_int.loc[
            (events_of_int['Instance in state'] / events_of_int['Max times in state'] == 1)]
    elif params.instance == 1:
        events_of_int = events_of_int.loc[(events_of_int['Instance in state'] == 1)]
        if params.no_repeats == 1:
            events_of_int = events_of_int.loc[events_of_int['Max times in state'] == 1]
    elif params.instance == 0:
        events_of_int = events_of_int

    if params.first_choice_correct != 0:
        events_of_int = events_of_int.loc[
            (events_of_int['First choice correct'] == 1)]

    event_times = events_of_int[params.align_to].values
    trial_nums = events_of_int['Trial num'].values
    state_name = events_of_int['State name'].values[0]
    other_event = np.asarray(
        np.squeeze(events_of_int[params.other_time_point].values) - np.squeeze(events_of_int[params.align_to].values))


    last_trial = np.max(trial_data['Trial num'])
    last_trial_num = events_of_int['Trial num'].unique()[-1]
    events_reset_indx = events_of_int.reset_index(drop=True)
    last_trial_event_indx = events_reset_indx.loc[(events_reset_indx['Trial num'] == last_trial_num)].index
    next_centre_poke = get_next_centre_poke(trial_data, events_of_int, last_trial_num==last_trial)
    outcome_times = get_next_reward_time(trial_data, events_of_int)
    outcome_times = outcome_times - event_times

    last_trial_num = events_of_int['Trial num'].unique()[-1]
    events_reset_indx = events_of_int.reset_index(drop=True)
    last_trial_event_indx = events_reset_indx.loc[(events_reset_indx['Trial num'] == last_trial_num)].index
    next_centre_poke[last_trial_event_indx] = events_reset_indx[params.align_to].values[last_trial_event_indx]
    next_centre_poke_norm = next_centre_poke - event_times

    # this all deals with getting photometry data
    if get_photometry_data == True:
        event_photo_traces = get_photometry_around_event(event_times, demod_signal, pre_window=norm_window,
                                                         post_window=norm_window)
        norm_traces = stats.zscore(event_photo_traces.T, axis=0)

        if len(other_event) != norm_traces.shape[1]:
            other_event = other_event[:norm_traces.shape[1]]
        if sort:
            arr1inds = other_event.argsort()
            sorted_other_event = other_event[arr1inds[::-1]]
            sorted_traces = norm_traces.T[arr1inds[::-1]]
            sorted_next_poke = next_centre_poke_norm[arr1inds[::-1]]
        else:
            sorted_other_event = other_event
            sorted_traces = norm_traces.T
            sorted_next_poke = next_centre_poke_norm

        time_points = np.linspace(-norm_window, norm_window, norm_traces.shape[0], endpoint=True, retstep=False, dtype=None,
                             axis=0)
        mean_trace = np.mean(sorted_traces, axis=0)

        return time_points, mean_trace, sorted_traces, sorted_other_event, state_name, title, sorted_next_poke, trial_nums, event_times, outcome_times
    else:
        if sort:
            arr1inds = other_event.argsort()
            sorted_other_event = other_event[arr1inds[::-1]]
            sorted_next_poke = next_centre_poke_norm[arr1inds[::-1]]
        else:
            sorted_other_event = other_event
            sorted_next_poke = next_centre_poke_norm
        return sorted_other_event, state_name, title, sorted_next_poke, trial_nums, event_times


def get_block_change_info(behavioural_data, block_types, contra_side):
    first_contra_trial_post_block_change = {}
    first_contra_trial = []
    current_reward_amounts = []
    change_in_reward_amounts = []
    relative_values = []
    change_in_relative_values = []
    block_diff_value = behavioural_data['Reward block'].diff()
    block_changes = behavioural_data[block_diff_value != 0]
    block_changes = block_changes.reset_index()
    previous_blocks = block_changes['Reward block']
    block_changes = block_changes.drop(block_changes.index[0])
    previous_blocks = previous_blocks.drop(previous_blocks.index[-1])
    block_changes = block_changes.reset_index()
    first_contra_trial_post_block_change['previous block type'] = previous_blocks
    first_contra_trial_post_block_change['new block type'] = block_changes['Reward block']
    for index, block_change in block_changes.iterrows():
        trial_num = block_change['Trial num']
        counter = 0
        current_trial = block_change
        current_response = current_trial['Response']
        current_trial_num = current_trial['Trial num']
        while current_response != contra_side:
            counter += 1
            if trial_num + counter < max(behavioural_data['Trial num']):
                current_trial = behavioural_data[(behavioural_data['Trial num'] == trial_num + counter)]
                current_trial = current_trial[(current_trial['State name'] == 'TrialStart')]
                current_response = current_trial['Response'].values
                current_trial_num = current_trial['Trial num'].values[0]
            else:
                current_trial_num = -1
                current_response = contra_side

        if contra_side == 2:
            new_reward_amount = block_types[block_types['block type'] == block_change['Reward block']]['right reward'].values[0]
            new_other_amount = block_types[block_types['block type'] == block_change['Reward block']]['left reward'].values[0]
            old_reward_amount = block_types[block_types['block type'] == previous_blocks[index]]['right reward'].values[0]
            old_other_amount = block_types[block_types['block type'] == previous_blocks[index]]['left reward'].values[0]
        else:
            new_reward_amount = block_types[block_types['block type'] == block_change['Reward block']]['left reward'].values[0]
            new_other_amount = block_types[block_types['block type'] == block_change['Reward block']]['right reward'].values[0]
            old_reward_amount = block_types[block_types['block type'] == previous_blocks[index]]['left reward'].values[0]
            old_other_amount = block_types[block_types['block type'] == previous_blocks[index]]['right reward'].values[0]

        current_reward_amounts.append(new_reward_amount)
        change_in_reward_amounts.append(new_reward_amount - old_reward_amount)
        relative_values.append(new_reward_amount - new_other_amount)
        change_in_relative_values.append(
            (new_reward_amount - new_other_amount) - (old_reward_amount - old_other_amount))
        first_contra_trial.append(current_trial_num)

    first_contra_trial_post_block_change['first contra trial num'] = first_contra_trial
    first_contra_trial_post_block_change['block change trial num'] = block_changes['Trial num']
    first_contra_trial_post_block_change['new reward amounts'] = current_reward_amounts
    first_contra_trial_post_block_change['change in reward amounts'] = change_in_reward_amounts
    first_contra_trial_post_block_change['new relative value'] = relative_values
    first_contra_trial_post_block_change['change in relative value'] = change_in_relative_values
    block_change_info = pd.DataFrame(first_contra_trial_post_block_change)
    return block_change_info


def add_traces_and_peaks(block_change_info, all_trials_traces):
    changes_with_no_contra_trials = block_change_info.index[(block_change_info['first contra trial num'] == -1)]
    block_change_info = block_change_info.drop(block_change_info.index[changes_with_no_contra_trials])
    traces = []
    peaks = []
    for index, change in block_change_info.iterrows():
        trial_num = change['first contra trial num']
        indx = np.where(all_trials_traces.contra_data.trial_nums == trial_num)[0]
        trace = all_trials_traces.contra_data.sorted_traces[indx,:]
        peak = all_trials_traces.contra_data.trial_peaks[int(indx)]
        traces.append(np.squeeze(trace.T))
        peaks.append(peak)
    block_change_info['traces'] = pd.Series(traces, index=block_change_info.index)
    block_change_info['peak size'] = pd.Series(peaks, index=block_change_info.index)
    return block_change_info, all_trials_traces.contra_data.time_points


def one_session_get_block_changes(mouse_id, date, block_types):
    all_experiments = get_all_experimental_records()
    experiment_to_process = all_experiments[
        (all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse_id)]
    session_data, behavioural_data = open_experiment(experiment_to_process)
    reward_block = 'all'
    params = {'state_type_of_interest': 5,
        'outcome': 2,
        'last_outcome': 0,  # NOT USED CURRENTLY
        'no_repeats' : 1,
        'last_response': 0,
        'align_to' : 'Time start',
        'instance':- 1,
        'plot_range': [-6, 6],
        'first_choice_correct': 0,
         'cue': 'None'}
    all_trials_traces = CustomAlignedDataRewardBlocks(session_data, params, reward_block)
    block_change_info = get_block_change_info(behavioural_data, block_types, all_trials_traces.contra_fiber_side_numeric)
    block_change_info, timepoints = add_traces_and_peaks(block_change_info, all_trials_traces)
    return block_change_info, timepoints


def plot_mean_trace_for_condition(ax, site, block_change_info, time_points, key, error_bar_method=None, save_location=None):
    mouse = block_change_info['mouse'].iloc[0]
    if key == 'contra reward amount':
        condition = 'abs'
        leg_title = 'Absolute reward value (uL)'
    elif key == 'relative reward amount':
        condition = 'rel'
        leg_title = 'Relative reward value (uL)'
    else:
        raise ValueError('Condition not recognised')

    reward_amounts = np.sort(block_change_info[key].unique())
    colours = cm.inferno(np.linspace(0, 0.8, reward_amounts.shape[0]))

    for reward_indx, reward_amount in enumerate(reward_amounts):
        rows = block_change_info[(block_change_info[key] == reward_amount)]
        traces = rows['traces'].values
        flat_traces = np.zeros([traces.shape[0], traces[0].shape[0]])
        for idx, trace in enumerate(traces):
            flat_traces[idx, :] = trace
        subfig = 'I' if site == 'tail' else 'K'
        csv_file = os.path.join(spreadsheet_path, 'ED_fig7', f'EDfig7{subfig}_reward_amount_{reward_amount}_traces_{site}.csv')
        if not os.path.exists(csv_file):
            df_for_spreadsheet = pd.DataFrame(flat_traces.T)
            df_for_spreadsheet.insert(0, "Timepoints", time_points)
            df_for_spreadsheet.to_csv(csv_file)
        mean_trace = np.mean(flat_traces, axis=0)
        ax.plot(time_points, mean_trace, lw=1.5, color=colours[reward_indx], label=reward_amount)

        if error_bar_method is not None:
            # bootstrapping takes a long time. calculate once and save:
            filename = 'errors_{}_{}_{}_clipped.npz'.format(mouse, condition, int(reward_amount))
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
                             facecolor=colours[reward_indx], linewidth=0)

    ax.axvline(0, color='k')
    ax.set_xlim([-2, 2])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('z-scored fluorescence')


def plot_mean_trace_for_condition_value_switch(ax, block_change_info, time_points, key, session_id, possible_values, error_bar_method=None, save_location=None):
    mouse = block_change_info['mouse'].iloc[0]
    if key == 'contra reward amount':
        condition = 'abs'
        leg_title = 'Absolute reward value (uL)'
    elif key == 'relative reward amount':
        condition = 'rel'
        leg_title = 'Relative reward value (uL)'
    else:
        raise ValueError('Condition not recognised')

    reward_amounts = np.sort(block_change_info[key].unique())
    colours = cm.inferno(np.linspace(0, 0.8, np.shape(possible_values)[0]))
    all_time_points = decimate(time_points, 10)
    start_plot = int(all_time_points.shape[0] / 2 - 2 * 1000)
    end_plot = int(all_time_points.shape[0] / 2 + 2 * 1000)
    time_points = all_time_points[start_plot: end_plot]

    for reward_indx, reward_amount in enumerate(reward_amounts):
        rows = block_change_info[(block_change_info[key] == reward_amount)]
        traces = rows['traces'].values
        flat_traces = np.zeros([traces.shape[0], traces[0].shape[0]])
        for idx, trace in enumerate(traces):
            flat_traces[idx, :] = trace
        mean_trace = decimate(np.mean(flat_traces, axis=0), 10)[start_plot:end_plot]
        colour_ind = np.where(possible_values == reward_amount)[0][0]
        print(colours[colour_ind])
        ax.plot(time_points, mean_trace, lw=1.5, color=colours[colour_ind], label=reward_amount)
        if error_bar_method is not None:
            # bootstrapping takes a long time. calculate once and save:
            filename = 'errors_{}_{}_{}_{}_{}_clipped.npz'.format(mouse, condition, int(reward_amount), 'value_switch', session_id)
            if not os.path.isfile(os.path.join(save_location, filename)):
                error_bar_lower, error_bar_upper = calculate_error_bars(mean_trace,
                                                                        decimate(flat_traces, 10)[:, start_plot:end_plot],
                                                                        error_bar_method=error_bar_method)
                np.savez(os.path.join(save_location, filename), error_bar_lower=error_bar_lower,
                         error_bar_upper=error_bar_upper)
            else:
                error_info = np.load(os.path.join(save_location, filename))
                error_bar_lower = error_info['error_bar_lower']
                error_bar_upper = error_info['error_bar_upper']
            ax.fill_between(time_points, error_bar_lower, error_bar_upper, alpha=0.5,
                            facecolor=colours[colour_ind], linewidth=0)

    ax.axvline(0, color='k')
    ax.set_xlim([-2, 2])
    ax.set_xlabel('time (s)')
    ax.set_ylabel('z-scored fluorescence')
    # keys.set_title(key)
    # lg = keys.legend(title=leg_title, bbox_to_anchor=(1., 1.), fontsize=14)
    # lg.get_title().set_fontsize(14)


