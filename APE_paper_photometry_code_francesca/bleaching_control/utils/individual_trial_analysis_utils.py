import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
import peakutils
import pandas as pd
from set_global_params import processed_data_path, daq_sample_rate


class HeatMapParams(object):
    def __init__(self, params, response, first_choice):
        """
        Sets the parameters for aligning traces to behavioural events
        Args:
            params (dict): parameters, e.g.
                        {'state_type_of_interest': 5, # Bpod state number to align to
                        'outcome': 2,                 # 0=incorrect, 1=correct, 2=correct and incorrect
                        'last_outcome': 0,            # NOT USED CURRENTLY
                        'no_repeats': 0,              # don't include bpod states that are repeated? 0=no, 1=yes (can only be used with instance=1)
                        'last_response': 0,           # 0=left and right, 1=left, 2=right
                        'align_to': 'Time end',       # What time point of bpod state to align to ('Time start' or 'Time end' )
                        'instance': -1,               # What instance of bpod state to choose for alignment. 1=first, -1=last
                        'plot_range': [-6, 6],        # How many seconds of photometry data to normalise to and save out aroung event
                        'first_choice_correct': 0,    # Trials where the first choice is correct? (1=yes, 0=no) Not the same as outcome as in non-punished sessions outcome is always 1.
                        'cue': 'None'                 # 'None', 'High', 'Low'. If you want a specific cue type.}
            response (int): 1 (left) or 2 (right) - this is the final choice
            first_choice (int): 1 (left) or 2 (right) - the first choice (not always the
             same as final choice in unpunished sessions)
        """
        self.state = params['state_type_of_interest']
        self.outcome = params['outcome']
        #self.last_outcome = params['last_outcome']
        self.response = response
        self.last_response = params['last_response']
        self.align_to = params['align_to']
        self.other_time_point = np.array(['Time start', 'Time end'])[np.where(np.array(['Time start', 'Time end']) != params['align_to'])]
        self.instance = params['instance']
        self.plot_range = params['plot_range']
        self.no_repeats = params['no_repeats']
        self.first_choice_correct = params['first_choice_correct']
        self.first_choice = first_choice
        self.cue = params['cue']


def get_photometry_around_event(all_trial_event_times, demodulated_trace, pre_window=5, post_window=5, sample_rate=daq_sample_rate):
    """
    Gets photometry traces around behavioural event times
    Args:
        all_trial_event_times (list): all the time points where there is an event to align to
        demodulated_trace (np.array): demodulated photometry signal
        pre_window (float): time (seconds) before event to get photometry data for
        post_window (float): time (seconds) after event to get photometry data for
        sample_rate (float): photometry data sample rate (Hz)

    Returns:
        event_photo_traces (np.array): traces aligned to events with pre_window and post_window seconds before and after
    """
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
    return event_photo_traces


def get_next_centre_poke(trial_data, events_of_int, last_trial):
    """
    Finds the initiation of the next trial (triggered by a centre poke)
    Args:
        trial_data (pd.dataframe): behavioural data
        events_of_int (np.array): event times
        last_trial (bool): is the last trial of the session present in the events?

    Returns:
        next_centre_poke_times (np.array): times of the next centre pokes after aligned to events
    """
    next_centre_poke_times = np.zeros(events_of_int.shape[0])
    events_of_int = events_of_int.reset_index(drop=True)
    for i, event in events_of_int[:-1].iterrows():
        trial_num = event['Trial num']
        if last_trial:
            next_centre_poke_times[i] = events_of_int['Trial end'].values[i] + 2
        else:
            next_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num + 1)]
            wait_for_pokes = next_trial_events.loc[(next_trial_events['State type'] == 2)]
            next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)]
            next_centre_poke_times[i] = next_wait_for_poke['Time end'].values[0]
    if last_trial:
        next_centre_poke_times[-1] = events_of_int['Trial end'].values[-1] + 2
    else:
        event = events_of_int.tail(1)
        trial_num = event['Trial num'].values[0]
        next_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num + 1)]
        wait_for_pokes = next_trial_events.loc[(next_trial_events['State type'] == 2)]
        next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)]
        next_centre_poke_times[-1] = next_wait_for_poke['Time end'].values[0]
    return next_centre_poke_times


def get_first_poke(trial_data, events_of_int):
    """
    Gets trial start times for each event
    Args:
        trial_data (pd.dataframe): behavioural data
        events_of_int (np.array): event times

    Returns:
        first_poke_times (np.array): times of trial starts for each event
    """
    trial_numbers = events_of_int['Trial num'].unique()
    first_poke_times = np.zeros(events_of_int.shape[0])
    events_of_int = events_of_int.reset_index(drop=True)
    for event_trial_num in trial_numbers:
        trial_num = event_trial_num
        event_indx_for_that_trial = events_of_int.loc[(events_of_int['Trial num'] == trial_num)].index
        trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num)]
        wait_for_pokes = trial_events.loc[(trial_events['State type'] == 2)]
        next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)]
        first_poke_times[event_indx_for_that_trial] = next_wait_for_poke['Time end'].values[0]-1
    return first_poke_times


def get_next_reward_time(trial_data, events_of_int):
    """
    Finds the time of the next reward (or lack of reward on incorrect trials) for each event
    Args:
        trial_data (pd.dataframe): behavioural data
        events_of_int (np.array): event times

    Returns:
        next_reward_times (list): times of the next outcome for each event
    """
    trial_numbers = events_of_int['Trial num'].values
    next_reward_times = []
    for event_trial_num in range(len(trial_numbers)):
        trial_num = trial_numbers[event_trial_num]
        other_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num)]
        choices = other_trial_events.loc[(other_trial_events['State type'] == 5)]
        max_times_in_state_choices = choices['Max times in state'].unique()
        choice = choices.loc[(choices['Instance in state'] == max_times_in_state_choices)]
        next_reward_times.append(choice['Time end'].values[0])
    return next_reward_times


def find_and_z_score_traces(trial_data, demod_signal, params, norm_window=8, sort=False, get_photometry_data=True):
    """
    Aligns and extracts photometry data to behavioural events based on params
    Args:
        trial_data (pd.dataframe): behavioural data (one session)
        demod_signal (np.array): demodulated photometry signal
        params (HeatMapParams): parameters for aligning photometry data
        norm_window (float): window (seconds) around event to zscore photometry data to
        sort (bool): whether to sort trials by reaction time
        get_photometry_data (bool): whether to get the photometry traces or just behavioural timestamps

    Returns:
        time_points (list): time points in seconds for the mean and single trial traces
        mean_trace (np.array): average trace, aligned to behaviuoral event
        sorted_traces (np.array): all traces for trials that meet these params, aligned to behavioural event
        sorted_other_event (list): either end of bpod state or reward time, or next trial start
        state_name (str): name of bpod state that events are aligned to
        title (str): title for plots
        sorted_next_poke (list): other relevant behavioural event timestamp (normally either bpod state start or end)
        trial_nums (list): trial numbers that correspond to sored_traces
        event_times (list): actual event times relative to raw data time
        trial_starts (list): time of trial starts
        trial_ends (list): time of trial ends
    """
    response_names = ['both left and right', 'left', 'right']
    outcome_names = ['incorrect', 'correct', 'both correct and incorrect']

    if params.state == 10 or params.state == 12 or params.state == 13:  # omissions, large rewards left and right
        if params.state == 10:
            omission_events = trial_data.loc[(trial_data['State type'] == params.state)]
        else:
            left_large_reward_events = trial_data.loc[(trial_data['State type'] == 12)]
            right_large_reward_events = trial_data.loc[(trial_data['State type'] == 13)]
            omission_events = pd.concat([left_large_reward_events, right_large_reward_events])

        trials_of_int = omission_events['Trial num'].values
        omission_trials_all_states = trial_data.loc[(trial_data['Trial num'].isin(trials_of_int))]
        events_of_int = omission_trials_all_states.loc[
            (omission_trials_all_states['State type'] == 5)]  # get the action aligned trace
    elif params.state == 5.5: # this is the time of the first incorrect response if the punishment isn't on
        events_of_int = trial_data.loc[
            np.logical_or((trial_data['State type'] == 5.5), (trial_data['State type'] == 5))]
        trial_nums = events_of_int['Trial num'].unique()
        for trial in trial_nums:
            events = events_of_int[events_of_int['Trial num'] == trial]
            if events.shape[0] > 1:
                second_event = pd.to_numeric(events['Time end']).idxmax()
                if events['State type'][second_event] == 5.5:
                    events_of_int = events_of_int.drop(second_event)

    else:
        print(trial_data.shape)
        events_of_int = trial_data.loc[(trial_data['State type'] == params.state)]

    if params.response != 0:
        if params.state == 5.5:
            correct_choices = np.logical_and(events_of_int['Response'] == params.response,
                                             events_of_int['State type'] == 5)
            incorrect_first_choices = np.logical_and(events_of_int['First response'] == params.response,
                                                     events_of_int['State type'] == 5.5)
            events_of_int = events_of_int.loc[np.logical_or(correct_choices, incorrect_first_choices)]
        else:
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

    if params.first_choice_correct == 1:
        events_of_int = events_of_int.loc[
            (events_of_int['First choice correct'] == 1)]
    elif params.first_choice_correct == -1:
        events_of_int = events_of_int.loc[np.logical_or(
            (events_of_int['First choice correct'] == 0), (events_of_int['First choice correct'].isnull()))]
        if events_of_int['State type'].isin([5.5]).any():
            events_of_int = events_of_int.loc[events_of_int['First choice correct'].isnull()]

    events_of_int_reset = events_of_int.reset_index(drop=True)
    if events_of_int['State type'].isin([5.5]).any():
        event_times = np.zeros(events_of_int_reset.shape[0])
        for i, event in events_of_int_reset.iterrows():
            if event['State type'] == 5.5:
                align_to = 'Time start'
                event_times[i] = event[align_to]
            else:
                event_times[i] = event[params.align_to]
    else:
        event_times = events_of_int[params.align_to].values
    trial_nums = events_of_int['Trial num'].values
    #trial_starts = events_of_int['Trial start'].values
    trial_ends = events_of_int['Trial end'].values

    if params.state == 5.5:

        other_event = np.zeros([events_of_int_reset.shape[0]])
        for i, event in events_of_int_reset.iterrows():
            if event['State type'] == 5.5:
                trial_num = event['Trial num']
                this_trial_data = trial_data[trial_data['Trial num'] == trial_num]
                out_of_centre = this_trial_data[this_trial_data['State type'] == 4].tail(1)['Time end'].values[0]
                other_event[i] = out_of_centre - np.squeeze(event[params.align_to])
            else:
                other_event[i] = np.squeeze(event[params.other_time_point]) - np.squeeze(event[params.align_to])
    else:
        other_event = np.asarray(
            np.squeeze(events_of_int[params.other_time_point].values) - np.squeeze(events_of_int[params.align_to].values))
    if params.state == 12 or params.state == 13:
        state_name = 'LargeReward'
    else:
        state_name = events_of_int['State name'].values[0]

    last_trial = np.max(trial_data['Trial num'])
    last_trial_num = events_of_int['Trial num'].unique()[-1]
    events_reset_indx = events_of_int.reset_index(drop=True)
    last_trial_event_indx = events_reset_indx.loc[(events_reset_indx['Trial num'] == last_trial_num)].index
    next_centre_poke = get_next_centre_poke(trial_data, events_of_int, last_trial_num==last_trial)
    trial_starts = get_first_poke(trial_data, events_of_int)
    outcome_times = get_next_reward_time(trial_data, events_of_int)
    outcome_times = outcome_times - event_times

    #print(events_of_int.shape)
    # this all deals with getting photometry data
    if get_photometry_data == True:
        next_centre_poke[last_trial_event_indx] = events_reset_indx[params.align_to].values[last_trial_event_indx] + 1 # so that you can find reward peak
        next_centre_poke_norm = next_centre_poke - event_times

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
            sorted_next_poke = next_centre_poke[arr1inds[::-1]]
        else:
            sorted_other_event = other_event
            sorted_next_poke = next_centre_poke
        return sorted_other_event, state_name, title, sorted_next_poke, trial_nums, event_times, trial_starts, trial_ends


def get_peak_each_trial(sorted_traces, time_points, sorted_other_events):
    """
    Finds peak of photometry traces between time_points and sorted_other_events.
    Can return some events where no peak is found which will be empty lists []
    Args:
        sorted_traces (np.array): photometry traces (may be sorted by reaction time or not)
        time_points (list): time points for the traces
        (window around event in seconds - 0 is the time of aligned to event)
        sorted_other_events (list): time of next event relative to aligned to event

    Returns:
        flat_peaks (list):
    """
    all_trials_peaks = []
    for trial_num in range(0, len(sorted_other_events)):
        indices_to_integrate = np.where(np.logical_and(np.greater_equal(time_points, 0), np.less_equal(time_points, sorted_other_events[trial_num])))
        trial_trace = (sorted_traces[trial_num, indices_to_integrate]).T
        trial_trace = trial_trace # - trial_trace[0]s
        trial_peak_inds = peakutils.indexes(trial_trace.flatten('F'))
        if len(trial_peak_inds>1):
            trial_peak_inds = trial_peak_inds[0]
        trial_peaks = trial_trace.flatten('F')[trial_peak_inds]
        all_trials_peaks.append(trial_peaks)
    flat_peaks = all_trials_peaks
    return flat_peaks


def get_peak_each_trial_no_nans(sorted_traces, time_points, sorted_other_events):
    """
    Finds peak of photometry traces between time_points and sorted_other_events. If peak finding fails, takes max.
    Args:
        sorted_traces (np.array): photometry traces (may be sorted by reaction time or not)
        time_points (list): time points for the traces
        (window around event in seconds - 0 is the time of aligned to event)
        sorted_other_events (list): time of next event relative to aligned to event

    Returns:
        flat_peaks (list):
    """
    all_trials_peaks = []
    for trial_num in range(0, len(sorted_other_events)):
        indices_to_integrate = np.where(np.logical_and(np.greater_equal(time_points, 0), np.less_equal(time_points, sorted_other_events[trial_num])))
        trial_trace = (sorted_traces[trial_num, indices_to_integrate]).T
        trial_trace = trial_trace # - trial_trace[0]s
        trial_peak_inds = peakutils.indexes(trial_trace.flatten('F'))
        if trial_peak_inds.shape[0] > 0 or len(trial_peak_inds > 1):
            trial_peak_inds = trial_peak_inds[0]
            trial_peaks = trial_trace.flatten('F')[trial_peak_inds]
        else:
            trial_peak_inds = np.argmax(trial_trace)
            trial_peaks = np.max(trial_trace)
        all_trials_peaks.append(trial_peaks)
    flat_peaks = all_trials_peaks
    return flat_peaks


def get_peak_each_trial_with_nans(sorted_traces, time_points, sorted_other_events):
    """
    Finds peak of photometry traces between time_points and sorted_other_events.
    Can return some events where no peak is found but replaces with nans
    Args:
        sorted_traces (np.array): photometry traces (may be sorted by reaction time or not)
        time_points (list): time points for the traces
        (window around event in seconds - 0 is the time of aligned to event)
        sorted_other_events (list): time of next event relative to aligned to event

    Returns:
        flat_peaks (list):
    """
    all_trials_peaks = []
    for trial_num in range(0, len(sorted_other_events)):
        indices_to_integrate = np.where(np.logical_and(np.greater_equal(time_points, 0),
                                                       np.less_equal(time_points, sorted_other_events[trial_num])))
        trial_trace = (sorted_traces[trial_num, indices_to_integrate]).T
        trial_trace = trial_trace  #- trial_trace[0]
        trial_peak_inds = peakutils.indexes(trial_trace.flatten('F'), thres=0.3)
        if trial_peak_inds.shape[0] > 0:
            if len(trial_peak_inds > 1):
                trial_peak_inds = trial_peak_inds[0]
            trial_peaks = trial_trace.flatten('F')[trial_peak_inds]
        else:
            trial_peaks = np.nan
        all_trials_peaks.append(trial_peaks)
    flat_peaks = all_trials_peaks
    return flat_peaks


class SessionData(object):
    def __init__(self, fiber_side, recording_site, mouse_id, date):
        """
        Creates object that will contain photometry aligned to all sorts of behavioural events
        Args:
            fiber_side (str): 'left' or 'right'
            recording_site (str): 'Nacc' or 'tail'
            mouse_id (str): normally something like 'SNL_photo17'
            date (str): 'YYYYMMDD'
        """
        self.mouse = mouse_id
        self.fiber_side = fiber_side
        self.recording_site = recording_site
        self.date = date
        self.choice_data = None
        self.cue_data = None
        self.reward_data = None
        self.outcome_data = None

    def get_choice_responses(self, save_traces=True):
        self.choice_data = ChoiceAlignedData(self, save_traces=save_traces)

    def get_cue_responses(self, save_traces=True):
        self.cue_data = CueAlignedData(self, save_traces=save_traces)

    def get_reward_responses(self, save_traces=True):
        self.reward_data = RewardAlignedData(self, save_traces=save_traces)

    def get_outcome_responses(self, save_traces=True):
        self.outcome_data = RewardAndNoRewardAlignedData(self, save_traces=save_traces)


class SessionDataPsychometric(object):
    def __init__(self, fiber_side, recording_site, mouse_id, date):
        self.mouse = mouse_id
        self.fiber_side = fiber_side
        self.recording_site = recording_site
        self.date = date
        self.choice_data = None
        self.cue_data = None
        self.reward_data = None
        self.outcome_data = None

    def get_choice_responses(self, save_traces=True):
        self.choice_data = ChoiceAlignedData(self, save_traces=save_traces)

    def get_cue_responses(self, save_traces=True):
        self.cue_data = CueAlignedData(self, save_traces=save_traces)

    def get_reward_responses(self, save_traces=True):
        self.reward_data = RewardAlignedData(self, save_traces=save_traces)

    def get_outcome_responses(self, save_traces=True):
        self.outcome_data = RewardAndNoRewardAlignedDataPsychometric(self, save_traces=save_traces)


class SessionEvents(object):
    def __init__(self, fiber_side, recording_site, mouse_id, date):
        self.mouse = mouse_id
        self.fiber_side = fiber_side
        self.recording_site = recording_site
        self.date = date
        self.choice_data = None
        self.cue_data = None
        self.reward_data = None

    def get_reaction_times(self, dff, trial_data):
        self.ipsi_reaction_times, state_name, title, ipsi_sorted_next_poke, self.ipsi_trial_nums = find_and_z_score_traces(
        trial_data, dff, self.ipsi_params, sort=True, get_photometry_data=False)
        self.contra_reaction_times, state_name, title, contra_sorted_next_poke, self.contra_trial_nums = find_and_z_score_traces(
        trial_data, dff, self.ipsi_params, sort=True, get_photometry_data=False)

    def get_choice_events(self):
        self.choice_data = ChoiceAlignedEvents(self)

    def get_cue_events(self):
        self.cue_data = CueAlignedEvents(self)

    def get_reward_events(self):
        self.reward_data = RewardAlignedEvents(self)


class ZScoredTraces(object):
    def __init__(self,  trial_data, dff, params, response, first_choice):
        """
        Creates object with traces and behavioural information for each event that is aligned to.
        Args:
            trial_data (pd.dataframe):
            dff (np.array):
            params (dict):
            response (int): 1 (left) or 2 (right)
            first_choice (int): 1 (left) or 2 (right)
        """
        self.trial_peaks = None
        self.params = HeatMapParams(params, response, first_choice)
        self.time_points, self.mean_trace, self.sorted_traces, self.reaction_times, self.state_name, title, self.sorted_next_poke, self.trial_nums, self.event_times, self.outcome_times = find_and_z_score_traces(
            trial_data, dff, self.params, sort=False)


    def get_peaks(self, save_traces=True):
        """
        Determines time window to look for peak photometry response and finds peaks for traces
        Args:
            save_traces (bool): if you only want the response size for each event but don;t want to save the traces
            (large file - takes time to load and save), you can set this to False

        Returns:
        """
        if self.params.align_to == 'Time start':
            other_time_point = self.outcome_times
        else: # for outcome aligned data
            other_time_point = self.sorted_next_poke # uses the start of the next trial
        self.trial_peaks = get_peak_each_trial_no_nans(self.sorted_traces, self.time_points, other_time_point)
        if not save_traces:
            self.sorted_traces = None


class BehaviouralEvents(object):
    def __init__(self,  trial_data, dff, params, response, first_choice):
        self.params = HeatMapParams(params, response, first_choice)
        self.reaction_times, self.state_name, title, self.sorted_next_poke, self.trial_nums, self.event_times, self.trial_starts, self.trial_ends = find_and_z_score_traces(
            trial_data, dff, self.params, sort=False,  get_photometry_data=False)

    def filter_by_trial_nums(self, trial_nums):
        inds = np.isin(self.trial_nums, trial_nums)
        self.trial_nums = self.trial_nums[inds]
        self.trial_ends = self.trial_ends[inds]
        self.trial_starts = self.trial_starts[inds]
        self.reaction_times = self.reaction_times[inds]
        self.sorted_next_poke = self.sorted_next_poke[inds]
        self.event_times = self.event_times[inds]


class ChoiceAlignedData(object):
    def __init__(self, session_data, save_traces=True):
        """
        Creates object and loads session photometry and behavioural data.
        Then aligns to choice for ipsi and contra choices.
        Args:
            session_data (SessionData): the session object with information about fiber side, dat, mouse ID etc...
            save_traces (bool): should photometry traces be aligned and saved or only behavioural data?
        """
        saving_folder = os.path.join(processed_data_path, session_data.mouse)
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(os.path.join(saving_folder, restructured_data_filename))
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(os.path.join(saving_folder, dff_trace_filename))

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 5,
            'outcome': 2,
            'last_outcome': 0,  # NOT USED CURRENTLY
            'no_repeats' : 1,
            'last_response': 0,
            'align_to' : 'Time start',
            'instance': -1,
            'plot_range': [-6, 6],
            'first_choice_correct': 0,
            'cue': None}

        self.ipsi_data = ZScoredTraces(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.ipsi_data.get_peaks(save_traces=save_traces)

        self.contra_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        self.contra_data.get_peaks(save_traces=save_traces)


class ChoiceAlignedEvents(object):
    def __init__(self, session_data):
        saving_folder = processed_data_path + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 5,
            'outcome': 2,
            'last_outcome': 0,  # NOT USED CURRENTLY
            'no_repeats' : 1,
            'last_response': 0,
            'align_to' : 'Time start',
            'instance': 1,#used to be -1 but now better to not allow repeats
            'plot_range': [-6, 6],
            'first_choice_correct': 0,
            'cue': None}


        self.ipsi_data = BehaviouralEvents(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.contra_data = BehaviouralEvents(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)


class CueAlignedData(object):
    def __init__(self, session_data, save_traces=True):
        """
        Creates object and loads session photometry and behavioural data.
        Then aligns to cue for ipsi and contra choices.
        Args:
            session_data (SessionData): the session object with information about fiber side, dat, mouse ID etc...
            save_traces (bool): should photometry traces be aligned and saved or only behavioural data?
        """
        saving_folder = processed_data_path + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 3,
            'outcome': 2,
            'last_outcome': 0,  # NOT USED CURRENTLY
            'no_repeats' : 1,
            'last_response': 0,
            'align_to' : 'Time start',
            'instance': -1,
            'plot_range': [-6, 6],
            'first_choice_correct': 0,
            'cue': None}

        self.ipsi_data = ZScoredTraces(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.ipsi_data.get_peaks(save_traces=save_traces)
        self.contra_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        self.contra_data.get_peaks(save_traces=save_traces)

        params['cue'] = 'high'
        self.high_cue_data = ZScoredTraces(trial_data, dff, params, 0, 0)
        self.high_cue_data.get_peaks(save_traces=save_traces)
        params['cue'] = 'low'
        self.low_cue_data = ZScoredTraces(trial_data, dff, params, 0, 0)
        self.low_cue_data.get_peaks(save_traces=save_traces)


class CueAlignedSidedData(object):
    def __init__(self, session_data, save_traces=True):
        saving_folder = processed_data_path + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 3,
            'outcome': 2,
            'last_outcome': 0,  # NOT USED CURRENTLY
            'no_repeats' : 1,
            'last_response': 0,
            'align_to' : 'Time start',
            'instance': -1,
            'plot_range': [-6, 6],
            'first_choice_correct': 0,
            'cue': None}

        self.ipsi_data = ZScoredTraces(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.ipsi_data.get_peaks(save_traces=save_traces)
        self.contra_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        self.contra_data.get_peaks(save_traces=save_traces)

        params['cue'] = 'high'
        self.high_cue_ipsi_data =  ZScoredTraces(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.high_cue_contra_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        self.high_cue_ipsi_data.get_peaks(save_traces=save_traces)
        self.high_cue_ipsi_data.get_peaks(save_traces=save_traces)

        params['cue'] = 'low'
        self.low_cue_ipsi_data = ZScoredTraces(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.low_cue_contra_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric,
                                                  contra_fiber_side_numeric)
        self.low_cue_ipsi_data.get_peaks(save_traces=save_traces)
        self.low_cue_contra_data.get_peaks(save_traces=save_traces)


class CueAlignedEvents(object):
    def __init__(self, session_data):
        saving_folder = processed_data_path + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 3,
            'outcome': 2,
            'last_outcome': 0,  # NOT USED CURRENTLY
            'no_repeats' : 1,
            'last_response': 0,
            'align_to' : 'Time start',
            'instance': 1,
            'plot_range': [-6, 6],
            'first_choice_correct': 0,
             'cue': None}

        self.ipsi_data = BehaviouralEvents(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)
        self.contra_data = BehaviouralEvents(trial_data, dff,params, contra_fiber_side_numeric, contra_fiber_side_numeric)
        params['cue'] = 'high'
        self.high_cue_data = BehaviouralEvents(trial_data, dff, params, 0, 0)
        params['cue'] = 'low'
        self.low_cue_data = BehaviouralEvents(trial_data, dff, params, 0, 0)


class RewardAlignedData(object):
    def __init__(self, session_data, save_traces=True):
        """
        Creates object and loads session photometry and behavioural data.
        Then aligns to reward for ipsi and contra choices.
        Args:
            session_data (SessionData): the session object with information about fiber side, dat, mouse ID etc...
            save_traces (bool): should photometry traces be aligned and saved or only behavioural data?
        """
        saving_folder = processed_data_path + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        fiber_options = np.array(['left', 'right'])
        fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]

        params = {'state_type_of_interest': 5,
                  'outcome': 1,
                  'last_outcome': 0,  # NOT USED CURRENTLY
                  'no_repeats': 0,
                  'last_response': 0,
                  'align_to': 'Time end',
                  'instance': -1,
                  'plot_range': [-6, 6],
                  'first_choice_correct': 1,
                  'cue': 'None'}

        self.ipsi_data = ZScoredTraces(trial_data, dff, params, fiber_side_numeric, fiber_side_numeric)

        self.contra_data = ZScoredTraces(trial_data, dff, params, contra_fiber_side_numeric, contra_fiber_side_numeric)


class RewardAlignedEvents(object):
    def __init__(self, session_data):
        saving_folder = processed_data_path + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)

        params = {'state_type_of_interest': 5.5, #5.5
                  'outcome': 2,
                  'last_outcome': 0,  # NOT USED CURRENTLY
                  'no_repeats': 0,
                  'last_response': 0,
                  'align_to': 'Time end',
                  'instance': -1,
                  'plot_range': [-6, 6],
                  'first_choice_correct': -1,
                  'cue': 'None'}
        self.no_reward_data = BehaviouralEvents(trial_data, dff, params, 0, 0)
        params = {'state_type_of_interest': 5,
                  'outcome': 1,
                  'last_outcome': 0,  # NOT USED CURRENTLY
                  'no_repeats': 0,
                  'last_response': 0,
                  'align_to': 'Time end',
                  'instance': -1,
                  'plot_range': [-6, 6],
                  'first_choice_correct': 1,
                  'cue': 'None'}
        self.reward_data = BehaviouralEvents(trial_data, dff, params, 0, 0)


class RewardAndNoRewardAlignedData(object):
    def __init__(self, session_data, save_traces=True):
        """
        Creates object and loads session photometry and behavioural data.
        Then aligns to outcome for ipsi and contra choices.
        Outcome incudes incorrect trials and can be used in sessions where punishment (timeout) is not used.
        Here the outcome time on incorrect choice trials is the first incorrect poke.
        Args:
            session_data (SessionData): the session object with information about fiber side, dat, mouse ID etc...
            save_traces (bool): should photometry traces be aligned and saved or only behavioural data?
        """
        saving_folder = processed_data_path + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)
        response = 0

        params = {'state_type_of_interest': 5.5, #5.5
                  'outcome': 2,
                  'last_outcome': 0,  # NOT USED CURRENTLY
                  'no_repeats': 0,
                  'last_response': 0,
                  'align_to': 'Time end',
                  'instance': -1,
                  'plot_range': [-6, 6],
                  'first_choice_correct': -1,
                  'cue': 'None'}
        self.no_reward_data = ZScoredTraces(trial_data, dff, params, response, response)
        if not save_traces:
            self.no_reward_data.sorted_traces=[]
        params = {'state_type_of_interest': 5,
                  'outcome': 1,
                  'last_outcome': 0,  # NOT USED CURRENTLY
                  'no_repeats': 0,
                  'last_response': 0,
                  'align_to': 'Time end',
                  'instance': -1,
                  'plot_range': [-6, 6],
                  'first_choice_correct': 1,
                  'cue': 'None'}
        self.reward_data = ZScoredTraces(trial_data, dff, params, response, response)
        if not save_traces:
            self.reward_data.sorted_traces = []


class RewardAndNoRewardAlignedDataPsychometric(object):
    def __init__(self, session_data, save_traces=True):
        saving_folder = processed_data_path + session_data.mouse + '\\'
        restructured_data_filename = session_data.mouse + '_' + session_data.date + '_' + 'restructured_data.pkl'
        trial_data = pd.read_pickle(saving_folder + restructured_data_filename)
        dff_trace_filename = session_data.mouse + '_' + session_data.date + '_' + 'smoothed_signal.npy'
        dff = np.load(saving_folder + dff_trace_filename)
        response = 0

        params = {'state_type_of_interest': 5, #5.5
                  'outcome': 0,
                  'last_outcome': 0,  # NOT USED CURRENTLY
                  'no_repeats': 0,
                  'last_response': 0,
                  'align_to': 'Time end',
                  'instance': -1,
                  'plot_range': [-6, 6],
                  'first_choice_correct': -1,
                  'cue': 'None'}
        self.no_reward_data = ZScoredTraces(trial_data, dff, params, response, response)
        self.no_reward_data.get_peaks(save_traces=save_traces)
        if not save_traces:
            self.no_reward_data.sorted_traces=[]
        params = {'state_type_of_interest': 5,
                  'outcome': 1,
                  'last_outcome': 0,  # NOT USED CURRENTLY
                  'no_repeats': 0,
                  'last_response': 0,
                  'align_to': 'Time end',
                  'instance': -1,
                  'plot_range': [-6, 6],
                  'first_choice_correct': 1,
                  'cue': 'None'}
        self.reward_data = ZScoredTraces(trial_data, dff, params, response, response)
        self.reward_data.get_peaks(save_traces=save_traces)
        if not save_traces:
            self.reward_data.sorted_traces = []




