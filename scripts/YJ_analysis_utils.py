
from scipy import stats
import numpy as np

def test(a, b):
    c = a + b
    return c

class SessionData(object):
    def __int__(self, mouse, date, fiber_side, protocol, trial_data, photometry_data):
        self.mouse = mouse
        self.date = date
        self.fiber_side = fiber_side
        self.protocol = protocol
        self.choice = None
        self.cue = None
        self.reward = None

        if protocol != 'SOR':
            #self.choice = ChoiceAlignedData(self, trial_data, photometry_data)
            #self.cue = CueAlignedData(self,trial_data, photometry_data, save_traces=True)
            #self.reward = RewardAlignedData(self, trial_data, photometry_data, save_traces=True)


class ChoiceAlignedData(object):
    """
    Traces for standard analysis aligned to choice (=movement from center port to left or right port)
    """

    def __init__(self, session_data, trial_data, photometry_data):

        # "RESPONSE": RIGHT = 2, LEFT = 1: hence ipsi and contra need to be assigned accordingly:
        fiber_options = np.array(['left', 'right'])  # left = (0+1) = 1; right = (1+1) == 2
        ipsi_fiber_side_numeric = (np.where(fiber_options == session_data.fiber_side)[0] + 1)[0]  # if fiber on right ipsi = 2; if fiber on left ipsi  = 1
        contra_fiber_side_numeric = (np.where(fiber_options != session_data.fiber_side)[0] + 1)[0]  # if fiber on right contra = 1, if fiber on left contra = 2

        params = {'state_type_of_interest': 5,
                  'outcome': 2,             # 2 = doesn't matter for choice aligned data
                  'no_repeats': 1,          # 1 = no repeats allowed
                  'last_response': 0,       # the previous trial doesn't matter
                  'align_to': 'Time start',
                  'instance': -1,           # -1 = last instance; 1 = first instance
                  'plot_range': [-6, 6],
                  'first_choice_correct': 2,  # 2 = doesnt matter
                  'SOR': 0,                 # 0 = nonSOR; 2 = doesnt matter, 1 = SOR
                  'psycho': 0,              # only Trial type 1 and 7 (no intermediate values, psychometric sounds)
                  'LRO': 0,                 # 0 = nonLRO;
                  'LargeRewards': 0,        # 1 = LR
                  'Omissions': 0,           # 1 = Omission
                  'cue': None}

        self.ipsi_data = ZScoredTraces(trial_data, photometry_data, params, ipsi_fiber_side_numeric, ipsi_fiber_side_numeric)
        self.contra_data = ZScoredTraces(trial_data, photometry_data, params, contra_fiber_side_numeric, contra_fiber_side_numeric)


class ZScoredTraces(object):
    def __init__(self, trial_data, dff, params, response, first_choice, curr_run):
        self.params = HeatMapParams(params, response, first_choice)
        self.time_points, self.mean_trace, self.sorted_traces, self.reaction_times, self.state_name, self.title, self.sorted_next_poke, self.trial_nums, self.event_times, self.outcome_times = find_and_z_score_traces(
            trial_data, dff, self.params, sort=False)

class HeatMapParams(object):
    def __init__(self, params, response, first_choice):
        self.state = params['state_type_of_interest']
        self.outcome = params['outcome']
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
        self.SOR = params['SOR']
        self.psycho = params['psycho']
        self.LRO = params['LRO']
        self.LR = params['LargeRewards']
        self.O = params['Omissions']


def find_and_z_score_traces(trial_data, dff, params, norm_window=8, sort=False, get_photometry_data=True):
    response_names = ['both left and right', 'left', 'right']
    outcome_names = ['incorrect', 'correct', 'both correct and incorrect']
    title = ''

    # Filter trial data according to selection of special trials (e.g. SOR, LRO, psychometric etc)
    if params.SOR == 0:
        events_of_int = getNonSORtrials(trial_data)
    elif params.SOR == 1:
        events_of_int = getSORtrials(trial_data)
    elif params.SOR == 2:
        events_of_int = trial_data
    if params.psycho == 0:
        events_of_int = getNonPsychotrials(events_of_int)
    if params.LRO == 0:
        events_of_int = getNonLROtrials(events_of_int)



    # 1) State type (e.g. corresp. State name = CueDelay, WaitforResponse...)
    events_of_int = events_of_int.loc[(events_of_int['State type'] == params.state)]  # State type = number of state of interest
    state_name = events_of_int['State name'].values[0]
    title = title + 'State type = ' + str(params.state) + ' =  state_name ' + state_name + ';'
    # --------------

    # 2) Response, trials to the left or to the right side
    if params.response != 0:    # 0 = don't care, 1 = left, 2 = right, selection of ipsi an contra side depends on fiber side
        events_of_int = events_of_int.loc[events_of_int['Response'] == params.response]
        title = title + ' Response = ' + str(params.response) + ';'
    # --------------

    # 3) First and last response:
    if params.first_choice != 0:
        events_of_int = events_of_int.loc[events_of_int['First response'] == params.first_choice]
        title = title + ' 1st response = ' + str(params.first_choice) + ';'
    if params.last_response != 0:
        events_of_int = events_of_int.loc[events_of_int['Last response'] == params.last_response]
        title = title + ' last response = ' + str(params.last_response) + ';'
    # --------------

    # 4) Outcome:
    if not params.outcome == 2:  # 2 = outcome doesn't matter
        events_of_int = events_of_int.loc[events_of_int['Trial outcome'] == params.outcome]
        title = title + ' Outcome = ' + str(params.outcome) + ';'
    # --------------

    # 5) Cues / Sounds:
    if params.cue == 'high':
        events_of_int = events_of_int.loc[events_of_int['Trial type'] == 7]
        title = title + ' Cue = high;'
    elif params.cue == 'low':
        events_of_int = events_of_int.loc[events_of_int['Trial type'] == 1]
        title = title + ' Cue = low;'
    # --------------

    # 6) Instance in State & Repeats:
    if params.instance == -1:   # Last time in State
        events_of_int = events_of_int.loc[
            (events_of_int['Instance in state'] / events_of_int['Max times in state'] == 1)]
        title = title + ' instance (' + str(params.instance) + ') last time in state (no matter the repetitions);'
    elif params.instance == 1:  # First time in State
        events_of_int = events_of_int.loc[(events_of_int['Instance in state'] == 1)]
        title = title + ' instance (' + str(params.instance) + ') first time in state;'
    if params.no_repeats == 1:
        events_of_int = events_of_int.loc[events_of_int['Max times in state'] == 1]
        title = title + ' no repetitions allowed (' + str(params.no_repeats) + ')'
    # --------------

    # 7) First choice directly in/correct?
    if params.first_choice_correct == 1:    # only first choice correct
        events_of_int = events_of_int.loc[
            (events_of_int['First choice correct'] == 1)]
        title = title + ' 1st choice correct (' + str(params.first_choice_correct) + ') only'
    elif params.first_choice_correct == -1: # only first choice incorrect
        events_of_int = events_of_int.loc[np.logical_or(
            (events_of_int['First choice correct'] == 0), (events_of_int['First choice correct'].isnull()))]
        title = title + ' 1st choice incorrect (' + str(params.first_choice_correct) + ') only'
        if events_of_int['State type'].isin([5.5]).any():   # first incorrect choice?
            events_of_int = events_of_int.loc[events_of_int['First choice correct'].isnull()]
    elif params.first_choice_correct == 0:  # first choice incorrect
        events_of_int = events_of_int.loc[(events_of_int['First choice correct'] == 0)]
    # --------------

    event_times = events_of_int[params.align_to].values # start or end of state of interest time points
    trial_nums = events_of_int['Trial num'].values
    trial_starts = events_of_int['Trial start'].values
    trial_ends = events_of_int['Trial end'].values

    other_event = np.asarray(np.squeeze(events_of_int[params.other_time_point].values) - np.squeeze(events_of_int[params.align_to].values))
    # for ex. time end - time start of state of interest

    last_trial = np.max(trial_data['Trial num'])                # absolutely last trial in session
    last_trial_num = events_of_int['Trial num'].unique()[-1]    # last trial that is considered in analysis meeting params requirements
    events_reset_index = events_of_int.reset_index(drop=True)
    last_trial_event_index = events_reset_index.loc[(events_reset_index['Trial num'] == last_trial_num)].index
            # index of the last event in the last trial that is considered in analysis meeting params requirements
    next_centre_poke = get_next_centre_poke(trial_data, events_of_int, last_trial_num == last_trial)
    trial_starts = get_first_poke(trial_data, events_of_int)
    absolute_outcome_times = get_outcome_time(trial_data, events_of_int)
    relative_outcome_times = absolute_outcome_times - event_times

    if get_photometry_data == True:
        next_centre_poke[last_trial_event_index] = events_reset_index[params.align_to].values[
                                                         last_trial_event_index] + 1  # so that you can find reward peak

        next_centre_poke_norm = next_centre_poke - event_times
        event_photo_traces = get_photometry_around_event(event_times, dff, pre_window=norm_window,
                                                             post_window=norm_window)
        norm_traces = stats.zscore(event_photo_traces.T, axis=0)


        if other_event.size == 1:
            print('Only one event for ' + title + ' so no sorting')
            sort = False
        elif len(other_event) != norm_traces.shape[1]:
            other_event = other_event[:norm_traces.shape[1]]
            print('Mismatch between #events and #other_event')
        if sort:
            arr1inds = other_event.argsort()
            sorted_other_event = other_event[arr1inds[::-1]]
            sorted_traces = norm_traces.T[arr1inds[::-1]]
            sorted_next_poke = next_centre_poke_norm[arr1inds[::-1]]
        else:
            sorted_other_event = other_event
            sorted_traces = norm_traces.T
            sorted_next_poke = next_centre_poke_norm

        time_points = np.linspace(-norm_window, norm_window, norm_traces.shape[0], endpoint=True, retstep=False,
                                      dtype=None, axis=0)
        mean_trace = np.mean(sorted_traces, axis=0)

    return time_points, mean_trace, sorted_traces, sorted_other_event, state_name, title, sorted_next_poke, trial_nums, event_times, relative_outcome_times





def getSORtrials(trial_data):
    trials_2AC = trial_data[trial_data['State name'] == 'WaitForPoke']
    trial_num_2AC = trials_2AC['Trial num'].unique()
    trials_SOR = trial_data
    for trialnum in trial_num_2AC:
        trials_SOR = trials_SOR[trials_SOR['Trial num']!=trialnum]
    return(trials_SOR)

def getNonSORtrials(trial_data):
    trialnums_trial_data = trial_data['Trial num'].unique() # all trial numbers
    events_2AC = trial_data[trial_data['State name'] == 'WaitForPoke'] # 2AC WaitForPoke events
    trialnums_2AC = events_2AC['Trial num'].unique() # 2AC trial numbers
    trial_data_2AC = trial_data
    for trial_all in trialnums_trial_data:
        include = 0
        for trial_2AC in trialnums_2AC:
            if trial_all == trial_2AC:
                include = 1
        if include == 0:
            trial_data_2AC = trial_data_2AC[trial_data_2AC['Trial num'] != trial_all]
    return trial_data_2AC

def getNonPsychotrials(trial_data):
    for sound_number in [2, 3, 4, 5, 6]:
        trial_data = trial_data[trial_data['Trial type'] != sound_number]   # Trial type == 1 or 7 == classic high and low frequency
    return trial_data

def getNonLROtrials(trial_data):
    LR_trial_numbers = trial_data[(trial_data['State type'] == 12) | (trial_data['State type'] == 13)][
        'Trial num'].values  # 13 = RightLargeReward, 12 = LeftLargeReward
    O_trial_numbers = trial_data[(trial_data['State type'] == 10) & (trial_data['State name'] == 'Omission')]['Trial num'].values  # 10 = Omission or REturnCuePlay
    LRO_trial_numbers = np.concatenate((LR_trial_numbers, O_trial_numbers))
    #NonLRO_trial_numbers = trial_data[~trial_data['Trial num'].isin(LRO_trial_numbers)]['Trial num'].values
    NonLRO_trial_data = trial_data
    for trial_num in trial_data['Trial num'].unique():
        if trial_num in LRO_trial_numbers:
            NonLRO_trial_data = NonLRO_trial_data[NonLRO_trial_data['Trial num'] != trial_num]
    return NonLRO_trial_data

def get_next_centre_poke(trial_data, events_of_int, last_trial):
    '''
    This function returns the time of the first centre poke in the subsequent trial for each event of interest.

    last_trial is a boolean that is true if the last trial in the session is included in the events of interest
    '''

    next_centre_poke_times = np.zeros(events_of_int.shape[0])
    events_of_int = events_of_int.reset_index(drop=True)
    for i, event in events_of_int.iterrows():
        trial_num = event['Trial num']
        if trial_num == trial_data['Trial num'].values[-1]:
            next_centre_poke_times[i] = events_of_int['Trial end'].values[i] + 2
        else:
            next_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num + 1)]
            wait_for_poke_state = next_trial_events.loc[(next_trial_events['State type'] == 2)] # wait for pokes

            if(len(wait_for_poke_state) > 0): # Classic 2AC:
                wait_for_pokes = next_trial_events.loc[(next_trial_events['State type'] == 2)] # wait for pokes
                next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)] # first wait for poke
                next_centre_poke_times[i] = next_wait_for_poke['Time end'].values[0] # time of first wait for poke ending
            elif len(wait_for_poke_state) == 0: # SOR: (SOR trials don't have WaitForPoke state)
                wait_for_pokes = next_trial_events.loc[(next_trial_events['State type'] == 3)] # CueDelay
                next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)] # first wait for poke
                next_centre_poke_times[i] = next_wait_for_poke['Time start'].values[0] # start time of first poke


    if last_trial: # last trial in events of interest == last trial in session, last_tial is true or false, not a number
        next_centre_poke_times[-1] = events_of_int['Trial end'].values[-1] + 2
    else: # last trial in events of interest != last trial in session
        event = events_of_int.tail(1)
        trial_num = event['Trial num'].values[0]
        next_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num + 1)]

        wait_for_poke_state = next_trial_events.loc[(next_trial_events['State type'] == 2)]  # wait for pokes
        if (len(wait_for_poke_state) > 0):  # Classic 2AC:
            wait_for_pokes = next_trial_events.loc[(next_trial_events['State type'] == 2)]
            next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)]
            next_centre_poke_times[-1] = next_wait_for_poke['Time end'].values[0] # end time of wait for poke
        elif len(wait_for_poke_state) == 0:  # SOR: (SOR trials don't have WaitForPoke state)
            wait_for_pokes = next_trial_events.loc[(next_trial_events['State type'] == 3)]  # CueDelay
            next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)]  # first wait for poke
            next_centre_poke_times[i] = next_wait_for_poke['Time start'].values[0]  # start time of first poke
    return next_centre_poke_times


def get_first_poke(trial_data, events_of_int): # get first poke in each trial of events of interest
    trial_numbers = events_of_int['Trial num'].unique()
    next_centre_poke_times = np.zeros(events_of_int.shape[0])
    events_of_int = events_of_int.reset_index(drop=True)
    for trial_num in trial_numbers:
        event_indx_for_that_trial = events_of_int.loc[(events_of_int['Trial num'] == trial_num)].index
        trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num)]
        wait_for_pokes = trial_events.loc[(trial_events['State type'] == 2)]
        if len(wait_for_pokes) > 0: #Classic 2AC:
            next_wait_for_poke = wait_for_pokes.loc[(wait_for_pokes['Instance in state'] == 1)]
            #next_centre_poke_times[event_indx_for_that_trial] = next_wait_for_poke['Time end'].values[0]-1 #why -1 in FG code?
            next_centre_poke_times[event_indx_for_that_trial] = next_wait_for_poke['Time end'].values[0]

        elif len(wait_for_pokes) == 0: #SOR: (SOR trials don't have WaitForPoke state)
            next_wait_for_poke = trial_events.loc[(trial_events['State type'] == 3) & (trial_events['Instance in state'] == 1)] #First CueDelay
            next_centre_poke_times[event_indx_for_that_trial] = next_wait_for_poke['Time start'].values[0]

    return next_centre_poke_times

def get_outcome_time(trial_data, events_of_int): # returns the time of the outcome of the current trial, indep of rewarded or punished
    trial_numbers = events_of_int['Trial num'].values
    outcome_times = []
    for event_trial_num in range(len(trial_numbers)):
        trial_num = trial_numbers[event_trial_num]
        other_trial_events = trial_data.loc[(trial_data['Trial num'] == trial_num)]
        choices = other_trial_events.loc[(other_trial_events['State type'] == 5)] # 5 is the state type for choices / wait for response
        max_times_in_state_choices = choices['Max times in state'].unique() # all values in max times in state available for this trial an state type
        choice = choices.loc[(choices['Instance in state'] == max_times_in_state_choices)] # last time wait for response
        outcome_times.append(choice['Time end'].values[0])
    return outcome_times


def get_photometry_around_event(all_trial_event_times, demodulated_trace, pre_window=5, post_window=5, sample_rate=10000):
    num_events = len(all_trial_event_times)
    event_photo_traces = np.zeros((num_events, sample_rate*(pre_window + post_window)))
    for event_num, event_time in enumerate(all_trial_event_times):
        plot_start = int(round(event_time*sample_rate)) - pre_window*sample_rate
        plot_end = int(round(event_time*sample_rate)) + post_window*sample_rate
        event_photo_traces[event_num, :] = demodulated_trace[plot_start:plot_end]
    return event_photo_traces