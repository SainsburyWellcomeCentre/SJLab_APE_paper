import pandas as pd
import nptdms
import numpy as np
import scipy
import math
import os


def GetExperimentsToProcess(mice, dates, protocol, experimental_record):

    df_mice = []
    df_dates = []
    df_protocol = []
    df_fiber_side = []
    for i, mouse in enumerate(mice):
        if protocol == '':
            df_mice.append(mouse)
            df_dates.append(dates[i])
            df_fiber_side.append(experimental_record[(experimental_record['mouse'] == mouse) & (experimental_record['date'] == dates[i])]['fiber_side'].values[0])
            protocol1 = experimental_record[(experimental_record['mouse'] == mouse) & (experimental_record['date'] == dates[i])]['protocol1'].values[0]
            df_protocol.append(protocol1)
            protocol2 = experimental_record[(experimental_record['mouse'] == mouse) & (experimental_record['date'] == dates[i])]['protocol2'].values[0]
            if not pd.isna(protocol2):
                df_mice.append(mouse)
                df_dates.append(dates[i])
                df_fiber_side.append(experimental_record[(experimental_record['mouse'] == mouse) & (
                            experimental_record['date'] == dates[i])]['fiber_side'].values[0])
                df_protocol.append(protocol2)
        else:
            df_mice.append(mouse)
            df_dates.append(dates[i])
            df_fiber_side.append(experimental_record[(experimental_record['mouse'] == mouse) & (experimental_record['date'] == dates[i])]['fiber_side'].values[0])
            df_protocol.append(protocol)
    experiments_to_process = pd.DataFrame({'mouse': df_mice, 'date': df_dates, 'fiber_side': df_fiber_side, 'protocol': df_protocol})
    return experiments_to_process



def preprocessData(experiments_to_process):
    dataset_path = '../../data/'
    count = 0

    all_photometry_data = []
    all_restructured_data = []

    for mouse in experiments_to_process['mouse']:
        print('Processing ' + mouse + ' ' + experiments_to_process['date'].values[count] + ' ' + experiments_to_process['protocol'].values[count])
        date = experiments_to_process['date'].values[count]
        protocol = experiments_to_process['protocol'].values[count]

        # loading the photometry data
        photo_filename = dataset_path + mouse + '_' + date + '_' + protocol + '_AI.tdms'  # needs to change according to final raw data filename
        photo_data = nptdms.TdmsFile(photo_filename)
        sampling_rate = 10000

        chan_0 = photo_data['acq_task'].channels()[0]
        led405 = photo_data['acq_task'].channels()[2]
        led465 = photo_data['acq_task'].channels()[1]
        clock = photo_data['acq_task'].channels()[3]
        stim_trigger = photo_data['acq_task'].channels()[4]

        stim_trigger_gaps = np.diff(stim_trigger)
        trial_start_ttls_daq_samples = np.where(stim_trigger_gaps > 2.6)
        trial_start_ttls_daq = trial_start_ttls_daq_samples[0] / sampling_rate  # trial start times in seconds
        daq_num_trials = trial_start_ttls_daq.shape[0]

        df_clipped = lerner_deisseroth_preprocess(chan_0[sampling_rate * 6:], led465[sampling_rate * 6:],
                                                  led405[sampling_rate * 6:], sampling_rate)
        df = np.pad(df_clipped, (6 * sampling_rate, 0), mode='median')
        clock_diff = np.diff(clock)
        clock_pulses = np.where(clock_diff > 2.6)[0] / sampling_rate


        # loading the bpod behavioural data
        if protocol == '2AC':
            search_tool = mouse + '_Two_Alternative_Choice_' + date + '_'
        elif protocol == 'SOR':
            search_tool = mouse + '_Two_Alternative_Choice_Tones_On_Return_' + date + '_'
        elif protocol == 'RTC':
            search_tool = mouse + '_Random_Tone_Clouds_' + date + '_'
        elif protocol == 'WN':
            search_tool = mouse + '_Random_WN_' + date + '_'

        files_on_that_day = [f for f in os.listdir(dataset_path) if search_tool in f]
        mat_files_on_that_day = [f for f in files_on_that_day if f.endswith('.mat')]

        if len(mat_files_on_that_day) != 1:
            print(str(len(mat_files_on_that_day)) + ' files found for ' + mouse + ' on ' + date)
            print('files found: ' + str(mat_files_on_that_day))
        else:
            no_extension_files = [os.path.splitext(filename)[0] for filename in mat_files_on_that_day]
            file_times = [filename.split('_')[-1] for filename in no_extension_files]
            main_session_file = dataset_path + mat_files_on_that_day[file_times.index(max(file_times))]
            loaded_bpod_file = loadmat(main_session_file)
            trial_raw_events = loaded_bpod_file['SessionData']['RawEvents']['Trial']

            for trial_num, trial in enumerate(trial_raw_events):
                trial_raw_events[trial_num] = _todict(trial)
            loaded_bpod_file['SessionData']['RawEvents']['Trial'] = trial_raw_events
            bpod_num_trials = trial_raw_events.shape[0]

        if daq_num_trials != bpod_num_trials:
            print('numbers of trials do not match! lerner_deisseroth_preprocess')
            print('daq: ', daq_num_trials)
            print('bpod: ', bpod_num_trials)
        else:
            print(daq_num_trials, 'trials in session (lerner_deisseroth_preprocess)')


        if protocol == 'RTC':
            restructured_data = restructure_bpod_timestamps_random_tone_clouds(loaded_bpod_file, trial_start_ttls_daq, clock_pulses)
            smoothed_trace_filename = mouse + '_' + date + '_RTC_smoothed_signal.npy'
            restructured_data_filename = mouse + '_' + date + '_RTC_restructured_data.pkl'
        elif protocol == 'Random_WN':
            restructured_data = restructure_bpod_timestamps_random_tone_clouds(loaded_bpod_file,
                                                                                    trial_start_ttls_daq, clock_pulses)
            smoothed_trace_filename = mouse + '_' + date + '_RWN_smoothed_signal.npy'
            restructured_data_filename = mouse + '_' + date + '_RWN_restructured_data.pkl'
        else:  # standard analysis
            restructured_data = restructure_bpod_timestamps(loaded_bpod_file, trial_start_ttls_daq, clock_pulses)
            smoothed_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
            restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'

        # save output:
        saving_folder = '../../data/'  #+ 'processed_data\\' + mouse + '\\'
        np.save(saving_folder + smoothed_trace_filename, df)
        restructured_data.to_pickle(saving_folder + restructured_data_filename)
        all_photometry_data.append(df)
        all_restructured_data.append(restructured_data)
        count += 1

    return all_photometry_data, all_restructured_data
def lerner_deisseroth_preprocess(
    photodetector_raw_data,
    reference_channel_211hz,
    reference_channel_531hz,
    sampling_rate):
    """
    Code from Steve Lenzi
    process data according to https://www.ncbi.nlm.nih.gov/pubmed/26232229 , supplement 11
    :param photodetector_raw_data: the raw signal from the photodetector
    :param reference_channel_211hz:  a copy of the reference signal sent to the signal LED (Ca2+ dependent)
    :param reference_channel_531hz:  a copy of the reference signal sent to the background LED (Ca2+ independent)
    :param sampling_rate: the sampling rate of the photodetector
    :return: deltaF / F
    """
    demodulated_211, demodulated_531 = demodulate(
        photodetector_raw_data,
        reference_channel_211hz,
        reference_channel_531hz,
        sampling_rate,
    )

    signal = _apply_butterworth_lowpass_filter(
        demodulated_211, 2, sampling_rate, order=2
    )
    background = _apply_butterworth_lowpass_filter(
        demodulated_531, 2, sampling_rate, order=2
    )

    regression_params = np.polyfit(background, signal, 1)
    bg_fit = regression_params[0] * background + regression_params[1]

    delta_f = (signal - bg_fit) / bg_fit
    return delta_f


def demodulate(raw, ref_211, ref_531, sampling_rate):
    """
    gets demodulated signals for 211hz and 531hz am modulated signal
    Code from Steve Lenzi
    :param raw:
    :param ref_211:
    :param ref_531:
    :return:
    """

    q211, i211 = am_demodulate(raw, ref_211, 211, sampling_rate=sampling_rate)
    q531, i531 = am_demodulate(raw, ref_531, 531, sampling_rate=sampling_rate)
    demodulated_211 = _demodulate_quadrature(q211, i211)
    demodulated_531 = _demodulate_quadrature(q531, i531)

    return demodulated_211, demodulated_531

def am_demodulate(signal, reference, modulation_frequency, sampling_rate=10000, low_cut=15, order=5):
    # Code from Steve Lenzi
    normalised_reference = reference - reference.mean()
    samples_per_period = sampling_rate / modulation_frequency
    samples_per_quarter_period = round(samples_per_period / 4)

    shift_90_degrees = np.roll(normalised_reference, samples_per_quarter_period)
    in_phase = np.pad(signal * normalised_reference, (sampling_rate, 0), mode='median')
    in_phase_filtered_pad = _apply_butterworth_lowpass_filter(in_phase, low_cut_off=low_cut, fs=sampling_rate,
                                                              order=order)
    in_phase_filtered = in_phase_filtered_pad[sampling_rate:]

    quadrature = np.pad(signal * shift_90_degrees, (sampling_rate, 0), mode='median')
    quadrature_filtered_pad = _apply_butterworth_lowpass_filter(quadrature, low_cut_off=low_cut, fs=sampling_rate,
                                                                order=order)
    quadrature_filtered = quadrature_filtered_pad[sampling_rate:]

    return quadrature_filtered, in_phase_filtered

def _demodulate_quadrature(quadrature, in_phase):
    # Code from Steve Lenzi
    return (quadrature ** 2 + in_phase ** 2) ** 0.5
def _apply_butterworth_lowpass_filter(
    demod_signal, low_cut_off=15, fs=10000, order=5):
    # Code from Steve Lenzi
    w = low_cut_off / (fs / 2)  # Normalize the frequency
    b, a = scipy.signal.butter(order, w, "low")
    output = scipy.signal.filtfilt(b, a, demod_signal)
    return output




def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(temp_dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in temp_dict:
        if isinstance(temp_dict[key], scipy.io.matlab.mio5_params.mat_struct):
            temp_dict[key] = _todict(temp_dict[key])

    return temp_dict
def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    temp_dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            temp_dict[strg] = _todict(elem)
        else:
            temp_dict[strg] = elem
    return temp_dict


def restructure_bpod_timestamps(loaded_bpod_file, trial_start_ttls_daq, clock_pulses):
    original_state_data_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateData']
    original_state_timestamps_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateTimestamps'] #an array of arrays, each array consists of the timestamps of a trial
    original_raw_events = loaded_bpod_file['SessionData']['RawEvents']['Trial']

    daq_trials_start_ttls = trial_start_ttls_daq # an array of the actual timestamps of the trial start TTLs
    # loops through all the trials and pulls out all the states
    for trial, state_timestamps in enumerate(original_state_timestamps_all_trials):
        # now trial as in pycharm (starts with 0), not as in matlab (starts with counting at 1):
        state_info = {} # a dict containing trial by trial all info; each trial has the length of the number of states in that trial
        event_info = {}
        trial_states = original_state_data_all_trials[trial]
        num_states = (len(trial_states))
        sound_types = {'COT': 0, 'NA': 1} # 2AC classic: all COT, all 0; SOR: COT = 0, NA = 1 = SOR
        try:
            state_info['Sound type'] = np.ones(num_states) * sound_types[loaded_bpod_file['SessionData']['SoundType'][trial]] # 0 = COT, 1 = NA
            # matlab sound type is a string of COT or NA;
        except:
            state_info['Sound type'] = np.ones(num_states) * 10

        state_info['Trial num'] = np.ones(num_states) * trial
        state_info['Trial type'] = np.ones(num_states) * loaded_bpod_file['SessionData']['TrialSequence'][trial] # 1 or 7 depending on high or low frequency sound
        state_info['State type'] = trial_states # all states in that trial
        num_times_in_state = find_num_times_in_state(trial_states)
        state_info['Instance in state'] = num_times_in_state[0]     # 1st, 2nd, 3rd time state X is happening
        state_info['Max times in state'] = num_times_in_state[1]    # max N that state X has happened
        state_info['State name'] = loaded_bpod_file['SessionData']['RawData']['OriginalStateNamesByNumber'][0][
            trial_states - 1] # -1 because MATLAB is 1-indexed while python is 0-indexed
        state_info['Time start'] = state_timestamps[0:-1] + daq_trials_start_ttls[trial]    # all but the last time stamp + the trial start TTL
        state_info['Time end'] = state_timestamps[1:] + daq_trials_start_ttls[trial]        # all but the first time stamp + the trial start TTL
        state_info['Response'] = np.ones(num_states) * loaded_bpod_file['SessionData']['ChosenSide'][trial] # 2 = right, 1 = left

        if trial > 0:
            state_info['Last response'] = np.ones(num_states) * loaded_bpod_file['SessionData']['ChosenSide'][
                trial - 1]
            state_info['Last outcome'] = np.ones(num_states) * loaded_bpod_file['SessionData']['Outcomes'][trial - 1]
        else:
            state_info['Last response'] = np.ones(num_states) * -1
            state_info['Last outcome'] = np.ones(num_states) * -1

        state_info['Trial start'] = np.ones(num_states) * daq_trials_start_ttls[trial] # trial start TTL
        state_info['Trial end'] = np.ones(num_states) * (state_timestamps[-1] + daq_trials_start_ttls[trial]) # last state timestamp + trial start TTL
        state_info['Trial outcome'] = np.ones(num_states) * loaded_bpod_file['SessionData']['Outcomes'][trial] # 1 = correct, 3 = in SOR: didn't poke in time
        state_info['First response'] = np.ones(num_states) * loaded_bpod_file['SessionData']['FirstPoke'][trial]
        if hasattr(loaded_bpod_file['SessionData']['TrialSettings'][trial], 'RewardChangeBlock'):
            state_info['Reward block'] = np.ones(num_states) * loaded_bpod_file['SessionData']['TrialSettings'][trial].RewardChangeBlock

        if loaded_bpod_file['SessionData']['FirstPoke'][trial] == loaded_bpod_file['SessionData']['TrialSide'][trial]: # correct
            state_info['First choice correct'] = np.ones(num_states)
            event_info['First choice correct'] = [1]
            event_info['Trial num'] = [trial]
            try:
                event_info['Sound type'] = sound_types[loaded_bpod_file['SessionData']['SoundType'][trial]]
            except KeyError:
                event_info['Sound type'] = 10

            event_info['Trial type'] = [loaded_bpod_file['SessionData']['TrialSequence'][trial]]
            event_info['State type'] = [8.5]    # correct choice, State = leaving reward port
            event_info['Instance in state'] = [1]
            event_info['Max times in state'] = [1]
            event_info['State name'] = ['Leaving reward port']
            if hasattr(loaded_bpod_file['SessionData']['TrialSettings'][trial], 'RewardChangeBlock'):
                event_info['Reward block'] = loaded_bpod_file['SessionData']['TrialSettings'][trial].RewardChangeBlock
            event_info['Response'] = [loaded_bpod_file['SessionData']['ChosenSide'][trial]]
            if trial > 0:
                event_info['Last response'] = [loaded_bpod_file['SessionData']['ChosenSide'][
                                                   trial - 1]]
                event_info['Last outcome'] = [loaded_bpod_file['SessionData']['Outcomes'][
                                                  trial - 1]]
            else:
                event_info['Last response'] = [-1]
                event_info['Last outcome'] = [-1]
            event_info['Trial outcome'] = [loaded_bpod_file['SessionData']['Outcomes'][trial]]
            event_info['First response'] = [loaded_bpod_file['SessionData']['FirstPoke'][trial]]

            correct_side = loaded_bpod_file['SessionData']['TrialSide'][trial] # 1 = left was correct, 2 = right was correct
            if correct_side == 1: # mouse should have gone left
                correct_port_in = 'Port1In' # left port
                correct_port_out = 'Port1Out' # left port
                reward_time = original_raw_events[trial]['States']['LeftReward'][0]
            else:
                correct_port_in = 'Port3In'
                correct_port_out = 'Port3Out'
                reward_time = original_raw_events[trial]['States']['RightReward'][0]
            all_correct_pokes_in = np.squeeze(np.asarray([original_raw_events[trial]['Events'][correct_port_in]]))
            if all_correct_pokes_in.size == 1 and all_correct_pokes_in >= reward_time:
                event_info['Time start'] = all_correct_pokes_in
            elif all_correct_pokes_in.size > 1:
                event_info['Time start'] = all_correct_pokes_in[
                    np.squeeze(np.where(all_correct_pokes_in > reward_time)[0])]
                if (event_info['Time start']).size > 1:
                    event_info['Time start'] = event_info['Time start'][0]
            else:
                event_info['Time start'] = np.empty(0)

            if trial < original_state_timestamps_all_trials.shape[0] - 1: # if not the last trial
                if correct_port_out in original_raw_events[trial + 1]['Events']:
                    all_correct_pokes_out = np.squeeze(np.asarray([original_raw_events[trial + 1]['Events'][correct_port_out]]))
                    if event_info['Time start'].size != 0:
                        if all_correct_pokes_out.size == 1:
                            event_info['Time end'] = all_correct_pokes_out
                        elif (all_correct_pokes_out).size > 1:
                            indices = np.where(all_correct_pokes_out > 0)
                            if len(indices) > 1:
                                event_info['Time end'] = all_correct_pokes_out[0]
                            else:
                                event_info['Time end'] = all_correct_pokes_out[0]
                        else: # all_correct_pokes_out.size = 0
                            event_info['Time end'] = np.empty(0)

                        if (event_info['Time end']).size > 1:
                            event_info['Time end'] = event_info['Time end'][0]

                        event_info['Time end'] = [event_info['Time end'] + daq_trials_start_ttls[trial + 1]]
                    else: # if there was no start time for correct poke in
                        event_info['Time end'] = [np.empty(0)]
                else: # if the next trial doesn't have a correct_port_out
                    event_info['Time end'] = [np.empty(0)]
            else: # if it is the last trial
                event_info['Time end'] = [np.empty(0)]
            event_info['Time start'] = [event_info['Time start'] + daq_trials_start_ttls[trial]]


        else: # incorrect trial or a missed trial
            out_of_centre_time = original_raw_events[trial]['States']['WaitForResponse'][0]
            if math.isnan(out_of_centre_time) == False:
                state_info['First choice correct'] = np.zeros(num_states) # 0 for incorrect
                event_info['Trial num'] = [trial]
                event_info['Trial type'] = [loaded_bpod_file['SessionData']['TrialSequence'][trial]] # 1 or 7, high or low f
                event_info['State type'] = [5.5] # wrong choice, first incorrect choice
                try:
                    event_info['Sound type'] = sound_types[
                        loaded_bpod_file['SessionData']['SoundType'][trial]] # COT or NA
                except KeyError:
                    event_info['Sound type'] = 'not available'


                event_info['Instance in state'] = [1]
                event_info['Max times in state'] = [1]
                if hasattr(loaded_bpod_file['SessionData']['TrialSettings'][trial], 'RewardChangeBlock'):
                    event_info['Reward block'] = loaded_bpod_file['SessionData']['TrialSettings'][trial].RewardChangeBlock
                event_info['State name'] = ['First incorrect choice']
                correct_side = loaded_bpod_file['SessionData']['TrialSide'][trial]
                if correct_side == 1:
                    incorrect_port_in = 'Port3In'
                    incorrect_port_out = 'Port3Out'
                else:
                    incorrect_port_in = 'Port1In'
                    incorrect_port_out = 'Port1Out'

                if incorrect_port_in in original_raw_events[trial]['Events'] and incorrect_port_out in original_raw_events[trial]['Events']:
                    all_incorrect_pokes_in = np.squeeze(np.asarray([original_raw_events[trial]['Events'][incorrect_port_in]]))
                    all_incorrect_pokes_out = np.squeeze(np.asarray([original_raw_events[trial]['Events'][incorrect_port_out]]))
                    if all_incorrect_pokes_in.size == 1 and all_incorrect_pokes_in > out_of_centre_time: # an incorrect poke after wait for response has started
                        event_info['Time start'] = all_incorrect_pokes_in
                    elif all_incorrect_pokes_in.size > 1:
                        event_info['Time start'] = all_incorrect_pokes_in[np.squeeze(np.where(all_incorrect_pokes_in > out_of_centre_time)[0])]
                        if (event_info['Time start']).size > 1:
                            event_info['Time start'] = event_info['Time start'][0]
                    else:
                        event_info['Time start'] = np.empty(0)

                    if event_info['Time start'].size != 0:
                        if all_incorrect_pokes_out.size == 1 and all_incorrect_pokes_out > event_info['Time start']:
                            event_info['Time end'] = all_incorrect_pokes_out
                        elif all_incorrect_pokes_out.size > 1:
                            indices = np.where(all_incorrect_pokes_out > event_info['Time start'])
                            if len(indices) > 1:
                                event_info['Time end'] = all_incorrect_pokes_out[np.squeeze(np.where(all_incorrect_pokes_out > event_info['Time start'])[0])]
                            else:
                                event_info['Time end'] = all_incorrect_pokes_out[
                                    np.squeeze(np.where(all_incorrect_pokes_out > event_info['Time start']))]
                        else: event_info['Time end'] = np.empty(0)

                        if (event_info['Time end']).size > 1:
                            event_info['Time end'] = event_info['Time end'][0]

                        event_info['Time end'] = [event_info['Time end'] + daq_trials_start_ttls[trial]]

                    else: event_info['Time end'] = []
                    event_info['Time start'] = [event_info['Time start'] + daq_trials_start_ttls[trial]]

                    event_info['Response'] = [loaded_bpod_file['SessionData']['ChosenSide'][trial]]
                    if trial > 0:
                        event_info['Last response'] = [loaded_bpod_file['SessionData']['ChosenSide'][
                            trial - 1]]
                        event_info['Last outcome'] = [loaded_bpod_file['SessionData']['Outcomes'][
                            trial - 1]]
                    else:
                        event_info['Last response'] = [-1]
                        event_info['Last outcome'] = [-1]
                    event_info['Trial start'] = [daq_trials_start_ttls[trial]]
                    event_info['Trial end'] = [(state_timestamps[-1] + daq_trials_start_ttls[trial])]
                    event_info['Trial outcome'] = [loaded_bpod_file['SessionData']['Outcomes'][trial]]
                    event_info['First response'] = [loaded_bpod_file['SessionData']['FirstPoke'][trial]]
                else:
                    event_info = {}
        # analysis of incorrect trials end

        trial_data = pd.DataFrame(state_info)
        if trial == 0:
            restructured_data = trial_data
        else:
            restructured_data = pd.concat([restructured_data, trial_data], ignore_index=True)
            if event_info != {} and event_info['Time start'][0].size != 0 and event_info['Time end'][0].size != 0:
                event_data = pd.DataFrame(event_info)
                restructured_data = pd.concat([restructured_data, event_data], ignore_index=True)
    return restructured_data

def restructure_bpod_timestamps_random_tone_clouds(loaded_bpod_file, trial_start_ttls_daq, clock_pulses):
    original_state_data_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateData']
    original_state_timestamps_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateTimestamps']
    original_raw_events = loaded_bpod_file['SessionData']['RawEvents']['Trial']

    daq_trials_start_ttls = trial_start_ttls_daq
    # loops through all the trials and pulls out all the states
    for trial, state_timestamps in enumerate(original_state_timestamps_all_trials):
        state_info = {}
        event_info = {}
        trial_states = original_state_data_all_trials[trial]
        num_states = (len(trial_states))
        sound_types = {'COT': 0, 'NA': 1, 'WN':2} # the parameter sound types was introduced later and is not available in early recordings.
        try:
            state_info['Sound type'] = np.ones((num_states)) * sound_types[
                loaded_bpod_file['SessionData']['SoundType'][trial]]
        except KeyError:
            event_info['Sound type'] = 'not available'
        state_info['Trial num'] = np.ones(num_states) * trial
        state_info['Trial type'] = np.ones(num_states) * loaded_bpod_file['SessionData']['TrialSequence'][trial]
        state_info['State type'] = trial_states
        num_times_in_state = find_num_times_in_state(trial_states)
        state_info['Instance in state'] = num_times_in_state[0]
        state_info['Max times in state'] = num_times_in_state[1]
        state_info['State name'] = loaded_bpod_file['SessionData']['RawData']['OriginalStateNamesByNumber'][0][
            trial_states - 1]
        state_info['Time start'] = state_timestamps[0:-1] + daq_trials_start_ttls[trial]
        state_info['Time end'] = state_timestamps[1:] + daq_trials_start_ttls[trial]
        state_info['Trial start'] = np.ones((num_states)) * daq_trials_start_ttls[trial]
        state_info['Trial end'] = np.ones((num_states)) * (state_timestamps[-1] + daq_trials_start_ttls[trial])
        trial_data = pd.DataFrame(state_info)

        if trial == 0:
            restructured_data = trial_data
        else:
            restructured_data = pd.concat([restructured_data, trial_data], ignore_index=True)
    return restructured_data
def find_num_times_in_state(trial_states):
    unique_states = np.unique(trial_states)
    state_occurences = np.zeros(trial_states.shape)
    max_occurences = np.zeros(trial_states.shape)
    for state in unique_states:
        total_occurences = np.where(trial_states==state)[0].shape[0]
        num_occurences = 0
        for idx, val in enumerate(trial_states):
            if val==state:
                num_occurences+=1
                state_occurences[idx] = num_occurences
                max_occurences[idx] = total_occurences
    return state_occurences, max_occurences

