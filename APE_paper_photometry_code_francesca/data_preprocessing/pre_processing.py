import nptdms
import numpy as np
from data_preprocessing.demodulation import lerner_deisseroth_preprocess
from data_preprocessing.demodulation import demodulate
import data_preprocessing.bpod_data_processing as bpod
from utils.post_processing_utils import get_all_experimental_records
import os
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from set_global_params import processed_data_path, daq_sample_rate


def pre_process_experiment_lerner_deissroth(mouse, date, protocol):
    """
    Pre processes the data according to what is described in the Lerner Deissroth paper.
    Code written mostly by Steve Lenzi.
    Args:
        mouse (str): mouse name
        date (str): YYYYMMDD
        protocol (str): bpod protocol

    Returns:

    """
    daq_file = bpod.find_daq_file(mouse, date)
    data = nptdms.TdmsFile(daq_file)

    main_session_file = bpod.find_bpod_file(mouse, date, protocol)
    loaded_bpod_file, trial_raw_events = bpod.load_bpod_file(main_session_file)

    chan_0 = data['acq_task'].channels()[0].data
    led405 = data['acq_task'].channels()[2].data
    led465 = data['acq_task'].channels()[1].data
    clock = data['acq_task'].channels()[3].data
    stim_trigger = data['acq_task'].channels()[4].data
    #for python 3.6 with tdms version 0.27.0
    #chan_0 = data.group_channels('acq_task')[0].data
    #led405 = data.group_channels('acq_task')[2].data
    #led465 = data.group_channels('acq_task')[1].data
    #clock = data.group_channels('acq_task')[3].data
    #stim_trigger = data.group_channels('acq_task')[4].data

    stim_trigger_gaps = np.diff(stim_trigger)
    trial_start_ttls_daq_samples = np.where(stim_trigger_gaps > 2.6)
    trial_start_ttls_daq = trial_start_ttls_daq_samples[0] / daq_sample_rate
    daq_num_trials = trial_start_ttls_daq.shape[0]
    bpod_num_trials = trial_raw_events.shape[0]
    if daq_num_trials != bpod_num_trials:
        print('numbers of trials do not match!')
        print('daq: ', daq_num_trials)
        print('bpod: ', bpod_num_trials)
    else:
        print(daq_num_trials, 'trials in session')

    df_clipped = lerner_deisseroth_preprocess(chan_0[daq_sample_rate * 6:], led465[daq_sample_rate * 6:],
                                              led405[daq_sample_rate * 6:], daq_sample_rate)
    df = np.pad(df_clipped, (6 * daq_sample_rate, 0), mode='median')
    clock_diff = np.diff(clock)
    clock_pulses = np.where(clock_diff > 2.6)[0] / daq_sample_rate

    restructured_data = bpod.restructure_bpod_timestamps(loaded_bpod_file, trial_start_ttls_daq, clock_pulses)

    saving_folder = processed_data_path + mouse + '\\'

    smoothed_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'

    restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
    np.save(saving_folder + smoothed_trace_filename, df)
    restructured_data.to_pickle(saving_folder + restructured_data_filename)


def pre_process_experiment_pyphotometry(mouse, date, protocol, daq_sample_rate=daq_sample_rate):
    """
    Preprocesses the photometry signal using demodulation and then filtering similar to that used by pyphotometry paper.
    Used on all photometry data for the cloud of tones task.
    Args:
        mouse (str): mouse name 
        date (str): date (YYYYMMDD)
        protocol (str): bpod protocol

    Returns:

    """
    
    daq_file = bpod.find_daq_file(mouse, date)
    data = nptdms.TdmsFile(daq_file)

    main_session_file = bpod.find_bpod_file(mouse, date, protocol)
    loaded_bpod_file, trial_raw_events = bpod.load_bpod_file(main_session_file)

    chan_0 = data['acq_task'].channels()[0].data
    led405 = data['acq_task'].channels()[2].data
    led465 = data['acq_task'].channels()[1].data
    clock = data['acq_task'].channels()[3].data
    stim_trigger = data['acq_task'].channels()[4].data
    #for python 3.6 with tdms version 0.27.0
    #chan_0 = data.group_channels('acq_task')[0].data
    #led405 = data.group_channels('acq_task')[2].data
    #led465 = data.group_channels('acq_task')[1].data
    #clock = data.group_channels('acq_task')[3].data
    #stim_trigger = data.group_channels('acq_task')[4].data
    stim_trigger_gaps = np.diff(stim_trigger)
    trial_start_ttls_daq_samples = np.where(stim_trigger_gaps > 2.595)
    trial_start_ttls_daq = trial_start_ttls_daq_samples[0] / daq_sample_rate
    daq_num_trials = trial_start_ttls_daq.shape[0]
    bpod_num_trials = trial_raw_events.shape[0]
    if daq_num_trials != bpod_num_trials:
        print('numbers of trials do not match!')
        print('daq: ', daq_num_trials)
        print('bpod: ', bpod_num_trials)
    else:
        print(daq_num_trials, 'trials in session')


    signal, background = demodulate(chan_0[daq_sample_rate * 6:], led465[daq_sample_rate * 6:], led405[daq_sample_rate * 6:], daq_sample_rate)

    # Median filtering to remove electrical artifact.
    signal_denoised = medfilt(signal, kernel_size=5)
    back_denoised = medfilt(background, kernel_size=5)

    # Lowpass filter - zero phase filtering (with filtfilt) is used to avoid distorting the signal.
    b, a = butter(2, 10, btype='low', fs=daq_sample_rate)
    signal_denoised = filtfilt(b, a, signal_denoised)
    back_denoised = filtfilt(b, a, back_denoised)

    b, a = butter(2, 0.001, btype='high', fs=daq_sample_rate)
    signal_highpass = filtfilt(b, a, signal_denoised, padtype='even')
    back_highpass = filtfilt(b, a, back_denoised, padtype='even')

    slope, intercept, r_value, p_value, std_err = linregress(x=back_highpass, y=signal_highpass)

    print('Slope    : {:.3f}'.format(slope))
    print('R-squared: {:.3f}'.format(r_value ** 2))

    signal_est_motion = intercept + slope * back_highpass
    signal_corrected = signal_highpass - signal_est_motion

    b, a = butter(2, 0.001, btype='low', fs=daq_sample_rate)
    baseline_fluorescence = filtfilt(b, a, signal_denoised, padtype='even')

    signal_dF_F = signal_corrected / baseline_fluorescence
    b, a = butter(2, 3, btype='low', fs=daq_sample_rate)
    smoothed_signal = filtfilt(b, a, signal_dF_F, padtype='even')
    smoothed_signal = np.pad(smoothed_signal, (6 * daq_sample_rate, 0), mode='median')
    clock_diff = np.diff(clock)
    clock_pulses = np.where(clock_diff > 2.6)[0] / daq_sample_rate

    restructured_data = bpod.restructure_bpod_timestamps(loaded_bpod_file, trial_start_ttls_daq, clock_pulses)

    saving_folder = processed_data_path + mouse + '\\'
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    smoothed_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
    restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
    np.save(saving_folder + smoothed_trace_filename, smoothed_signal)
    restructured_data.to_pickle(saving_folder + restructured_data_filename)


def pre_process_experiments(experiments, method='pyphotometry', protocol='Two_Alternative_Choice'):
    """
    Demodulates photometry signal and extracts behavioural events and saves out files.
    Args:
        experiments (pd.dataframe): experiments to pre-process
        method (str): pyphotometry or lerner
        protocol (str): bpod protocol

    Returns:

    """
    for index, experiment in experiments.iterrows():
        mouse = experiment['mouse_id']
        date = experiment['date']
        if method == 'pyphotometry':
            pre_process_experiment_pyphotometry(mouse, date, protocol)
        elif method == 'lerner':
            pre_process_experiment_lerner_deissroth(mouse, date, protocol)


if __name__ == "__main__":
    mouse_ids = ['SNL_photo18']
    date = '20200311'
    for mouse_id in mouse_ids:
        all_experiments = get_all_experimental_records()
        if (mouse_id =='all') & (date == 'all'):
            experiments_to_process = all_experiments
        elif (mouse_id == 'all') & (date != 'all'):
            experiments_to_process = all_experiments[all_experiments['date'] == date]
        elif (mouse_id != 'all') & (date == 'all'):
            experiments_to_process = all_experiments[all_experiments['mouse_id'] == mouse_id]
        elif (mouse_id != 'all') & (date != 'all'):
            experiments_to_process = all_experiments[(all_experiments['date'] == date) & (all_experiments['mouse_id'] == mouse_id)]
        pre_process_experiments(experiments_to_process, method='pyphotometry', protocol='Two_Alternative_Choice')

