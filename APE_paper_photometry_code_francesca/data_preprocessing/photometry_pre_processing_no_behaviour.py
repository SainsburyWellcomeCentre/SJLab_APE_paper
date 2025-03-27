import nptdms
import numpy as np
from data_preprocessing.demodulation import demodulate
import os
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from set_global_params import daq_sample_rate, photometry_data_path, out_of_task_movement_mice_dates, running_in_box_dir


def pre_process_experiment_pyphotometry_only_photometry(file, mouse, date, sampling_rate=daq_sample_rate):
    data = nptdms.TdmsFile(file)
    # for python 3.6 with tdms version 0.27.0
    # chan_0 = data.group_channels('acq_task')[3].data
    # led405 = data.group_channels('acq_task')[5].data
    # led465 = data.group_channels('acq_task')[4].data
    # clock = data.group_channels('acq_task')[1].data
    # stim_trigger = data.group_channels('acq_task')[0].data

    # this data was recorded on a different setup from main data
    chan_0 = data['acq_task'].channels()[3]
    led405 = data['acq_task'].channels()[5]
    led465 = data['acq_task'].channels()[4]
    clock = data['acq_task'].channels()[1]
    stim_trigger = data['acq_task'].channels()[0]

    signal, back = demodulate(chan_0[sampling_rate * 6:], led465[sampling_rate * 6:], led405[sampling_rate * 6:], 10000)

    GCaMP_raw = signal
    back_raw = back


    # Median filtering to remove electrical artifact.
    GCaMP_denoised = medfilt(GCaMP_raw, kernel_size=5)
    back_denoised = medfilt(back_raw, kernel_size=5)

    # Lowpass filter - zero phase filtering (with filtfilt) is used to avoid distorting the signal.
    b, a = butter(2, 10, btype='low', fs=sampling_rate)
    GCaMP_denoised = filtfilt(b, a, GCaMP_denoised)
    back_denoised = filtfilt(b, a, back_denoised)

    b, a = butter(2, 0.001, btype='high', fs=sampling_rate)
    GCaMP_highpass = filtfilt(b, a, GCaMP_denoised, padtype='even')
    back_highpass = filtfilt(b, a, back_denoised, padtype='even')

    slope, intercept, r_value, p_value, std_err = linregress(x=back_highpass, y=GCaMP_highpass)

    print('Slope    : {:.3f}'.format(slope))
    print('R-squared: {:.3f}'.format(r_value ** 2))

    GCaMP_est_motion = intercept + slope * back_highpass
    GCaMP_corrected = GCaMP_highpass - GCaMP_est_motion

    b, a = butter(2, 0.001, btype='low', fs=sampling_rate)
    baseline_fluorescence = filtfilt(b, a, GCaMP_denoised, padtype='even')

    GCaMP_dF_F = GCaMP_corrected / baseline_fluorescence
    b, a = butter(2, 3, btype='low', fs=sampling_rate)
    smoothed_GCaMP = filtfilt(b, a, GCaMP_dF_F, padtype='even')
    smoothed_GCaMP = np.pad(smoothed_GCaMP, (6 * sampling_rate, 0), mode='median')
    clock_diff = np.diff(clock)
    clock_pulses = np.where(clock_diff > 2.4)[0]

    saving_folder = os.path.join(running_in_box_dir, 'processed_data\\{}\\'.format(mouse))
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    smoothed_trace_filename = mouse + '_' + date + '_' + 'smoothed_signal.npy'
    clock_filename = mouse + '_' + date + '_' + 'clock.npy'
    stimulus_filename = mouse + '_' + date + '_' + 'stimulus.npy'
    np.save(saving_folder + smoothed_trace_filename, smoothed_GCaMP)
    np.save(saving_folder + clock_filename, clock_pulses)
    np.save(saving_folder + stimulus_filename, stim_trigger)


if __name__ == "__main__":
    for mouse, date_time in out_of_task_movement_mice_dates.items():
        file = os.path.join(photometry_data_path, '\\running_in_box\\{}\\{}\\AI.tdms'.format(mouse, date_time))
        date = date_time[0:8]
        pre_process_experiment_pyphotometry_only_photometry(file, mouse, date)