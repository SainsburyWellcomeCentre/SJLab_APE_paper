import numpy as np
import scipy.signal
from scipy.stats import halfnorm


def smooth_with_half_gauss(signal, window=1000):
    x = np.linspace(halfnorm.ppf(0.01), halfnorm.ppf(0.99), window)
    k = halfnorm.pdf(x)
    s = np.convolve(signal, k, mode='valid')
    return s


def smooth_with_window_mean(signal, window=1000):
    cumulativ = signal.cumsum()
    smoothed_signal = (cumulativ[window:] - cumulativ[:-window]) / float(window)
    padded_smoothed_signal = np.pad(smoothed_signal, (window, 0), 'constant', constant_values=(0, 0))
    return padded_smoothed_signal


def am_demodulate_unpad(
    signal,
    reference,
    modulation_frequency,
    sampling_rate=10000,
    low_cut=15,
    order=5):

    """
    demodulates photodetector input to get quadrature and in-phase components
    Code from Steve Lenzi
    :param signal:
    :param reference:
    :param modulation_frequency:
    :param sampling_rate:
    :param low_cut:
    :param order:
    :return:
    """
    normalised_reference = reference - reference.mean()
    samples_per_period = sampling_rate / modulation_frequency
    samples_per_quarter_period = round(samples_per_period / 4)

    shift_90_degrees = np.roll(
        normalised_reference, samples_per_quarter_period
    )

    in_phase = signal * normalised_reference
    in_phase_filtered = _apply_butterworth_lowpass_filter(
        in_phase, low_cut_off=low_cut, fs=sampling_rate, order=order
    )

    quadrature = signal * shift_90_degrees
    quadrature_filtered = _apply_butterworth_lowpass_filter(
        quadrature, low_cut_off=low_cut, fs=sampling_rate, order=order
    )

    return quadrature_filtered, in_phase_filtered


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


def lerner_deisseroth_preprocess(
    photodetector_raw_data,
    reference_channel_211hz,
    reference_channel_531hz,
    sampling_rate,
):
    """
    process data according to https://www.ncbi.nlm.nih.gov/pubmed/26232229 , supplement 11
    Code from Steve Lenzi
    :param photodetector_raw_data: the raw signal from the photodetector
    :param reference_channel_211hz:  a copy of the reference signal sent to the signal LED (Ca2+ dependent)
    :param reference_channel_531hz:  a copy of the reference signal sent to the background LED (Ca2+ independent)
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
