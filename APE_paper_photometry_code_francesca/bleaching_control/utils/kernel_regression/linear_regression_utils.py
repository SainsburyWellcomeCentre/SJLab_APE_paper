import numpy as np
from scipy.ndimage.interpolation import shift
from matplotlib import pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression
from set_global_params import daq_sample_rate


def rolling_zscore(x, window=10*daq_sample_rate):
    r = x.rolling(window=window)
    m = r.mean().shift(1)
    s = r.std(ddof=0).shift(1)
    z = (x-m)/s
    return z


def turn_timestamps_into_continuous(num_samples, *behavioural_events):
    continuous_parameters = []
    for event_type_timestamps in behavioural_events:
        continuous_time_version = np.zeros([num_samples])
        continuous_time_version[event_type_timestamps] = 1
        continuous_parameters.append(continuous_time_version)
    return continuous_parameters


def convert_behavioural_timestamps_into_samples(time_stamps, window_to_remove, sample_rate=daq_sample_rate, decimate_factor=100):
    adjusted_stamps = (time_stamps - window_to_remove)*sample_rate/decimate_factor
    adjusted_stamps = np.round(np.vstack(adjusted_stamps).astype(np.float)).astype(int)
    return adjusted_stamps


def make_design_matrix(parameters, window_min=-1*daq_sample_rate/100, window_max=1.5*daq_sample_rate/100):
    num_parameters = len(parameters)
    shifts = np.arange(window_min, window_max + 1)
    shift_window_size = shifts.shape[0]
    X = np.zeros([parameters[0].shape[0], shift_window_size*num_parameters])
    all_param_indices = []
    for shift_num, shift_val in enumerate(shifts):
        for param_num, param in enumerate(parameters):
            param_indices = range(param_num*shift_window_size, param_num*shift_window_size + shift_window_size)
            all_param_indices.append(param_indices)
            shifted_param = shift(param, shift_val, cval=0)
            X[:, param_indices[shift_num]] = shifted_param
    return all_param_indices, X


def plot_kernels(parameter_names, regression_results, window_min=-1 * daq_sample_rate / 100, window_max=1.5 * daq_sample_rate / 100):
    fig, axs = plt.subplots(nrows=1, ncols=len(parameter_names), sharey=True, figsize=(15, 8))
    axs[0].set_ylabel('Regression coefficient')
    shifts = np.arange(window_min, window_max + 1) / 100
    shift_window_size = shifts.shape[0]
    for param_num, param_name in enumerate(parameter_names):
        param_kernel = regression_results.coef_[param_num * shift_window_size:(param_num + 1) * shift_window_size]
        axs[param_num].plot(shifts, param_kernel, label=param_name)
        axs[param_num].set_title(param_name)
        axs[param_num].set_xlabel('Time (s)')
        axs[param_name].axvline(0, color='k')


def save_kernels(save_filename, parameter_names, regression_results, downsampled_dff, X, window_min=-1 * daq_sample_rate / 100, window_max=1.5 * daq_sample_rate/ 100):
    shifts = np.arange(window_min, window_max + 1) / 100
    shift_window_size = shifts.shape[0]
    param_kernels = {}
    for param_num, param_name in enumerate(parameter_names):
        kernel_name = parameter_names[param_num]
        param_kernels[kernel_name] = regression_results.coef_[param_num * shift_window_size:(param_num + 1) * shift_window_size]
    session_kernels = {}
    session_kernels['kernels'] = param_kernels
    session_kernels['time_stamps'] = shifts
    session_kernels['intercept'] = regression_results.intercept_

    kernel_filename = save_filename + 'linear_regression_kernels_no_repeated_cues_both_cues.p'
    inputs_X_filename = save_filename + 'linear_regression_X.p'
    inputs_y_filename = save_filename + 'linear_regression_y.p'
    with open(kernel_filename, "wb") as f:
        pickle.dump(session_kernels, f)
    with open(inputs_X_filename, "wb") as f:
        pickle.dump(X, f)
    with open(inputs_y_filename, "wb") as f:
        pickle.dump(downsampled_dff, f)


def save_kernels_different_shifts(save_filename, parameter_names,params, regression_results, downsampled_dff, X, all_shifts, shift_window_sizes):
    param_kernels = {}
    shifts_for_saving = {}
    shift_window_lengths = {}
    for param_num, param_name in enumerate(parameter_names):
        kernel_name = parameter_names[param_num]
        shifts = all_shifts[param_num]
        shift_window_size = shift_window_sizes[param_num]
        starting_ind = int(np.sum(shift_window_sizes[:param_num]))
        param_kernels[kernel_name] = regression_results.coef_[starting_ind: starting_ind + shift_window_size]
        shifts_for_saving[kernel_name] = shifts
        shift_window_lengths[kernel_name] = shift_window_size
    session_kernels = {}
    session_kernels['kernels'] = param_kernels
    session_kernels['shifts'] = shifts_for_saving
    session_kernels['shift_window_lengths'] = shift_window_lengths
    session_kernels['intercept'] = regression_results.intercept_
    kernel_filename = save_filename + 'linear_regression_kernels_different_shifts_all_cues_matched_trials.p'
    params_filename = save_filename + 'linear_regression_parameters_all_cues_matched_trials.p'
    inputs_X_filename = save_filename + 'linear_regression_different_shifts_X_all_cues_matched_trials.p'
    inputs_y_filename = save_filename + 'linear_regression_different_shifts_y_all_cues_matched_trials.p'
    with open(kernel_filename, "wb") as f:
        pickle.dump(session_kernels, f)
    with open(params_filename, "wb") as f:
        pickle.dump(params, f)
    with open(inputs_X_filename, "wb") as f:
        pickle.dump(X, f)
    with open(inputs_y_filename, "wb") as f:
        pickle.dump(downsampled_dff, f)


def save_kernels_different_shifts_diff_reg_types(save_filename, parameter_names,params, regression_results, downsampled_dff, X, all_shifts, shift_window_sizes, reg_type=''):
    param_kernels = {}
    shifts_for_saving = {}
    shift_window_lengths = {}
    for param_num, param_name in enumerate(parameter_names):
        kernel_name = parameter_names[param_num]
        shifts = all_shifts[param_num]
        shift_window_size = shift_window_sizes[param_num]
        starting_ind = int(np.sum(shift_window_sizes[:param_num]))
        param_kernels[kernel_name] = regression_results.coef_[starting_ind: starting_ind + shift_window_size]
        shifts_for_saving[kernel_name] = shifts
        shift_window_lengths[kernel_name] = shift_window_size
    session_kernels = {}
    session_kernels['kernels'] = param_kernels
    session_kernels['shifts'] = shifts_for_saving
    session_kernels['shift_window_lengths'] = shift_window_lengths
    session_kernels['intercept'] = regression_results.intercept_
    kernel_filename = save_filename + 'linear_regression_kernels{}.p'.format(reg_type)
    params_filename = save_filename + 'linear_regression_parameters{}.p'.format(reg_type)
    inputs_X_filename = save_filename + 'linear_regression_different_shifts{}.p'.format(reg_type)
    inputs_y_filename = save_filename + 'linear_regression_different_shifts_y{}.p'.format(reg_type)
    with open(kernel_filename, "wb") as f:
        pickle.dump(session_kernels, f)
    with open(params_filename, "wb") as f:
        pickle.dump(params, f)
    with open(inputs_X_filename, "wb") as f:
        pickle.dump(X, f)
    with open(inputs_y_filename, "wb") as f:
        pickle.dump(downsampled_dff, f)


def get_first_x_sessions(sorted_experiment_record, x=3):
    i = []
    inds = []
    for mouse in np.unique(sorted_experiment_record['mouse_id']):
        i.append(sorted_experiment_record[sorted_experiment_record['mouse_id'] == mouse][0:x].index)
        inds += range(0, x)
    flattened_i = [val for sublist in i for val in sublist]
    exps = sorted_experiment_record.loc[flattened_i].reset_index(drop=True)
    exps['session number'] = inds
    return exps


def remove_one_parameter(param_names, params_to_remove, old_coefs, old_X, window_min=-0.5*daq_sample_rate/100, window_max=1.5*daq_sample_rate/100):
    param_df = pd.DataFrame({'parameter': param_names})
    params_to_include = param_df[~param_df['parameter'].isin(params_to_remove)]
    params_to_include = params_to_include.reset_index(drop=False)
    num_parameters = params_to_include.shape[0]
    shifts = np.arange(window_min, window_max + 1)/100
    shift_window_size = shifts.shape[0]
    new_coefs = np.zeros([shift_window_size*num_parameters])
    new_X = np.zeros([old_X.shape[0], shift_window_size*num_parameters])
    for param_num, param_row in params_to_include.iterrows():
        old_index = param_row['index']
        new_index = param_num
        param_kernel = old_coefs[old_index*shift_window_size:(old_index+1)*shift_window_size]
        param_indices = range(new_index*shift_window_size,new_index*shift_window_size + shift_window_size)
        new_coefs[param_indices] = param_kernel
        old_X_for_param = old_X[:, old_index*shift_window_size:(old_index+1)*shift_window_size]
        new_X[:, param_indices] = old_X_for_param
    return new_coefs, new_X, params_to_include


def remove_one_parameter_different_shifts(param_names, params_to_remove, old_coefs, old_X, all_shifts, shift_window_sizes, remove=True):
    param_df = pd.DataFrame({'parameter': param_names, 'shifts': all_shifts, 'shift window sizes': shift_window_sizes})
    if remove:
        params_to_include = param_df[~param_df['parameter'].isin(params_to_remove)]
    else:
        params_to_include = param_df[param_df['parameter'].isin(params_to_remove)]
    params_to_include = params_to_include.reset_index(drop=False)
    new_coefs = np.zeros([np.sum(params_to_include['shift window sizes'])])
    new_X = np.zeros([old_X.shape[0], np.sum(params_to_include['shift window sizes'])])
    for param_num, param_row in params_to_include.iterrows():
        shift_window_size = param_row['shift window sizes']
        old_index = param_row['index']
        new_index = param_num
        old_starting_ind = int(np.sum(shift_window_sizes[:old_index]))
        new_starting_ind = int(np.sum(params_to_include['shift window sizes'][:new_index]))
        param_kernel = old_coefs[old_starting_ind: old_starting_ind + shift_window_size]
        param_indices = range(new_starting_ind, new_starting_ind + shift_window_size)
        new_coefs[param_indices] = param_kernel
        old_X_for_param = old_X[:, old_starting_ind: old_starting_ind + shift_window_size]
        new_X[:, param_indices] = old_X_for_param
    return new_coefs, new_X, params_to_include


def remove_param_and_calculate_r2(param_names, param_to_remove, old_coefs, old_X, intercept, dff, shifts, window_sizes, remove=True):
    new_coefs, new_X, params = remove_one_parameter_different_shifts(param_names, param_to_remove, old_coefs, old_X, shifts, window_sizes, remove=remove)
    new_pred = np.dot(new_X, new_coefs) + intercept
    old_pred = np.dot(old_X, old_coefs) + intercept
    old_r2 = explained_variance_score(dff, old_pred)
    new_r2 = explained_variance_score(dff, new_pred)
    prop_due_to_param = (old_r2 - new_r2)/old_r2 * 100
    return new_pred, prop_due_to_param


def remove_param_and_refit_r2(param_names, param_to_remove, old_coefs, old_X, intercept, dff):
    _, new_X, params = remove_one_parameter(param_names, param_to_remove, old_coefs, old_X)
    refitted_results = LinearRegression().fit(new_X, dff)
    new_coefs = refitted_results.coef_
    new_intercept = refitted_results.intercept_
    new_pred = np.dot(new_X, new_coefs) + new_intercept
    old_pred = np.dot(old_X, old_coefs) + intercept
    old_r2 = explained_variance_score(dff, old_pred)
    new_r2 = explained_variance_score(dff, new_pred)
    prop_due_to_param = (old_r2 - new_r2)/old_r2 * 100
    return new_pred, prop_due_to_param


def make_design_matrix_different_shifts(parameters, all_shifts, shift_window_sizes):
    num_parameters = len(parameters)
    total_num_regressors = np.sum(shift_window_sizes)
    X = np.zeros([parameters[0].shape[0], total_num_regressors])
    all_param_indices = []
    for param_num, param in enumerate(parameters):
        shifts = all_shifts[param_num]
        shift_window_size = shift_window_sizes[param_num]
        starting_ind = int(np.sum(shift_window_sizes[:param_num]))
        for shift_num, shift_val in enumerate(shifts):
                param_indices = range(starting_ind, starting_ind + shift_window_size)
                all_param_indices.append(param_indices)
                shifted_param = shift(param, shift_val, cval=0)
                X[:, param_indices[shift_num]] = shifted_param
    return(all_param_indices, X)


def make_shifts_for_params(param_names):
    shifts_for_params = []
    shift_window_sizes = []
    shifts = {'high cues': np.arange(0, 1*daq_sample_rate/100 + 1),
              'low cues': np.arange(0, 1*daq_sample_rate/100 + 1),
              'ipsi choices': np.arange(-0.5*daq_sample_rate/100, 1.5*daq_sample_rate/100 + 1),
              'contra choices': np.arange(-0.5*daq_sample_rate/100, 1.5*daq_sample_rate/100 + 1),
              'rewards': np.arange(0, 1*daq_sample_rate/100 + 1),
              'no rewards': np.arange(0, 1*daq_sample_rate/100 + 1),
              'contra returns': np.arange(-0.2*daq_sample_rate/100, 1.5*daq_sample_rate/100 + 1),
              'ipsi returns': np.arange(-0.2 * daq_sample_rate / 100, 1.5 * daq_sample_rate / 100 + 1)
             }
    for param in param_names:
        shifts_for_params.append(shifts[param])
        shift_window_sizes.append(shifts[param].shape[0])
    return shifts_for_params, shift_window_sizes