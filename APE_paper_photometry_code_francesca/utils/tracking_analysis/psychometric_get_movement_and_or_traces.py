import os
import pandas as pd
from utils.tracking_analysis.camera_trigger_preprocessing_utils import *
import scipy as sp
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings
from utils.kernel_regression.linear_regression_utils import get_first_x_sessions
from utils.tracking_analysis import dlc_processing_utils
from set_global_params import experiment_record_path, post_processed_tracking_data_path, psychometric_data_path


def get_fit_slopes(quantile_data, experiment):
    mouse = experiment['mouse_id']
    date = experiment['date']
    quantile_means = []
    quantile_ranges = []
    quantile_midpoints = []
    quantile_nums = []
    quantile_slopes = []
    num_trials = []

    for i, q in enumerate(quantile_data['APE quantile'].unique()[::-1]):
        quantile_midpoint = q.mid
        trials = quantile_data.loc[quantile_data['APE quantile'] == q]
        num_trials.append(trials.shape[0])
        quantile_means.append(np.nanmedian(trials['fitted max cumsum ang vel'].values))
        quantile_slopes.append(np.nanmedian(trials['turn slopes'].values))
        quantile_midpoints.append(quantile_midpoint)
        quantile_ranges.append(q)
        quantile_nums.append(i)
    norm_quantile_means = np.abs(quantile_means) / np.max(np.abs(quantile_means))
    norm_quantile_midpoints = quantile_midpoints / np.max(quantile_midpoints)
    norm_slopes = np.abs(quantile_slopes) / np.max(np.abs(quantile_slopes))
    fit_slope = np.polyfit(norm_quantile_midpoints, norm_quantile_means, 1)[0]
    fit_slope_slopes = np.polyfit(norm_quantile_midpoints, norm_slopes, 1)[0]
    session_numbers = np.ones([len(quantile_midpoints)]) * experiment['session number']
    quantile_df = pd.DataFrame({'mean max cumsum ang vel': quantile_means, 'quantile range': quantile_ranges,
                                'APE quantile midpoint': quantile_midpoints,
                                'quantile num': quantile_nums, 'mouse': mouse, 'session': date,
                                'normalised cumsum ang vel': norm_quantile_means, 'slopes': quantile_slopes,
                                'norm slopes': norm_slopes, 'normalised APE quantile midpoint': norm_quantile_midpoints,
                                'fit slope': fit_slope, 'session number': session_numbers, 'num trials': num_trials,
                                'fit slope slopes': fit_slope_slopes})
    return quantile_df


def get_all_mice_data(experiments_to_process, exp_type='', key='fitted max cumsum ang vel', shuffle=True, num_shuffles=100, load_saved=True, get_movement=True, align_to='choice'):
    for index, experiment in experiments_to_process.iterrows():
        mouse = experiment['mouse_id']
        date = experiment['date']
        save_out_folder = os.path.join(psychometric_data_path, mouse)
        if not os.path.exists(save_out_folder):
            os.makedirs(save_out_folder)
        movement_param_file = os.path.join(save_out_folder, 'contra_APE_tracking{}_{}{}.pkl'.format(mouse, date, exp_type))
        if os.path.isfile(movement_param_file) and load_saved:
            quantile_data = pd.read_pickle(movement_param_file)
        else:
            if get_movement:
                quantile_data, trial_data = dlc_processing_utils.get_movement_properties_for_session(mouse, date)
                quantile_data.to_pickle(movement_param_file)
            else:
                quantile_data, trial_data = dlc_processing_utils.get_peaks_and_trial_types(mouse, date, align_to=align_to)
                quantile_data.to_pickle(movement_param_file)
        if get_movement:
            non_nan_data = quantile_data[np.invert(np.isnan(quantile_data[key]))]
            slope, intercept, r_value, p_value, std_err = sp.stats.linregress(non_nan_data[key], non_nan_data['APE peaks'])

            shuffled_data = quantile_data.copy(deep=False)
            shuffled_data['APE quantile'] = np.random.permutation(quantile_data['APE quantile'].values)

            quantile_df = get_fit_slopes(quantile_data, experiment)
            quantile_df['recording site'] = experiment['recording_site']
            quantile_df['shuffle number'] = np.nan
            quantile_df['r squared'] = r_value ** 2
            if shuffle:
                for i in range(0, num_shuffles):
                    shuffled_data = quantile_data.copy(deep=False)
                    shuffled_data['APE quantile'] = np.random.permutation(quantile_data['APE quantile'].values)

                    shuffled_df = get_fit_slopes(shuffled_data, experiment)
                    shuffled_df['shuffle number'] = i
                    s_data = shuffled_df.drop_duplicates(subset=['mouse', 'session number'])

                    if index == 0 & i == 0:
                        shuffled_fit_slopes = s_data
                    else:
                        shuffled_fit_slopes = pd.concat([shuffled_fit_slopes, s_data])
            else:
                shuffled_fit_slopes = False
        quantile_data['mouse'] = experiment['mouse_id']
        quantile_data['session'] = experiment['date']
        if index == 0:
            if get_movement:
                restructured_data = quantile_df
            all_trials_data = quantile_data
        else:
            if get_movement:
                restructured_data = pd.concat([restructured_data, quantile_df], ignore_index=True)
            all_trials_data = pd.concat([all_trials_data, quantile_data])
            if get_movement:
                q_data = restructured_data.drop_duplicates(subset=['mouse', 'session number', 'recording site'])
            else:
                restructured_data = None
                q_data = None
                shuffled_fit_slopes = None
    return restructured_data, q_data, shuffled_fit_slopes, all_trials_data


def get_first_three_sessions_dlc(mouse_ids, site, num_sessions=3, save=False, load_saved=True):
    save_out_folder = post_processed_tracking_data_path
    mouse_names = '_'.join(mouse_ids)
    save_out_file_shuffles = os.path.join(save_out_folder, 'contra_APE_tracking_first_{}_sessions_{}_with_shuffles.pkl'.format(num_sessions, mouse_names))
    save_out_file = os.path.join(save_out_folder, 'contra_APE_tracking_first_{}_sessions_{}.pkl'.format(num_sessions, mouse_names))
    if os.path.isfile(save_out_file_shuffles) and os.path.isfile(save_out_file) and load_saved:
        data_to_save = pd.read_pickle(save_out_file)
        all_data = pd.read_pickle(save_out_file_shuffles)
    else:
        experiment_record = pd.read_csv(experiment_record_path, dtype='str')
        experiment_record['date'] = experiment_record['date'].astype(str)
        clean_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
        all_experiments_to_process = clean_experiments[
            (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
            drop=True)
        all_experiments_to_process = all_experiments_to_process[all_experiments_to_process['tracking include'] != 'no'].reset_index(drop=True)
        experiments_to_process = get_first_x_sessions(all_experiments_to_process, x=num_sessions).reset_index(
            drop=True)


        data_to_save, q_data, s_data, _ = get_all_mice_data(experiments_to_process)
        s_data['recording site'] = 'shuffled ' + site
        all_data = pd.concat([s_data, q_data])


        data_to_save['mean max cumsum ang vel'] = data_to_save['mean max cumsum ang vel'].abs()
        data_to_save['norm by mouse'] = data_to_save.groupby(['mouse'])['mean max cumsum ang vel'].transform(lambda x: (x/ x.max()))

        #plot_data = q_data
        #sns.catplot(x='recording site', y='fit slope', data=plot_data, hue='mouse', palette='pastel')
        #fig, axs = plt.subplots(1,1)
        #make_box_plot(plot_data, dx='recording site', dy='fit slope', fig_ax=axs)
        #print(sp.stats.ttest_ind(plot_data[plot_data['recording site'] == 'tail']['fit slope'], plot_data[plot_data['recording site'] == 'tail shuffled']['fit slope']))
        if save:
            all_data.to_pickle(save_out_file_shuffles)
            data_to_save.to_pickle(save_out_file)
    return data_to_save, all_data

