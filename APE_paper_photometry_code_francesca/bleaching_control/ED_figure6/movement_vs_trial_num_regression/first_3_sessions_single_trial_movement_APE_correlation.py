import pandas as pd
import os
from get_regression_slopes_for_turn_angle_speed_trial_number_vs_APE import create_movement_param_and_APE_df_just_first_3_sessions, correlate_movement_with_APE
from utils.kernel_regression.linear_regression_utils import get_first_x_sessions
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings
from set_global_params import experiment_record_path, processed_data_path, change_over_time_mice


if __name__ == '__main__':
    load_saved = True
    recording_site = 'tail'
    mice = change_over_time_mice[recording_site]
    experiment_record = pd.read_csv(experiment_record_path, dtype='str')
    good_experiments = remove_exps_after_manipulations(experiment_record, mice)
    clean_experiments = remove_unsuitable_recordings(good_experiments)

    for i, mouse in enumerate(mice):
        df_save_dir = r'{}{}\turn_angle_over_time'.format(processed_data_path, mouse)
        if not os.path.isdir(df_save_dir):
            os.makedirs(df_save_dir)
        df_save_file = os.path.join(df_save_dir, 'movement_params_all_trials_vs_APE_{}.pkl'.format(mouse))
        if os.path.isfile(df_save_file) & load_saved:
            valid_contra_data = pd.read_pickle(df_save_file)
        else:
            valid_contra_data = create_movement_param_and_APE_df_just_first_3_sessions(mouse)
            valid_contra_data.to_pickle(df_save_file)

        experiments_to_process = clean_experiments[
            (clean_experiments['mouse_id'] == mouse) & (
                        clean_experiments['recording_site'] == recording_site)].reset_index(drop=True)
        first_3_dates = get_first_x_sessions(experiments_to_process, x=3)['date'].values
        first_3_dates_data = valid_contra_data[valid_contra_data.date.isin(first_3_dates)]
        first_3_save_file = os.path.join(df_save_dir, 'movement_params_first_3_sessions_vs_APE_{}.pkl'.format(mouse))
        first_3_dates_data.to_pickle(first_3_save_file)

        speeds = []
        turn_angles = []
        mouse_ids = []
        dates_for_df = []

        speed_slope, speed_pval = correlate_movement_with_APE(first_3_dates_data, movement_type='average speed')
        turn_slope, turn_pval = correlate_movement_with_APE(first_3_dates_data, movement_type='abs fitted max cumsum ang vel')

        if i == 0:
            all_mice_df = pd.DataFrame({'mouse': [mouse], 'turn angle': [turn_slope], 'speed': [speed_slope],
                                        'turn angle pval': [turn_pval], 'speed pval': [speed_pval]})
        else:
            mouse_df = pd.DataFrame({'mouse': [mouse], 'turn angle': [turn_slope], 'speed': [speed_slope],
                                     'turn angle pval': [turn_pval], 'speed pval': [speed_pval]})
            all_mice_df = pd.concat([all_mice_df, mouse_df])

    all_mice_df = all_mice_df.reset_index(drop=True)
    all_mice_df_save_dir = os.path.join(processed_data_path, 'turn_angle_over_time')
    if not os.path.isdir(all_mice_df_save_dir):
        os.makedirs(all_mice_df_save_dir)
    all_mice_df_save_file = os.path.join(all_mice_df_save_dir, 'movement_params_first_3_sessions_vs_APE_regression_coefs_and_pvals.pkl')
    all_mice_df.to_pickle(all_mice_df_save_file)