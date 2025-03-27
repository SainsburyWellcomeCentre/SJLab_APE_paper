import os
import statsmodels.api as sm
from sklearn.metrics import explained_variance_score
from utils.plotting import *
from utils.tracking_analysis.extract_movement_for_all_sessions_utils import *
from utils.reaction_time_utils import get_bpod_trial_nums_per_session
from utils.post_processing_utils import get_all_experimental_records
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings, remove_manipulation_days
from utils.kernel_regression.linear_regression_utils import get_first_x_sessions
from set_global_params import processed_data_path, change_over_time_mice, raw_tracking_path

def get_session_with_10000th_trial(mouse, experiments):
    dates = experiments[experiments['mouse_id']==mouse]['date'].unique()
    session_starts = get_bpod_trial_nums_per_session(mouse, dates)
    if session_starts[-1] >= 10000:
        last_session_idx = np.where(np.asarray(session_starts) >=10000)[0][0]
    else:
        last_session_idx = -1
    last_session_date = dates[last_session_idx]
    return(last_session_date)


def filter_trials_around_reaction_time(side_data, window_around_mean=0.2):
    data = side_data.reset_index(drop=True)
    reaction_times = data['reaction times'].values
    median_reaction_time = np.nanmedian(reaction_times)
    valid_trials = np.where(
            np.logical_and(np.greater_equal(reaction_times, median_reaction_time - window_around_mean),
                           np.less_equal(reaction_times, median_reaction_time + window_around_mean)))
    valid_data = data.loc[valid_trials]
    return valid_data


def create_movement_param_and_APE_df(mouse, recording_site='tail'):
    all_experiments = get_all_experimental_records()
    all_experiments = remove_exps_after_manipulations(all_experiments, [mouse])
    all_experiments = remove_manipulation_days(all_experiments)
    all_experiments = remove_unsuitable_recordings(all_experiments)
    experiments_to_process = all_experiments[
        (all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == recording_site)]
    last_session = get_session_with_10000th_trial(mouse, experiments_to_process)
    dates = experiments_to_process['date'].values
    last_session_ind = int(np.where(dates == last_session)[0])
    for i, date in enumerate(dates[0: last_session_ind + 1]):
        save_out_folder = os.path.join(raw_tracking_path, mouse, date)
        movement_param_file = os.path.join(save_out_folder, 'APE_tracking{}_{}.pkl'.format(mouse, date))
        if os.path.isfile(movement_param_file):
            session_data = pd.read_pickle(movement_param_file)
            session_data['date'] = date
            session_data.mouse = mouse
            print(date, session_data.shape)
            if i == 0:
                all_session_data = session_data
            else:
                all_session_data = pd.concat([all_session_data, session_data])
        else:
            print('{} not found'.format(date))
            all_session_data, trial_data = get_movement_properties_for_session(mouse, date,
                                                                                protocol='Two_Alternative_Choice',
                                                                                multi_session=False)
            all_session_data.to_pickle(movement_param_file)
    all_session_data = all_session_data.reset_index(drop=True)
    all_session_data['abs fitted max cumsum ang vel'] = all_session_data['fitted max cumsum ang vel'].abs()
    contra_data = all_session_data[all_session_data.side == 'contra']
    valid_contra_data = filter_trials_around_reaction_time(contra_data)
    return valid_contra_data


def create_movement_param_and_APE_df_just_first_3_sessions(mouse, recording_site='tail'):
    all_experiments = get_all_experimental_records()
    all_experiments = remove_exps_after_manipulations(all_experiments, [mouse])
    all_experiments = remove_manipulation_days(all_experiments)
    all_experiments = remove_unsuitable_recordings(all_experiments)
    experiments_to_process = all_experiments[
        (all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == recording_site) & (all_experiments['tracking include'] != 'no')]
    if mouse == 'SNL_photo57':
        x=2
    else:
        x=3
    first_3_sessions = get_first_x_sessions(experiments_to_process, x=x)
    dates = first_3_sessions['date'].values
    for i, date in enumerate(dates):
        save_out_folder = os.path.join(raw_tracking_path, mouse, date)
        movement_param_file = os.path.join(save_out_folder, 'APE_tracking{}_{}.pkl'.format(mouse, date))
        if os.path.isfile(movement_param_file):
            session_data = pd.read_pickle(movement_param_file)
            print(date, session_data.shape)
        else:
            print('{} not found'.format(date))
            session_data, trial_data = get_movement_properties_for_session(mouse, date,
                                                                                protocol='Two_Alternative_Choice',
                                                                                multi_session=False)
            session_data.to_pickle(movement_param_file)
        session_data['date'] = date
        session_data.mouse = mouse
        if i == 0:
            all_session_data = session_data
        else:
            all_session_data = pd.concat([all_session_data, session_data])

    all_session_data = all_session_data.reset_index(drop=True)
    all_session_data['abs fitted max cumsum ang vel'] = all_session_data['fitted max cumsum ang vel'].abs()
    contra_data = all_session_data[all_session_data.side == 'contra']
    valid_contra_data = filter_trials_around_reaction_time(contra_data)
    return valid_contra_data


def correlate_movement_with_APE(valid_contra_data, movement_type='average speed'):
    data = valid_contra_data.dropna()
    slope, intercept, r, p_value, st_err = stats.linregress(data[movement_type], data['APE peaks'])
    return slope, p_value


def correlate_trial_num_with_movement(valid_contra_data, movement_type='average speed'):
    data = valid_contra_data.dropna()
    slope, intercept, r, p_value, st_err = stats.linregress(data['actual trial numbers'], data[movement_type])
    return slope, p_value

def regression_with_and_without_movement(valid_contra_data):
    df = valid_contra_data.sort_values(by='actual trial numbers')
    key = 'average speed'
    key2 = 'abs fitted max cumsum ang vel'
    mouse_data = df.dropna(axis=0)

    Y = mouse_data['APE peaks'].reset_index(drop=True)
    X = mouse_data[[key, key2]].reset_index(drop=True)
    X = sm.add_constant(X).reset_index(drop=True)
    model = sm.OLS(Y, X)
    result = model.fit()
    movement_model_prediction = model.predict(result.params)
    movement_model_diff_data = Y - movement_model_prediction

    key4 = 'LogTrialN'
    mouse_data = df.dropna(axis=0)
    mouse_data['LogTrialN'] = np.log(mouse_data['actual trial numbers'])
    residuals_X = mouse_data[[key4]].reset_index(drop=True)
    residuals_X = sm.add_constant(residuals_X).reset_index(drop=True)
    residuals_model = sm.OLS(movement_model_diff_data, residuals_X)
    residuals_result = residuals_model.fit()
    coef = residuals_result.params[key4]
    pval = residuals_result.pvalues[key4]
    return coef, pval

def regression_with_all_params(valid_contra_data):
    df = valid_contra_data.sort_values(by='actual trial numbers')
    key = 'average speed'
    key1 = 'abs fitted max cumsum ang vel'
    key2 = 'LogTrialN'

    mouse_data = df.dropna(axis=0)
    mouse_data['LogTrialN'] = np.log(mouse_data['actual trial numbers'])
    Y = mouse_data['APE peaks'].reset_index(drop=True)
    full_X = mouse_data[[key, key1, key2]].reset_index(drop=True)
    full_X = sm.add_constant(full_X).reset_index(drop=True)
    full_model = sm.OLS(Y, full_X)
    full_result = full_model.fit()
    speed_coef = full_result.params[key]
    turn_coef = full_result.params[key1]
    trial_num_coef = full_result.params[key2]
    return speed_coef, turn_coef, trial_num_coef


def regression_with_and_without_trial_num(valid_contra_data):
    df = valid_contra_data.sort_values(by='actual trial numbers')
    mouse_data = df.dropna(axis=0)
    mouse_data['LogTrialN'] = np.log(mouse_data['actual trial numbers'])
    key4 = 'LogTrialN'
    Y = mouse_data['APE peaks'].reset_index(drop=True)
    X = mouse_data[[key4]].reset_index(drop=True)
    X = sm.add_constant(X).reset_index(drop=True)
    model = sm.OLS(Y, X)
    result = model.fit()
    trial_num_model_prediction = model.predict(result.params)
    trial_num_model_diff_data = Y - trial_num_model_prediction

    key = 'average speed'
    key1 = 'abs fitted max cumsum ang vel'
    mouse_data = df.dropna(axis=0)

    residuals_X = mouse_data[[key, key1]].reset_index(drop=True)
    residuals_X = sm.add_constant(residuals_X).reset_index(drop=True)
    residuals_model = sm.OLS(trial_num_model_diff_data, residuals_X)
    residuals_result = residuals_model.fit()
    speed_coef = residuals_result.params[key]
    turn_coef = residuals_result.params[key1]
    return speed_coef, turn_coef

def full_model_regression(valid_contra_data):
    df = valid_contra_data.sort_values(by='actual trial numbers')
    mouse_data = df.dropna(axis=0)
    mouse_data['LogTrialN'] = np.log(mouse_data['actual trial numbers'])
    key4 = 'LogTrialN'
    key = 'average speed'
    key1 = 'abs fitted max cumsum ang vel'
    Y = mouse_data['APE peaks'].reset_index(drop=True)
    X = mouse_data[[key4, key, key1]].reset_index(drop=True)
    X = sm.add_constant(X).reset_index(drop=True)
    model = sm.OLS(Y, X)
    result = model.fit()
    full_model_prediction = model.predict(result.params)
    r2 = result.rsquared * 100
    return r2


def remove_one_parameter(param_names, param_to_remove, old_coefs, old_X):
    param_names.remove(param_to_remove)
    new_params = param_names
    new_coefs = old_coefs[new_params]
    new_X = old_X[new_params]
    return new_X, new_coefs


def remove_param_and_calculate_r2(param_to_remove, old_coefs, old_X, Y):
    param_names = old_X.columns.values.tolist()
    new_X, new_coefs = remove_one_parameter(param_names, param_to_remove, old_coefs, old_X)
    new_pred = np.dot(new_X, new_coefs)
    old_pred = np.dot(old_X, old_coefs)
    old_r2 = explained_variance_score(Y, old_pred)
    new_r2 = explained_variance_score(Y, new_pred)
    prop_due_to_param = (old_r2 - new_r2)/old_r2 * 100
    return new_pred, prop_due_to_param


def calculate_r2(valid_contra_data):
    df = valid_contra_data.sort_values(by='actual trial numbers')
    mouse_data = df.dropna(axis=0)
    mouse_data['LogTrialN'] = np.log(mouse_data['actual trial numbers'])
    key4 = 'LogTrialN'
    key = 'average speed'
    key1 = 'abs fitted max cumsum ang vel'
    params = [key, key1, key4]
    Y = mouse_data['APE peaks'].reset_index(drop=True)
    X = mouse_data[params].reset_index(drop=True)
    X = sm.add_constant(X).reset_index(drop=True)
    model = sm.OLS(Y, X)
    result = model.fit()
    coefs = result.params
    r2s = {}
    param_labels = {key4: 'log trial number', key: 'speed', key1: 'turn angle'}
    for param in params:
        _, r2 = remove_param_and_calculate_r2(param, coefs, X, Y)
        r2s[param_labels[param] + ' r2'] = r2
    r2s['full model'] = result.rsquared * 100
    return r2s


if __name__ == '__main__':
    load_saved = True
    mice = change_over_time_mice['tail']
    for i, mouse in enumerate(mice):
        df_save_dir = os.path.join(processed_data_path, mouse,'turn_angle_over_time')
        if not os.path.isdir(df_save_dir):
            os.makedirs(df_save_dir)
        df_save_file = os.path.join(df_save_dir, 'movement_params_all_trials_vs_APE_{}.pkl'.format(mouse)) # this pickle might need python 3.6 env
        if os.path.isfile(df_save_file) & load_saved:
            valid_contra_data = pd.read_pickle(df_save_file)
        else:
            valid_contra_data = create_movement_param_and_APE_df(mouse)
            valid_contra_data.to_pickle(df_save_file)
        trial_num_slope_speed, trial_num_speed_pval = correlate_trial_num_with_movement(valid_contra_data, movement_type='average speed')
        trial_num_slope_turn_ang, trial_num_turn_pval = correlate_trial_num_with_movement(valid_contra_data, movement_type='abs fitted max cumsum ang vel')
        speed_slope, speed_pval = correlate_movement_with_APE(valid_contra_data, movement_type='average speed')
        turn_slope, turn_pval = correlate_movement_with_APE(valid_contra_data, movement_type='abs fitted max cumsum ang vel')
        trial_num_slope, trial_num_pval = regression_with_and_without_movement(valid_contra_data)
        full_model_speed_slope, full_model_turn_slope = regression_with_and_without_trial_num(valid_contra_data)
        reg_dict = {'mouse': [mouse], 'turn angle': [turn_slope], 'speed': [speed_slope], 'log trial number': [trial_num_slope], 'full model speed': [full_model_speed_slope],
                                         'full model turn angle': [full_model_turn_slope], 'turn angle pval': [turn_pval], 'speed pval': [speed_pval], 'log trial number pval': [trial_num_pval],
                    'trial_num_slope_turn_ang': trial_num_slope_turn_ang, 'trial_num_turn_pval': trial_num_turn_pval, 'trial_num_slope_speed': trial_num_slope_speed, 'trial_num_speed_pval': trial_num_speed_pval}
        r2s = calculate_r2(valid_contra_data)
        reg_dict.update(r2s)

        if i == 0:
             all_mice_df = pd.DataFrame(reg_dict)
        else:
            mouse_df = pd.DataFrame(reg_dict)
            all_mice_df = pd.concat([all_mice_df, mouse_df])
    all_mice_df = all_mice_df.reset_index(drop=True)
    all_mice_df_save_dir = os.path.join(processed_data_path, 'turn_angle_over_time')
    if not os.path.isdir(all_mice_df_save_dir):
        os.makedirs(all_mice_df_save_dir)
    all_mice_df_save_file = os.path.join(all_mice_df_save_dir, 'movement_params_all_trials_vs_APE_regression_coefs_pvals_r2_and_trial_num_correlation_and_full_model.pkl')
    all_mice_df.to_pickle(all_mice_df_save_file)


