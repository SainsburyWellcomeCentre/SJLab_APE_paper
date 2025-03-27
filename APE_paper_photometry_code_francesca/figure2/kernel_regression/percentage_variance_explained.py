import os
from utils.kernel_regression.linear_regression_utils import *
import gc
import pickle
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings
from sklearn.metrics import explained_variance_score
from set_global_params import experiment_record_path, processed_data_path, mice_average_traces

site = 'tail'
mouse_ids = mice_average_traces[site]
experiment_record = pd.read_csv(experiment_record_path)
experiment_record['date'] = experiment_record['date'].astype(str)
good_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
clean_experiments = remove_unsuitable_recordings(good_experiments)
all_experiments_to_process = clean_experiments[(clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(drop=True)
experiments_to_process = get_first_x_sessions(all_experiments_to_process)

file_name = site + '_explained_variances_all_cues.p'
processed_data_dir = os.path.join(processed_data_path, 'linear_regression_data')
saving_filename = os.path.join(processed_data_dir, file_name)
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

choice = []
cue = []
outcome = []
model_total = []
for index, experiment in experiments_to_process.iterrows():
    mouse = experiment['mouse_id']
    date = experiment['date']
    print('proccessing' + mouse + date)
    saving_folder = processed_data_path + mouse + '\\'
    save_filename = saving_folder + mouse + '_' + date + '_'
    kernel_filename = save_filename + 'linear_regression_kernels_different_shifts_all_cues_matched_trials.p'
    inputs_X_filename = save_filename + 'linear_regression_different_shifts_X_all_cues_matched_trials.p'
    inputs_y_filename = save_filename + 'linear_regression_different_shifts_y_all_cues_matched_trials.p'
    X = pickle.load(open(inputs_X_filename, 'rb'))
    y = pickle.load(open(inputs_y_filename, 'rb'))
    kernels = pickle.load(open(kernel_filename, 'rb'))
    kernel_list = []
    param_names = ['high cues', 'low cues', 'ipsi choices', 'contra choices', 'rewards', 'no rewards']
    for param_name in param_names:
        kernel = kernels['kernels'][param_name]
        kernel_list.append(kernel)
    coefs = np.array([item for sublist in kernel_list for item in sublist])
    intercept = kernels['intercept']
    shifts = kernels['shifts']
    windows = kernels['shift_window_lengths']
    all_shifts = [shifts[param_name] for param_name in param_names]
    all_windows = [windows[param_name] for param_name in param_names]
    params_to_remove = ['high cues', 'low cues']
    cue_pred, prop_due_to_cue = remove_param_and_calculate_r2(param_names, params_to_remove, coefs, X,
                                                                intercept, y, all_shifts, all_windows)
    print('cue: ', prop_due_to_cue)
    params_to_remove = ['ipsi choices', 'contra choices']
    choice_pred, prop_due_to_choice = remove_param_and_calculate_r2(param_names, params_to_remove, coefs, X,
                                                                intercept, y, all_shifts, all_windows)
    print('choice: ', prop_due_to_choice)
    params_to_remove = ['rewards', 'no rewards']
    reward_pred, prop_due_to_outcome = remove_param_and_calculate_r2(param_names, params_to_remove, coefs, X,
                                                                intercept, y, all_shifts, all_windows)
    print('outcome: ', prop_due_to_outcome)
    full_model = explained_variance_score(y, np.dot(X, coefs) + intercept)*100
    print('full model: ', full_model)
    model_total.append(full_model)
    choice.append(prop_due_to_choice)
    cue.append(prop_due_to_cue)
    outcome.append(prop_due_to_outcome)
    gc.collect()
regression_stats = experiments_to_process[['mouse_id', 'date']].reset_index(drop=True)
regression_stats['cue explained variance'] = cue
regression_stats['choice explained variance'] = choice
regression_stats['outcome explained variance'] = outcome
regression_stats['full model explained variance'] = model_total
with open(saving_filename, "wb") as f:
    pickle.dump(regression_stats, f)

