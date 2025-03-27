import os
from utils.kernel_regression.linear_regression_utils import *
import gc
from set_global_params import experiment_record_path, processed_data_path
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings
from utils.kernel_regression.return_to_centre_regression_utils import get_first_x_sessions_reg_rtc, calculate_prop_data_covered_by_design_matrix
from set_global_params import processed_data_path, mice_average_traces

site = 'tail'
mouse_ids = mice_average_traces[site]
experiment_record = pd.read_csv(experiment_record_path)
experiment_record['date'] = experiment_record['date'].astype(str)
good_experiments = remove_exps_after_manipulations(experiment_record, mouse_ids)
clean_experiments = remove_unsuitable_recordings(good_experiments)
all_experiments_to_process = clean_experiments[(clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(drop=True)
experiments_to_process = get_first_x_sessions_reg_rtc(all_experiments_to_process)
perc_covered = []

for index, experiment in experiments_to_process.iterrows():
    perc_covered.append(calculate_prop_data_covered_by_design_matrix(experiment))

file_name = site + '_perc_covereed_by_design_matrix_all_cues_trimmed_traces_only_tracking_mice.p' #'_explained_variances_not_cleaned.p' #'_explained_variances.p'
processed_data_dir = os.path.join(processed_data_path, 'linear_regression_data')
saving_filename = os.path.join(processed_data_path, 'linear_regression_data', file_name)
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

regression_stats = experiments_to_process[['mouse_id', 'date']].reset_index(drop=True)
regression_stats['perc covered by design matrix'] = perc_covered

with open(saving_filename, "wb") as f:
    pickle.dump(regression_stats, f)

    gc.collect()

