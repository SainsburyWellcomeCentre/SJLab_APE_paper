import pandas as pd
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings, get_first_x_sessions, get_all_experimental_records, add_experiment_to_aligned_data
from set_global_params import experiment_record_path, mice_average_traces


site = 'tail_ant' # can also be 'tail', 'nacc'.
mouse_ids = mice_average_traces[site]
experiment_record = pd.read_csv(experiment_record_path)

experiments_to_process = get_first_x_sessions(experiment_record, mouse_ids, site).reset_index(drop=True)
add_experiment_to_aligned_data(experiments_to_process, for_heat_map_figure=True)
