import pickle
import pandas as pd
import numpy as np
from utils.post_processing_utils import remove_exps_after_manipulations, get_first_x_sessions
from set_global_params import experiment_record_path, processed_data_path, mice_average_traces
from utils.average_trace_processing_utils import get_all_mice_average_movement_times


site = 'tail'
mouse_ids = mice_average_traces[site]

# get the first three sessions
experiment_record = pd.read_csv(experiment_record_path, dtype='str')
experiments_to_process = get_first_x_sessions(experiment_record, mouse_ids, site).reset_index(drop=True)

tail_reaction_times = get_all_mice_average_movement_times(experiments_to_process)


site = 'Nacc'
mouse_ids = mice_average_traces[site]

# get the first three sessions
experiment_record = pd.read_csv(experiment_record_path, dtype='str')
experiments_to_process = get_first_x_sessions(experiment_record, mouse_ids, site).reset_index(drop=True)

nacc_reaction_times = get_all_mice_average_movement_times(experiments_to_process)
print(np.mean(np.concatenate([tail_reaction_times, nacc_reaction_times])))