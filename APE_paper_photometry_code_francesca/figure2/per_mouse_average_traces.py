import pickle
import os
import pandas as pd
import numpy as np
from utils.post_processing_utils import remove_exps_after_manipulations, get_first_x_sessions
from set_global_params import experiment_record_path, processed_data_path, mice_average_traces
from utils.average_trace_processing_utils import get_all_mice_average_data


site = 'tail_ant' #'tail_ant'  # tail # nacc
mouse_ids = mice_average_traces[site]

# get the first three sessions
experiment_record = pd.read_csv(experiment_record_path, dtype='str')
experiments_to_process = get_first_x_sessions(experiment_record, mouse_ids, site).reset_index(drop=True)

#get trace data from these sessions
ipsi_choice, contra_choice, reward, no_reward, cue, time_stamps = get_all_mice_average_data(experiments_to_process)

#save out per mouse average traces
dir = os.path.join(processed_data_path, 'for_figure')
file_name = 'group_data_avg_across_sessions_' + site +'_new_mice_added_with_cues.npz'
np.savez(dir + file_name, ipsi_choice=ipsi_choice, contra_choice=contra_choice, reward=reward, no_reward=no_reward, time_stamps=time_stamps, cue=cue)

