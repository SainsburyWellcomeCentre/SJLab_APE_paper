import pickle
import pandas as pd
import numpy as np
from utils.post_processing_utils import remove_exps_after_manipulations, get_first_x_sessions
from set_global_params import experiment_record_path, processed_data_path, mice_average_traces
from utils.average_trace_processing_utils import get_all_mice_average_data_high_low_cues
import os


site = 'Nacc'
mouse_ids = mice_average_traces[site]

# get the first three sessions
experiment_record = pd.read_csv(experiment_record_path, dtype='str')
experiments_to_process = get_first_x_sessions(experiment_record, mouse_ids, site).reset_index(drop=True)

#get trace data from these sessions
contra_high_cues, contra_low_cues, ipsi_high_cues, ipsi_low_cues,  time_stamps = get_all_mice_average_data_high_low_cues(experiments_to_process)

#save out per mouse average traces
dir = os.path.join(processed_data_path, 'for_figure')
file_name = 'group_data_avg_across_sessions_' + site +'_new_mice_added_high_low_cues_ipsi_contra.npz'
np.savez(os.path.join(dir, file_name), contra_high_cues=contra_high_cues, contra_low_cues=contra_low_cues, ipsi_high_cues=ipsi_high_cues, ipsi_low_cues=ipsi_low_cues, time_stamps=time_stamps)

