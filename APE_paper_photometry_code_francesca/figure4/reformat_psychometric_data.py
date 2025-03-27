import os
from set_global_params import bias_analysis_mice, processed_data_path
from utils.psychometric_post_processing_utils import get_all_psychometric_session_dlc

""" This analysis was originally written to look at movement parameters, so we call them here with get_movement=False
The resulting df has a column named 'APE peaks' this is just the dopamine response aligned to whichever behavioural event 
you have chosen. It is not exclusively APE, it depends on the brain region of course.
"""
site = 'Nacc'
mouse_ids = bias_analysis_mice[site]
save = False
num_sessions = 2
key = 'fitted max cumsum ang vel'
alignment = 'reward'
all_trial_data = get_all_psychometric_session_dlc(mouse_ids, site, save=False, load_saved=False, get_movement=False, align_to=alignment, num_sessions=num_sessions)
save_path = os.path.join(processed_data_path, 'psychometric_data')
all_trial_data.to_pickle(os.path.join(save_path, "all_trial_data_{}_contra_ipsi_last_trial_confidence_and_traces_no_tracking_{}_aligned_pk5.pkl".format(site, alignment)))

# first round of tail mice
site = 'tail'
mouse_ids = bias_analysis_mice[site][0]
num_sessions = 3
alignment = 'choice'
all_trial_data = get_all_psychometric_session_dlc(mouse_ids, site, save=False, load_saved=False, get_movement=False, align_to=alignment, num_sessions=num_sessions)
save_path = os.path.join(processed_data_path, 'psychometric_data')
all_trial_data.to_pickle(os.path.join(save_path, "all_trial_data_{}_contra_ipsi_last_trial_confidence_and_traces_no_tracking_{}_aligned_old_data_pk5.pkl".format(site, alignment)))

# this is for the second round of tail psychometric mice
mouse_ids = bias_analysis_mice[site][1]
num_sessions = 4
all_trial_data = get_all_psychometric_session_dlc(mouse_ids, site, save=False, load_saved=False, get_movement=False, align_to=alignment, num_sessions=num_sessions)
save_path = os.path.join(processed_data_path, 'psychometric_data')
all_trial_data.to_pickle(os.path.join(save_path, "all_trial_data_{}_contra_ipsi_last_trial_confidence_and_traces_no_tracking_{}_aligned_pk5.pkl".format(site, alignment)))
