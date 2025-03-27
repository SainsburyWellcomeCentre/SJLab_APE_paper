import pandas as pd
import numpy as np
import os
from set_global_params import processed_data_path, silence_mice, behavioural_data_path
import data_preprocessing.bpod_data_processing as bpod
from utils.post_processing_utils import get_all_experimental_records

"""
Finds the number of times mice have poked into the side ports during the punishment time out in regular training 
"""

mouse_names_for_df = []
labels_for_df = []
values_for_df = []

for mouse in silence_mice:
    num_centre_pokes_in_punishment = 0
    num_contra_pokes_after_silence = 0
    num_ipsi_pokes_after_silence = 0
    contra_choice_count = 0
    ipsi_choice_count = 0

    recording_site = 'tail'
    protocol = 'Two_Alternative_Choice'

    # get fiber side
    all_experiments = get_all_experimental_records()
    experiments_to_process = all_experiments[
        (all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == recording_site)]

    fiber_side = experiments_to_process['fiber_side'].unique()[0]

    # get dates of training (not just recording up to silence day)
    Bpod_data_path = behavioural_data_path + mouse + '\\' + protocol + '\\Session Data\\'
    files_in_bpod_path = os.listdir(Bpod_data_path)
    behaviour_files = sorted([s for s in files_in_bpod_path if '.mat' in s])

    for i, file in enumerate(behaviour_files[0:]):
        fiber_options = ['left', 'right']
        if fiber_side == 'left':
            contra_port_in = 'Port3In'
            ipsi_port_in = 'Port1In'
        else:
            contra_port_in = 'Port1In'
            ipsi_port_in = 'Port3In'
        contra_fiber_ind = np.where(np.array(fiber_options) == fiber_side)[0][0] + 1
        ipsi_fiber_ind = np.where(np.array(fiber_options) != fiber_side)[0][0] + 1

        main_session_file = Bpod_data_path + file
        loaded_bpod_file, trial_raw_events = bpod.load_bpod_file(main_session_file)

        original_state_data_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateData']
        original_state_timestamps_all_trials = loaded_bpod_file['SessionData']['RawData']['OriginalStateTimestamps']
        original_raw_events = loaded_bpod_file['SessionData']['RawEvents']['Trial']
        if np.all(loaded_bpod_file['SessionData']['Stimulus'][0] ==0): #it's habituation
            session_silence_contra_choices = (loaded_bpod_file['SessionData']['ChosenSide'] == ipsi_fiber_ind).sum()
            session_silence_ipsi_choices = (loaded_bpod_file['SessionData']['ChosenSide'] == contra_fiber_ind).sum()
            num_ipsi_pokes_after_silence += session_silence_ipsi_choices
        else:
            for trial, state_timestamps in enumerate(original_state_timestamps_all_trials):

                if (loaded_bpod_file['SessionData']['ChosenSide'][trial] == contra_fiber_ind) & (loaded_bpod_file['SessionData']['TrialSide'][trial] == contra_fiber_ind):
                    contra_choice_count += 1
                elif (loaded_bpod_file['SessionData']['ChosenSide'][trial] == ipsi_fiber_ind) & (loaded_bpod_file['SessionData']['TrialSide'][trial] == contra_fiber_ind):
                    ipsi_choice_count += 1
                else:
                    pass

                if 'Punish' in original_raw_events[trial]['States']:
                    punish_start = original_raw_events[trial]['States']['Punish'][0]

                    if 'Port2In' in original_raw_events[trial]['Events']:
                        centre_pokes = np.asarray(original_raw_events[trial]['Events']['Port2In'])

                        if contra_port_in in original_raw_events[trial]['Events']:
                            contra_pokes = np.asarray(original_raw_events[trial]['Events'][contra_port_in])
                        else:
                            contra_pokes = np.array([])  # Empty array if key is missing

                        if ipsi_port_in in original_raw_events[trial]['Events']:
                            ipsi_pokes = np.asarray(original_raw_events[trial]['Events'][ipsi_port_in])
                        else:
                            ipsi_pokes = np.array([])  # Empty array if key is missing

                        centre_pokes_in_punish = centre_pokes[centre_pokes > punish_start]
                        num_centre_pokes = centre_pokes_in_punish.shape[0]
                        num_centre_pokes_in_punishment += num_centre_pokes

                    if num_centre_pokes >= 1:
                        for i, silence_poke in enumerate(centre_pokes_in_punish):
                            if (num_centre_pokes > 1) & (i < (num_centre_pokes - 1)):
                                next_centre_poke = centre_pokes_in_punish[i + 1]
                                contra_pokes_after_centre = contra_pokes[(contra_pokes > silence_poke) & (contra_pokes < next_centre_poke)]
                                ipsi_pokes_after_centre = ipsi_pokes[(ipsi_pokes > silence_poke) & (ipsi_pokes < next_centre_poke)]
                            else:
                                contra_pokes_after_centre = contra_pokes[(contra_pokes > silence_poke)]
                                ipsi_pokes_after_centre = ipsi_pokes[(ipsi_pokes > silence_poke)]
                            num_contra_pokes_after_silence += contra_pokes_after_centre.shape[0]
                            num_ipsi_pokes_after_silence += ipsi_pokes_after_centre.shape[0]

    tone_pokes = contra_choice_count + ipsi_choice_count
    print('tone ipsi contra: ', ipsi_choice_count/contra_choice_count)
    silence_pokes = num_contra_pokes_after_silence + num_ipsi_pokes_after_silence
    print('silence ipsi contra: ', num_ipsi_pokes_after_silence/num_contra_pokes_after_silence)
    print((silence_pokes/ tone_pokes))
    mouse_names_for_df.extend([mouse] * 2)
    labels_for_df.extend(['tones', 'silence'])
    values_for_df.extend([tone_pokes, silence_pokes])
df_to_save = pd.DataFrame({'mouse': mouse_names_for_df, 'stimulus': labels_for_df, 'count': values_for_df})
filename = os.path.join(processed_data_path, 'num_pokes_in_punishment.pkl')
df_to_save.to_pickle(filename)

