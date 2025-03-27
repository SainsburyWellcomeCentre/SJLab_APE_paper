import os
import pandas as pd
from utils.tracking_analysis.psychometric_get_movement_and_or_traces import get_all_mice_data
from utils.plotting import *
from set_global_params import experiment_record_path, post_processed_tracking_data_path


def get_first_x_sessions(sorted_experiment_record, x=3):
    i = []
    inds = []
    for mouse in np.unique(sorted_experiment_record['mouse_id']):
        i.append(sorted_experiment_record[sorted_experiment_record['mouse_id'] == mouse][0:x].index)
        inds += range(0, x)
    flattened_i = [val for sublist in i for val in sublist]
    exps = sorted_experiment_record.loc[flattened_i].reset_index(drop=True)
    exps['session number'] = inds
    return exps


def get_all_psychometric_session_dlc(mouse_ids, site, num_sessions=3, save=False, load_saved=True, get_movement=True, align_to='choice'):
    save_out_folder = post_processed_tracking_data_path
    mouse_names = '_'.join(mouse_ids)
    save_out_file = os.path.join(save_out_folder, 'contra_APE_tracking_psychometric_sessions_{}.pkl'.format(num_sessions, mouse_names))
    if os.path.isfile(save_out_file) and load_saved:
        data_to_save = pd.read_pickle(save_out_file)
    else:
        experiment_record = pd.read_csv(experiment_record_path, dtype='str')
        experiment_record['date'] = experiment_record['date'].astype(str)
        clean_experiments = experiment_record[(experiment_record['experiment_notes'] == 'psychometric')].reset_index(
            drop=True)
        all_experiments_to_process = clean_experiments[
            (clean_experiments['mouse_id'].isin(mouse_ids)) & (clean_experiments['recording_site'] == site)].reset_index(
            drop=True)
        all_experiments_to_process = all_experiments_to_process[all_experiments_to_process['include'] != 'no'].reset_index(drop=True)
        experiments_to_process = get_first_x_sessions(all_experiments_to_process, x=num_sessions).reset_index(
            drop=True)

        data_to_save, q_data, _, all_trial_data = get_all_mice_data(experiments_to_process, exp_type='_psychometric', shuffle=False, load_saved=False, get_movement=get_movement, align_to=align_to)
        norm_cumsum_data = norm_data_for_param(all_trial_data, experiment_record, site=site)
        if save:
            norm_cumsum_data.to_pickle(save_out_file)
    return norm_cumsum_data


def norm_data_for_param(all_trial_data, experiment_record, site='tail', key='fitted max cumsum ang vel', norm_APE_only=True):
    for m, mouse in enumerate(all_trial_data['mouse'].unique()):
        mouse_data = all_trial_data[all_trial_data['mouse'] == mouse]
        fiber_side = \
            experiment_record[(experiment_record['mouse_id'] == mouse) & (experiment_record['recording_site'] == site)][
                'fiber_side'].unique()
        for i, session in enumerate(mouse_data['session'].unique()):
            session_data = mouse_data[mouse_data['session'] == session]
            norm_data = session_data.copy(deep=False)
            norm_data['fiber side'] = fiber_side[0]
            norm_data['norm APE'] = session_data['APE peaks'] / np.nanmedian(session_data['APE peaks'])
            if not norm_APE_only:
                norm_data['norm ' + key] = np.abs(
                    session_data[key] / np.nanmedian(session_data[key]))

            # if fiber_side == 'left':
            #     for a, row in norm_data.iterrows():
            #         t = row['trial type']
            #         idx = trial_types[::-1] == t
            #         norm_data.loc[a, 'trial type'] = trial_types[idx][0]

            if (i == 0) & (m == 0):
                norm_all_data = norm_data
            else:
                norm_all_data = pd.concat([norm_all_data, norm_data])
    return norm_all_data