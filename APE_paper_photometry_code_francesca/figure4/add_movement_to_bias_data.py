from scipy import stats
import os
import sys
sys.path.append('..')
import seaborn as sns
from utils.tracking_analysis.tracking_plotting import *
from set_global_params import raw_tracking_path, processed_data_path, bias_path, change_over_time_mice, plotting_colours
from utils.reaction_time_utils import get_bpod_trial_nums_per_session
from utils.post_processing_utils import get_all_experimental_records
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings, remove_manipulation_days
from utils.plotting_visuals import makes_plots_pretty
from utils.plotting import output_significance_stars_from_pval
import statsmodels.api as sm

def get_session_with_10000th_trial(mouse, experiments):
    dates = experiments[experiments['mouse_id']==mouse]['date'].unique()
    session_starts = get_bpod_trial_nums_per_session(mouse, dates)
    if session_starts[-1] >= 10000:
        last_session_idx = np.where(np.asarray(session_starts) >=10000)[0][0]
    else:
        last_session_idx = -1
    last_session_date = dates[last_session_idx]
    return(last_session_date)

def get_movement_data(mouse, all_experiments):
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
    all_session_data = all_session_data.reset_index(drop=True)
    all_session_data['abs fitted max cumsum ang vel'] = all_session_data['fitted max cumsum ang vel'].abs()
    return all_session_data

all_experiments = get_all_experimental_records()

for recording_site in ['tail']:

    mice = change_over_time_mice[recording_site]
    num_mice = len(mice)

    for m, mouse in enumerate(mice):
        movement_data = get_movement_data(mouse, all_experiments)
        movement_data = movement_data.set_index('actual trial numbers')
        bias_file = os.path.join(bias_path, 'pre_processing_bias_{}.pkl'.format(mouse))
        no_movement_data = pd.read_pickle(bias_file).drop_duplicates(subset='actual trial nums', keep='first').set_index('actual trial nums')


        # Get the common and all indices between the two dataframes
        common_indices = movement_data.index.intersection(no_movement_data.index)
        no_movement_data.loc[common_indices, 'turn angle'] = movement_data.loc[common_indices,'abs fitted max cumsum ang vel']
        no_movement_data.loc[common_indices, 'head x'] = movement_data.loc[common_indices,'head x']
        no_movement_data.loc[common_indices, 'head y'] = movement_data.loc[common_indices, 'head y']
        with_movement_data = no_movement_data.copy()
        bias_file = os.path.join(bias_path, 'pre_processing_movement_bias_{}.pkl'.format(mouse))
        with_movement_data.to_pickle(bias_file)