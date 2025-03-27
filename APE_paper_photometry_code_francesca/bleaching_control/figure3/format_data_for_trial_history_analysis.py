from scipy import stats
import os
import sys
sys.path.append('../..')
from scipy.signal import decimate
from utils.tracking_analysis.tracking_plotting import *
from set_global_params import processed_data_path, bias_path, change_over_time_mice
from utils.reaction_time_utils import get_bpod_trial_nums_per_session
from utils.post_processing_utils import get_all_experimental_records
from utils.post_processing_utils import remove_exps_after_manipulations, remove_unsuitable_recordings, remove_manipulation_days
from utils.tracking_analysis.extract_movement_for_all_sessions_utils import get_actual_trial_numbers, get_photometry_data
from utils.individual_trial_analysis_utils import get_peak_each_trial_no_nans

def get_session_with_10000th_trial(mouse, experiments):
    dates = experiments[experiments['mouse_id']==mouse]['date'].unique()
    session_starts = get_bpod_trial_nums_per_session(mouse, dates)
    if session_starts[-1] >= 10000:
        last_session_idx = np.where(np.asarray(session_starts) >=10000)[0][0]
    else:
        last_session_idx = -1
    last_session_date = dates[last_session_idx]
    return(last_session_date)


def downsample_and_clip_taces_for_plotting(traces, time_points, t_min=0, t_max=1):
    time_points = decimate(time_points, 10)
    indices = np.where(np.logical_and(np.greater_equal(time_points, t_min), np.less_equal(time_points, t_max)))[0]
    clipped_time_points = time_points[indices]
    clipped_traces = np.zeros([traces.shape[0], len(indices)])
    for t, trial in enumerate(traces):
        trial_trace = traces[t, :]
        downsampled_trace = decimate(trial_trace, 10)
        clipped_traces[t, :] = downsampled_trace[indices]
    return clipped_traces, clipped_time_points



recording_site = 'Nacc'
align_to_reward = False
for mouse in change_over_time_mice[recording_site]:
    print(mouse)
    all_experiments = get_all_experimental_records()
    all_experiments = remove_exps_after_manipulations(all_experiments, [mouse])
    all_experiments = remove_manipulation_days(all_experiments)
    all_experiments = remove_unsuitable_recordings(all_experiments)
    experiments_to_process = all_experiments[
        (all_experiments['mouse_id'] == mouse) & (all_experiments['recording_site'] == recording_site)]
    last_session = get_session_with_10000th_trial(mouse, experiments_to_process)

    dates = experiments_to_process['date'].values
    last_session_ind = int(np.where(dates == last_session)[0])
    for i, date in enumerate(dates[0:last_session_ind + 1]):
            photometry_data = get_photometry_data(mouse, date)
            saving_folder = os.path.join(processed_data_path, mouse)
            restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
            trial_data = pd.read_pickle(os.path.join(saving_folder, restructured_data_filename))

            # Initialize empty lists to collect data
            trial_nums = []
            reaction_times = []
            peaks = []
            traces = []
            trial_types = []
            time_points = []

            if recording_site == 'tail':
                aligned_data = photometry_data.choice_data
            elif align_to_reward:
                aligned_data = photometry_data.reward_data
            else:
                aligned_data = photometry_data.cue_data
            # Single loop to iterate over trial types and collect data

            for trial_type in ['contra_data', 'ipsi_data']:
                trial_type_data = getattr(aligned_data, trial_type)

                # Extend lists with data from the current trial type
                trial_nums.extend(trial_type_data.trial_nums)
                reaction_times.extend(trial_type_data.reaction_times)
                if trial_type_data.trial_peaks is not None:
                    peaks.extend(trial_type_data.trial_peaks)
                    trial_type_traces, trial_type_time_points = downsample_and_clip_taces_for_plotting(trial_type_data.sorted_traces, trial_type_data.time_points, t_min=-0.5, t_max=1.5)
                    traces.extend(trial_type_traces)
                    time_points.extend(np.ones(trial_type_traces.shape) * trial_type_time_points)
                else:
                    trial_peaks = get_peak_each_trial_no_nans(trial_type_data.sorted_traces, trial_type_data.time_points, np.ones(trial_type_data.event_times.shape))
                    peaks.extend(trial_peaks)
                # Add the trial type for each entry
                trial_types.extend([trial_type] * len(trial_type_data.trial_nums))

            # Create the DataFrame
            df = pd.DataFrame({
                'trial_nums': trial_nums,
                'reaction_times': reaction_times,
                'APE_peaks': peaks,
                'trial_type': trial_types,
                'traces': traces,
                'time_points': time_points
            })


            df_sorted = df.sort_values(by='trial_nums')

            # Assuming df_sorted is the sorted DataFrame with 'trial_nums', 'reaction_times', 'peaks', 'trial_type'
            # Assuming trial_data is the DataFrame with many columns, one of which is 'Trial num'

            # First, rename the 'Trial num' column in trial_data to match 'trial_nums' in df_sorted for merging
            trial_data = trial_data.rename(columns={'Trial num': 'trial_nums'})

            # Merge the two DataFrames on 'trial_nums', using a left join to keep all rows in trial_data
            merged_df = pd.merge(trial_data, df_sorted, on='trial_nums', how='left')
            merged_df['actual trial nums'] = get_actual_trial_numbers(merged_df['trial_nums'], date, mouse, recording_site=recording_site)
            if i == 0:
                all_session_data = merged_df
            else:
                all_session_data = pd.concat([all_session_data, merged_df])

    all_session_data = all_session_data.reset_index(drop=True)
    if not os.path.exists(bias_path):
        os.makedirs(bias_path)
    if align_to_reward:
        bias_file = os.path.join(bias_path, 'pre_processing_bias_{}_reward.pkl'.format(mouse))
    else:
        bias_file = os.path.join(bias_path, 'pre_processing_bias_{}.pkl'.format(mouse))
    all_session_data.to_pickle(bias_file)


