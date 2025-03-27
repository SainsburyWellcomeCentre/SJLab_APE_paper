import os
import numpy as np
import pickle
from utils.reaction_time_utils import get_bpod_trial_nums_per_session
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from utils.post_processing_utils import remove_unsuitable_recordings, remove_exps_after_manipulations_not_including_psychometric, get_all_experimental_records
from utils.plotting import multi_conditions_plot, output_significance_stars_from_pval
from set_global_params import processed_data_path, beginning_and_end_comparison_mice, reproduce_figures_path
from utils.stats import cohen_d_paired

def get_mean_contra_peak(session_record):
    """
    Finds the peak for contralateral trials for a given session
    Args:
        session_record (pd.dataframe): line of experimental record with session information

    Returns:
        mean_peak (float): mean peak response of session
    """
    mouse_id = session_record['mouse_id']
    date = session_record['date']
    saving_folder = os.path.join(processed_data_path, mouse_id)
    aligned_filename = mouse_id + '_' + date + '_' + 'aligned_traces.p'
    save_filename = os.path.join(saving_folder, aligned_filename)
    session_data = pickle.load(open(save_filename, "rb"))
    _trial_peaks = session_data.choice_data.contra_data.trial_peaks
    trial_peaks = [p if not isinstance(p, np.ndarray) else np.nan for p in _trial_peaks]
    mean_peak = np.nanmean(trial_peaks)
    return mean_peak


def get_session_with_10000th_trial(mouse, experiments):
    """
    Finds session with 10000th trial or the final session if mouse did fewer than 10000 trials.
    Args:
        mouse (str): mouse name
        experiments (pd.dataframe): experimental records

    Returns:
        last_session_date (str): date of last session in YYYYMMDD format
    """
    dates = experiments[experiments['mouse_id']==mouse]['date'].unique()
    session_starts = get_bpod_trial_nums_per_session(mouse, dates)
    if session_starts[-1] >= 10000:
        last_session_idx = np.where(np.asarray(session_starts) >=10000)[0][0]
    else:
        last_session_idx = -1
    last_session_date = dates[last_session_idx]
    return last_session_date


def get_first_and_10000th_peaks(mouse, records, site='tail'):
    """
    Gets average response in first session and session with 10000th trial. If mouse did not quite complete 10000 trials,
    the last training session is taken.
    Args:
        mouse (str): mouse name
        records (pd.dataframe): experimental records
        site (str): recording site (tail or Nacc)

    Returns:
        first_session_peak (float): mean response in first session
        last_session_peak (float): mean response in last session
    """
    experiments_to_process = records[(records['mouse_id'] == mouse) & (records['recording_site'] == site)]
    sorted_records = experiments_to_process.sort_values('date').reset_index(drop=True)
    first_recording = sorted_records.iloc[0]
    last_recording_date = get_session_with_10000th_trial(mouse, sorted_records)
    last_recording_ind = sorted_records[sorted_records['date'] == last_recording_date].index.values[0]
    last_recording = sorted_records.iloc[last_recording_ind]
    first_session_peak = get_mean_contra_peak(first_recording)
    last_session_peak = get_mean_contra_peak(last_recording)
    return first_session_peak, last_session_peak


def make_beginning_and_end_comparison_plot(ax, site='tail', colour='grey'):
    """
    Makes plot comparing size of responses at beginning and end of training and performs significance tests
    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): axes for plot
        site (str): recording site (tail or Nacc)
        colour (str): line colour

    Returns:

    """
    fn = f'start_and_end_{site}.csv'
    df_path = os.path.join(reproduce_figures_path, 'fig3', fn)
    mice = beginning_and_end_comparison_mice[site]

    if not os.path.exists(df_path):
        print('start and end file does not exist yet. creating dataframe...')
        records = get_all_experimental_records()
        first_peaks = []
        last_peaks = []
        for mouse in mice:
            all_experiments = remove_exps_after_manipulations_not_including_psychometric(records, [mouse])
            all_experiments = remove_unsuitable_recordings(all_experiments)
            first, last = get_first_and_10000th_peaks(mouse, all_experiments, site=site)
            first_peaks.append(first)
            last_peaks.append(last)

        data = pd.DataFrame({'mouse': mice, 'first session peak mean': first_peaks, 'last session peak mean': last_peaks})
        data.to_csv(df_path)
    else:
        print(f'Loading dataframe from {df_path}')
        data = pd.read_csv(df_path)

    first_data = data['first session peak mean']
    last_data = data['last session peak mean']
    stat, pval = stats.ttest_rel(first_data, last_data)
    cohend = cohen_d_paired(first_data, last_data)
    multi_conditions_plot(ax, data.set_index('mouse').T, colour=colour, mean_linewidth=0, show_err_bar=False)

    ax.set_xticks([0, 1], ['First session peak', 'Last session peak'], fontsize=6)
    ax.set_ylabel('Z-scored fluorescence', fontsize=6)

    # significance stars
    y = data.set_index('mouse').T.to_numpy().max() + .2
    h = .1
    ax.plot([0, 0, 1, 1], [y, y + h, y + h, y], c='k', lw=1)
    significance_stars = output_significance_stars_from_pval(pval)
    ax.text(.5, y + h, significance_stars, ha='center', fontsize=8)
    ax.set_ylim([0.3, 2.6])
    plt.tight_layout()
    print(pval)
    return data

