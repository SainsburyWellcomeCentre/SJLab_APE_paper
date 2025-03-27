import pandas as pd
import numpy as np
import matplotlib.cm as cm


def align_PEs(PEs, trial_types, choices):
    trial_lengths = [len(trial) for trial in PEs]
    max_trial_length = max(trial_lengths)
    cue_aligned_PEs = np.full([len(PEs), max_trial_length], np.nan)
    reward_aligned_PEs = np.full([len(PEs), max_trial_length], np.nan)
    for trial in range(0, len(PEs)):
        trial_PE = PEs[trial]
        trial_length = len(trial_PE)
        cue_aligned_PEs[trial, :trial_length] = trial_PE
        reward_aligned_PEs[trial, -trial_length:] = trial_PE
    cue_aligned_df = pd.DataFrame(cue_aligned_PEs)
    cue_aligned_df['Type'] = trial_types
    cue_aligned_df['Choice'] = choices
    reward_aligned_df = pd.DataFrame(reward_aligned_PEs)
    reward_aligned_df['Type'] = trial_types
    reward_aligned_df['Choice'] = choices
    return cue_aligned_df, reward_aligned_df


def plot_heat_maps_over_trials(PEs, time_stamps, ax, title, window=10, delta_range=[-1, 1]):
    aligned_PEs = np.zeros([len(time_stamps), window])
    for trial_num, time_stamp in enumerate(time_stamps[1:-2]):
        aligned_PEs[trial_num] = PEs[time_stamp - int(window / 2): time_stamp + int(window / 2)]
    im = ax.imshow(aligned_PEs, extent=[-(window / 2), (window / 2), trial_num, 0], aspect='auto', vmin=delta_range[0],
                   vmax=delta_range[1])
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Trial number')
    ax.set_title(title)
    return


def plot_early_and_late(PEs, time_stamps, ax, max_val, window=10, chunk_prop=.33):
    colours = cm.viridis(np.linspace(0, 0.8, 3))
    aligned_PEs = np.zeros([len(time_stamps), window])
    for trial_num, time_stamp in enumerate(time_stamps[1:-2]):
        aligned_PEs[trial_num] = PEs[time_stamp - int(window / 2): time_stamp + int(window / 2)]
    early_aligned_PEs = aligned_PEs[:int(aligned_PEs.shape[0] * chunk_prop)].mean(axis=0)
    mid_aligned_PEs = aligned_PEs[int(aligned_PEs.shape[0] * chunk_prop):-int(aligned_PEs.shape[0] * chunk_prop)].mean(
        axis=0)
    late_aligned_PEs = aligned_PEs[-int(aligned_PEs.shape[0] * chunk_prop):].mean(axis=0)
    timesteps = np.arange(-(window / 2), (window / 2))
    ax.plot(timesteps, early_aligned_PEs, color=colours[0], label='early')
    ax.plot(timesteps, mid_aligned_PEs, color=colours[1], label='mid')
    ax.plot(timesteps, late_aligned_PEs, color=colours[2], label='late')
    ax.set_ylim([0, 1])
    #ax.set_xlabel('Time steps')
   # ax.set_ylabel('Response')
    return


def plot_average_response(PEs, time_stamps, ax, window=10, color='k', label='late'):
    aligned_PEs = np.zeros([len(time_stamps), window])
    for trial_num, time_stamp in enumerate(time_stamps[1:-2]):
        aligned_PEs[trial_num] = PEs[time_stamp - int(window / 2): time_stamp + int(window / 2)]
    average_PEs = aligned_PEs.mean(axis=0)
    timesteps = np.arange(-(window / 2), (window / 2))
    ax.plot(timesteps, average_PEs, color=color, label=label)
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Response')
    return

def plot_average_response_no_cutting(PEs, time_stamps, ax, window=10, color='k', label='late'):
    aligned_PEs = np.zeros([len(time_stamps), window])
    for trial_num, time_stamp in enumerate(time_stamps):
        aligned_PEs[trial_num] = PEs[time_stamp - int(window / 2): time_stamp + int(window / 2)]
    average_PEs = aligned_PEs.mean(axis=0)
    timesteps = np.arange(-(window / 2), (window / 2))
    ax.plot(timesteps, average_PEs, color=color, label=label)
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Response')
    return
def plot_change_over_time(PEs, stamps,ax):
    PEs_peak = np.zeros([len(stamps)-3])
    for trial_num, time_stamp in enumerate(stamps[1:-2]):
        PEs_peak[trial_num] = PEs[time_stamp]
    rolling_av_peaks = moving_average(PEs_peak, n=50)
    ax.plot(rolling_av_peaks, color='#3F888F')
    return


def get_one_agent_stamps(PEs, stamps):
    PEs_peak = np.zeros([len(stamps)-3])
    for trial_num, time_stamp in enumerate(stamps[1:-2]):
        PEs_peak[trial_num] = PEs[time_stamp]
    return PEs_peak


def get_all_agents_peaks(all_agents):
    max_num_stamps = max([len(i) for i in all_agents])
    all_peaks = np.empty([len(all_agents), max_num_stamps])
    all_peaks[:] = np.nan
    for agent_num, agent in enumerate(all_agents):
        all_peaks[agent_num, :len(agent)] = agent
    return all_peaks


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def label_axes_change_over_time(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Response size')
    return ax