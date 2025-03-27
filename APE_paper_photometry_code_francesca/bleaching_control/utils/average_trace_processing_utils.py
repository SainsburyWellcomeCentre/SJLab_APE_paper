import pickle
from tqdm import tqdm
import numpy as np
from set_global_params import processed_data_path
import pandas as pd
import os

def get_all_mice_average_movement_times(experiments_to_process):
    reaction_times = []

    for mouse in tqdm(experiments_to_process['mouse_id'].unique(), desc='Mouse: '):
        df = experiments_to_process[experiments_to_process.mouse_id == mouse]
        data_dir = os.path.join(processed_data_path, 'for_figure', mouse)

        mouse_reaction_times = []

        for date in df['date']:
            filename = mouse + '_' + date + '_' + 'aligned_traces_for_fig.p'
            with open(os.path.join(data_dir, filename), 'rb') as f:
                session_data = pickle.load(f)
            ipsi_choice = session_data.choice_data.ipsi_data.reaction_times
            contra_choice = session_data.choice_data.contra_data.reaction_times
            all_reaction_times = np.concatenate([ipsi_choice, contra_choice])
            mouse_reaction_times.append(np.mean(all_reaction_times))

        reaction_times.append(np.mean(mouse_reaction_times))
    reaction_times = np.array(reaction_times)
    return reaction_times



def get_all_mice_average_reaction_times(experiments_to_process):
    reaction_times = []

    for mouse in tqdm(experiments_to_process['mouse_id'].unique(), desc='Mouse: '):
        df = experiments_to_process[experiments_to_process.mouse_id == mouse]
        data_dir = os.path.join(processed_data_path, 'for_figure', mouse)

        mouse_reaction_times = []

        for date in df['date']:
            filename = mouse + '_' + date + '_' + 'aligned_traces_for_fig.p'
            with open(os.path.join(data_dir, filename), 'rb') as f:
                session_data = pickle.load(f)
            ipsi_trial_nums = session_data.choice_data.ipsi_data.trial_nums
            contra_trial_nums = session_data.choice_data.contra_data.trial_nums
            all_trial_nums = np.concatenate([ipsi_trial_nums, contra_trial_nums])
            trial_data_folder = os.path.join(processed_data_path, mouse)
            restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
            trial_data = pd.read_pickle(os.path.join(trial_data_folder, restructured_data_filename))
            trials_of_interest = trial_data[trial_data['Trial num'].isin(all_trial_nums)]
            states_of_interest = trials_of_interest[trials_of_interest['State type'] == 4]
            session_reaction_times = states_of_interest['Time end'] - states_of_interest['Time start']
            mouse_reaction_times.append(np.mean(session_reaction_times.values +  0.1))

        reaction_times.append(np.mean(mouse_reaction_times))
    reaction_times = np.array(reaction_times)
    return reaction_times


def get_all_mice_average_data(experiments_to_process, time_range=(-1.5, 1.5)):
    """
    This version takes the average across session for each mouse and then stacks the mouse averages.
    Args:
        experiments_to_process (pd.dataframe): experimental records for all the mice you want to average the traces for
        time_range (tuple): time window seconds before and after event to get traces for

    Returns:
        ipsi_choices (np.array): per mouse mean traces aligned to ipsi choices centre port out
        contra_choices (np.array): per mouse mean traces aligned to contra choices centre port out
        rewards (np.array): per mouse mean traces aligned to reward delivery
        no_rewards (np.array): per mouse mean traces aligned to time of choice when there is no reward
        cues (np.array): per mouse mean traces aligned to cues (ipsi and contra)
        time_stamps (np.array): time stamps that correspond to traces (seconds)
    """

    ipsi_choices = []
    contra_choices = []
    rewards = []
    no_rewards = []
    cues = []

    for mouse in tqdm(experiments_to_process['mouse_id'].unique(), desc='Mouse: '):
        df = experiments_to_process[experiments_to_process.mouse_id == mouse]
        data_dir = os.path.join(processed_data_path, 'for_figure', mouse)

        mouse_ipsi_choices = []
        mouse_contra_choices = []
        mouse_reward = []
        mouse_no_reward = []
        mouse_cues = []

        for date in df['date']:
            filename = mouse + '_' + date + '_' + 'aligned_traces_for_fig.p'
            with open(os.path.join(data_dir, filename), 'rb') as f:
                session_data = pickle.load(f)
            time_mask = (session_data.choice_data.contra_data.time_points >= time_range[0]) & (session_data.choice_data.contra_data.time_points <= time_range[-1])
            ipsi_choice = session_data.choice_data.ipsi_data.mean_trace[time_mask]
            contra_choice = session_data.choice_data.contra_data.mean_trace[time_mask]
            reward = session_data.outcome_data.reward_data.mean_trace[time_mask]
            no_reward = session_data.outcome_data.no_reward_data.mean_trace[time_mask]
            # for cues you need to take the mean after combining ipsi and contra
            contra_cues = session_data.cue_data.contra_data.sorted_traces[:, time_mask]
            ipsi_cues = session_data.cue_data.ipsi_data.sorted_traces[:, time_mask]
            all_cues = np.concatenate([ipsi_cues, contra_cues])
            mean_trace_cues = np.mean(all_cues, axis=0)
            time_stamps = session_data.choice_data.contra_data.time_points[time_mask]

            mouse_ipsi_choices.append(ipsi_choice)
            mouse_contra_choices.append(contra_choice)
            mouse_reward.append(reward)
            mouse_no_reward.append(no_reward)
            mouse_cues.append(mean_trace_cues)

        ipsi_choices.append(np.mean(mouse_ipsi_choices, axis=0))
        contra_choices.append(np.mean(mouse_contra_choices, axis=0))
        rewards.append(np.mean(mouse_reward, axis=0))
        no_rewards.append(np.mean(mouse_no_reward, axis=0))
        cues.append(np.mean(mouse_cues, axis=0))
    ipsi_choices = np.array(ipsi_choices)
    contra_choices = np.array(contra_choices)
    rewards = np.array(rewards)
    no_rewards = np.array(no_rewards)
    cues = np.array(cues)
    return ipsi_choices, contra_choices, rewards, no_rewards, cues, time_stamps


def get_all_mice_average_data_only_contra_cues(experiments_to_process, time_range=(-1.5, 1.5)):
    """
    This version takes the average across session for each mouse and then stacks the mouse averages.
    Args:
        experiments_to_process (pd.dataframe): experimental records for all the mice you want to average the traces for
        time_range (tuple): time window seconds before and after event to get traces for

    Returns:
        ipsi_choices (np.array): per mouse mean traces aligned to ipsi choices centre port out
        contra_choices (np.array): per mouse mean traces aligned to contra choices centre port out
        rewards (np.array): per mouse mean traces aligned to reward delivery
        no_rewards (np.array): per mouse mean traces aligned to time of choice when there is no reward
        contra_cues (np.array): per mouse mean traces aligned to contra cues
        time_stamps (np.array): time stamps that correspond to traces (seconds)
    """

    ipsi_choices = []
    contra_choices = []
    rewards = []
    no_rewards = []
    cues = []

    for mouse in tqdm(experiments_to_process['mouse_id'].unique(), desc='Mouse: '):
        df = experiments_to_process[experiments_to_process.mouse_id == mouse]
        data_dir = processed_data_path + 'for_figure\\' + mouse + '\\'

        mouse_ipsi_choices = []
        mouse_contra_choices = []
        mouse_reward = []
        mouse_no_reward = []
        mouse_cues = []

        for date in df['date']:
            filename = mouse + '_' + date + '_' + 'aligned_traces_for_fig.p'
            with open(data_dir + filename, 'rb') as f:
                session_data = pickle.load(f)
                time_mask = (session_data.choice_data.contra_data.time_points >= time_range[0]) & (session_data.choice_data.contra_data.time_points <= time_range[-1])
                ipsi_choice = session_data.choice_data.ipsi_data.mean_trace[time_mask]
                contra_choice = session_data.choice_data.contra_data.mean_trace[time_mask]
                reward = session_data.outcome_data.reward_data.mean_trace[time_mask]
                no_reward = session_data.outcome_data.no_reward_data.mean_trace[time_mask]
                # for cues you need to take the mean after combining ipsi and contra
                contra_cues = session_data.cue_data.contra_data.sorted_traces[:, time_mask]
                time_stamps = session_data.choice_data.contra_data.time_points[time_mask]
                mean_trace_cues = np.mean(contra_cues, axis=0)
                mouse_contra_choices.append(contra_choice)
                mouse_reward.append(reward)
                mouse_no_reward.append(no_reward)
                mouse_cues.append(mean_trace_cues)

        ipsi_choices.append(np.mean(mouse_ipsi_choices, axis=0))
        contra_choices.append(np.mean(mouse_contra_choices, axis=0))
        rewards.append(np.mean(mouse_reward, axis=0))
        no_rewards.append(np.mean(mouse_no_reward, axis=0))
        cues.append(np.mean(mouse_cues, axis=0))
    ipsi_choices = np.array(ipsi_choices)
    contra_choices = np.array(contra_choices)
    rewards = np.array(rewards)
    no_rewards = np.array(no_rewards)
    cues = np.array(cues)
    return ipsi_choices, contra_choices, rewards, no_rewards, cues, time_stamps


def get_all_mice_average_data_high_low_cues(experiments_to_process, time_range=(-1.5, 1.5)):
    """
    This version takes the average across session for each mouse and then stacks the mouse averages (only for high and low cue aligned).
    Args:
        experiments_to_process (pd.dataframe): experimental records for all the mice you want to average the traces for
        time_range (tuple): time window seconds before and after event to get traces for

    Returns:
        high_cues (np.array): per mouse mean traces aligned to high cues
        low_cues (np.array): per mouse mean traces aligned to low cues
        time_stamps (np.array): time stamps that correspond to traces (seconds)
    """

    contra_high_cues = []
    contra_low_cues = []
    ipsi_high_cues = []
    ipsi_low_cues = []

    for mouse in tqdm(experiments_to_process['mouse_id'].unique(), desc='Mouse: '):
        df = experiments_to_process[experiments_to_process.mouse_id == mouse]
        data_dir = os.path.join(processed_data_path, 'for_figure', mouse)
        trial_data_folder = os.path.join(processed_data_path, mouse)
        fiber_side = df['fiber_side'].values[0]
        if fiber_side == 'left':
            contra_cue = 'low'
        elif fiber_side == 'right':
            contra_cue = 'high'
        mouse_ipsi_high_cues = []
        mouse_ipsi_low_cues = []
        mouse_contra_high_cues = []
        mouse_contra_low_cues = []

        for date in df['date']:
            restructured_data_filename = mouse + '_' + date + '_' + 'restructured_data.pkl'
            trial_data = pd.read_pickle(os.path.join(trial_data_folder, restructured_data_filename))
            filename = mouse + '_' + date + '_' + 'aligned_traces_for_fig.p'
            with open(os.path.join(data_dir, filename), 'rb') as f:
                session_data = pickle.load(f)
                time_mask = (session_data.cue_data.high_cue_data.time_points >= time_range[0]) & (session_data.cue_data.high_cue_data.time_points <= time_range[-1])
                contra_trial_nums = session_data.choice_data.contra_data.trial_nums
                ipsi_trial_nums = session_data.choice_data.ipsi_data.trial_nums
                high_cue_trial_nums = trial_data[trial_data['Trial type'] == 1]['Trial num'].unique()
                low_cue_trial_nums = trial_data[trial_data['Trial type'] == 7]['Trial num'].unique()
            if contra_cue == 'high':
                _, contra_high_cue_inds, _ = np.intersect1d(contra_trial_nums, high_cue_trial_nums,
                                                            return_indices=True)
                session_contra_high_cues = session_data.cue_data.contra_data.sorted_traces[contra_high_cue_inds,
                                           :][:, time_mask]
                mouse_contra_high_cues.append(np.mean(session_contra_high_cues, axis=0))
                _, ipsi_low_cue_inds, _ = np.intersect1d(ipsi_trial_nums, low_cue_trial_nums, return_indices=True)
                session_ipsi_low_cues = session_data.cue_data.ipsi_data.sorted_traces[
                                        ipsi_low_cue_inds, :][:, time_mask]
                mouse_ipsi_low_cues.append(np.mean(session_ipsi_low_cues, axis=0))

            elif contra_cue == 'low':
                _, contra_low_cue_inds, _ = np.intersect1d(contra_trial_nums, low_cue_trial_nums,
                                                           return_indices=True)
                session_contra_low_cues = session_data.cue_data.contra_data.sorted_traces[contra_low_cue_inds, :][
                                          :, time_mask]
                mouse_contra_low_cues.append(np.mean(session_contra_low_cues, axis=0))

                _, ipsi_high_cue_inds, _ = np.intersect1d(ipsi_trial_nums, high_cue_trial_nums, return_indices=True)
                session_ipsi_high_cues = session_data.cue_data.ipsi_data.sorted_traces[
                                         ipsi_high_cue_inds, :][:, time_mask]
                mouse_ipsi_high_cues.append(np.mean(session_ipsi_high_cues, axis=0))

                time_stamps = session_data.cue_data.high_cue_data.time_points[time_mask]

        if contra_cue == 'high':
            contra_high_cues.append(np.mean(mouse_contra_high_cues, axis=0))
            ipsi_low_cues.append(np.mean(mouse_ipsi_low_cues, axis=0))
        elif contra_cue == 'low':
            contra_low_cues.append(np.mean(mouse_contra_low_cues, axis=0))
            ipsi_high_cues.append(np.mean(mouse_ipsi_high_cues, axis=0))


    contra_high_cues = np.array(contra_high_cues)
    contra_low_cues = np.array(contra_low_cues)
    ipsi_high_cues = np.array(ipsi_high_cues)
    ipsi_low_cues = np.array(ipsi_low_cues)
    return contra_high_cues, contra_low_cues, ipsi_high_cues, ipsi_low_cues, time_stamps

