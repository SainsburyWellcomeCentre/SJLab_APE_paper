import os
import gc
from tqdm import trange
from agent import Mouse
from classic_task import Box
from helper_functions.plotting_functions import *
from helper_functions.trial_matched_comparisons import get_early_mid_late_left_trials
from directories import save_dir
np.random.seed(0)

"""
Saves out several agents (100 used in paper) to be used by plot_average_agent.py (this creates learning curves seen in 
figures 3 C G and ED fig8 CDE
"""

if __name__ == '__main__':
    import pandas as pd
    num_agents = 11
    RPE_cues = []
    Salience_cues = []
    Novelty_cues = []
    Movement_contra = []
    APE_contra = []

    for agent_num in range(num_agents):
        n_trials = 2000
        x = np.linspace(0,50, n_trials * 10)
        cue_reaction_times = np.random.geometric(0.01, x.shape[0])
        movement_times = np.random.geometric(0.01, x.shape[0]) * 2
        e = Box(punish=True)
        a = Mouse(cue_reaction_times, movement_times, env=e, critic_learning_rate=0.005, actor_learning_rate=0.005, habitisation_rate=0.01, psi=0.2)

        all_PEs = []
        all_APEs = []
        all_trial_types = []
        all_actions = []
        all_states = []
        all_state_changes = pd.DataFrame(columns=['state name', 'time stamp', 'dwell time', 'action taken'])

        all_rewards, all_MSs, all_Ns, all_Ss, all_Vs = [], [], [], [], []
        with trange(n_trials) as t:
            for i in t:
                _, PEs, trial_type, action, states, state_changes, apes, trial_r, m_signals, values, novelties, saliences, _ = a.one_trial(i)
                # pdb.set_trace()
                all_rewards.append(trial_r)
                mean_r = sum(all_rewards) / (i + 1.)
                t.set_description(f"avg. reward = {mean_r}")
                e.reset()
                all_states.append(states)
                all_PEs.append(PEs)
                all_trial_types.append(trial_type)
                #################################
                all_MSs.append(m_signals)
                all_Ns.append(novelties)
                all_Ss.append(saliences)
                all_Vs.append(values)
                ################################
                all_actions.append(action)
                all_state_changes = all_state_changes.append(state_changes, ignore_index=True)
                all_APEs.append(apes)

        continuous_time_PEs = np.concatenate(all_PEs).ravel()
        continuous_time_APEs = np.concatenate(all_APEs).ravel()
        continuous_time_MSs = np.squeeze(np.concatenate(all_MSs))[:,0] # np.concatenate(all_MSs).ravel()
        continuous_time_Ns = np.squeeze(np.concatenate(all_Ns))
        continuous_time_Ss = np.squeeze(np.concatenate(all_Ss))
        continuous_time_Vs = np.squeeze(np.concatenate(all_Vs))

        reward_times = np.where(np.asarray(a.reward_history) == 1)[0]

        left_inds = all_state_changes[all_state_changes['action taken'] == 'Left'].index
        cue_left = all_state_changes.shift(periods=1).iloc[left_inds]
        cue_left_times = cue_left['time stamp'].values

        right_inds = all_state_changes[all_state_changes['action taken'] == 'Right'].index
        cue_right = all_state_changes.shift(periods=1).iloc[left_inds]
        cue_right_times = cue_right['time stamp'].values


        low_tone_times = all_state_changes['time stamp'][
            all_state_changes[all_state_changes['state name'] == 'HighLeft'].index.values].values
        high_tone_times = all_state_changes['time stamp'][
            all_state_changes[all_state_changes['state name'] == 'LowRight'].index.values].values
        left_choices = all_state_changes['time stamp'][
            all_state_changes[all_state_changes['action taken'] == 'Left'].index.values].values
        right_choices = all_state_changes['time stamp'][
            all_state_changes[all_state_changes['action taken'] == 'Right'].index.values].values
        incorrect_left_trials = all_state_changes['trial number'][
            all_state_changes[all_state_changes['state name'] == 'LowLeft'].index.values].values
        correct_left_trials = all_state_changes['trial number'][
            all_state_changes[all_state_changes['state name'] == 'HighLeft'].index.values].values
        early_incorrect_times, mid_incorrect_times, late_incorrect_times = get_early_mid_late_left_trials(incorrect_left_trials, n_trials, all_state_changes)
        early_correct_times, mid_correct_times, late_correct_times = get_early_mid_late_left_trials(correct_left_trials, n_trials, all_state_changes)


        states = ['Left cues', 'Right cues', 'Reward', 'Contra', 'Ipsi']
        time_stamps = {'Left cues': cue_left_times, 'Right cues': cue_right_times, 'Reward': reward_times,
                       'Contra': left_choices, 'Ipsi': right_choices}
        models = {'APE': continuous_time_APEs, 'RPE': continuous_time_PEs, 'Novelty': np.sum(continuous_time_Ns, axis=1),
                  'Salience': np.sum(continuous_time_Ss, axis=1), 'Movement': continuous_time_MSs}

        RPE_cues.append(get_one_agent_stamps(models['RPE'], time_stamps['Left cues']))
        Salience_cues.append(get_one_agent_stamps(models['Salience'], time_stamps['Left cues']))
        Novelty_cues.append(get_one_agent_stamps(models['Novelty'], time_stamps['Left cues']))
        Movement_contra.append(get_one_agent_stamps(models['Movement'], time_stamps['Contra']))
        APE_contra.append(get_one_agent_stamps(models['APE'], time_stamps['Contra']))
    gc.collect()
    peaks = {}
    peaks['RPE cues'] = get_all_agents_peaks(RPE_cues)
    peaks['Salience cues'] = get_all_agents_peaks(Salience_cues)
    peaks['Novelty cues'] = get_all_agents_peaks(Novelty_cues)
    peaks['Movement contra'] = get_all_agents_peaks(Movement_contra)
    peaks['APE contra'] = get_all_agents_peaks(APE_contra)
    np.save(os.path.join(save_dir, '{}_agents_classic_exp.npy'.format(num_agents), peaks))



