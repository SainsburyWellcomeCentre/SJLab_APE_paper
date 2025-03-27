import matplotlib.pyplot as plt
from tqdm import trange
from agent import Mouse
from omissions_large_rewards.omissions_large_rewards_env import OmissionsLargeRewardBox
from helper_functions.plotting_functions import *
import os
from directories import save_dir
np.random.seed(0)



if __name__ == '__main__':
    import pandas as pd

    n_trials = 2300
    x = np.linspace(0, 50, n_trials  * 10)
    cue_reaction_times = np.random.geometric(0.01, x.shape[0]) #(np.exp(-x)*3+ np.random.rand(x.shape[0])) + 5
    movement_times = np.random.geometric(0.01, x.shape[0]) * 2
    e = OmissionsLargeRewardBox(punish=True)
    a = Mouse(cue_reaction_times, movement_times, env=e, critic_learning_rate=0.005, actor_learning_rate=0.005, habitisation_rate=0.01, psi=0.2)

    all_PEs = []
    all_APEs = []
    all_trial_types = []
    all_actions = []
    all_states = []
    all_state_changes = pd.DataFrame(columns=['state name', 'time stamp', 'dwell time', 'action taken'])

    all_rewards, all_MSs, all_Ns, all_Ss, all_Vs, all_reward_types = [], [], [], [], [], []
    with trange(n_trials) as t:
        for i in t:
            _, PEs, trial_type, action, states, state_changes, apes, trial_r, m_signals, values, novelties, saliences, reward_types = a.one_trial(i)
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
            all_reward_types.append(reward_types)

continuous_time_PEs = np.concatenate(all_PEs).ravel()
continuous_time_APEs = np.concatenate(all_APEs).ravel()
continuous_time_MSs = np.squeeze(np.concatenate(all_MSs))[:,0] # np.concatenate(all_MSs).ravel()
continuous_time_Ns = np.squeeze(np.concatenate(all_Ns))
continuous_time_Ss = np.squeeze(np.concatenate(all_Ss))
continuous_time_Vs = np.squeeze(np.concatenate(all_Vs))
continuous_time_reward_types = np.concatenate(all_reward_types).ravel()

reward_times = np.where(np.asarray(a.reward_history) == 1)[0]
omission_times = np.where(np.asarray(continuous_time_reward_types) == 'omission')[0]
normal_reward_times = np.where(np.asarray(continuous_time_reward_types) == 'normal')[0]
large_reward_times = np.where(np.asarray(continuous_time_reward_types) == 'large reward')[0]

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



states = ['Left cues', 'Right cues', 'Reward', 'Contra', 'Ipsi']
time_stamps = {'Omissions': omission_times, 'Normal rewards': normal_reward_times, 'Large rewards': large_reward_times}
models = {'APE': continuous_time_APEs, 'RPE': continuous_time_PEs, 'Novelty': np.sum(continuous_time_Ns, axis=1),
          'Salience': np.sum(continuous_time_Ss, axis=1), 'Movement': continuous_time_MSs}
fig, axs = plt.subplots(5, 1, figsize=[2, 6])


colours = cm.inferno(np.linspace(0, 0.8, 3))
for ax_num, ax in enumerate(axs):
    model_type = list(models)[ax_num]
    plot_average_response(models[model_type], time_stamps['Omissions'], ax, color=colours[2])
    plot_average_response(models[model_type], time_stamps['Normal rewards'], ax, color=colours[1])
    plot_average_response(models[model_type], time_stamps['Large rewards'], ax, color=colours[0])
    ax.set_ylabel(model_type)
    ax.set_ylim([-0.5, 2])


for ax in axs.ravel():
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


plt.tight_layout()
plt.savefig(os.path.join(save_dir, "omissions_large_rewards.pdf"))
plt.show()