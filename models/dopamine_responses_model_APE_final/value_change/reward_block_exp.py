import matplotlib.pyplot as plt
import matplotlib
from tqdm import trange
from agent import Mouse
from value_change.reward_block_env import RewardBlockBox
from helper_functions.plotting_functions import *
import os
from directories import save_dir
np.random.seed(0)


if __name__ == '__main__':
    import pandas as pd

    n_trials = 8000
    x = np.linspace(0, 50, n_trials  * 10)
    cue_reaction_times = np.random.geometric(0.01, x.shape[0]) #(np.exp(-x)*3+ np.random.rand(x.shape[0])) + 5
    movement_times = np.random.geometric(0.01, x.shape[0]) * 2
    e = RewardBlockBox(punish=True)
    a = Mouse(cue_reaction_times, movement_times, env=e, critic_learning_rate=0.005, actor_learning_rate=0.005, habitisation_rate=0.001, psi=0.2)

    all_PEs = []
    all_APEs = []
    all_APEs_ipsi = []
    all_trial_types = []
    all_actions = []
    all_states = []
    all_state_changes = pd.DataFrame(columns=['state name', 'time stamp', 'dwell time', 'action taken'])

    all_rewards, all_MSs, all_Ns, all_Ss, all_Vs, all_reward_types = [], [], [], [], [], []
    with trange(n_trials) as t:
        for i in t:
            _, PEs, trial_type, action, states, state_changes, apes_l, apes_r, trial_r, m_signals, values, novelties, saliences, reward_types = a.one_trial_reward_blocks(i)
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
            all_APEs.append(apes_l)
            all_APEs_ipsi.append(apes_r)
            all_reward_types.append(reward_types)


def find(c):
    for i, trial_type in enumerate(all_reward_types):
        try:
            j = trial_type.index(c)
        except ValueError:
            continue
        yield i
small_reward_trials = [match for match in find('small reward side')]
large_reward_trials = [match for match in find('large reward side')]
normal_reward_trials = [match for match in find('normal')]
continuous_time_PEs = np.concatenate(all_PEs).ravel()
continuous_time_APEs = np.concatenate(all_APEs).ravel()
continuous_time_APEs_ipsi = np.concatenate(all_APEs_ipsi).ravel()
continuous_time_MSs = np.squeeze(np.concatenate(all_MSs))[:,0] # np.concatenate(all_MSs).ravel()
continuous_time_Ns = np.squeeze(np.concatenate(all_Ns))
continuous_time_Ss = np.squeeze(np.concatenate(all_Ss))
continuous_time_Vs = np.squeeze(np.concatenate(all_Vs))
continuous_time_reward_types = np.concatenate(all_reward_types).ravel()
average_APE = np.sum(np.vstack([continuous_time_APEs_ipsi, continuous_time_APEs]), axis=0)

left_inds = all_state_changes[all_state_changes['action taken'] == 'Left'].index
BIG_side_inds = all_state_changes[(all_state_changes['action taken'] == 'Left') & (all_state_changes['trial number'].isin(large_reward_trials))].index[-50:]
normal_left_inds = all_state_changes[(all_state_changes['action taken'] == 'Left') & (all_state_changes['trial number'].isin(normal_reward_trials))].index[-50:]
cue_left_BIG = all_state_changes.shift(periods=1).iloc[BIG_side_inds]
cue_left_BIG_times = cue_left_BIG['time stamp'].values
movement_left_BIG_times = all_state_changes['time stamp'].iloc[BIG_side_inds]
movement_left_normal_times = all_state_changes['time stamp'].iloc[normal_left_inds]
cue_left_normal = all_state_changes.shift(periods=1).iloc[normal_left_inds]
cue_left_normal_times = cue_left_normal['time stamp'].values


right_inds = all_state_changes[all_state_changes['action taken'] == 'Right'].index
SMALL_side_inds = all_state_changes[(all_state_changes['action taken'] == 'Right') & (all_state_changes['trial number'].isin(small_reward_trials))].index[-50:]
normal_right_inds = all_state_changes[(all_state_changes['action taken'] == 'Right') & (all_state_changes['trial number'].isin(normal_reward_trials))].index[-50:]
cue_right_SMALL = all_state_changes.shift(periods=1).iloc[SMALL_side_inds]
cue_right_SMALL_times = cue_right_SMALL['time stamp'].values
cue_right_normal = all_state_changes.shift(periods=1).iloc[normal_right_inds]
cue_right_normal_times = cue_right_normal['time stamp'].values
movement_right_SMALL_times = all_state_changes['time stamp'].iloc[SMALL_side_inds]
movement_right_normal_times = all_state_changes['time stamp'].iloc[normal_right_inds]

states = ['Left cues', 'Right cues', 'Reward', 'Contra', 'Ipsi']
time_stamps = {'Normal left cue': cue_left_normal_times, 'Normal right cue': cue_right_normal_times, 'BIG left cue': cue_left_BIG_times, 'SMALL right cue': cue_right_SMALL_times,
               'Normal left movement': movement_left_normal_times, 'BIG left movement': movement_left_BIG_times,
               'SMALL right movement': movement_right_SMALL_times, 'normal right movement': movement_right_normal_times,
               'All cue times normal': np.sort(np.concatenate([cue_left_normal_times, cue_right_normal_times])),
               'All movement times normal': np.sort(np.concatenate([movement_left_normal_times, movement_left_normal_times]))}
models = {'APE': average_APE, 'RPE': continuous_time_PEs, 'Novelty': np.sum(continuous_time_Ns, axis=1),
          'Salience': np.sum(continuous_time_Ss, axis=1), 'Movement': continuous_time_MSs, 'APE ipsi': continuous_time_APEs_ipsi}

cue_stamps = ('SMALL right cue', 'All cue times normal', 'BIG left cue')
movement_stamps = ('SMALL right movement', 'All movement times normal', 'BIG left movement')


model_timestamp_type = {'APE': movement_stamps, 'RPE': cue_stamps, 'Novelty': cue_stamps,
          'Salience': cue_stamps, 'Movement': movement_stamps}

fig, axs = plt.subplots(5, 2, figsize=[4, 6])


colours = cm.inferno(np.linspace(0, 0.8, 3))
for ax_num, ax in enumerate(axs[:, 0]):
    model_type = list(models)[ax_num]
    plot_average_response(models[model_type], time_stamps['Normal left cue'], ax, color=colours[1])
    plot_average_response(models[model_type], time_stamps['BIG left cue'], ax, color=colours[2])
    plot_average_response(models[model_type], time_stamps['BIG left cue'], ax, color=colours[2])
    ax.set_ylabel(model_type)
    ax.set_ylim([0, 1.2])

for ax_num, ax in enumerate(axs[:, 1]):
    model_type = list(models)[ax_num]
    plot_average_response(models[model_type], time_stamps['Normal right cue'], ax, color=colours[1])
    plot_average_response(models[model_type], time_stamps['SMALL right cue'], ax, color=colours[0])
    ax.set_ylabel(model_type)
    ax.set_ylim([0, 1])


for ax in axs.ravel():
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
plt.tight_layout()

#on the same axis with the cue or movement alignment
fig, axs = plt.subplots(5, 1, figsize=[2, 6])


colours = cm.inferno(np.linspace(0, 0.8, 3))
for ax_num, ax in enumerate(axs):
    model_type = list(models)[ax_num]
    plot_average_response(models[model_type], time_stamps[model_timestamp_type[model_type][0]], ax, color=colours[0])
    plot_average_response(models[model_type], time_stamps[model_timestamp_type[model_type][1]], ax, color=colours[1])
    plot_average_response(models[model_type], time_stamps[model_timestamp_type[model_type][2]], ax, color=colours[2])
    ax.set_ylabel(model_type)
    ax.set_ylim([0, 1.2])


for ax in axs.ravel():
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
plt.tight_layout()

plt.savefig(os.path.join(save_dir, "reward_blocks_slow_habitisation_more_training_all_models.pdf"))

#Movement aligned APE
fig, axs = plt.subplots(1, 1, figsize=[2.5, 2])
font = {'size': 7}
matplotlib.rc('font', **font)
plot_average_response(models['APE'], time_stamps['Normal left movement'], axs, color=colours[1])
plot_average_response(models['APE'], time_stamps['SMALL right movement'], axs, color=colours[0])
plot_average_response(models['APE'], time_stamps['BIG left movement'], axs, color=colours[2])


axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.spines['bottom'].set_visible(False)
axs.spines['left'].set_visible(False)
axs.set_ylim([0, 0.6])
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "reward_blocks_slow_habitisation_more_training_APE.pdf"))
#Cue aligned RPE
fig, axs = plt.subplots(1, 1, figsize=[2.5, 2])
plot_average_response(models['RPE'], time_stamps['All cue times normal'], axs, color=colours[1])
plot_average_response(models['RPE'], time_stamps['SMALL right cue'], axs, color=colours[0])
plot_average_response(models['RPE'], time_stamps['BIG left cue'], axs, color=colours[2])
axs.set_ylim([0, 0.6])

axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)
axs.spines['bottom'].set_visible(False)
axs.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "reward_blocks_slow_habitisation_more_training_RPE.pdf"))
plt.show()