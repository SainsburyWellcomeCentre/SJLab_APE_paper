import matplotlib.pyplot as plt
from tqdm import trange
from agent import Mouse
from classic_task import Box
from helper_functions.plotting_functions import *
from helper_functions.trial_matched_comparisons import get_early_mid_late_left_trials
import os
from directories import save_dir
from helper_functions.next_trial_effect_helpers import get_previous_choice_same_stim, run_multilag_regression_past_choice_on_signal
np.random.seed(0)

"""
Runs simulation for one agent, all models of dopamine. Used to produce plots seen in fig3 panel i and l
"""

if __name__ == '__main__':
    import pandas as pd

    n_trials = 2000
    x = np.linspace(0,50, n_trials * 10)
    cue_reaction_times = np.random.geometric(0.01, x.shape[0])
    movement_times = np.random.geometric(0.01, x.shape[0]) * 2 #for almost all simulations but not for regression
    e = Box(punish=True)
    inv_temp = .5
    a = Mouse(cue_reaction_times, movement_times, env=e, critic_learning_rate=0.05, actor_learning_rate=0.05,
              habitisation_rate=0.01, psi=0.2, inv_temp=inv_temp)


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
continuous_time_MSs = np.squeeze(np.concatenate(all_MSs))[:, 0]
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
left_choices_previous_choice = get_previous_choice_same_stim(all_state_changes, continuous_time_APEs,
                                                             continuous_time_PEs, choice='Left', stim='High',
                                                             num_lags=10)
tail_coefs, tail_pvals = run_multilag_regression_past_choice_on_signal(left_choices_previous_choice, y_var='current APE', num_lags=10)
vs_coefs, vs_pvals = run_multilag_regression_past_choice_on_signal(left_choices_previous_choice, y_var='current RPE', num_lags=10)
fig, ax = plt.subplots(1,1, figsize=(3, 3))

ax.plot(tail_coefs, color='#00343a')
ax.plot(vs_coefs, color='#E95F32')
ax.axhline(0, color='gray', linestyle='--')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "previous choice regression {} inv temp.pdf".format(inv_temp)))
fig, axs = plt.subplots(1, 2, figsize=(4, 3))
same_choice = left_choices_previous_choice[(left_choices_previous_choice['previous action'] == 'Left')
                                           & (left_choices_previous_choice['lag'] == 1)]
different_choice = left_choices_previous_choice[(left_choices_previous_choice['previous action'] == 'Right') &
                                                (left_choices_previous_choice['lag'] == 1)]
plot_average_response_no_cutting(models['APE'],  same_choice['action time stamp'], axs[0], color='#00343a', window=6)
plot_average_response_no_cutting(models['APE'],  different_choice['action time stamp'], axs[0], color= '#62b3c4', window=6)

plot_average_response_no_cutting(models['RPE'],  same_choice['cue time stamp'], axs[1], color='#E95F32', window=6)
plot_average_response_no_cutting(models['RPE'],  different_choice['cue time stamp'], axs[1], color='#f78b43', window=6)

for ax in axs.ravel():
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "previous choice traces {} inv temp.pdf".format(inv_temp)))
plt.tight_layout()
plt.show()