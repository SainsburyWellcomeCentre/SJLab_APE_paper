import numpy as np


def get_early_mid_late(trial_numbers, n_trials):
    early_trials = trial_numbers[trial_numbers < int(0.33*n_trials)]
    mid_trials = trial_numbers[np.logical_and(trial_numbers < int(0.33*2*n_trials), trial_numbers > int(0.33*n_trials))]
    late_trials = trial_numbers[np.logical_and(trial_numbers < int(n_trials), trial_numbers > int(0.33*2*n_trials))]
    return early_trials, mid_trials, late_trials


def get_early_mid_late_left_trials(trial_numbers, n_trials, all_state_changes):
    early_trials, mid_trials, late_trials = get_early_mid_late(trial_numbers, n_trials)
    early_lefts = all_state_changes['time stamp'][all_state_changes[(all_state_changes['action taken'] == 'Left') & (all_state_changes['trial number'].isin(early_trials))].index.values].values
    mid_lefts = all_state_changes['time stamp'][all_state_changes[(all_state_changes['action taken'] == 'Left') & (all_state_changes['trial number'].isin(mid_trials))].index.values].values
    late_lefts = all_state_changes['time stamp'][all_state_changes[(all_state_changes['action taken'] == 'Left') & (all_state_changes['trial number'].isin(late_trials))].index.values].values
    return early_lefts, mid_lefts, late_lefts