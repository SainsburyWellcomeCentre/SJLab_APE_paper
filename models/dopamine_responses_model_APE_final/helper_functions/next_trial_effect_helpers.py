import numpy as np
import pandas as pd
import statsmodels.api as sm

def get_previous_choice_same_stim(all_state_changes, APEs, RPEs, choice='Left', stim='High', num_lags=5):
    left_choices = all_state_changes[all_state_changes['state name'] == stim + choice]
    left_trial_numbers = left_choices['trial number'].values
    left_inds = all_state_changes[all_state_changes['action taken'] == 'Left'].index
    cue_left = all_state_changes.shift(periods=1).iloc[left_inds]
    correct_cue_left = cue_left[cue_left['state name'] == stim]
    left_cue_time_stamps = correct_cue_left['time stamp'].values
    left_action_time_stamps = left_choices['time stamp'].values
    same_stim_trials = all_state_changes[
                    (all_state_changes['state name'] == stim + 'Left') |
                    (all_state_changes['state name'] == stim + 'Right')]
    last_same_stim_trial_numbers = []
    last_stim_actions = []
    last_stim_lags = []
    APE_trial_numbers = []
    left_APEs = []
    left_RPEs = []
    cue_time_stamps = []
    action_time_stamps = []
    for i, left_trial_number in enumerate(left_trial_numbers):
        for lag in range(1, num_lags + 1):
            previous_same_stim_trials = same_stim_trials[same_stim_trials['trial number'] < left_trial_number]
            if previous_same_stim_trials.shape[0] >= lag:
                last_same_stim_trial_number = previous_same_stim_trials['trial number'].values[-lag]
                last_same_stim_trial_numbers.append(last_same_stim_trial_number)
                last_same_stim_trial = all_state_changes[all_state_changes['trial number'] == last_same_stim_trial_number]
                action_state = last_same_stim_trial[
                    (last_same_stim_trial['state name'] == stim + 'Left') |
                    (last_same_stim_trial['state name'] == stim + 'Right')]
                action = action_state['action taken'].values[0]
                last_stim_actions.append(action)
                last_stim_lags.append(lag)
                APE_trial_numbers.append(left_trial_number)
                left_APEs.append(APEs[left_action_time_stamps[i]])
                left_RPEs.append(RPEs[left_cue_time_stamps[i]])
                cue_time_stamps.append(left_cue_time_stamps[i])
                action_time_stamps.append(left_action_time_stamps[i])
    past_trial_df = pd.DataFrame({'previous trial number': last_same_stim_trial_numbers, 'lag': last_stim_lags, 'previous action': last_stim_actions,
                                  'current trial number': APE_trial_numbers, 'current APE': left_APEs, 'current RPE': left_RPEs,
                                 'cue time stamp': cue_time_stamps, 'action time stamp': action_time_stamps})
    return past_trial_df

def run_multilag_regression_past_choice_on_signal(previous_choice_df, num_lags=5, y_var='current APE'):
    coefs = np.zeros([num_lags])
    pvals = np.zeros([num_lags])
    for lag in range(1, num_lags + 1):
        lag_df = previous_choice_df[previous_choice_df['lag'] == lag]
        df = lag_df[[y_var, 'previous action']].copy()
        df = df.replace({'Left': 1, 'Right': 0})

        df.loc[:, 'previous action'] = df['previous action'].astype(float)
        df.loc[:, y_var] = df[y_var].apply(
            lambda x: np.nan if isinstance(x, np.ndarray) and x.size == 0 else x)

        df.loc[:, y_var] = df[y_var].astype(float)
        df = df.dropna().reset_index(drop=True)

        y = df[y_var].astype(float)
        X = df['previous action']
        X = sm.add_constant(X)
        # Fit the regression model
        model = sm.OLS(y, X).fit()
        coefs[lag - 1] = model.params[1:]
        pvals[lag - 1] = model.pvalues[1:]
    return coefs, pvals