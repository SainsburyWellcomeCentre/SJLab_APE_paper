import numpy as np
import pandas as pd
from classic_task import Box, TaskWithTimeOuts


class Mouse(object):
    def __init__(self, reaction_times, movement_times, env=Box(), critic_learning_rate=.2, actor_learning_rate=.1,
                 habitisation_rate=.1, inv_temp=5., psi=0.1):
        self.env = env
        self.inv_temp = inv_temp  # inverse temperature param for softmax decision func
        self.critic_lr = critic_learning_rate
        self.critic_value = np.zeros(self.env.n_states)  # weights parameterising value function
        self.actor_lr = actor_learning_rate
        self.actor_value = np.zeros([self.env.n_actions, self.env.n_states])
        self.habit_strength = np.zeros([self.env.n_actions, self.env.n_states])
        self.habitisation_rate = habitisation_rate
        self.reward_history = []
        self.dwell_time_history = []
        self.k = 0
        self.dwell_timer = movement_times[0]
        self.t_per_state = np.zeros(self.env.n_states)
        self.instances_in_state = np.zeros(self.env.n_states)
        self.saliences = np.zeros(self.env.n_states)
        self.cue_reaction_times = reaction_times
        self.movement_times = movement_times
        self.rho = 0
        self.r_k = []
        self.psi = psi  # Threshold for rectification of negative PEs
        self.num_cues = 0
        self.num_movements = 1

    def rectify(self, num_to_rectify):
        if num_to_rectify < 0:
            return 0
        else:
            return num_to_rectify

    def get_dwell_time(self, state):
        if state == 'High' or state == 'Low':
            dwell_time = self.cue_reaction_times[self.num_cues]
            self.num_cues += 1
        else:
            dwell_time = self.movement_times[self.num_movements]
            self.num_movements += 1
        return dwell_time

    def compute_value(self, features):
        return np.dot(self.critic_weights, features)

    def compute_action_value(self, features, action_idx):
        return np.dot(features, self.actor_weights[action_idx])  # is this right?

    def compute_habit_prediction_error(self, action, state_num):
        action_idx = self.env.action_idx[action]
        if action == 'Idle':
            delta_a = np.zeros([self.env.n_actions])  # means action 'Idle' does not cause APEs
        elif action == 'Centre':
            delta_a = np.zeros([self.env.n_actions])
        else:
            action_vector = np.eye(self.env.n_actions)[action_idx]
            delta_a = action_vector - self.habit_strength[:, state_num]
        if np.any(delta_a < 0):
            delta_a[np.where(delta_a < 0)] = 0
        return delta_a

    def compute_movement_signal(self, action, *args):
        """fire in response to movement"""
        action_idx = self.env.action_idx[action]
        if action == 'Idle':
            signal = np.zeros([self.env.n_actions])  # no movement
        elif action == 'Centre':
            signal = np.zeros([self.env.n_actions])  # no movement
        else:
            signal = np.eye(self.env.n_actions)[action_idx]

        return signal

    def compute_novelty(self, gamma=0.01):  # .33 if reset every trial, 0.01 if maintain across
        # just a decaying exponential of time spent in a state
        return np.exp(-gamma * self.instances_in_state)

    def compute_salience(self, value, novelty, state_idx, beta=1):
        # TODO: novelty + rpe
        # return beta * value + (1 - beta) * novelty
        return value[state_idx] / beta + novelty[state_idx]

    def compute_average_reward_per_timestep(self, n=500):
        if self.k == 0:
            rho_k = 0
        else:
            rho_k = sum(self.r_k[-n:]) / sum(self.dwell_time_history[-n:])
        return rho_k

    def one_trial_reward_blocks(self, trial_num):
        k = 0
        t = 0
        rectified_prediction_errors, prediction_errors, apes_l, apes_r, actions, states, m_signals, novelties, salience_hist, values, reward_types = [], [], [], [], [], [], [], [], [], [], []
        tone = None
        a = None
        state_changes = pd.DataFrame(columns=['state name', 'time stamp', 'dwell time', 'action taken'])
        total_reward = 0.
        self.instances_in_state[0] += 1
        novelty = self.compute_novelty()
        self.saliences[0] = self.compute_salience(self.critic_value, novelty, 0)
        while not self.env.in_terminal_state() and t < 1000:
            current_state_num = self.env.state_idx[self.env.current_state]
            self.t_per_state[current_state_num] += 1
            current_state = self.env.current_state
            dwell_time = self.env.time_in_state[current_state_num]
            policy = self.softmax(self.actor_value[:, current_state_num])
            a = self.choose_action(policy, dwell_time)
            next_state, reward, trial_type = self.env.act(a, dwell_time < self.dwell_timer, trial_num)
            next_state_num = self.env.state_idx[next_state]
            rho2 = 0
            delta_k = 0
            rectified_delta_k = self.rectify(0 + self.psi)
            ################################################################
            movement_signal = self.compute_movement_signal(a)
            novelty = np.zeros(self.env.n_states)
            self.saliences = np.zeros(self.env.n_states)

            delta_action = np.zeros([self.env.n_actions])

            ################################################################
            movement_states = ['HighLeft', 'HighRight', 'LowLeft', 'LowRight']
            if current_state != next_state:  # only updates value at state transitions
                novelty[next_state_num] = self.compute_novelty()[next_state_num]
                self.saliences[next_state_num] = self.compute_salience(self.critic_value, novelty, next_state_num)
                self.instances_in_state[next_state_num] += 1
                self.r_k.append(reward)
                rho2 = self.compute_average_reward_per_timestep()
                delta_k = reward - rho2 * dwell_time + self.critic_value[next_state_num] - self.critic_value[
                    current_state_num]
                rectified_delta_k = self.rectify(delta_k + self.psi)
                self.k += 1
                self.dwell_timer = self.get_dwell_time(next_state)
                self.critic_value[current_state_num] += self.critic_lr * delta_k
                self.actor_value[self.env.action_idx[a], current_state_num] += self.actor_lr * delta_k
                delta_action = self.compute_habit_prediction_error(a, current_state_num)
                self.habit_strength[:, current_state_num] += self.habitisation_rate * delta_action
                k += 1  # transition index increases
                new_state_changes = pd.DataFrame([[next_state, self.env.timer, dwell_time, a, trial_num]],
                                                 columns=['state name', 'time stamp', 'dwell time', 'action taken',
                                                          'trial number'])
                state_changes = state_changes.append(new_state_changes)
                self.dwell_time_history.append(dwell_time)

            if next_state == 'High':
                tone = 'High'
            elif next_state == 'Low':
                tone = 'Low'
            self.last_action = a
            prediction_errors.append(delta_k)
            reward_types.append(trial_type)
            rectified_prediction_errors.append(rectified_delta_k)
            apes_l.append(delta_action[0])
            apes_r.append(delta_action[2])
            actions.append(a)
            values.append(self.critic_value.reshape(-1, 1))
            m_signals.append(movement_signal.reshape(-1, 1))
            novelties.append(novelty.reshape(-1, 1))
            salience_hist.append(self.saliences.reshape(-1, 1))
            states.append(self.env.current_state)
            total_reward += reward
            self.reward_history.append(reward)
            t += 1
            self.env.timer += 1
        return prediction_errors, rectified_prediction_errors, tone, actions, states, state_changes, apes_l, apes_r, \
               total_reward, m_signals, values, novelties, salience_hist, reward_types  # novelties, saliences is 4 x trial_len

    def one_trial(self, trial_num):
        k = 0
        t = 0
        rectified_prediction_errors, prediction_errors, apes, actions, states, m_signals, novelties, salience_hist, values, reward_types = [], [], [], [], [], [], [], [], [], []
        tone = None
        a = None
        state_changes = pd.DataFrame(columns=['state name', 'time stamp', 'dwell time', 'action taken'])
        total_reward = 0.
        self.instances_in_state[0] += 1
        novelty = self.compute_novelty()
        self.saliences[0] = self.compute_salience(self.critic_value, novelty, 0)
        while not self.env.in_terminal_state() and t < 1000:
            current_state_num = self.env.state_idx[self.env.current_state]
            self.t_per_state[current_state_num] += 1
            current_state = self.env.current_state
            dwell_time = self.env.time_in_state[current_state_num]
            policy = self.softmax(self.actor_value[:, current_state_num])
            a = self.choose_action(policy, dwell_time)
            next_state, reward, trial_type = self.env.act(a, dwell_time < self.dwell_timer, trial_num)
            next_state_num = self.env.state_idx[next_state]
            rho2 = 0
            delta_k = 0
            rectified_delta_k = self.rectify(0 + self.psi)
            ################################################################
            movement_signal = self.compute_movement_signal(a)
            novelty = np.zeros(self.env.n_states)
            self.saliences = np.zeros(self.env.n_states)

            delta_action = np.zeros([self.env.n_actions])

            ################################################################
            movement_states = ['HighLeft', 'HighRight', 'LowLeft', 'LowRight', 'SilenceLeft', 'SilenceRight']
            if current_state != next_state:  # only updates value at state transitions
                novelty[next_state_num] = self.compute_novelty()[next_state_num]
                self.saliences[next_state_num] = self.compute_salience(self.critic_value, novelty, next_state_num)
                self.instances_in_state[next_state_num] += 1
                self.r_k.append(reward)
                rho2 = self.compute_average_reward_per_timestep()
                delta_k = reward - rho2 * dwell_time + self.critic_value[next_state_num] - self.critic_value[
                    current_state_num]
                rectified_delta_k = self.rectify(delta_k + self.psi)
                self.k += 1
                self.dwell_timer = self.get_dwell_time(next_state)
                self.critic_value[current_state_num] += self.critic_lr * delta_k
                self.actor_value[self.env.action_idx[a], current_state_num] += self.actor_lr * delta_k
                delta_action = self.compute_habit_prediction_error(a, current_state_num)
                self.habit_strength[:, current_state_num] += self.habitisation_rate * delta_action
                k += 1  # transition index increases
                new_state_changes = pd.DataFrame([[next_state, self.env.timer, dwell_time, a, trial_num]],
                                                 columns=['state name', 'time stamp', 'dwell time', 'action taken',
                                                          'trial number'])
                state_changes = state_changes.append(new_state_changes)
                self.dwell_time_history.append(dwell_time)

            if next_state == 'High':
                tone = 'High'
            elif next_state == 'Low':
                tone = 'Low'
            self.last_action = a
            prediction_errors.append(delta_k)
            reward_types.append(trial_type)
            rectified_prediction_errors.append(rectified_delta_k)
            apes.append(delta_action[0])
            actions.append(a)
            values.append(self.critic_value.reshape(-1, 1))
            m_signals.append(movement_signal.reshape(-1, 1))
            novelties.append(novelty.reshape(-1, 1))
            salience_hist.append(self.saliences.reshape(-1, 1))
            states.append(self.env.current_state)
            total_reward += reward
            self.reward_history.append(reward)
            t += 1
            self.env.timer += 1
        return prediction_errors, rectified_prediction_errors, tone, actions, states, state_changes, apes, \
               total_reward, m_signals, values, novelties, salience_hist, reward_types  # novelties, saliences is 4 x trial_len

    def choose_action(self, policy, dwell_time, random_policy=False, optimal_policy=False):
        if dwell_time < self.dwell_timer:
            a = 'Idle'
        elif self.env.current_state == 'HighLeft' or self.env.current_state == 'LowLeft':
            a = 'Idle'
        elif self.env.current_state == 'HighRight' or self.env.current_state == 'LowRight':
            a = 'Idle'
        elif self.env.current_state == 'SilenceLeft' or self.env.current_state == 'SilenceRight':
            a = 'Idle'
        else:
            if random_policy:
                a = np.random.choice(self.env.actions)
            elif optimal_policy:
                if self.env.current_state == 'High':
                    a = 'Left'
                elif self.env.current_state == 'Low':
                    a = 'Right'
                else:
                    a = 'Centre'
            else:
                a = np.random.choice(self.env.actions, p=policy)

        return a

    def softmax(self, state_action_values):
        exps = np.exp(self.inv_temp * state_action_values)
        policy = [e / sum(exps) for e in exps]
        return policy


class AgentWithTimeOuts(Mouse):
    def __init__(self, reaction_times, movement_times, env=TaskWithTimeOuts(), critic_learning_rate=.2,
                 actor_learning_rate=.1, habitisation_rate=.1, inv_temp=5., psi=0.1):
        super().__init__(reaction_times, movement_times, env=env, critic_learning_rate=critic_learning_rate,
                         actor_learning_rate=actor_learning_rate, habitisation_rate=habitisation_rate,
                         inv_temp=inv_temp, psi=psi)

    def one_trial(self, trial_num):
        k = 0
        t = 0
        rectified_prediction_errors, prediction_errors, apes, actions, states, m_signals, novelties, salience_hist, values, reward_types = [], [], [], [], [], [], [], [], [], []
        tone = None
        state_changes = pd.DataFrame(columns=['state name', 'time stamp', 'dwell time', 'action taken'])
        total_reward = 0.
        self.instances_in_state[0] += 1  # need this for the semi-Markov model update
        novelty = self.compute_novelty()
        self.saliences[0] = self.compute_salience(self.critic_value, novelty, 0)

        while not self.env.in_terminal_state() and t < 1000:

            # Update dwell time
            current_state_num = self.env.state_idx[self.env.current_state]
            self.t_per_state[self.env.state_idx[self.env.current_state]] += 1
            dwell_time = self.env.time_in_state[current_state_num]

            # choose and take action given policy
            policy = self.softmax(self.actor_value[:, current_state_num])
            a = self.choose_action(policy, dwell_time)
            next_state, reward, trial_type = self.env.act(a, dwell_time < self.dwell_timer, trial_num)

            if self.env.current_state == 'NoSound':
                break

            next_state_num = self.env.state_idx[next_state]
            delta_k = 0

            # rectify the prediction error
            rectified_delta_k = self.rectify(0 + self.psi)
            ################################################################
            # movement_signal = self.compute_movement_signal(a)  # basically just an index saying which movement was taken for saving later
            novelty = np.zeros(self.env.n_states)
            self.saliences = np.zeros(self.env.n_states)

            delta_action = np.zeros([self.env.n_actions])

            ################################################################
            # movement_states = ['HighLeft', 'HighRight', 'LowLeft', 'LowRight']
            if self.env.current_state != next_state:  # only updates value at state transitions
                novelty[next_state_num] = self.compute_novelty()[next_state_num]
                self.saliences[next_state_num] = self.compute_salience(self.critic_value, novelty, next_state_num)
                self.instances_in_state[next_state_num] += 1
                self.r_k.append(reward)
                rho2 = self.compute_average_reward_per_timestep()
                delta_k = reward - rho2 * dwell_time + self.critic_value[next_state_num] - self.critic_value[
                    current_state_num]
                rectified_delta_k = self.rectify(delta_k + self.psi)
                self.k += 1
                self.dwell_timer = self.get_dwell_time(next_state)
                self.critic_value[current_state_num] += self.critic_lr * delta_k
                self.actor_value[self.env.action_idx[a], current_state_num] += self.actor_lr * delta_k
                delta_action = self.compute_habit_prediction_error(a, current_state_num)
                self.habit_strength[:, current_state_num] += self.habitisation_rate * delta_action
                k += 1  # transition index increases
                new_state_changes = pd.DataFrame([[next_state, self.env.timer, dwell_time, a, trial_num]],
                                                 columns=['state name', 'time stamp', 'dwell time', 'action taken',
                                                          'trial number'])
                state_changes = state_changes.append(new_state_changes)
                self.dwell_time_history.append(dwell_time)

            if next_state == 'High':
                tone = 'High'
            elif next_state == 'Low':
                tone = 'Low'
            self.last_action = a
            prediction_errors.append(delta_k)
            reward_types.append(trial_type)
            rectified_prediction_errors.append(rectified_delta_k)
            apes.append(delta_action[0])
            actions.append(a)
            values.append(self.critic_value.reshape(-1, 1))
            #  m_signals.append(movement_signal.reshape(-1, 1))
            novelties.append(novelty.reshape(-1, 1))
            salience_hist.append(self.saliences.reshape(-1, 1))
            states.append(self.env.current_state)
            total_reward += reward
            self.reward_history.append(reward)
            t += 1
            self.env.timer += 1
        return prediction_errors, rectified_prediction_errors, tone, actions, states, state_changes, apes, \
               total_reward, m_signals, values, novelties, salience_hist, reward_types  # novelties, saliences is 4 x trial_len
