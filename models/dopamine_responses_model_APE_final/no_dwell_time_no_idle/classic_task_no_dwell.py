import numpy as np


class Box(object):
    def __init__(self, punish=False):
        self.high_to_left = True  # toggles whether High tone corresponds to correct action being left
        self.high_sound_prob = .5
        self.env_states = ['Start', 'High', 'Low', 'Outcome']
        self.actions = ['Left', 'Centre', 'Right', 'Idle']
        action_states = ['HighLeft', 'HighRight', 'LowLeft', 'LowRight']
        self.states = self.env_states + action_states
        self.n_states = len(self.states)
        self.state_idx = {s: idx for s, idx in zip(self.states, range(self.n_states))}
        self.n_actions = len(self.actions)
        self.action_idx = {a: idx for a, idx in zip(self.actions, range(self.n_actions))}

        self.current_state = self.states[0]
        self.time_in_state = np.zeros(self.n_states, dtype=int)
        self.timer = 0
        self.punish = punish

    def act(self, action, time_is_not_up, trial_num):
        # maybe make animals not able to act every time step
        next_state = self.get_next_state(self.current_state, action, time_is_not_up, trial_num)
        reward, trial_type = self.get_reward(self.current_state, action, next_state)

        # adjust state timer
        if next_state != self.current_state:
            self.time_in_state = np.zeros(self.n_states)
        else:
            self.time_in_state[self.state_idx[self.current_state]] += 1

        self.current_state = next_state
        return next_state, reward, trial_type

    def get_reward(self, state, action, next_state):
        if next_state != 'Outcome':
            reward_amount = 0
            trial_type = None
        else:
            if self.high_to_left:
                if state == 'HighLeft' or state == 'LowRight':
                    trial_type = 'normal'
                    reward_amount = 1
                else:
                    trial_type = 'incorrect'
                    reward_amount = 0
            else:
                if state == 'HighRight' or state == 'LowLeft':
                    trial_type = 'normal'
                    reward_amount = 1
                else:
                    trial_type = 'incorrect'
                    reward_amount = 0
        return reward_amount, trial_type

    def get_next_state(self, state, action, timer_is_not_up, trial_num):
        if state == 'Start':
            if action == 'Centre':
                if np.random.rand() <= self.high_sound_prob:
                    next_state = 'High'
                else:
                    next_state = 'Low'
            else:
                next_state = 'Start'

        elif state == 'High' or state == 'Low':
            if action == 'Idle':
                next_state = state
            elif action == 'Centre':
                if self.punish:
                    next_state = 'Outcome'
                else:
                    next_state = state
            else:
                next_state = state + action
        elif state == 'HighLeft' or state == 'HighRight' or state == 'LowLeft' or state == 'LowRight':
            if timer_is_not_up:
                next_state = state
            else:
                next_state = 'Outcome'
        elif state == 'Outcome':
            next_state = None

        else:
            raise ValueError('No valid state input.')

        return next_state

    def reset(self):
        """Reset environment to initial state.

        :return:
        """
        self.current_state = self.states[0]

    def in_terminal_state(self):

        """Return True if current state is terminal, False otherwise.

        :return:
        """
        return self.current_state == 'Outcome'


