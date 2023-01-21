from RL.agent.AgentBasis import AgentBasisClass
from collections import defaultdict
import numpy as np
import pandas as pd
import random


class QLearningAgent(AgentBasisClass):
    def __init__(self,
                 actions,
                 name="QLearningAgent",
                 alpha=0.5,
                 gamma=0.99,
                 epsilon=0.1,
                 explore="uniform",
                 **kwargs):
        super().__init__(name, actions, gamma)
        self.alpha, self.init_alpha = alpha, alpha
        self.epsilon, self.init_epsilon = epsilon, epsilon
        self.explore = explore

        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))

    # Accessors

    def get_params(self):
        params = self.get_params()
        params["alpha"] = self.alpha
        params["epsilon"] = self.epsilon
        params["explore"] = self.explore
        params["Q"] = self.Q
        return params

    def get_alpha(self):
        return self.alpha

    def get_q_val(self, state, action):
        return self.Q[state][action]

    def get_policy(self, state):
        return self._get_max_q_key(state)

    def get_value(self, state):
        return self._get_max_q_val(state)

    # Setters

    def set_alpha(self, new_alpha):
        self.alpha = new_alpha

    # Core

    def act(self, state):
        if self.explore == "uniform":
            action = self._epsilon_greedy_policy(state)
        elif self.explore == "softmax":
            action = self._soft_max_policy(state)
        elif self.explore == "random":
            action = random.choice(list(self.actions[state.get_state()]))
        else:
            action = self._epsilon_greedy_policy(state)  # default

        self.step_number += 1

        return action

    def update(self, state, action, reward, learning=True, **kwargs):
        pre_state = self.get_pre_state()
        pre_action = self.get_pre_action()
        if learning:
            if pre_state is None:
                self.set_pre_state(state.get_state())
                self.set_pre_action(action)
                return

            diff = self.gamma * self._get_max_q_val(state.get_state()) - self.get_q_val(pre_state, pre_action)
            self.Q[pre_state][pre_action] += self.alpha * (reward + diff)
        # print(pre_state, pre_action, self.Q[pre_state][pre_action])
        self.set_pre_state(state.get_state())
        self.set_pre_action(action)

    def reset(self):
        self.alpha = self.init_alpha
        self.epsilon = self.init_epsilon
        self.episode_number = 0
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))

    def _get_max_q_key(self, state):
        return self._get_max_q(state)[0]

    def _get_max_q_val(self, state):
        return self._get_max_q(state)[1]

    def _get_max_q(self, state):
        tmp = list(self.actions[state])
        best_action = random.choice(tmp)
        actions = tmp[:]
        np.random.shuffle(actions)
        max_q_val = float("-inf")
        for key in actions:
            q_val = self.get_q_val(state, key)
            if q_val > max_q_val:
                best_action = key
                max_q_val = q_val
        return best_action, max_q_val

    def _soft_max_policy(self, state):
        pass

    def _epsilon_greedy_policy(self, state):
        if self.epsilon > np.random.random():
            action = random.choice(list(self.actions[state.get_state()]))
        else:
            action = self._get_max_q_key(state.get_state())
        return action