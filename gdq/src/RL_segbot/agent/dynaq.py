#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from builtins import open, range, object, str, super
from builtins import bytes, zip, round, input, int, pow
from RL_segbot.agent.AgentBasis import AgentBasisClass
from collections import defaultdict
import numpy as np
import random


class DynaQAgent(AgentBasisClass):
    def __init__(self,
                 actions,
                 name="DynaQAgent",
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=0.1,
                 lookahead=30,
                 explore="uniform",
                 **kwargs):
        AgentBasisClass.__init__(self, name, actions, gamma)

        self.alpha, self.init_alpha = alpha, alpha
        self.epsilon, self.init_epsilon = epsilon, epsilon
        self.explore = explore
        self.lookahead = lookahead

        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.V = defaultdict(lambda: 0.0)
        self.C_sa = defaultdict(lambda: defaultdict(lambda: 0))
        self.C_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.rewards = defaultdict(lambda: defaultdict(list))

    # Accessors

    def get_params(self):
        params = self.get_params()
        params["alpha"] = self.alpha
        params["epsilon"] = self.epsilon
        params["explore"] = self.explore
        params["Q"] = self.Q
        params["C_sa"] = self.C_sa
        params["C_sas"] = self.C_sas
        return params

    def get_q_val(self, state, action):
        return self.Q[state][action]

    def get_policy(self, state):
        return self._get_max_q_key(state)

    def get_value(self, state):
        return self._get_max_q_val(state)

    def get_reward(self, state, action):
        if self.get_count(state, action) == 0:
            return 0.0
        return float(sum(self.rewards[state][action])) / self.get_count(state, action)

    def get_transition(self, state, action, next_state):
        return self.get_count(state, action, next_state) / self.get_count(state, action)

    def get_count(self, state, action, next_state=None):
        if next_state is None:
            return self.C_sa[state][action]
        else:
            return self.C_sas[state][action][next_state]

    def get_next_state(self, state, action):
        rnd = random.random()
        tmp = list(self.C_sas[state][action].keys())
        next_states = tmp[:]
        np.random.shuffle(next_states)
        sp = next_states[0]
        for next_state in next_states:
            if rnd < self.get_transition(state, action, next_state):
                sp = next_state
        return sp

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

        if learning and reward is not None:
            if pre_state is None and pre_action is None:
                self.set_pre_state(state.get_state())
                self.set_pre_action(action)
                return

            self.C_sa[pre_state][pre_action] += 1
            self.C_sas[pre_state][pre_action][state.get_state()] += 1
            self.rewards[pre_state][pre_action] += [reward]

            # real experience
            diff = self.gamma * self._get_max_q_val(state.get_state()) - self.get_q_val(pre_state, pre_action)
            self.Q[pre_state][pre_action] += self.alpha * (reward + diff)

            # simulated experience
            for n in range(self.lookahead):
                s = random.choice(list(self.C_sas.keys()))
                a = random.choice(list(self.C_sas[s].keys()))
                r = self.get_reward(s, a)
                sp = self.get_next_state(s, a)
                diff = self.gamma * self._get_max_q_val(sp) - self.get_q_val(s, a)
                self.Q[s][a] += self.alpha * (r + diff)

        self.set_pre_state(state.get_state())
        self.set_pre_action(action)

    def reset(self):
        self.episode_number = 0
        self.alpha = self.init_alpha
        self.epsilon = self.init_epsilon
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.C_sa = defaultdict(lambda: defaultdict(lambda: 0))
        self.C_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.rewards = defaultdict(lambda: defaultdict(list))

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