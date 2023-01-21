#!/usr/bin/env python
from RL_segbot.agent.AgentBasis import AgentBasisClass
from planning import Planner

from collections import defaultdict
import numpy as np
import random
import itertools


class MBRLPAgent(AgentBasisClass):
    def __init__(self,
                 actions,
                 goal_state,
                 name="MBRLPAgent",
                 is_initialize=True,
                 total_episode=100,
                 rmax=1.0,
                 u_count=2,
                 lookahead=15,
                 alpha=0.5,
                 gamma=0.99,
                 epsilon=0.1,
                 explore="greedy"):
        AgentBasisClass.__init__(self, name, actions, gamma)
        self.alpha, self.init_alpha = alpha, alpha
        self.u_count, self.init_urate = u_count, u_count
        self.epsilon, self.init_epsilon = epsilon, epsilon
        self.rmax, self.init_rmax = rmax, rmax
        self.goal_state = goal_state
        self.planner = Planner(goal_state)
        self.is_initialize, self.init_is_initialize = is_initialize, is_initialize
        self.lookahead = lookahead
        self.explore = explore

        self.tau = 10
        self.total_episode = total_episode
        exps = [np.exp(-e * self.tau / self.total_episode) for e in range(self.total_episode)]
        self.softmax = [j / sum(exps) for j in exps]

        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.V = defaultdict(lambda: 0.0)
        self.C_sa = defaultdict(lambda: defaultdict(lambda: 0))
        self.C_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.rewards = defaultdict(lambda: defaultdict(list))

        self.limited_C_sa = defaultdict(lambda: defaultdict(lambda: 0))
        self.limited_C_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

    # Accessors

    def get_params(self):
        params = self.get_params()
        params["goal"] = self.goal_state
        params["urate"] = self.u_count
        params["alpha"] = self.alpha
        params["Q"] = self.Q
        params["C_sa"] = self.C_sa
        params["C_sas"] = self.C_sas
        return params

    def get_urate(self):
        return self.u_count

    def get_q_val(self, state, action):
        return self.Q[state][action]

    def get_policy(self, state):
        return self._get_max_q_key(state)

    def get_value(self, state):
        return self._get_max_q_val(state)

    def get_reward(self, state, action):
        if self.get_count(state, action) >= self.u_count:
            return float(sum(self.rewards[state][action])) / self.get_count(state, action)
        else:
            return self.rmax

    def get_transition(self, state, action, next_state):
        return self.get_count(state, action, next_state) / self.get_count(state, action)

    def get_count(self, state, action, next_state=None):
        if next_state is None:
            return self.C_sa[state][action]
        else:
            return self.C_sas[state][action][next_state]

    def get_decay_prob(self, e):
        return sum(self.softmax[:e])
        # para = 0
        # if e < para:
        #     return 0
        # else:
        #     return (e - para) / (self.total_episode - para)

    # Setters

    def set_urate(self, new_urate):
        self.u_count = new_urate

    # Core

    def act(self, state):
        if self.is_initialize:
            self.is_initialize = False
            self.init_q_val(state)

        if self.explore == "uniform":
            action = self._epsilon_greedy_policy(state.get_state())
        elif self.explore == "greedy":
            action = self._get_max_q_key(state.get_state())
        elif self.explore == "softmax":
            action = self._soft_max_policy(state.get_state())
        elif self.explore == "random":
            action = random.choice(list(self.actions[state.get_state()]))
        else:
            action = self._epsilon_greedy_policy(state.get_state())  # default

        self.step_number += 1
        # print(state.get_state(), action, self.get_reward(state.get_state(), action))

        return action

    def _soft_max_policy(self, state):
        return NotImplemented

    def _epsilon_greedy_policy(self, state):
        if self.epsilon > np.random.random():
            action = random.choice(list(self.actions[state]))
        else:
            action = self._get_max_q_key(state)
        return action

    def init_q_val(self, state):
        plans = self.planner.get_plan(str(state))
        for plan in plans:
            s, a, sp = plan
            self.limited_C_sa[s][a] += 1
            self.limited_C_sas[s][a][sp] += 1
            self.Q[s][a] = self.get_reward(s, a)
            # print(s, a)

    def update(self, state, action, reward, learning=True, goal=None, episode=None):
        # print(state.get_state(), action)
        pre_state = self.get_pre_state()
        pre_action = self.get_pre_action()

        if goal is not None:
            self.goal_state = goal
            self.planner.set_goal_state(self.goal_state)
        if learning and reward is not None:
            if pre_state is None and pre_action is None:
                self.set_pre_state(state.get_state())
                self.set_pre_action(action)
                return

            self.C_sa[pre_state][pre_action] += 1
            self.C_sas[pre_state][pre_action][state.get_state()] += 1
            self.rewards[pre_state][pre_action] += [reward]

            # self.value_iteration(pre_state)

            diff = self.gamma * self._get_max_q_val(state.get_state()) - self.get_q_val(pre_state, pre_action)
            self.Q[pre_state][pre_action] += self.alpha * (reward + diff)
            # print(pre_state, pre_action, self.get_transition(pre_state, pre_action, state.get_state()))
            # Simulated experience
            self.simulated_update_with_guide(pre_state)

        self.set_pre_state(state.get_state())
        self.set_pre_action(action)

    def value_iteration(self, state):
        plans = self.planner.get_plan(state.split("_")[0])
        for plan in plans:
            s, a, sp = plan
            if self.limited_C_sa[s][a] == 0:
                self.Q[s][a] = self.get_reward(s, a)
            self.limited_C_sa[s][a] += 1
            self.limited_C_sas[s][a][sp] += 1

        # TODO: only limited_C_sa does not work because other states are not to be updated
        lim = int(np.log(1 / (self.epsilon * (1 - self.gamma))) / (1 - self.gamma))
        tmp = list(map(lambda x: itertools.product(self.limited_C_sa.keys(), self.limited_C_sa[x].keys()),
                       self.limited_C_sa.keys())) * lim
        for l in range(0, lim):
            for s, a in tmp[l]:
                if self.get_count(s, a) >= self.u_count:
                    self.Q[s][a] = self.get_reward(s, a) + self.gamma * \
                                   sum([self.get_transition(s, a, sp) * self._get_max_q_val(sp) for sp in
                                        self.Q.keys()])

    def simulated_update_with_guide(self, pre_state):
        tmp_state = pre_state.split("_")[0]
        plans = self.planner.get_plan(tmp_state)
        for plan in plans:
            s, a, sp = plan
            if self.limited_C_sa[s][a] == 0:
                self.Q[s][a] = self.get_reward(s, a)
            self.limited_C_sa[s][a] += 1
            self.limited_C_sas[s][a][sp] += 1

        all_plans = self.planner.plans_memory.values()
        all_trajectory = set()
        for plans in all_plans:
            for plan in plans:
                s, a, sp = plan
                all_trajectory.add((s, a))

        # print(all_trajectory)
        for n in range(self.lookahead):
            s = random.choice(list(self.C_sas.keys()))
            a = random.choice(list(self.C_sas[s].keys()))
            r = self.get_reward(s, a)
            sp = self.get_next_state(s, a)
            if (s, a, sp) in all_trajectory:
                diff = self.gamma * self._get_max_q_val(sp) - self.get_q_val(s, a)
                # ap = self._epsilon_greedy_policy(sp)
                # diff = self.gamma * self.get_q_val(sp, ap) - self.get_q_val(s, a)
                self.Q[s][a] = self.get_q_val(s, a) + self.alpha * (r + diff)

    def simulated_update_with_planning(self):
        for n in range(self.lookahead):
            s = random.choice(list(self.C_sas.keys()))
            a = random.choice(list(self.C_sas[s].keys()))
            r = self.get_reward(s, a)
            sp = self.get_next_state(s, a)
            diff = self.gamma * self._get_max_q_val(sp) - self.get_q_val(s, a)
            self.Q[s][a] = self.get_q_val(s, a) + self.alpha * (r + diff)

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

    def reset(self):
        self.u_count = self.init_urate
        self.alpha = self.init_alpha
        self.epsilon = self.init_epsilon
        self.episode_number = 0
        self.is_initialize = self.init_is_initialize
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.C_sa = defaultdict(lambda: defaultdict(lambda: 0))
        self.C_sas = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.rewards = defaultdict(lambda: defaultdict(list))

    def _get_max_q_key(self, state):
        return self._get_max_q(state)[0]

    def _get_max_q_val(self, state):
        return self._get_max_q(state)[1]

    def _get_max_q(self, state):
        # print(state, self.actions[state])
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
