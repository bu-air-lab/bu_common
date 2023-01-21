#!/usr/bin/python
from __future__ import absolute_import, division, print_function
from builtins import str, open, super, range, object
# from builtins import bytes, zip, round, input, int, pow
import pandas as pd
import dill
import copy
import seaborn as sns;
import pathlib
import sys
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../')

from RL_segbot.testConstants import *

sns.set()


def run_episodes(_mdp, _agent, step=50, episode=100, s=0, decision_cb=None, display_cb=None, pause_cb=None):
    if decision_cb is None: decision_cb = _agent
    df_list = list()
    for e in range(episode):
        print("-------- new episode: {0:04} starts --------".format(e))
        # INIT ENV AND AGENT
        _mdp.reset()
        _agent.reset_of_episode()
        cumulative_reward = 0.0
        # GET STATE
        state = _mdp.get_cur_state()
        # SELECT ACTION
        action = decision_cb.act(state)
        _agent.update(state, action, 0.0, learning=False, goal=_mdp.goal_query)
        # _mdp.print_gird()

        # DISPLAY CURRENT STATE
        if display_cb is not None:
            display_cb(state)
            pause_cb()
        for t in range(1, step):
            # EXECUTE ACTION AND UPDATE ENV
            _mdp, reward, done, info = _mdp.step(action)
            cumulative_reward += reward
            print(state.get_state(), action, reward, _mdp.get_visited(state))

            # GET STATE
            state = copy.deepcopy(_mdp.get_cur_state())

            # DISPLAY CURRENT STATE
            if display_cb is not None:
                display_cb(state)
                pause_cb()
            # _mdp.print_gird()

            # SELECT ACTION
            action = decision_cb.act(state)

            # UPDATE LEARNER
            _agent.update(state, action, reward, episode=e)

            # END IF DONE
            if done:
                # print(reward)
                # UPDATE LEARNER
                # print("The agent arrived at tearminal state.")
                # print("Exit")
                break

        #############
        # Logging
        #############
        # if e % int(episode/2 + 0.5) == 0:  # record agent's log every 250 episode
        #     _mdp.to_pickle(PKL_DIR + "mdp_{0}_{1}_{2}_{3}.pkl".format(_agent.name, _mdp.name, s, e))
        #     _agent.to_pickle(PKL_DIR + "{0}_{1}_{2}_{3}.pkl".format(_agent.name, _mdp.name, s, e))
        #     _agent.q_to_csv(CSV_DIR + "q_{0}_{1}_{2}_{3}.csv".format(_agent.name, _mdp.name, s, e))
        df_list.append([e, _agent.step_number, cumulative_reward, s, _agent.name, _mdp.name, _agent.alpha, _agent.gamma, _agent.epsilon, _agent.rmax, _agent.u_count, _agent.lookahead])
    df = pd.DataFrame(df_list, columns=['Episode', 'Timestep', 'Cumulative Reward', 'seed', 'AgentName', 'MDPName', 'alpha', 'gamma', 'epsilon', 'rmax', 'ucount', 'lookahead'])
    df.to_csv(CSV_DIR + "{0}_{1}_{2}_fin.csv".format(_agent.name, _mdp.name, s))
    _mdp.to_pickle(PKL_DIR + "mdp_{0}_{1}_{2}_fin.pkl".format(_agent.name, _mdp.name, s))
    _agent.q_to_csv(CSV_DIR + "q_{0}_{1}_{2}_fin.csv".format(_agent.name, _mdp.name, s))
    _agent.to_pickle(PKL_DIR + "{0}_{1}_{2}_fin.pkl".format(_agent.name, _mdp.name, s))



def runs_episodes(_mdp, _agent, step=50, episode=100, seed=10):
    print("Running experiment: {0} in {1}".format(_agent.name, _mdp.name))
    for s in range(0, seed):
        _agent.reset()
        print("-------- new seed: {0:02} starts --------".format(s))
        run_episodes(_mdp, _agent, step, episode, s)


def run_experiments(_mdp, _agents, step=50, episode=100, seed=10):
    for a in _agents:
        runs_episodes(_mdp, a, step, episode, seed)


def episode_data_to_df(tmp, df, agent, seed, columns=("Timestep", "episode", "seed", "agent")):
    plot_df = pd.DataFrame(columns=columns)
    plot_df.loc[:, columns[0]] = tmp[seed]
    plot_df.loc[:, columns[1]] = range(len(tmp[seed]))
    plot_df.loc[:, columns[2]] = seed
    plot_df.loc[:, columns[3]] = str(agent)
    df = df.append(plot_df)
    return df


def load_agent(filename):
    with open(filename, "rb") as f:
        return dill.load(f)
