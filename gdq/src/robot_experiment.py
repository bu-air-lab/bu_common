#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from builtins import open, range, object, str, super
from builtins import bytes, zip, round, input, int, pow

from RL_segbot.testConstants import *
from asp_navigator import *
import rospy

from RL_segbot.agent.dynaq import DynaQAgent
from mbrlp import MBRLPAgent

import pandas as pd
import dill
import copy
from os import path
import sys
from signal import signal, SIGINT
import seaborn as sns

sns.set()

program_flag = True


def run_episodes(_mdp, _agent, step=50, episode=100, start_eps=0, s=0):
    global program_flag
    if path.exists(CSV_DIR + "{0}_{1}_{2}_fin.csv".format(_agent.name, _mdp.name, s)):
        df = pd.read_csv(CSV_DIR + "{0}_{1}_{2}_fin.csv".format(_agent.name, _mdp.name, s))
        df_list = list(df.values())
    else:
        df_list = list()

    for e in range(start_eps, episode):
        print("-------- new episode: {0:04} starts --------".format(e))

        with open('/home/yohei/rl_ws/src/rl_robot/src/robot_location.txt', 'a') as f:
            f.write(unicode("------init--------\n"))

        with open('/home/yohei/rl_ws/src/rl_robot/src/robot_action.txt', 'a') as f:
            f.write(unicode("------init--------\n"))
        location_list = list()

        # INIT ENV AND AGENT
        _mdp.reset()
        _agent.reset_of_episode()
        cumulative_reward = 0.0
        start_time = time.clock()

        # GET STATE
        state = _mdp.get_cur_state()
        # SELECT ACTION
        action = _agent.act(state)
        _agent.update(state, action, 0.0, learning=False, goal=_mdp.goal_query)
        print("[0 step] robot take action {0} in {1}".format(action, state.get_state()))

        with open('/home/yohei/rl_ws/src/rl_robot/src/robot_location.txt', 'a') as f:
            f.write(unicode("------------------\n"))
            f.write(unicode("episode {0}\n".format(e)))
        with open('/home/yohei/rl_ws/src/rl_robot/src/robot_action.txt', 'a') as f:
            f.write(unicode("------------------\n"))
            f.write(unicode("episode {0}\n".format(e)))
        location_list = list()

        #############
        # Logging
        #############
        x, y, ox, oy, oz, ow = _mdp.controller.get_pose(True)
        location_list.append((0, x, y, ox, oy, oz, ow))

        for t in range(1, step):
            # EXECUTE ACTION AND UPDATE ENV
            if program_flag:
                _mdp, reward, done, info = _mdp.step(action)
            if not program_flag:
                _mdp, reward, done, info = _mdp, -1000, True, None
            cumulative_reward += reward
            # GET STATE
            state = copy.deepcopy(_mdp.get_cur_state())
            action = _agent.act(state)
            print("Update Q({0},{1}) in {2} with reward {3}".format(_agent.get_pre_state(), _agent.get_pre_action(),
                                                                    state, reward))

            _agent.update(state, action, reward, episode=e)  # UPDATE LEARNER
            
            #############
            # Logging
            #############
            x, y, ox, oy, oz, ow = _mdp.controller.get_pose(True)
            location_list.append((t, x, y, ox, oy, oz, ow))
            with open('/home/yohei/rl_ws/src/rl_robot/src/robot_action.txt', 'a') as f:
                f.write(unicode("t: {0}\tstate: {2}\taction: {1}\treward: {3}\n".format(t-1, _agent.get_pre_action(), _agent.get_pre_state(), reward)))

            # END IF DONE
            if done:
                # _agent.update(state, action, reward, episode=e)  # UPDATE LEARNER
                break
            print("[{2} step] robot take action {0} in {1}".format(action, state.get_state(), t))


        end_time = time.clock()
        total_time = end_time - start_time
        
        #############
        # Logging
        #############

        location_df = pd.DataFrame(location_list, columns=['Timestep', 'x', 'y', 'ox', 'oy', 'oz', 'ow'])
        location_df.to_csv(CSV_DIR + "location_{0}_{1}_{2}_{3}.csv".format(_agent.name, _mdp.name, s, e))

        _mdp.to_pickle(PKL_DIR + "mdp_{0}_{1}_{2}_{3}.pkl".format(_agent.name, _mdp.name, s, e))
        _agent.q_to_csv(CSV_DIR + "q_{0}_{1}_{2}_{3}.csv".format(_agent.name, _mdp.name, s, e))
        _agent.to_pickle(PKL_DIR + "{0}_{1}_{2}_{3}.pkl".format(_agent.name, _mdp.name, s, e))
        df_list.append([e, _agent.step_number, cumulative_reward, total_time, s,
                        _agent.name, _mdp.name, _agent.alpha, _agent.gamma,
                        _agent.epsilon, _agent.rmax, _agent.u_count, _agent.lookahead])

        tmp_df = pd.DataFrame(df_list, columns=['Episode', 'Timestep', 'Cumulative Reward', 'Time(s)', 'seed',
                                                'AgentName', 'MDPName', 'alpha', 'gamma',
                                                'epsilon', 'rmax', 'ucount', 'lookahead'])
        tmp_df.to_csv(CSV_DIR + "{0}_{1}_{2}_{3}.csv".format(_agent.name, _mdp.name, s, e))

        # operation time to go to init
        if program_flag:
            for t in range(120):
                print("Bring this robot to initial room in {0} sec".format(t))
                rospy.sleep(1)
        else:
            location_df = pd.DataFrame(location_list, columns=['Timestep', 'x', 'y', 'ox', 'oy', 'oz', 'ow'])
            location_df.to_csv(CSV_DIR + "location_{0}_{1}_{2}_{3}.csv".format(_agent.name, _mdp.name, s, e))

            df = pd.DataFrame(df_list, columns=['Episode', 'Timestep', 'Cumulative Reward', 'Time(s)', 'seed',
                                                'AgentName', 'MDPName', 'alpha', 'gamma',
                                                'epsilon', 'rmax', 'ucount', 'lookahead'])
            df.to_csv(CSV_DIR + "{0}_{1}_{2}_fin.csv".format(_agent.name, _mdp.name, s))
            _mdp.to_pickle(PKL_DIR + "mdp_{0}_{1}_{2}_fin.pkl".format(_agent.name, _mdp.name, s))
            _agent.q_to_csv(CSV_DIR + "q_{0}_{1}_{2}_fin.csv".format(_agent.name, _mdp.name, s))
            _agent.to_pickle(PKL_DIR + "{0}_{1}_{2}_fin.pkl".format(_agent.name, _mdp.name, s))
            print("finished successfully by KeyboardInterrupt")
            sys.exit(0)

    df = pd.DataFrame(df_list, columns=['Episode', 'Timestep', 'Cumulative Reward', 'Time(s)', 'seed',
                                        'AgentName', 'MDPName', 'alpha', 'gamma',
                                        'epsilon', 'rmax', 'ucount', 'lookahead'])
    df.to_csv(CSV_DIR + "{0}_{1}_{2}_fin.csv".format(_agent.name, _mdp.name, s))
    _mdp.to_pickle(PKL_DIR + "mdp_{0}_{1}_{2}_fin.pkl".format(_agent.name, _mdp.name, s))
    _agent.q_to_csv(CSV_DIR + "q_{0}_{1}_{2}_fin.csv".format(_agent.name, _mdp.name, s))
    _agent.to_pickle(PKL_DIR + "{0}_{1}_{2}_fin.pkl".format(_agent.name, _mdp.name, s))


def experiment_start(_mdp, method, step=25, episode=15, s=0, start_eps=0):
    # method.reset()
    print("-------- id: {0:02} starts --------".format(s))
    run_episodes(_mdp, method, step, episode, start_eps)
    print("finished successfully")


def check_status(signal_received, frame):
    global program_flag
    program_flag = False
    print("pressed KeyboardInterrupt")


if __name__ == '__main__':
    try:
        signal(SIGINT, check_status)

        rospy.init_node('robot_experiment_py')
        controller = ControllerClass()
        mdp = MDPRealWorld(node_num=16,
                           is_goal_terminal=True,
                           step_cost=1.0,
                           name="realS0G15E25T25",
                           controller=controller)

        GAMMA = 0.95
        EPSILON = 0.1
        ALPHA = 0.5
        RMAX = 1000
        UCOUNT = 1

        dynaq = DynaQAgent(actions=mdp.get_actions(),
                           alpha=ALPHA,
                           gamma=GAMMA,
                           epsilon=EPSILON,
                           lookahead=10,
                           explore="uniform",
                           name="DynaQ")

        gdq = MBRLPAgent(actions=mdp.get_actions(),
                         alpha=ALPHA,
                         gamma=GAMMA,
                         epsilon=EPSILON,
                         rmax=RMAX,
                         u_count=UCOUNT,
                         lookahead=100,
                         goal_state="s15",
                         is_initialize=True,
                         explore="uniform",
                         name="GDQ")

        ##############
        # SELECT AGENT
        ##############
        agent = gdq

        #########################
        # SELECT SEED AND EPISODE
        #########################
        print("input the run No. and episode No.: ")
        s, e = list(map(int, raw_input().split()))
        sys.stdout.flush()

        ####################
        # LOAD AGENT AND MDP
        ####################
        agent_name = PKL_DIR + "{0}_{1}_{2}_{3}.pkl".format(agent.name, mdp.name, s, e)
        mdp_name = PKL_DIR + "mdp_{0}_{1}_{2}_{3}.pkl".format(agent.name, mdp.name, s, e)
        if path.exists(agent_name) and path.exists(mdp_name):
            print("load {0}".format(agent_name))
            print("load {0}".format(mdp_name))
            with open(agent_name, "rb") as f:
                agent = dill.load(f)
            with open(mdp_name, "rb") as f:
                tmp_mdp = dill.load(f)
            for node in mdp.G: mdp.G.nodes[node]['count'] = tmp_mdp.G.nodes[node]['count']
        else:
            e -= 1
            if not path.exists(agent_name):
                print("there is no such file {0}".format(agent_name))
            if not path.exists(mdp_name):
                print("there is no such file {0}".format(mdp_name))

        ####################
        # START EXPERIMENT
        ####################
        print("experiment start (MBRLP)")
        experiment_start(mdp, agent, step=10, episode=30, s=s, start_eps=e + 1)
        print("end")

        rospy.spin()

    except rospy.ROSInterruptException:
        print("finished!")
