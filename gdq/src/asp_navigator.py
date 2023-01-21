#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from builtins import open, range, object, str, super
# from builtins import bytes, zip, round, input, int, pow
import time
import networkx as nx
from collections import defaultdict

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction
# from move_base_msgs.msg import MoveBaseGoal
# from move_base_msgs.msg import MoveBaseFeedback
# from geometry_msgs.msg import Twist
# from actionlib_msgs.msg import *
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
# from geometry_msgs.msg import Quaternion, Pose, Point
from nav_msgs.srv import GetPlan
from sound_play.libsoundplay import SoundClient
import std_srvs.srv
from bwi_services.srv import GoToLocation, GoToLocationRequest, GoToLocationResponse

from location import locations
from RL_segbot.mdp.MDPBasis import MDPBasisClass
from RL_segbot.mdp.MDPState import MDPStateClass
from RL_segbot.mdp.GraphWorldConstants_hard import *


class MDPRealWorld(MDPBasisClass):
    def __init__(self,
                 node_num=NODE_NUM,
                 init_node=START_NODES[0],
                 goal_node=GOAL_NODES[0],
                 start_nodes=START_NODES,
                 goal_nodes=GOAL_NODES,
                 has_door_nodes=has_door_nodes_tuple,
                 door_open_nodes=door_open_nodes_dict,
                 door_id=door_id_dict,
                 step_cost=0.0,
                 goal_reward=goal_reward,
                 stack_cost=stack_cost,
                 is_goal_terminal=True,
                 name="Graphworld",
                 controller=None
                 ):
        self.node_num = node_num
        self.nodes = [MDPRealWorldNode(i, is_terminal=False) for i in range(self.node_num)]
        self.door_id = door_id
        self.has_door_nodes = has_door_nodes
        self.door_open_nodes = door_open_nodes
        self.is_goal_terminal = is_goal_terminal

        self.start_states = start_nodes
        self.goal_states = goal_nodes
        self.init_node = init_node
        self.goal_node = goal_node

        self.set_nodes()
        self.G = self.set_graph()

        self.init_state = self.nodes[self.init_node]
        self.goal_query = str(self.nodes[self.goal_node])
        self.cur_state = self.init_state
        self.init_actions()
        self.goal_reward = goal_reward
        self.stack_cost = stack_cost
        self.name = name
        super(MDPRealWorld, self).__init__(self.init_state, self.actions, self._transition_func, self._reward_func, step_cost)

        self.controller = controller

    def __str__(self):
        return self.name + "_n-" + str(self.node_num)

    def __repr__(self):
        return self.__str__()

    # Accessors

    def get_params(self):
        get_params = super(MDPRealWorld, self).get_params()
        get_params["node_num"] = self.node_num
        get_params["init_state"] = self.init_state
        get_params["goal_query"] = self.goal_query
        get_params["goal_states"] = self.goal_states
        get_params["start_states"] = self.start_states
        get_params["has_door_nodes"] = self.has_door_nodes
        get_params["cur_state"] = self.cur_state
        get_params["is_goal_terminal"] = self.is_goal_terminal

    def get_neighbor(self, node):
        return list(self.G[node])

    def get_actions(self, state=None):
        if state is None:
            return self.actions
        return self.actions[state]

    def get_nodes(self):
        nodes = dict()
        for node in self.nodes:
            nodes[str(node)] = node
        return nodes

    def get_cur_state(self):
        self.set_cur_state()
        self.door_open(self.cur_state)
        print("current state is {0}".format(self.cur_state.get_state()))
        return self.cur_state

    # Setter

    def set_cur_state(self):
        x1, y1 = self.controller.get_pose()
        min_dist = distance(x1, y1, locations[str(self.cur_state)]["x"], locations[str(self.cur_state)]["y"])
        for node in self.nodes:
            x2 = locations[str(node)]["x"]
            y2 = locations[str(node)]["y"]
            dist = distance(x1, y1, x2, y2)
            if dist < min_dist:
                min_dist = dist
                self.cur_state = node

    # print("set the current state. move to its cordination")
    # goto_location(locations[str(self.cur_state)])

    def init_actions(self):
        self.actions = defaultdict(lambda: set())
        for node in self.nodes:
            neighbor = self.get_neighbor(node)
            neighbor_id = [neighbor_node.id for neighbor_node in neighbor]
            for a in ACTIONS:
                for n in neighbor_id + [node.id]:
                    node.set_door(node.has_door(), node.get_door_id(), True)
                    self.actions[node.get_state()].add((a, n))
                    node.set_door(node.has_door(), node.get_door_id(), False)
                    self.actions[node.get_state()].add((a, n))
        self.set_nodes()

    def set_nodes(self):
        for i in self.has_door_nodes:
            self.nodes[i].set_door(True, self.door_id[i], self.door_open_nodes[i])

        if self.is_goal_terminal:
            for i in self.goal_states:
                self.nodes[i].set_terminal(True)

    def door_open(self, node):
        if node.has_door():
            print("[Call] checking d{0}: {1} ".format(node._door_id, node._door_open), end="")
            node._door_open = self.controller.checkdoor_open(locations["s" + str(node.id)],
                                                             locations["s" + str(get_match_door(node.id))])
        else:
            node._door_open = False
        return node._door_open

    def set_graph(self):
        node = self.nodes
        graph_dist = {node[0]: [node[1], node[2]],
                      node[1]: [node[0], node[2], node[3]],
                      node[2]: [node[0], node[1], node[4]],
                      node[3]: [node[1], node[5]],
                      node[4]: [node[2], node[8]],
                      node[5]: [node[3], node[6], node[8]],
                      node[6]: [node[5], node[7]],
                      node[7]: [node[6], node[15]],
                      node[8]: [node[4], node[5], node[9], node[11]],
                      node[9]: [node[8], node[10]],
                      node[10]: [node[9], node[15]],
                      node[11]: [node[8], node[12]],
                      node[12]: [node[11], node[13]],
                      node[13]: [node[12], node[14]],
                      node[14]: [node[13], node[15]],
                      node[15]: [node[7], node[10], node[14]]
                      }

        graph = nx.Graph(graph_dist)
        nx.set_node_attributes(graph, 0, "count")
        return graph

    # Core

    def _transition_func(self, state, action):
        """
        transition function. it returns next state
        :param state: <State>
        :param action: <tuple <str, id>> action discription and node id
        :return: next_state <State>
        """
        self.G.nodes[state]['count'] += 1

        if state.is_terminal():
            return state

        next_state = state
        status = False
        start = rospy.get_time()

        is_door_open = self.door_open(state)

        if action[0] == "goto" and not self.nodes[action[1]].has_door():
            ns = self.nodes[action[1]]
            self.door_open(ns)
            # status = self.controller.goto_location(locations[str(ns)])
            status = self.controller.goto_location(ns)
            print("executed {0}".format(action), end="")

        elif action[0] == "approach" and self.nodes[action[1]].has_door():
            ns = self.nodes[action[1]]
            self.door_open(ns)
            # status = self.controller.approach(ns.get_state(), locations[str(ns)])
            status = self.controller.approach(ns, ns.get_state(), locations[str(ns)])
            print("executed {0}".format(action), end="")

        elif action[0] == "opendoor" and state == self.nodes[action[1]] and state.has_door():
            ns = state
            self.door_open(ns)
            status = self.controller.opendoor(locations[str(state)], locations["s" + str(get_match_door(state.id))], ns.get_state())
            print("executed {0}".format(action), end="")

        elif action[0] == "gothrough" and state.has_door() and is_door_open and self.nodes[action[1]].has_door():
            ns = self.nodes[get_match_door(action[1])]
            self.door_open(ns)
            # status = self.controller.gothrough(ns.get_state(), locations[str(state)], locations[str(ns)])
            status = self.controller.gothrough(ns, ns.get_state(), locations[str(state)], locations[str(ns)])
            print("executed {0}".format(action), end="")
        else:
            ns = state
            print("executed nothing", end="")

        rospy.sleep(1)
        end = rospy.get_time()
        self.execute_time = end - start
        if status == 1:
            print(" -> Success")
            next_state = ns
        else:
            print(" -> Fail")
            self.execute_time += 50
            self.set_cur_state()
        print("[Call] wait for 2 secondes for next time step")
        rospy.sleep(1)
        print("[Call] wait for 1 secondes for next time step")
        rospy.sleep(1)
        return next_state

    def _reward_func(self, state, action, next_state):
        """
        return rewards in next_state after taking action in state
        :param state: <State>
        :param action: <str>
        :param next_state: <State>
        :return: reward <float>
        """
        if state.is_terminal():
            return 1000
        else:
            return -self.execute_time

    def reset(self):
        super(MDPRealWorld, self).reset()
        self.set_nodes()
        self.set_graph()
        print("reset mdp (move robot tos): ", end="")
        x, y = self.controller.get_pose()
        to_x = locations["s" + str(self.init_node)]["x"]
        to_y = locations["s" + str(self.init_node)]["y"]
        print(to_x, to_y)
        self.controller.clear_obstacle_map()
        status = self.controller.goto_location(self.init_state)
        x, y = self.controller.get_pose()
        # while not (to_x - 0.25 < x < to_x + 0.25) or not (to_y - 0.25 < y < to_y + 0.25):
        #     self.controller.clear_obstacle_map()
        #     # self.controller.goto_location(locations["s" + str(self.init_node)])
        #     self.controller.goto_location(self.init_state)
        #     x, y = self.controller.get_pose()
        #     rospy.sleep(1)
        while not self.init_state == self.get_cur_state():
            status = self.controller.goto_location(self.init_state)
            x, y = self.controller.get_pose()
        print("robot is initially located in {2} ({0}, {1})".format(x, y, self.get_cur_state()))

    def print_graph(self):
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos, alpha=0.9, node_size=500)
        nodelist = [self.nodes[0], self.nodes[10]]
        nx.draw_networkx_nodes(self.G, pos, nodelist=nodelist, node_color='r', alpha=0.9,
                               node_size=500)
        nx.draw_networkx_labels(self.G, pos)
        nx.draw_networkx_edges(self.G, pos)
        plt.show()

    def save_graph_fig(self, filename="graph.png"):
        import matplotlib.pyplot as plt
        fix, ax = plt.subplots()
        pos = nx.spring_layout(self.G)
        nx.draw_networkx_nodes(self.G, pos, alpha=0.9, node_size=500)
        nodelist = [self.nodes[0], self.nodes[10]]
        nx.draw_networkx_nodes(self.G, pos, nodelist=nodelist, node_color='r', alpha=0.9,
                               node_size=500)
        nx.draw_networkx_labels(self.G, pos)
        nx.draw_networkx_edges(self.G, pos)
        plt.savefig(filename)
        del plt

    def save_graph(self, filename="graph.p"):
        with open(filename, "wb") as f:
            nx.write_gpickle(self.G, f)


class MDPRealWorldNode(MDPStateClass):
    def __init__(self, _id, is_terminal=False, has_door=False, door_id=None, door_open=False, success_rate=0.0):
        """
        A state in MDP
        :param _id: <str>
        :param is_terminal: <bool>
        :param has_door: <bool>
        :param success_rate: <float>
        """
        self.id = _id
        self.success_rate = success_rate
        self._has_door = has_door
        self._door_open = door_open
        self._door_id = door_id
        super(MDPRealWorldNode, self).__init__(data=self.id, is_terminal=is_terminal)

    def __hash__(self):
        return hash(self.data)

    def __str__(self):
        return "s{0}".format(self.id)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        assert isinstance(other, MDPRealWorldNode), "Arg object is not in" + type(self).__module__
        return self.id == other.id

    def get_param(self):
        params_dict = dict()
        params_dict["id"] = self.id
        params_dict["success_rate"] = self.success_rate
        params_dict["has_door"] = self._has_door
        params_dict["door_open"] = self._door_open
        params_dict["door_id"] = self._door_id
        return params_dict

    def get_slip_prob(self):
        return self.success_rate

    def get_door_id(self):
        return self._door_id

    def get_state(self):
        if self.has_door():
            if self._door_open:
                return "s{0}_d{1}_True".format(self.id, self._door_id)
            else:
                return "s{0}_d{1}_False".format(self.id, self._door_id)
        else:
            return "s{0}".format(self.id)

    def set_slip_prob(self, new_slip_prob):
        self.success_rate = new_slip_prob

    def has_door(self):
        return self._has_door

    def set_door(self, has_door, door_id, door_open):
        self._has_door = has_door
        self._door_id = door_id
        self._door_open = door_open


class ControllerClass(object):
    def __init__(self):
        self.to_x, self.to_y = None, None
        self.pose_x, self.pose_y = None, None
        self.pose_ox, self.pose_oy, self.pose_oz, self.pose_ow = None, None, None, None

        self.ac = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        while not self.ac.wait_for_server(rospy.Duration.from_sec(5.0)):
            rospy.loginfo("Waiting for the move_base action server to come up")

        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.callback_pose)

        self.clear_cost = rospy.ServiceProxy('/move_base/clear_costmaps', std_srvs.srv.Empty())

        self.sound_handler = SoundClient()
        rospy.sleep(3)
        self.sound_handler.say('Program starts.')

    def clear_obstacle_map(self):
        self.clear_cost()
        rospy.sleep(3)

    @staticmethod
    def goto_location(node):
        rospy.wait_for_service('/bwi_services/go_to_location')
        gotolocation = rospy.ServiceProxy('/bwi_services/go_to_location', GoToLocation)
        to = GoToLocationRequest()
        to.location = str(node).encode('ascii')
        status = gotolocation(to)
        # print(type(status.result))
        return status.result

    def active_cb(self):
        # rospy.loginfo("Goal pose "+"location ({0}, {1})".format(to_x, to_y)+" is now being processed by the Action Server...")
        pass

    def feedback_cb(self, feedback):
        # clear_obstacle_map()
        # rospy.sleep(60)
        pass

    def done_cb(self, status, result):
        pass
        # if status == 2:
        #     rospy.loginfo("Goal pose ({0}, {1})".format(self.to_x, self.to_y) +
        #                   " received a cancel request after it started executing, completed execution!")

        # if status == 3:
        #     rospy.loginfo("Goal pose ({0}, {1})".format(self.to_x, self.to_y) + " reached")

        # if status == 4:
        #     rospy.loginfo("Goal pose ({0}, {1})".format(self.to_x, self.to_y) + " was aborted by the Action Server")
        #     self.ac.cancel_all_goals()

        # if status == 5:
        #     rospy.loginfo(
        #         "Goal pose ({0}, {1})".format(self.to_x, self.to_y) + " has been rejected by the Action Server")
        #     self.ac.cancel_all_goals()

        # if status == 6:
        #     rospy.loginfo("Goal pose ({0}, {1})".format(self.to_x, self.to_y) +
        #                   "received a cancel request after it started executing and has not yet completed execution")
        #     self.ac.cancel_all_goals()

        # if status == 8:
        #     rospy.loginfo("Goal pose ({0}, {1})".format(self.to_x, self.to_y) +
        #                   " received a cancel request before it started executing, successfully cancelled!")

    def checkdoor_open(self, cur, tar):
        rospy.wait_for_service('move_base/NavfnROS/make_plan')
        make_plan = rospy.ServiceProxy('move_base/NavfnROS/make_plan', GetPlan)
        tolerance = 0.1
        start = self.set_pose(cur)
        goal = self.set_pose(tar)
        plan_response = make_plan(start=start, goal=goal, tolerance=tolerance)
        poses = plan_response.plan.poses
        print(len(poses))
        # print("({0}, {1}), ({2}, {3})".format(cur["x"], cur["y"], tar["x"], tar["y"]))
        return 0 < len(poses) < 500

    def approach(self, node, name, coordinate):
        print("[Call] approaching door: " + name + " which is at: ({0}, {1})".format(coordinate["x"], coordinate["y"]))
        return self.goto_location(node)

    def opendoor(self, cur, tar, node):
        done = False
        start = rospy.get_time()
        end = rospy.get_time()
        while (not done) and (end - start < 20):
            self.clear_obstacle_map()
            result = self.checkdoor_open(cur, tar)
            end = rospy.get_time()
            if not result:
                self.sound_handler.say('Please Open the door for me.')
                print("[Call] Please Open the door for me.")
                done = False
                self.goto_location(node)
                rospy.sleep(3)
            else:
                self.sound_handler.say('Thank you.')
                print("[Call] Thank you.")
                done = True
                rospy.sleep(1)
        return done

    def gothrough(self, node, name, cur, tar):
        print("[Call] Gothrough to door: " + name + " which is at: ({0}, {1})".format(tar["x"], tar["y"]))
        return self.goto_location(node)

    # for loop over the plan
    def callback_pose(self, data):
        self.pose_x = data.pose.pose.position.x
        self.pose_y = data.pose.pose.position.y
        self.pose_ox = data.pose.pose.orientation.x
        self.pose_oy = data.pose.pose.orientation.y
        self.pose_oz = data.pose.pose.orientation.z
        self.pose_ow = data.pose.pose.orientation.w
        with open('/home/yohei/rl_ws/src/rl_robot/src/robot_location.txt', 'a') as f:
            f.write(str(self.pose_x) + "," + str(self.pose_y) + "," +
                    str(self.pose_ox) + "," + str(self.pose_oy) + "," +
                    str(self.pose_oz) + "," + str(self.pose_ow) + "\n")

    def get_pose(self, _all=False):
        if not _all:
            return self.pose_x, self.pose_y
        else:
            return self.pose_x, self.pose_y, self.pose_ox, self.pose_oy, self.pose_oz, self.pose_ow

    def set_pose(self, location):
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.callback_pose)
        pos = PoseStamped()
        pos.header.frame_id = "level_mux_map"
        x, y, ox, oy, oz, ow = location["x"], location["y"], location["ox"], location["oy"], location["oz"], location[
            "ow"]
        pos.pose.position.x = x
        pos.pose.position.y = y
        pos.pose.orientation.x = ox
        pos.pose.orientation.y = oy
        pos.pose.orientation.z = oz
        pos.pose.orientation.w = ow
        return pos


if __name__ == "__main__":
    pass
