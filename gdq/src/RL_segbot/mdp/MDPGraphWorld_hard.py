from RL_segbot.mdp.MDPBasis import MDPBasisClass
from RL_segbot.mdp.MDPState import MDPStateClass
import RL_segbot.mdp.GraphWorldConstants_hard as const
from location import locations
import random
from collections import defaultdict
import networkx as nx


class MDPGraphWorld(MDPBasisClass):
    def __init__(self,
                 node_num=const.NODE_NUM,
                 init_node=const.START_NODES[0],
                 goal_node=const.GOAL_NODES[0],
                 start_nodes=const.START_NODES,
                 goal_nodes=const.GOAL_NODES,
                 has_door_nodes=const.has_door_nodes_tuple,
                 door_open_nodes=const.door_open_nodes_dict,
                 door_id=const.door_id_dict,
                 success_rate=const.env0,
                 step_cost=1.0,
                 goal_reward=const.goal_reward,
                 stack_cost=const.stack_cost,
                 is_goal_terminal=True,
                 is_rand_init=False,
                 is_rand_goal=False,
                 name="Graphworld"
                 ):
        self.node_num = node_num
        self.nodes = [MDPGraphWorldNode(i, is_terminal=False) for i in range(self.node_num)]
        self.success_rate = success_rate
        self.door_id = door_id
        self.has_door_nodes = has_door_nodes
        self.door_open_nodes = door_open_nodes
        self.is_goal_terminal = is_goal_terminal

        self.start_states = start_nodes
        self.goal_states = goal_nodes
        self.init_node = init_node
        self.goal_node = goal_node
        self.is_rand_init = is_rand_init
        self.is_rand_goal = is_rand_goal
        self.set_rand_init()
        self.set_rand_goal()

        self.set_nodes()
        self.G = self.set_graph()

        self.init_state = self.nodes[self.init_node]
        self.goal_query = str(self.nodes[self.goal_node])
        self.cur_state = self.init_state
        self.init_actions()
        self.goal_reward = goal_reward
        self.stack_cost = stack_cost
        self.name = name
        super().__init__(self.init_state, self.actions, self._transition_func, self._reward_func, step_cost)

    def __str__(self):
        return self.name + "_n-" + str(self.node_num)

    def __repr__(self):
        return self.__str__()

    # Accessors

    def get_params(self):
        get_params = super().get_params()
        get_params["node_num"] = self.node_num
        get_params["init_state"] = self.init_state
        get_params["goal_query"] = self.goal_query
        get_params["goal_states"] = self.goal_states
        get_params["start_states"] = self.start_states
        get_params["has_door_nodes"] = self.has_door_nodes
        get_params["cur_state"] = self.cur_state
        get_params["is_goal_terminal"] = self.is_goal_terminal
        get_params["success_rate"] = self.success_rate
        return get_params

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

    def get_action_cost(self, state, next_state):
        x1 = locations[str(state)]["x"]
        y1 = locations[str(state)]["y"]
        x2 = locations[str(next_state)]["x"]
        y2 = locations[str(next_state)]["y"]
        return const.distance(x1, y1, x2, y2)

    def get_visited(self, state):
        return self.G.nodes[state]['count']

    def get_stack_cost(self):
        return self.stack_cost

    def get_goal_reward(self):
        return self.goal_reward

    # Setter

    def init_actions(self):
        self.actions = defaultdict(lambda: set())
        for node in self.nodes:
            neighbor = self.get_neighbor(node)
            neighbor_id = [node.id for node in neighbor]
            for a in const.ACTIONS:
                for n in neighbor_id + [node.id]:
                    self.actions[node.get_state()].add((a, n))
                    node.set_door(node.has_door(), node.get_door_id(), not node.door_open())
                    self.actions[node.get_state()].add((a, n))
        self.set_nodes()

    def set_rand_init(self):
        if self.is_rand_init:
            self.init_node = random.choice(self.start_states)
        self.init_state = self.nodes[self.init_node]

    def set_rand_goal(self):
        if self.is_rand_goal:
            self.goal_node = random.choice(self.goal_states)
        self.goal_query = str(self.nodes[self.goal_node])

    def set_nodes(self):
        for node in self.nodes:
            node.is_stack = False
        for i in self.success_rate:
            self.nodes[i].set_slip_prob(self.success_rate[i])
        for i in self.has_door_nodes:
            self.nodes[i].set_door(True, self.door_id[i], self.door_open_nodes[i])
            # print(self.nodes[i].get_state())

        if self.is_goal_terminal:
            self.nodes[self.goal_node].set_terminal(True)

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

        G = nx.Graph(graph_dist)
        nx.set_node_attributes(G, 0, "count")
        return G

    # Core

    def _is_goal_state(self, state):
        return state.id == self.goal_node

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

        # print(self.get_neighbor(state))
        rand = random.random()
        if state.success_rate[0] < rand and not self._is_goal_state(state):
            if action[0] == "gothrough":
                action = ("fail", action[1])

            if action[0] == "opendoor":
                action = ("fail", action[1])

            if action[0] == "approach":
                miss = random.choice(self.get_neighbor(state) + [state])
                action = ("fail", miss.id)

            if action[0] == "goto":
                miss = random.choice(self.get_neighbor(state) + [state])
                action = ("fail", miss.id)

        next_state = state

        if action[0] == "opendoor" and state == self.nodes[action[1]] and state.has_door():
            state.set_door(state.has_door(), state.get_door_id(), True)
            next_state = state
        elif action[0] == "gothrough" and state.has_door() and state.door_open():
            for node in self.get_neighbor(state):
                if node.get_door_id() == state.get_door_id():
                    next_state = node
            next_state.set_door(state.has_door(), state.get_door_id(), True)
        elif action[0] == "approach" and self.nodes[action[1]].has_door():
            next_state = self.nodes[action[1]]
            if next_state.get_door_id() == state.get_door_id():
                next_state = state
        elif action[0] == "goto" and not self.nodes[action[1]].has_door():
            next_state = self.nodes[action[1]]
        else:
            next_state = state
            action = ("fail", action[1])

        # print("current goal is {0}".format(self.goal_query))
        if (action[0] == "gothrough" or action[0] == "opendoor" or action[0] == "fail") and \
                state.success_rate[0] + state.success_rate[1] < rand and \
                not self._is_goal_state(next_state):
            next_state.is_stack = True
            next_state.set_terminal()

        return next_state

    def _reward_func(self, state, action, next_state):
        """
        return rewards in next_state after taking action in state
        :param state: <State>
        :param action: <str>
        :param next_state: <State>
        :return: reward <float>
        """
        if self._is_goal_state(state):
            return self.get_goal_reward()
        elif next_state.get_is_stack():
            return -self.get_stack_cost()
        else:
            time = 0
            if str(state) == str(next_state):
                time = - 50 - random.random() * 20
            else:
                time = 0 - const.times[str(state)][str(next_state)] - random.random() * 20
            return time

    def reset(self):
        self.cur_state.set_terminal(False)
        self.set_rand_init()
        self.set_rand_goal()
        self.set_nodes()
        self.set_graph()
        super().reset()

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


class MDPGraphWorldNode(MDPStateClass):
    def __init__(self, id, is_terminal=False, has_door=False, door_id=None, door_open=False, success_rate=0.0):
        """
        A state in MDP
        :param id: <str>
        :param is_terminal: <bool>
        :param has_door: <bool>
        :param success_rate: <float>
        """
        self.id = id
        self.success_rate = success_rate
        self._has_door = has_door
        self._door_open = door_open
        self._door_id = door_id
        self.is_stack = False
        super().__init__(data=self.id, is_terminal=is_terminal)

    def __hash__(self):
        return hash(self.data)

    def __str__(self):
        return "s{0}".format(self.id)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        assert isinstance(other, MDPGraphWorldNode), "Arg object is not in" + type(self).__module__
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
            if self.door_open():
                return "s{0}_d{1}_True".format(self.id, self._door_id)
            else:
                return "s{0}_d{1}_False".format(self.id, self._door_id)
        else:
            return "s{0}".format(self.id)

    def get_is_stack(self):
        return self.is_stack

    def set_slip_prob(self, new_slip_prob):
        self.success_rate = new_slip_prob

    def has_door(self):
        return self._has_door

    def door_open(self):
        return self._door_open

    def set_door(self, has_door, door_id, door_open):
        self._has_door = has_door
        self._door_id = door_id
        self._door_open = door_open


if __name__ == "__main__":
    Graph_world = MDPGraphWorld(is_rand_init=False,
                                step_cost=1.0,
                                success_rate=const.env1)
    Graph_world.reset()
    observation = Graph_world
    # Graph_world.print_graph()
    for t in range(100):
        # print(Graph_world.get_actions())
        # print(observation.get_cur_state().get_state())
        # print(Graph_world.get_actions(observation.get_cur_state().get_state()))
        random_action = (random.choice(list(Graph_world.get_actions(observation.get_cur_state().get_state()))))
        print(observation.get_cur_state().get_state(), random_action, end=" ")
        observation, reward, done, info = Graph_world.step(random_action)
        print(observation.get_cur_state().get_state())
        # print(observation.get_cur_state().is_terminal(), done)
        # print(observation.get_params())
        if done:
            print("Goal!")
            print(observation.get_params())
            break
    print(observation.is_stack)
