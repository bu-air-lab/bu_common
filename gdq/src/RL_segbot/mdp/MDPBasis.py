import copy
import dill

class MDPBasisClass(object):
    """ abstract class for a MDP """

    def __init__(self, init_state, actions, transition_func, reward_func, step_cost):
        self.init_state = copy.deepcopy(init_state)
        self.cur_state = self.init_state
        self.actions = actions
        self.__transition_func = transition_func
        self.__reward_func = reward_func
        self.step_cost = step_cost

    # Accessors

    def get_params(self):
        """
        Returns:
            <dict> key -> param_name, val -> param_value
        """
        param_dict = dict()
        param_dict["step_cost"] = self.step_cost

        return param_dict

    def get_init_state(self):
        return self.init_state

    def get_cur_state(self):
        return self.cur_state

    def get_actions(self, state):
        return self.actions

    def get_transition_func(self):
        return self.__transition_func

    def get_reward_func(self):
        return self.__reward_func

    def get_step_cost(self):
        return self.step_cost

    # Setters

    def set_init_state(self, new_init_state):
        self.init_state = copy.deepcopy(new_init_state)

    def set_actions(self, new_actions):
        self.actions = new_actions

    def set_transition_func(self, new_transition_func):
        self.__transition_func = new_transition_func

    def set_step_cost(self, new_step_cost):
        self.step_cost = new_step_cost

    # Core

    def step(self, action):
        """
        :param action: <str>
        :return: observation: <MDPStateClass>,
                 reward: <float>,
                 done: <bool>,
                 info: <dict>
        """
        next_state = self.__transition_func(self.cur_state, action)
        reward = self.__reward_func(self.cur_state, action, next_state)
        done = self.cur_state.is_terminal()
        self.cur_state = next_state

        return self, reward, done, self.get_params()

    def reset(self):
        self.cur_state = self.init_state

    def to_pickle(self, filename):
        with open(filename, "wb") as f:
            dill.dump(self, f)