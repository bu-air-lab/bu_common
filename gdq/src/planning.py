from ASP_segbot.asp2py.asp_planning import find_plan, arrange_plan
# from ASP.asp2py.grid_planning import find_plan, arrange_plan
from collections import defaultdict

class Planner:
    def __init__(self, goal_state):
        self.goal_state = goal_state
        self.pre_state = None
        self.plans_memory = defaultdict(list)

    def find_plans(self, cur_state, tar_state=None):
        def flatten(_plan):
            return [item for sublist in _plan for item in sublist]
        
        def rearrange(_plan):
            t, s, d, ds, a, sp, nd, nds = arrange_plan(_plan)
            return [s, a, sp]
            
        raw_plans = self.find_path(cur_state, tar_state)
        plans = list()
        for raw_plan in flatten(raw_plans):
            plan = rearrange(raw_plan)
            plans.append(plan)
        self.plans_memory[cur_state] = plans
            
    def get_plan(self, cur_state):
        if cur_state in self.plans_memory.keys():
            return self.plans_memory[cur_state]
        else:
            self.find_plans(cur_state)
            return self.plans_memory[cur_state]

    def set_pre_state(self, state):
        self.pre_state = state

    def set_goal_state(self, new_goal):
        self.goal_state = new_goal

    def find_path(self, cur_state, tar_state=None):
        if tar_state is None:
            return find_plan(cur_state, self.goal_state)
        else:
            return find_plan(cur_state, tar_state)

    def print_plans(self):
        for p in self.plans:
            print(p)

    def print_paths(self):
        for p in self.plans:
            print(p[1])

    def print_costs(self):
        for c in self.plans:
            print(c[0])


if __name__ == "__main__":
    planner = Planner("s10")
    # planner.find_path("s0")
    planner.get_plans("s0")
    planner.print_plans()
