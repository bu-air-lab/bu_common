def parseOptions():
    import optparse
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--discount', action='store',
                         type='float', dest='discount', default=0.95,
                         help='Discount on future (default %default)')
    optParser.add_option('-e', '--epsilon', action='store',
                         type='float', dest='epsilon', default=0.1,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    optParser.add_option('-l', '--learningRate', action='store',
                         type='float', dest='alpha', default=0.5,
                         metavar="L", help='TD learning rate (default %default)')
    optParser.add_option('-i', '--iterations', action='store',
                         type='int', dest='iters', default=15,
                         metavar="I", help='Number of rounds of value iteration (default %default)')
    optParser.add_option('-k', '--episodes', action='store',
                         type='int', dest='episodes', default=25,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    optParser.add_option('-s', '--seed', action='store',
                         type='int', dest='seed', default=1,
                         metavar="s", help='Number of seeds of the experiment to run (default %default)')
    optParser.add_option('-n', '--lookahead', action='store',
                         type='int', dest='lookahead', default=10,
                         metavar="N", help='Number of times of the planning to look ahead (default %default)')
    optParser.add_option('-r', '--rmax', action='store',
                         type='float', dest='rmax', default=1000,
                         help='The upper bound of the reward function (default %default)')
    optParser.add_option('-u', '--update_count', action='store',
                         type='int', dest='update_count', default=1,
                         metavar="T", help='Number of count of updating q-values (default %default)')
    optParser.add_option('-a', '--agent', action='store', metavar="A",
                         type='string', dest='agent', default="gdq",
                         help='Agent type (options are \'dynaq\' and \'gdq\' default %default)')
    optParser.add_option('-p', '--pause', action='store_true',
                         dest='pause', default=False,
                         help='Pause GUI after each time step when running the MDP')
    optParser.add_option('-q', '--quiet', action='store_true',
                         dest='quiet', default=False,
                         help='Skip display of any learning episodes')
    optParser.add_option('-v', '--speed', action='store', metavar="V", type=float,
                         dest='speed', default=1,
                         help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)')
    optParser.add_option('-m', '--manual', action='store_true',
                         dest='manual', default=False,
                         help='Manually control agent')
    optParser.add_option('--mdp', action='store', metavar="M",
                         type='string', dest='mdp', default="testGraph",
                         help='MDP name (options are \'testGraph\', \'S0G14\', \'S0G16\', \'S10G14\', \'S10G16\', \'S21G14\', and \'S21G16\' default %default)')
    optParser.add_option('--env', action='store', metavar="E",
                         type='string', dest='env', default="env0",
                         help='env (options are \'env1\', \'env2\' default %default)')
    optParser.add_option('--start', action='store',
                         type='int', dest='start', default=mdp_const.START_NODES[0],
                         help='start node (options {0} default %default)'.format(mdp_const.START_NODES))
    optParser.add_option('--goal', action='store',
                         type='int', dest='goal', default=mdp_const.GOAL_NODES[0],
                         help='goal node (options {0} default %default)'.format(mdp_const.GOAL_NODES))

    opts, args = optParser.parse_args()

    if opts.manual and opts.agent != 'q':
        print('## Disabling Agents in Manual Mode (-m) ##')
        opts.agent = None

    if opts.manual:
        opts.pause = True

    return opts


def test_learned_policy(_mdp, _agent, seed=0):
    import RL_segbot.testConstants as const
    from RL_segbot.testMDPTask import load_agent

    _agent = load_agent(const.PKL_DIR + "{0}_{1}_{2}_fin.pkl".format(_agent.name, _mdp.name, seed))

    print()

    _mdp.reset()
    _agent.reset_of_episode()
    _agent.explore = "greedy"
    state = _mdp.get_cur_state()
    action = _agent.act(state)
    for t in range(1, 25):
        print(state.get_state(), action)
        _mdp, reward, done, info = _mdp.step(action)
        state = _mdp.get_cur_state()
        action = _agent.act(state)
        _agent.update(state, action, reward, learning=False)
        if done:
            print("The agent arrived at tearminal state.")
            print("Exit")
            break


def test_graphworld():
    opts = parseOptions()

    np.random.seed(0)

    env = mdp_const.env0
    if opts.env == "env0":
        env = mdp_const.env0

    route = [opts.start, opts.goal]

    envTest = MDPGraphWorld_hard(init_node=route[0],
                                 goal_node=route[1],
                                 step_cost=1.0,
                                 success_rate=env,
                                 name=opts.env + opts.mdp)

    mdp = envTest

    dynaq = DynaQAgent(actions=mdp.get_actions(),
                       alpha=opts.alpha,
                       gamma=opts.discount,
                       epsilon=opts.epsilon,
                       lookahead=opts.lookahead,
                       explore="uniform",
                       name="DynaQ")

    mbrlp = MBRLPAgent(actions=mdp.get_actions(),
                       alpha=opts.alpha,
                       gamma=opts.discount,
                       epsilon=opts.epsilon,
                       rmax=opts.rmax,
                       u_count=opts.update_count,
                       lookahead=opts.lookahead,
                       goal_state=mdp.goal_query,
                       is_initialize=True,
                       name="GDQ",
                       explore="uniform",
                       total_episode=opts.episodes)

    agent = None
    if opts.agent == 'dynaq':
        agent = dynaq
    elif opts.agent == 'gdq':
        agent = mbrlp

    runs_episodes(_mdp=mdp,
                  _agent=agent,
                  step=opts.iters,
                  episode=opts.episodes,
                  seed=opts.seed)

    test_learned_policy(mdp, agent)


if __name__ == "__main__":
    from RL_segbot.mdp.MDPGraphWorld_hard import MDPGraphWorld as MDPGraphWorld_hard
    import RL_segbot.mdp.GraphWorldConstants_hard as mdp_const
    from RL_segbot.testMDPTask import runs_episodes
    from RL_segbot.agent.dynaq import DynaQAgent
    from mbrlp import MBRLPAgent

    import numpy as np

    test_graphworld()