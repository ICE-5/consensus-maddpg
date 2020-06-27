from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios



def make_env(args):
    # load scenario from script
    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    # env setup
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    
    # add to args
    args.obs_dim = env.observation_space[0].shape[0]
    args.action_dim = env.action_space[0].n
    args.num_players = env.n
    args.num_agents = env.n - args.num_adversaries      # TODO: check adversary related
    args.action_bound_max = 1
    args.action_bound_min = -1

    return env, args