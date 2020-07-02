# Evaluate from MPE_Local environment
from mpe_local.multiagent.environment import MultiAgentEnv
import mpe_local.multiagent.scenarios as scenarios
# Evaluate from original MPE environments
# from mpe.multiagent.environment import MultiAgentEnv
# import mpe.multiagent.scenarios as scenarios

def make_env(args):
    # load scenario from script

    scenario = scenarios.load(args.scenario_name + ".py").Scenario()

    # env setup
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

    # add to args
    args.num_agents = env.n
    args.obs_dim_arr = [env.observation_space[i].shape[0] for i in range(args.num_agents)]
    args.act_dim = env.action_space[0].n

    args.num_friends = args.num_agents - args.num_adversaries

    # args.action_bound_max = 1
    # args.action_bound_min = -1

    print('-'*100)
    print(f'num of agent:\t\t {env.n}')
    print(f'num of friends:\t\t {args.num_friends}')
    print(f'obs dim:\t\t {args.obs_dim_arr}')
    print(f'action dim:\t\t {args.act_dim}')
    print(f'len action space:\t {len(env.action_space)}')
    print(f'len obs space:\t\t {len(env.observation_space)}')
    print('-'*100)

    return env, args
