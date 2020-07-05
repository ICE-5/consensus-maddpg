import argparse


def get_args():
    parser = argparse.ArgumentParser("Multi-agent reinforcement learning with actor-critic")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="adversarial", help="name of the scenario script")
    parser.add_argument("--map-size", type=int, default=2,
                        help="The size of the environment. 1 if normal and 2 otherwise. (default: normal)")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-good", type=int, default=10, help="number of good agents in the scenario (default: 2)")
    parser.add_argument("--num-adversaries", type=int, default=10,
                        help=" number of adversaries in the environment (default: 2)")
    parser.add_argument("--num-food", type=int, default=10,
                        help="number of food(resources) in the scenario (default: 4)")
    parser.add_argument("--num-forests", type=int, default=0,
                        help="number of forest in the scenario (default: 0)")
    parser.add_argument("--num-landmarks", type=int, default=0,
                        help="number of landmarks in the scenario (default: 0)")
    parser.add_argument("--sight", type=int, default=100, help="The agent's visibility radius. (default: 100)")
    parser.add_argument("--alpha", type=float, default=0., help="Reward shared weight. (default: 0.0)")

    # Network config
    parser.add_argument("--good-policy", type=str, default="ddpg",
                        help="algorithm used for the 'good' (non adversary) policies in the environment"
                             " (default: maddpg; options: {ddpg, maddpg, cmaddpgv1, cmaddpgv2})")
    parser.add_argument("--adv-policy", type=str, default="ddpg",
                        help="algorithm used for the adversary policies in the environment"
                             " (default: maddpg; options: {ddpg, maddpg, cmaddpgv1, cmaddpgv2})")
    parser.add_argument("--use-common", type=bool, default=True, help = "Whether use common knowledge network.")
    parser.add_argument("--common-agents", type=float, default=10, help="K in consensus MADDPG algorithm.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="hidden layer dimension")
    parser.add_argument("--normalize-input", type=bool, default=True, help="whether to normalize input for network")
    parser.add_argument("--discrete-action", type=bool, default=False, help="whether to output discrete action")
    parser.add_argument("--beta",type=float, default = 0.5, help="Combination factor between common knowledge and maddpg")

    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise-rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--update-rate-maddpg", type=int, default=100, help="num of steps between each network update")
    parser.add_argument("--target-update-rate-maddpg", type=int, default=500, help="num of steps between each target update")
    parser.add_argument("--update-rate-consensus", type=int, default=50, help="num of steps between each consensus netwrok update")
    parser.add_argument("--target-update-rate-consensus", type=int, default=500,
                        help="num of steps between each consensus target update")


    # Replay buffer and sample
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--burnin-size", type=int, default=int(5e2), help="number of transitions to burnin")
    parser.add_argument("--batch-size", type=int, default=256, help="number of transitions to sample at each train")
    

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-num-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=25, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=True, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    parser.add_argument("--render", type=bool, default=False, help="Whether to render when evaluating")

    # Device
    parser.add_argument("--device", type = str, default = "gpu", help = "Whether use GPU. Type 'gpu' or 'cpu'")

    args = parser.parse_args()

    return args
