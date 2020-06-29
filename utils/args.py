import argparse


def get_args():
    parser = argparse.ArgumentParser("Multi-agent reinforcement learning with actor-critic")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")

    # Network config
    parser.add_argument("--hidden-dim", type=int, default=64, help="hidden layer dimension")
    parser.add_argument("--normalize-input", type=bool, default=True, help="whether to normalize input for network")
    parser.add_argument("--discrete-action", type=bool, default=False, help="whether to output discrete action")

    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries, check MPE for specified info")

    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise-rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--update-rate", type=int, default=100, help="num of steps between each target update")
    parser.add_argument("--target-update-rate", type=int, default=500, help="num of steps between each target update")


    # Replay buffer and sample
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--burnin-size", type=int, default=int(1e3), help="number of transitions to burnin")
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
