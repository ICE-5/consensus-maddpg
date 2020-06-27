from utils.args import get_args
from utils.env import make_env
from maddpg.maddpg import MADDPG


if __name__ == "__main__":
    args = get_args()
    env, args = make_env(args)

    model = MADDPG(args, env)
    model.memory_burnin()
    model.train()

 
