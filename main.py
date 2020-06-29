from utils.args import get_args
from utils.env import make_env
from maddpg.maddpg import MADDPG

from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    args = get_args()
    env, args = make_env(args)

    writer = SummaryWriter('_log')
    model = MADDPG(args, env, writer=writer)
    model.memory_burnin()
    model.train()

 
