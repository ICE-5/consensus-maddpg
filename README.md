# Consensus-MADDPG

This is a pytorch implementation of MADDPG on [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs), the corresponding paper of MADDPG is [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275).

## Requirements

- python=3.6.5
- [Multi-Agent Particle Environment(MPE)](https://github.com/openai/multiagent-particle-envs)
- torch=1.1.0

## Quick Start

```shell
$ python main.py --scenario-name=simple_tag --evaluate-episodes=10
```

Directly run the main.py, then the algrithm will be tested on scenario 'simple_tag' for 10 episodes, using the pretrained model.

## Note

+ There are 4 agents in simple_tag, including 3 predators and 1 prey. we use MADDPG to train predators to catch the prey. The prey's action can be controlled by you, in our case we set it random. 

+ The default setting of Multi-Agent Particle Environment(MPE) is sparse reward, you can change it to dense reward by replacing 'shape=False' to 'shape=True' in file multiagent-particle-envs/multiagent/scenarios/simple_tag.py/.

+ If pop out no module found error, check whether PYTHONPATH is /home/haotb/DRL/MADDPG:/home/haotb/DRL/MADDPG/multiagent-particle-envs.
