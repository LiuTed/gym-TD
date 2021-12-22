import gym
from gym.vector.async_vector_env import AsyncVectorEnv
import Net
from DQN import Model
from DQN import EpsScheduler
import Config
import torch
from tensorboardX import SummaryWriter
import tqdm
from gym_TD import logger
import numpy as np

import gym_toys

# logger.set_level(logger.DEBUG)
logger.enable_all_region()

directory = 'gymtest-dqn'
config = Config.load_config('DQNConfig.json')
device = Config.get_device(config)
writer = SummaryWriter(directory+'/gymtest-result')

def env_fn(i):
    def closure(_i = i):
        # env = gym.make('LunarLander-v2')
        # env = gym.make('DistributionLearning-v0', nclass=4, discrete=True, nsample=10)
        env = gym.make('DiskRaising-v0')
        env = gym.wrappers.Monitor(env, directory+'/gymtest-log-{}'.format(_i), force=True)
        return env
    return closure

# env = AsyncVectorEnv([env_fn(i) for i in range(config.num_actors)])
env = env_fn('0')()

tenv = env_fn('test')()

net = Net.FullyConnected([2], None, [4], [64, 128]).to(config.device)
eps_sche = EpsScheduler(1, 'Linear', lower_bound=0.1, target_steps=100000)
dqn = Model.DQN(eps_sche, 4, net, config)

bar = tqdm.tqdm(total=1000000, unit='ts')

total_steps = 0
intv = 0

while total_steps < 1000000:
    state = env.reset()
    state = torch.tensor([state], dtype=torch.float32)
    done = False
    step = 0
    r = []
    acts = []
    while not done:
        val = dqn.get_value(state).numpy()[0]
        writer.add_scalars('Train/ActionValue', {
            'V0': val[0],
            'V1': val[1],
            'V2': val[2],
            'V3': val[3]
        }, total_steps)

        action = dqn.get_action(state)
        next_state, reward, done, info = env.step(action.item())
        next_state = torch.tensor([next_state], dtype=torch.float32)

        if done:
            dqn.push([state, action, None, reward])
        else:
            dqn.push([state, action, next_state, reward])
        acts.append(action.item())
        ret = dqn.learn()
        if ret is not None:
            loss, ls = ret
            writer.add_scalar('Train/Loss', loss, total_steps)

        state = next_state
        step += 1
        total_steps += 1
        r.append(reward)
        bar.update()

        writer.add_scalar('Train/Eps', eps_sche.eps, total_steps)
        writer.add_scalar('Train/Reward', reward, total_steps)

    writer.add_scalar('Train/EpisodeReward', sum(r), total_steps)
    writer.add_scalar('Train/Length', step, total_steps)
    cnt = [0, 0, 0, 0]
    for a in acts:
        cnt[a] += 1
    writer.add_scalars('Train/ActionProb', {
        'F0': cnt[0] / sum(cnt),
        'F1': cnt[1] / sum(cnt),
        'F2': cnt[2] / sum(cnt),
        'F3': cnt[3] / sum(cnt)
    }, total_steps)

    intv += step

    if intv > 10000:
        tr = []
        tlen = []
        for _ in range(10):
            state = tenv.reset()
            state = torch.tensor([state], dtype=torch.float32)
            done = False
            step = 0
            r = []
            while not done:
                val = dqn.get_value(state).numpy()[0]

                action = dqn.get_action(state, False)
                next_state, reward, done, info = tenv.step(action.item())
                next_state = torch.tensor([next_state], dtype=torch.float32)

                state = next_state
                step += 1
                r.append(reward)
            tr.append(sum(r))
            tlen.append(step)
        writer.add_scalar('Test/MeanReward', sum(tr)/len(tr), total_steps)
        writer.add_scalar('Test/MeanLength', sum(tlen)/len(tlen), total_steps)
        intv = 0

del bar
