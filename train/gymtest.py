import gym
from gym.vector.async_vector_env import AsyncVectorEnv
import Net
from SamplerPPO import Model
import Config
import torch
from tensorboardX import SummaryWriter
import tqdm
from gym_TD import logger
import numpy as np

import gym_toys

# logger.set_level(logger.DEBUG)
logger.enable_all_region()

config = Config.load_config('PPOConfig.json')
device = Config.get_device(config)
writer = SummaryWriter('gymtest/gymtest-result')

def env_fn(i):
    def closure(_i = i):
        # env = gym.make('LunarLander-v2')
        # env = gym.make('DistributionLearning-v0', nclass=4, discrete=True, nsample=10)
        env = gym.make('DiskRaising-v0')
        env = gym.wrappers.Monitor(env, 'gymtest/gymtest-log-{}'.format(_i), force=True)
        return env
    return closure

env = AsyncVectorEnv([env_fn(i) for i in range(config.num_actors)])

tenv = env_fn('test')()

ac = Net.FullyConnected([2], [4], [1], [64, 128]).to(config.device)
# c = Net.FullyConnected([8], None, [1], [64, 128]).to(config.device)

ppo = Model.SamplerPPO(None, None, ac, [2], [4], 0, config)

rsum = [0 for _ in range(config.num_actors)]
elen = [0 for _ in range(config.num_actors)]
states = env.reset()
states = torch.tensor(states, dtype=torch.float32)

bar = tqdm.tqdm(total=100000, unit='ts')
total_step = 0
intv = 0

while total_step < 100000:
    step = 0
    r = []
    lengths = []
    all_dones = [False for _ in range(config.num_actors)]
    while not all(all_dones):
        prob = ppo.get_prob(states).detach()
        prob = torch.mean(torch.softmax(prob, -1), 0).cpu().numpy()
        cnt = [0, 0, 0, 0]
        actions = ppo.get_action(states)
        # for row in actions:
        #     for act in row:
        #         cnt[act] += 1
        for act in actions:
            cnt[act] += 1
        writer.add_scalars('Train/ActionProb', {
            'E0': prob[0],
            'E1': prob[1],
            'E2': prob[2],
            'E3': prob[3]
        }, total_step)
        writer.add_scalars('Train/ActionFreq', {
            'P0': cnt[0] / config.num_actors,
            'P1': cnt[1] / config.num_actors,
            'P2': cnt[2] / config.num_actors,
            'P3': cnt[3] / config.num_actors
        }, total_step)
        logger.debug('M', 'actions: {}', actions)
        next_states, rewards, dones, infos = env.step(actions)

        next_states = torch.tensor(next_states, dtype=torch.float32)

        # for i in range(len(env.env_fns)):
        #     if dones[i]:
        #         if length[i] >= 200:
        #             rewards[i] = 1
        #         else:
        #             rewards[i] = -1
        #     else:
        #         rewards[i] = 0
        # real_acts = [info['RealAct'] for info in infos]
        # real_acts = np.asarray(real_acts)

        ppo.record(states, actions, rewards, dones)
        losses = None
        if ppo.len_trajectory % config.horizon == 0:
            ppo.flush(next_states)
            losses = ppo.learn()
        
        if losses is not None:
            for surr, vf, ent, mpent, ls, _step in losses:
                writer.add_scalar('Train/Surrogate', surr, _step)
                writer.add_scalar('Train/ValueFunction', vf, _step)
                writer.add_scalar('Train/Entropy', ent, _step)
                writer.add_scalar('Train/MeanProbEntropy', mpent, _step)
                writer.add_scalar('Train/Loss', ls, _step)
        
        for i, done in enumerate(dones):
            rsum[i] += rewards[i]
            elen[i] += 1
            if done:
                all_dones[i] = True
                r.append(rsum[i])
                lengths.append(elen[i])
                rsum[i] = 0
                elen[i] = 0
        
        states = next_states

        step += 1
        total_step += 1
        bar.update()
    intv += step
    
    if intv > 10000:
        tr = []
        rews = []
        for _ in range(10):
            done = False
            tstep = 0
            tstate = tenv.reset()
            tstate = torch.tensor([tstate], dtype=torch.float32)
            while not done:
                a = ppo.get_action(tstate, determined=True)
                tns, rew, done, __ = tenv.step(a[0])
                tstate = torch.tensor([tns], dtype=torch.float32)
                tstep += 1
                rews.append(rew)
            tr.append(tstep)
        writer.add_scalar('Test/MeanReward', sum(rews)/len(rews), ppo.step)
        writer.add_scalar('Test/MeanStep', sum(tr)/len(tr), ppo.step)
        intv = 0
            
    writer.add_scalar('Train/MeanReward', sum(r) / len(r), ppo.step)
    writer.add_scalar('Train/MeanLength', sum(lengths)/len(lengths), ppo.step)

del bar

