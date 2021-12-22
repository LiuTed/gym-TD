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

directory = 'gymtest-0'
config = Config.load_config('PPOConfig.json')
device = Config.get_device(config)
writer = SummaryWriter(directory+'/gymtest-result')

def env_fn(i):
    def closure(_i = i):
        # env = gym.make('LunarLander-v2')
        # env = gym.make('DistributionLearning-v0', nclass=4, discrete=True, nsample=10)
        # env = gym.make('DiskRaising-v0')
        env = gym.make('DiskRaisingM-v0')
        env = gym.wrappers.Monitor(env, directory+'/gymtest-log-{}'.format(_i), force=True)
        return env
    return closure

env = AsyncVectorEnv([env_fn(i) for i in range(config.num_actors)])

tenv = env_fn('test')()

a = Net.FullyConnected([4], [15], None, [64, 128]).to(config.device)
c = Net.FullyConnected([4], None, [1], [64, 128]).to(config.device)
ppo = Model.SamplerPPO(a, c, None, [4], [15], 0, config)

# ac = Net.FullyConnected([2], [4], [1], [64, 128]).to(config.device)
# ppo = Model.SamplerPPO(None, None, ac, [2], [4], 0, config)

rsum = [0 for _ in range(config.num_actors)]
elen = [0 for _ in range(config.num_actors)]
next_actions = [None for _ in range(config.num_actors)]
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
        cnt = np.zeros((3,5))
        actions = ppo.get_action(states)
        # for row in actions:
        #     for d in range(3):
        #         cnt[d,row[d,0]] += 1
        for act in actions:
            cnt[act // 5, act % 5] += 1
        for i in range(len(env.env_fns)):
            if next_actions[i] is not None:
                actions[i] = next_actions[i]

        logger.debug('M', 'actions: {}', actions)
        next_states, rewards, dones, infos = env.step(actions)

        next_states = torch.tensor(next_states, dtype=torch.float32)

        # for i in range(len(env.env_fns)):
        #     if infos[i]['RealAct'] != actions[i]:
        #         rewards[i] -= 0.1
        # real_acts = [info['RealAct'] for info in infos]
        # real_acts = np.asarray(real_acts)
        # ppo.record(states, real_acts, rewards, dones)

        for i in range(len(env.env_fns)):
            real_act = infos[i]['RealAct']
            if real_act != actions[i]:
                next_actions[i] = actions[i]
            # if np.any(real_act != actions[i]):
            #     next_actions[i] = np.where(real_act != actions[i], actions[i], np.zeros_like(actions[i]))
                # actions[i] = real_act
            else:
                next_actions[i] = None
        
        for d in range(3):
            prob_dict = {}
            # freq_dict = {}
            # real_act_dict = {}
            # rcnt = [0, 0, 0, 0, 0]
            # for i in range(len(env.env_fns)):
            #     real_act = infos[i]['RealAct']
            #     rcnt[real_act] += 1
            for i in range(5):
                prob_dict['E{}'.format(i)] = prob[d*5+i]
                # freq_dict['P{}'.format(i)] = cnt[d,i] / sum(cnt[d])
                # real_act_dict['P{}'.format(i)] = rcnt[i] / sum(rcnt)
            writer.add_scalars('Train/ActionProb_{}'.format(d), prob_dict, total_step)
            # writer.add_scalars('Train/ActionFreq_{}'.format(d), freq_dict, total_step)
            # writer.add_scalars('Train/RealActionFreq_{}'.format(d), real_act_dict, total_step)

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
                next_actions[i] = None
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
    writer.add_scalar('Train/MeanReward', sum(r) / len(r), ppo.step)
    writer.add_scalar('Train/MeanLength', sum(lengths)/len(lengths), ppo.step)
    
    if intv > 10000:
        tr = []
        rews = []
        for _ in range(10):
            done = False
            tstep = 0
            tstate = tenv.reset()
            tstate = torch.tensor([tstate], dtype=torch.float32)
            rew = 0
            while not done:
                a = ppo.get_action(tstate, determined=True)
                tns, r, done, __ = tenv.step(a[0])
                tstate = torch.tensor([tns], dtype=torch.float32)
                tstep += 1
                rew += r
            rews.append(rew)
            tr.append(tstep)
        writer.add_scalar('Test/MeanReward', sum(rews)/len(rews), ppo.step)
        writer.add_scalar('Test/MeanStep', sum(tr)/len(tr), ppo.step)
        intv = 0
            

del bar

