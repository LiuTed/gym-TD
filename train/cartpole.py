import gym
from gym.vector.async_vector_env import AsyncVectorEnv
import Net
from SamplerPPO import Model
import Config
import torch
from tensorboardX import SummaryWriter
import tqdm
from gym_TD import logger

# logger.set_level(logger.DEBUG)
logger.enable_all_region()

config = Config.load_config('PPOConfig.json')
device = Config.get_device(config)
writer = SummaryWriter('cartpole/cartpole-result')

def env_fn(i):
    def closure(_i = i):
        env = gym.make('CartPole-v0')
        env = gym.wrappers.Monitor(env, 'cartpole/cartpole-log-{}'.format(_i), force=True)
        return env
    return closure

env = AsyncVectorEnv([env_fn(i) for i in range(config.num_actors)])

tenv = env_fn('test')()

a = Net.FullyConnected([4], [2], None, [64, 128]).to(config.device)
c = Net.FullyConnected([4], None, [1], [64, 128]).to(config.device)

ppo = Model.SamplerPPO(a, c, None, [4], [2], 0, config)

length = [0] * config.num_actors
states = env.reset()
states = torch.tensor(states, dtype=torch.float32)

bar = tqdm.tqdm(total=100000, unit='ts')
total_step = 0
intv = 0
while total_step < 100000:
    step = 0
    r = []
    while step < 300:
        actions = ppo.get_action(states)
        logger.debug('M', 'actions: {}', actions)
        next_states, rewards, dones, infos = env.step(actions)

        next_states = torch.tensor(next_states, dtype=torch.float32)

        ppo.record(states, actions, rewards, dones)
        losses = None
        if ppo.len_trajectory % config.horizon == 0:
            ppo.flush(next_states)
            losses = ppo.learn()
        
        if losses is not None:
            for surr, vf, ent, ls, _step in losses:
                writer.add_scalar('Train/Surrogate', surr, _step)
                writer.add_scalar('Train/ValueFunction', vf, _step)
                writer.add_scalar('Train/Entropy', ent, _step)
                writer.add_scalar('Train/Loss', ls, _step)
        
        for i, done in enumerate(dones):
            length[i] += 1
            if done:
                r.append(length[i])
                length[i] = 0
        
        states = next_states

        step += 1
        total_step += 1
        bar.update()
    intv += step
    
    if intv > 10000:
        tr = []
        for _ in range(10):
            done = False
            tstep = 0
            tstate = tenv.reset()
            tstate = torch.tensor([tstate], dtype=torch.float32)
            while not done:
                a = ppo.get_action(tstate, determinated=True)
                tns, rew, done, __ = tenv.step(a[0])
                tstate = torch.tensor([tns], dtype=torch.float32)
                tstep += 1
            tr.append(tstep)
        writer.add_scalar('Test/MeanReward', sum(tr)/len(tr), ppo.step)
        intv = 0
            
    writer.add_scalar('Train/MeanReward', sum(r) / len(r), ppo.step)
    
del bar

