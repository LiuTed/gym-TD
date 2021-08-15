from gym_TD.envs import TDAttack
from gym_TD.envs import TDDefense
import Config
import torch
import numpy as np
import json
import os

import gym
from gym import wrappers
from gym import logger as gym_logger
import gym_TD
from gym_TD import logger
from gym_TD.envs import paramConfig, getConfig

import argparse
import tqdm
import time

from tensorboardX import SummaryWriter

def strtime():
    return time.asctime(time.localtime(time.time()))

def PPO_train(ppo, state, action, next_state, reward, done, info, writer, title, config):
    if (action != info['RealAction']).any():
        reward -= 0.3
    ppo.record(state, action, reward, done)
    if ppo.len_trajectory % config.horizon == 0:
        ppo.flush(next_state)
        logger.debug('M', 'PPO_train: flush one trajectory')
        if ppo.n_record == config.horizon * config.num_actors:
            logger.debug('M', 'PPO_train: start training')
            ts = time.perf_counter()
            losses = ppo.learn()
            te = time.perf_counter()
            logger.verbose('M', 'PPO_train: finish training, used {} seconds', te-ts)
            return losses
    return None

def PPO_loss_parse(losses, writer, title):
    surr, vf, ent, ls, step = [], [], [], [], []
    for loss in losses:
        surr += map(lambda x: x[0], loss)
        vf += map(lambda x: x[1], loss)
        ent += map(lambda x: x[2], loss)
        ls += map(lambda x: x[3], loss)
        step += map(lambda x: x[4], loss)
    for i in range(len(surr)):
        writer.add_scalar(title+'/Surrogate', surr[i], step[i])
        writer.add_scalar(title+'/VF', vf[i], step[i])
        writer.add_scalar(title+'/Entropy', ent[i], step[i])
        writer.add_scalar(title+'/Loss', ls[i], step[i])
    dict = {
        'SurrogateLoss': surr,
        'VFLoss': vf,
        'Entropy': ent,
        'TotalLoss': ls
    }
    return dict

def PPO_model(env, map_size, config):
    import PPO
    import Net
    if env.name == "TDDefense":
        net = Net.UNet(
            env.observation_space.shape[2], 64,
            env.observation_space.shape[0], env.observation_space.shape[1],
            4, 1
        ).to(config.device)
        ppo = PPO.Model(
            None, None, net,
            [env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1]],
            (),
            config
        )
    elif env.name == "TDAttack":
        net = Net.FCN(
            env.observation_space.shape[2],
            env.observation_space.shape[0], env.observation_space.shape[1],
            [4, *env.action_space.shape], [1]
        ).to(config.device)
        ppo = PPO.Model(
            None, None, net,
            [env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1]],
            env.action_space.shape,
            config
        )
    else:
        logger.error('M', 'Unknown Environment {} ({})', env, type(env))
    return ppo

def DQN_train(dqn, state, action, next_state, reward, done, info, writer, title, config):
    dqn.push([
        state,
        action,
        next_state,
        reward
    ])
    return dqn.learn()

def DQN_loss_parse(losses, writer, title):
    for loss, step in losses:
        writer.add_scalar(title+'/Loss', loss, step)
    return map(lambda x: x[0], losses)

def DQN_model(env, map_size, config):
    import DQN
    import Net
    eps_sche = DQN.EpsScheduler(1., 'Linear', lower_bound=0.1, target_steps=200000)
    if env.name == "TDDefense":
        net = Net.UNet(
            env.observation_space.shape[2], 64,
            map_size, map_size, None, 4, value_type='dependent'
        ).to(config.device)
    elif env.name == "TDAttack":
        net = Net.FCN(
            env.observation_space.shape[0], map_size, map_size,
            None, [4, *env.action_space.shape]
        )
    dqn = DQN.Model(eps_sche, env.action_space.n, net, config)
    return dqn

def game_loop(env, model, train_callback, loss_callback, writer, title, config):
    env.seed(0x12345)
    state = env.reset()
    state = torch.Tensor([state.transpose(2, 0, 1)])

    done = False
    step = 0
    rewards = []
    actions = []
    real_actions = []
    losses = []
    win = None
    allow_next_move = True

    while not done:
        if allow_next_move:
            action = model.get_action(state)[0]
        else:
            action = env.empty_action()
        next_state, r, done, info = env.step(action)
        
        next_state = torch.Tensor([next_state.transpose(2, 0, 1)])
        
        if train_callback is not None:
            loss = train_callback(model, state, action, next_state, r, done, info, writer, title, config)
            if loss is not None:
                losses.append(loss)

        if done:
            next_state = None
            win = info['Win']

        state = next_state

        rewards.append(r)
        if allow_next_move:
            actions.append(action)
            real_actions.append(info['RealAction'])
        step += 1
        allow_next_move = info['AllowNextMove']
    
    writer.add_scalar(title+'/Length', step, model.step)
    writer.add_scalar(title+'/TotalReward', sum(rewards), model.step)

    info = {
        'TotalReward': sum(rewards),
        'Win': win,
        'Actions': actions,
        'RealActions': real_actions
    }

    if loss_callback is not None and len(losses) > 0:
        info['Loss'] = loss_callback(losses, writer, title)

    return step, info


def train_loop(env, model, checkpoint, train_callback, loss_callback, writer, config):
    logger.info('M', 'train_loop: start')
    for i in tqdm.tqdm(range(1, config.total_loops+1), desc='Training', unit='epsd'):
        logger.debug('M', 'train_loop: {}: start train {}/{}', strtime(), i, config.total_loops)
        nsteps = 0
        bar = tqdm.tqdm(total=config.timesteps_per_loop, leave=False, desc='Collecting', unit='ts')
        while nsteps < config.timesteps_per_loop:
            step, info = game_loop(env, model, train_callback, loss_callback, writer, 'Train', config)
            nsteps += step
            bar.update(step)
        del bar

        logger.debug('M', 'train_loop: {}: train finished, start test', strtime())
        wins = []
        steps = []
        rewards = []
        legal_act_ratio = []
        for _ in tqdm.tqdm(range(config.test_episode), desc = 'Testing', leave=False, unit='game'):
            step, info = game_loop(env, model, None, loss_callback, writer, 'Test', config)
            wins.append(info['Win'])
            steps.append(step)
            rewards.append(info['TotalReward'])
            legal_actions = []
            for act, ract in zip(info['Actions'], info['RealActions']):
                legal_actions.append(np.all(act == ract))
            legal_act_ratio.append(sum(legal_actions)/len(legal_actions))

        writer.add_scalar('Test/WinningRate', sum(wins)/len(wins), model.step)
        writer.add_scalar('Test/AverageEpisodeLength', sum(steps)/len(steps), model.step)
        writer.add_scalar('Test/AverageTotalReward', sum(rewards)/len(rewards), model.step)
        writer.add_scalar('Test/LegalActionRatio', sum(legal_act_ratio)/len(legal_act_ratio), model.step)

        model.save(checkpoint)
        logger.info('M', 'train_loop: model saved')

def test_loop(env, model, loss_callback, writer, config):
    logger.info('M', 'test_loop: started')
    wins = []
    steps = []
    rewards = []
    legal_act_ratio = []
    for _ in tqdm.tqdm(range(config.test_episode), desc='Testing', unit='game'):
        step, info = game_loop(env, model, None, loss_callback, writer, 'Test', config)
        wins.append(info['Win'])
        steps.append(step)
        rewards.append(info['TotalReward'])
        legal_actions = []
        for act, ract in zip(info['Actions'], info['RealActions']):
            legal_actions.append(np.all(act == ract))
        legal_act_ratio.append(sum(legal_actions)/len(legal_actions))

    writer.add_scalar('Test/WinningRate', sum(wins)/len(wins), model.step)
    writer.add_scalar('Test/AverageEpisodeLength', sum(steps)/len(steps), model.step)
    writer.add_scalar('Test/AverageTotalReward', sum(rewards)/len(rewards), model.step)
    writer.add_scalar('Test/LegalActionRatio', sum(legal_act_ratio)/len(legal_act_ratio), model.step) 

    logger.info('M', 'test_loop: finished')
    logger.info('M',
        '''test_loop: Result:
        Winning Rate: {}
        Average Episode Length: {}
        Average Total Reward: {}
        Legal Action Ratio: {}
        '''.format(
            sum(wins)/len(wins),
            sum(steps)/len(steps),
            sum(rewards)/len(rewards),
            sum(legal_act_ratio)/len(legal_act_ratio)
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    train_args = parser.add_argument_group('Training Arguments')
    train_args.add_argument('-m', '--method', default='PPO', choices=['PPO', 'DQN'], type=str, help='The training method')
    train_args.add_argument('-c', '--config', default=None, type=str, help='The config file used for training')
    train_args.add_argument('-r', '--restore', action='store_true', help='Restore from checkpoint')
    train_args.add_argument('-s', '--checkpoint', default='./ckpt', type=str, help='The name of checkpoint')
    train_args.add_argument('-t', '--test', action='store_true', help='Run test')

    env_args = parser.add_argument_group('Environment Arguments')
    env_args.add_argument('-E', '--env', default='TD-def-v0', type=str, help='The name of environment')
    env_args.add_argument('--env-config', type=str, help='The config file of the environment')
    env_args.add_argument('-S', '--map-size', default=20, type=int, help='Map size of the environment')

    log_args = parser.add_argument_group('Logger Arguments')
    log_args.add_argument('-d', '--log-dir', default='./log', type=str, help='The directory of log')
    log_args.add_argument('--no-render', action='store_true', help='Run without rendering')
    log_args.add_argument('--regions', type=str, nargs='+', help='Specify information of which region(s) could be output')
    verb_lv_args = log_args.add_mutually_exclusive_group()
    verb_lv_args.add_argument('-V', '--verbose', action='store_true', help='Output more information')
    verb_lv_args.add_argument('-D', '--debug-output', action='store_true', help='Output debug information')
    verb_lv_args.add_argument('-q', '--quiet', action='store_true', help='Only output errors')

    args = parser.parse_args()

    # set verbose level
    if getattr(args, 'regions', None) is None:
        logger.enable_all_region()
    else:
        for r in args.regions:
            logger.add_region(r)

    if args.debug_output:
        logger.set_level(logger.DEBUG)
        gym_logger.set_level(gym_logger.DEBUG)
    elif args.verbose:
        logger.set_level(logger.FULL)
        gym_logger.set_level(gym_logger.INFO)
    elif args.quiet:
        logger.set_level(logger.ERROR)
        gym_logger.set_level(gym_logger.ERROR)
        import warnings
        warnings.simplefilter('ignore')
    else:
        logger.set_level(logger.INFO)
    
    try:
        logger.set_writer(tqdm.tqdm.write)
    except:
        pass

    # read config file
    if args.config is None:
        args.config = args.method + 'Config.json'
        logger.warn('M', 'No config file specified, try using {}'.format(args.config))

    config = Config.load_config(args.config)
    dev = Config.get_device(config)

    env_config_file = getattr(args, 'env_config', None)
    if env_config_file is not None:
        env_config = json.load(env_config_file)
        paramConfig(**env_config)

    logger.verbose('M', 'Config: {}', config)
    logger.verbose('M', 'EnvConfig: {}', getConfig())

    # prepare environment
    env = gym.make(args.env, map_size = args.map_size)
    env = wrappers.Monitor(env, directory=args.log_dir, force=True, video_callable=False if args.no_render else None)
    writer = SummaryWriter(args.log_dir)

    # prepare model
    if args.method == 'PPO':
        model = PPO_model(env, args.map_size, config)
        train_callback = PPO_train
        loss_callback = PPO_loss_parse
    elif args.method == 'DQN':
        model = DQN_model(env, args.map_size, config)
        train_callback = DQN_train
        loss_callback = DQN_loss_parse
    
    if not os.path.isdir(args.checkpoint):
        os.mkdir(args.checkpoint)
    
    if args.restore:
        model.restore(args.checkpoint)
    elif args.test:
        logger.warn('M', 'Testing with model not restored')
    
    # start training/testing
    if args.test:
        test_loop(env, model, loss_callback, writer, config)
    else:
        train_loop(env, model, args.checkpoint, train_callback, loss_callback, writer, config)
