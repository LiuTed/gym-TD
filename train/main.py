import torch
import numpy as np
import json
import os

import gym
import gym_TD
from gym_TD import logger

import tqdm
import time

from tensorboardX import SummaryWriter

def strtime():
    return time.asctime(time.localtime(time.time()))

def game_loop(env, model, train_callback, loss_callback, writer, title, config):
    state = env.reset()
    state = torch.Tensor([state])

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
        
        next_state = torch.Tensor([next_state])
        
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


__episode_rewards = None
__episode_length = None
__last_state = None
__allow_next_move = None
def game_loop_vec(env, dummy_env, model, train_callback, loss_callback, writer, title, config):
    global __episode_rewards, __episode_length
    global __last_state, __allow_next_move

    if __episode_rewards is None:
        __episode_rewards = [[] for _ in range(len(env.env_fns))]
        __episode_length = [0 for _ in range(len(env.env_fns))]
        __allow_next_move = [True for _ in range(len(env.env_fns))]
        states = env.reset()
        states = torch.tensor(states)
    else:
        states = __last_state

    have_dones = [False for _ in range(len(env.env_fns))]
    wins = []
    total_rewards = []
    length = []
    step = 0
    losses = []

    while not all(have_dones):
        logger.debug('M', 'states: {}', states.shape)
        prob = model.get_prob(states)
        prob = torch.softmax(prob, -1)
        prob = torch.mean(prob, 0).cpu().numpy()
        for i in range(3):
            writer.add_scalars(title+'/ActionProb_{}'.format(i), {
                '0': prob[i, 0],
                '1': prob[i, 1],
                '2': prob[i, 2],
                '3': prob[i, 3]
            })
        actions = model.get_action(states)
        logger.debug('M', 'actions: {}', actions)

        for i in range(len(env.env_fns)):
            if not __allow_next_move[i]:
                actions[i] = dummy_env.empty_action()
        logger.debug('M', 'actions: {}', actions)

        next_states, rewards, dones, infos = env.step(actions)
        next_states = torch.tensor(next_states)

        if train_callback is not None:
            loss = train_callback(model, states, actions, next_states, rewards, dones, infos, writer, title, config)
            if loss is not None:
                losses.append(loss)
        
        for i, done in enumerate(dones):
            __episode_rewards[i].append(rewards[i])
            __episode_length[i] += 1
            __allow_next_move[i] = infos[i]['AllowNextMove']
            if done:
                have_dones[i] = True
                wins.append(infos[i]['Win'])

                total_rewards.append(sum(__episode_rewards[i]))
                __episode_rewards[i] = []

                length.append(__episode_length[i])
                __episode_length[i] = 0

                writer.add_scalar(title+'/TotalReward', total_rewards[-1], model.step)
                writer.add_scalar(title+'/Length', length[-1], model.step)
        
        states = next_states
        step += 1

    writer.add_scalar(title+'/AvgTotalReward', sum(total_rewards)/len(total_rewards), model.step)
    writer.add_scalar(title+'/AvgLength', sum(length)/len(length), model.step)

    __last_state = states

    info = {
        'TotalRewards': total_rewards,
        'Lengths': length,
        'Wins': wins
    }
    if loss_callback is not None and len(losses) > 0:
        info['Loss'] = loss_callback(losses, writer, title)
    
    return step, info

def train_loop(env, dummy_env, model, checkpoint, train_callback, loss_callback, writer, config):
    logger.info('M', 'train_loop: start')
    for i in tqdm.tqdm(range(1, config.total_loops+1), desc='Training', unit='epsd'):
        logger.debug('M', 'train_loop: {}: start train {}/{}', strtime(), i, config.total_loops)
        nsteps = 0
        bar = tqdm.tqdm(total=config.timesteps_per_loop, leave=False, desc='Collecting', unit='ts')
        while nsteps < config.timesteps_per_loop:
            step, info = game_loop_vec(env, dummy_env, model, train_callback, loss_callback, writer, 'Train', config)
            nsteps += step
            bar.update(step)
        del bar

        logger.debug('M', 'train_loop: {}: train finished, start test', strtime())
        wins = []
        steps = []
        rewards = []
        legal_act_ratio = []
        for _ in tqdm.tqdm(range(config.test_episode), desc = 'Testing', leave=False, unit='game'):
            step, info = game_loop(dummy_env, model, None, loss_callback, writer, 'Test', config)
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

def _get_args():
    import argparse
    parser = argparse.ArgumentParser()

    train_args = parser.add_argument_group('Training Arguments')
    train_args.add_argument('-m', '--method', default='PPO', choices=['PPO', 'SamplerPPO'], type=str, help='The training method. Default: PPO')
    train_args.add_argument('-c', '--config', default=None, type=str, help='The config file used for training.')
    train_args.add_argument('-r', '--restore', action='store_true', help='Restore from checkpoint.')
    train_args.add_argument('-s', '--checkpoint', default='./ckpt', type=str, help='The name of checkpoint. Default: ./ckpt')
    train_args.add_argument('-t', '--test', action='store_true', help='Run test.')

    env_args = parser.add_argument_group('Environment Arguments')
    env_args.add_argument('-E', '--env', default='TD-def-v0', type=str, help='The name of environment. Default: TD-def-v0')
    env_args.add_argument('--env-config', type=str, default=None, help='The config file of the environment.')
    env_args.add_argument('-S', '--map-size', default=20, type=int, help='Map size of the environment. Default: 20')
    env_args.add_argument('-e', '--seed', type=int, default=None, help='Seed of environment')
    env_args.add_argument('-o', '--difficulty', default=1, type=int, help='Opponent AI level. Default: 1')

    log_args = parser.add_argument_group('Logger Arguments')
    log_args.add_argument('-d', '--log-dir', default='./log', type=str, help='The directory of log. Default: ./log')
    log_args.add_argument('--no-render', action='store_true', help='Run without rendering.')
    log_args.add_argument('--regions', type=str, nargs='+', help='Specify information of which region(s) could be output. Default: all regions')
    verb_lv_args = log_args.add_mutually_exclusive_group()
    verb_lv_args.add_argument('-V', '--verbose', action='store_true', help='Output more information.')
    verb_lv_args.add_argument('-D', '--debug-output', action='store_true', help='Output debug information.')
    verb_lv_args.add_argument('-w', '--disable-warning', action='store_true', help='Do not output warnings.')
    verb_lv_args.add_argument('-q', '--quiet', action='store_true', help='Only output errors.')

    args = parser.parse_args()
    return args


####################################################

def _set_output(args):
    # set verbose level
    if getattr(args, 'regions', None) is None:
        logger.enable_all_region()
    else:
        for r in args.regions:
            logger.add_region(r)
    
    def disable_warning():
        import warnings
        warnings.simplefilter('ignore')


    from gym import logger as gym_logger
    if args.debug_output:
        logger.set_level(logger.DEBUG)
        gym_logger.set_level(gym_logger.DEBUG)
    elif args.verbose:
        logger.set_level(logger.FULL)
        gym_logger.set_level(gym_logger.INFO)
    elif args.quiet:
        logger.set_level(logger.ERROR)
        gym_logger.set_level(gym_logger.ERROR)
        disable_warning()
    elif args.disable_warning:
        logger.set_level(logger.INFO)
        disable_warning()
    else:
        logger.set_level(logger.INFO)

    
    try:
        logger.set_writer(tqdm.tqdm.write)
    except:
        pass

def _get_config(args):
    # read config file
    if args.config is None:
        args.config = args.method + 'Config.json'
        logger.warn('M', 'No config file specified, try using {}'.format(args.config))
    import Config
    config = Config.load_config(args.config)
    dev = Config.get_device(config)
    return config, dev

def _get_environment(args):
    # prepare environment
    from gym.wrappers import Monitor
    from gym.vector.async_vector_env import AsyncVectorEnv
    def make_fn(i):
        def closure(_i = i):
            env = gym.make(
                args.env,
                map_size = args.map_size,
                difficulty = args.difficulty,
                seed = args.seed,
                fixed_seed = (args.seed is not None)
            )
            env = Monitor(env, directory=args.log_dir + '/' + str(_i), force=True, video_callable=False if args.no_render else None)
            return env
        return closure
    env = AsyncVectorEnv([make_fn(i) for i in range(config.num_actors)])
    dummy_env = make_fn('test')()
    return env, dummy_env

def _get_model(args, env):
    # prepare model
    if args.method == 'PPO':
        import PPO
        model = PPO.Callbacks.PPO_model(env, args.env, args.map_size, config)
        train_callback = PPO.Callbacks.PPO_train
        loss_callback = PPO.Callbacks.PPO_loss_parse
    elif args.method == 'SamplerPPO':
        import SamplerPPO
        model = SamplerPPO.Callbacks.SamplerPPO_model(env, args.env, args.map_size, config)
        train_callback = SamplerPPO.Callbacks.SamplerPPO_train
        loss_callback = SamplerPPO.Callbacks.SamplerPPO_loss_parse
    # elif args.method == 'DQN':
    #     import DQN
    #     model = DQN.Callbacks.DQN_model(env, args.map_size, config)
    #     train_callback = DQN.Callbacks.DQN_train
    #     loss_callback = DQN.Callbacks.DQN_loss_parse
    
    if not os.path.isdir(args.checkpoint):
        os.mkdir(args.checkpoint)
    
    if args.restore:
        model.restore(args.checkpoint)
    elif args.test:
        logger.warn('M', 'Testing with model not restored')
    return model, train_callback, loss_callback


if __name__ == "__main__":
    args = _get_args()

    _set_output(args)

    config, dev = _get_config(args)

    # get environment config
    from gym_TD.envs import paramConfig, getConfig
    if args.env_config is not None:
        env_config = json.load(args.env_config)
        paramConfig(**env_config)

    logger.verbose('M', 'Config: {}', config)
    logger.verbose('M', 'EnvConfig: {}', getConfig())

    env, dummy_env = _get_environment(args)

    writer = SummaryWriter(args.log_dir)

    model, train_callback, loss_callback = _get_model(args, dummy_env)

    writer.add_graph(model)
    
    # start training/testing
    if args.test:
        test_loop(dummy_env, model, loss_callback, writer, config)
    else:
        train_loop(env, dummy_env, model, args.checkpoint, train_callback, loss_callback, writer, config)
