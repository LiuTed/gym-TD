import Config
import torch

import gym
from gym import logger, wrappers
import gym_TD

import argparse
import tqdm
import time

from tensorboardX import SummaryWriter

def strtime():
    return time.asctime(time.localtime(time.time()))

def PPO_train(ppo, state, action, next_state, reward, done, info, writer, title, config):
    ppo.record(state, action, reward, done)
    if ppo.len_trajectory % config.horizon == 0:
        ppo.flush(next_state)
        logger.debug('PPO_train: flush one trajectory')
        if ppo.n_record == config.horizon * config.num_actors:
            logger.debug('PPO_train: start training')
            ts = time.perf_counter()
            losses = ppo.learn()
            te = time.perf_counter()
            logger.info('PPO_train: finish training, used {} seconds'.format(te-ts))
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
    actor = Net.UNet(env.observation_space.shape[2], map_size, map_size).to(config.device)
    actor_old = Net.UNet(env.observation_space.shape[2], map_size, map_size).to(config.device)
    critic = Net.FCN(env.observation_space.shape[2], map_size, map_size, 1, False).to(config.device)
    ppo = PPO.Model(
        actor, actor_old, critic,
        [env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1]],
        [env.action_space.n],
        config
    )
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
    net = Net.UNet(env.observation_space.shape[2], map_size, map_size, False).to(config.device)
    dqn = DQN.Model(eps_sche, env.action_space.n, net, config)
    return dqn

def game_loop(env, model, train_callback, loss_callback, writer, title, config):
    state = env.reset()
    state = torch.Tensor([state.transpose(2, 0, 1)])

    done = False
    step = 0
    rewards = []
    actions = []
    real_actions = []
    losses = []
    win = None

    while not done:
        action = model.get_action(state)
        next_state, r, done, info = env.step(action.item())

        if done:
            next_state = None
            win = info['Win']
        else:
            next_state = torch.Tensor([next_state.transpose(2, 0, 1)])
        
        if train_callback is not None:
            loss = train_callback(model, state, action, next_state, r, done, info, writer, title, config)
            if loss is not None:
                losses.append(loss)

        state = next_state

        rewards.append(r)
        actions.append(action.item())
        real_actions.append(info['RealAction'])
        step += 1
    
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
    for i in tqdm.tqdm(range(1, config.total_loops+1), desc='Training', unit='epsd'):
        logger.debug('train_loop: {}: start train {}/{}'.format(strtime(), i, config.total_loops))
        nsteps = 0
        while nsteps < config.timesteps_per_loop:
            step, info = game_loop(env, model, train_callback, loss_callback, writer, 'Train', config)
            nsteps += step

        logger.debug('train_loop: {}: train finished, start test'.format(strtime()))
        wins = []
        steps = []
        rewards = []
        legal_act_ratio = []
        for _ in range(config.test_episode):
            step, info = game_loop(env, model, None, loss_callback, writer, 'Test', config)
            wins.append(info['Win'])
            steps.append(step)
            rewards.append(info['TotalReward'])
            legal_actions = []
            for act, ract in zip(info['Actions'], info['RealActions']):
                legal_actions.append(act == ract)
            legal_act_ratio.append(sum(legal_actions)/len(legal_actions))

        writer.add_scalar('Test/WinningRate', sum(wins)/len(wins), model.step)
        writer.add_scalar('Test/AverageEpisodeLength', sum(steps)/len(steps), model.step)
        writer.add_scalar('Test/AverageTotalReward', sum(rewards)/len(rewards), model.step)
        writer.add_scalar('Test/LegalActionRatio', sum(legal_act_ratio)/len(legal_act_ratio), model.step)

        model.save(checkpoint)
        logger.info('train_loop: model saved')

def test_loop(env, model, loss_callback, writer, config):
    logger.info('test_loop: started')
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
            legal_actions.append(act == ract)
        legal_act_ratio.append(sum(legal_actions)/len(legal_actions))

    writer.add_scalar('Test/WinningRate', sum(wins)/len(wins), model.step)
    writer.add_scalar('Test/AverageEpisodeLength', sum(steps)/len(steps), model.step)
    writer.add_scalar('Test/AverageTotalReward', sum(rewards)/len(rewards), model.step)
    writer.add_scalar('Test/LegalActionRatio', sum(legal_act_ratio)/len(legal_act_ratio), model.step) 

    logger.info('test_loop: finished')
    print(
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
    env_args.add_argument('-S', '--map-size', default=20, type=int, help='Map size of the environment')

    log_args = parser.add_argument_group('Logger Arguments')
    log_args.add_argument('-d', '--log-dir', default='./log', type=str, help='The directory of log')
    log_args.add_argument('--no-render', action='store_true', help='Run without rendering')
    verb_lv_args = log_args.add_mutually_exclusive_group()
    verb_lv_args.add_argument('-V', '--verbose', action='store_true', help='Output more information')
    verb_lv_args.add_argument('-q', '--quiet', action='store_true', help='Try not to output messages')

    args = parser.parse_args()

    # set verbose level
    if args.verbose:
        logger.set_level(logger.DEBUG)
    elif args.quiet:
        logger.set_level(logger.DISABLED)
    else:
        logger.set_level(logger.INFO)

    # read config file
    if args.config is None:
        args.config = args.method + 'Config.json'
        logger.info('No config file specified, try using {}'.format(args.config))

    config = Config.load_config(args.config)
    dev = Config.get_device(config)

    logger.debug('Config: {}'.format(config.__dict__))

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
    
    if args.restore:
        model.restore(args.checkpoint)
    elif args.test:
        logger.warn('Testing with model not restored')
    
    # start training/testing
    if args.test:
        test_loop(env, model, loss_callback, writer, config)
    else:
        train_loop(env, model, args.checkpoint, train_callback, loss_callback, writer, config)
