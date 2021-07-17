import DQN
import Net
import Param
import EpsScheduler
import gym
from gym import logger, wrappers
import gym_TD
import torch
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import time

def game_loop(env, dqn, writer, title, train = True):
    state = env.reset()
    state = torch.Tensor([state.transpose(2, 0, 1)])
    done = False
    step = 0
    loss_sum = 0.
    loss_n = 0
    total_reward = 0.
    win = None
    while not done:
        action = dqn.get_action(state)
        next_state, reward, done, info = env.step(action.item())
        if done:
            next_state = None
            win = info['Win']
        else:
            next_state = torch.Tensor([next_state.transpose(2, 0, 1)])
        dqn.push([
            state,
            action,
            next_state,
            reward
        ])
        state = next_state
        if train:
            loss = dqn.learn()
        else:
            loss = None
        step += 1
        total_reward += reward
        writer.add_scalar(title+'/reward', reward, dqn.step)
        if loss is not None:
            writer.add_scalar(title+'/loss', loss.item(), dqn.step)
            loss_sum += loss
            loss_n += 1
    writer.add_scalar(title+'/length', step, dqn.step)
    writer.add_scalar(title+'/total_reward', total_reward, dqn.step)
    if loss_sum == 0.:
        loss_avg = 0.
    else:
        loss_avg = loss_sum / loss_n
    return step, loss_avg, total_reward, win

def train(env_name, map_size, logdir, restore, ckpt):
    env = gym.make(env_name, map_size=map_size)
    logger.set_level(logger.INFO)
    env = wrappers.Monitor(env, directory=logdir, force=True, video_callable=False)
    writer = SummaryWriter(logdir)
    eps_sche = EpsScheduler.EpsScheduler(1., 'Linear', lower_bound=0.1, target_steps=200000)

    if env_name.startswith('TD-atk'):
        net = Net.FCN(map_size, map_size, env.action_space.n).to(Param.device)
        dqn = DQN.DQN(Param.MEMORY_SIZE, eps_sche, env.action_space.n, net)
    elif env_name.startswith('TD-def'):
        net = Net.UNet(env.observation_space.shape[2], map_size, map_size).to(Param.device)
        dqn = DQN.DQN(Param.MEMORY_SIZE, eps_sche, env.action_space.n, net)
        
    if restore:
        dqn.restore(ckpt)

    i = 0
    loop = 0
    episode_step = 0
    while i < Param.NUM_EPISODE:
        step, loss, total_reward, win = game_loop(env, dqn, writer, 'Train')
        writer.add_scalar('Train/eps', dqn.eps_scheduler.eps, dqn.step)
        # for idx, param in enumerate(dqn.policy.parameters()):
        #     writer.add_histogram('Train/Param%d'%idx, param.data, dqn.step)

        episode_step += step
        if(episode_step >= Param.STEPS_PER_EPISODE):
            episode_step = 0
            print('{}: episode {} finished'.format(time.asctime(time.localtime(time.time())),i))
            i += 1

        if loop % Param.DO_TEST_EVERY_LOOP == 0:
            steps = []
            rewards = []
            wins = []
            for _ in range(Param.TEST_EPISODE):
                step, loss, total_reward, win = game_loop(env, dqn, writer, 'Test', False)
                steps.append(step)
                rewards.append(total_reward)
                wins.append(win)
            writer.add_scalar('Test/reward_average', sum(rewards)/len(rewards), dqn.step)
            writer.add_scalar('Test/episode_length_average', sum(steps)/len(steps), dqn.step)
            writer.add_scalar('Test/winning_rate', sum(wins)/len(wins), dqn.step)
            # if avgstep >= 195.0:
            #     print('Solved!')
            #     #break
            dqn.save(ckpt)
        loop += 1
        
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--logdir', type=str, help='the directory for logs', default='./log')
    parser.add_argument('-c', '--checkpoint', type=str, help='the name of checkpoint', default='net.pkl')
    parser.add_argument('-r', '--restore', action='store_true', help='load checkpoint')
    parser.add_argument('-t', '--test', action='store_true', help='run test')
    parser.add_argument('-e', '--env', type=str, help='gym environment', default='TD-def-v0')
    # parser.add_argument('-N', '--net', type=str, help='network structure', default=None)

    parser.add_argument('-S', '--map-size', type=int, help='the map size used in tower defense game', default=20)

    args = parser.parse_args()

    if not args.env.startswith('TD'):
        print('Unknown environment', args.env)
        exit(1)

    if not args.test:
        train(args.env, args.map_size, args.logdir, args.restore, args.checkpoint)
