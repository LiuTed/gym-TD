import PPO
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

def game_loop(env, ppo, writer, title, train = True):
    state = env.reset()
    state = torch.Tensor([state.transpose(2, 0, 1)])
    done = False
    step = 0
    total_reward = 0.
    win = None
    jppos = []
    lbls = []
    while not done:
        action = ppo.get_action(state)
        next_state, reward, done, info = env.step(action.item())
        if done:
            next_state = None
            win = info['Win']
        else:
            next_state = torch.Tensor([next_state.transpose(2, 0, 1)])
        ppo.record(state, action, reward)
        state = next_state
        if train:
            if done or ppo.n_record % Param.BATCH_SIZE == 0:
                JPPO, LBL = ppo.learn()
                jppos.append(JPPO)
                lbls.append(LBL)
            else:
                JPPO, LBL = None, None
        else:
            JPPO, LBL = None, None
        step += 1
        total_reward += reward
        writer.add_scalar(title+'/reward', reward, ppo.step)
        if JPPO is not None:
            writer.add_scalar(title+'/JPPO', JPPO, ppo.step)
            writer.add_scalar(title+'/LBL', LBL, ppo.step)
    writer.add_scalar(title+'/length', step, ppo.step)
    writer.add_scalar(title+'/total_reward', total_reward, ppo.step)

    if len(jppos) > 0:
        jppo_avg = sum(jppos)/len(jppos)
    else:
        jppo_avg = None
    if len(lbls) > 0:
        lbl_avg = sum(lbls)/len(lbls)
    else:
        lbl_avg = None

    return step, jppo_avg, lbl_avg, total_reward, win

def train(env_name, map_size, logdir, restore, ckpt):
    env = gym.make(env_name, map_size=map_size)
    logger.set_level(logger.INFO)
    env = wrappers.Monitor(env, directory=logdir, force=True, video_callable=False)
    writer = SummaryWriter(logdir)

    if env_name.startswith('TD-atk'):
        actor = Net.FCN(env.observation_space.shape[2], map_size, map_size, env.action_space.n).to(Param.device)
        actor_old = Net.FCN(env.observation_space.shape[2], map_size, map_size, env.action_space.n).to(Param.device)
        critic = Net.FCN(env.observation_space.shape[2], map_size, map_size, 1, False).to(Param.device)
        ppo = PPO.PPO(actor, actor_old, critic, env.action_space.n)
    elif env_name.startswith('TD-def'):
        actor = Net.UNet(env.observation_space.shape[2], map_size, map_size).to(Param.device)
        actor_old = Net.UNet(env.observation_space.shape[2], map_size, map_size).to(Param.device)
        critic = Net.FCN(env.observation_space.shape[2], map_size, map_size, 1, False).to(Param.device)
        ppo = PPO.PPO(actor, actor_old, critic, env.action_space.n)
        
    if restore:
        ppo.restore(ckpt)

    i = 0
    loop = 0
    episode_step = 0
    while i < Param.NUM_EPISODE:
        step, jppo, lbl, total_reward, win = game_loop(env, ppo, writer, 'Train')
        
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
                step, jppo, lbl, total_reward, win = game_loop(env, ppo, writer, 'Test', False)
                steps.append(step)
                rewards.append(total_reward)
                wins.append(win)
            writer.add_scalar('Test/reward_average', sum(rewards)/len(rewards), ppo.step)
            writer.add_scalar('Test/episode_length_average', sum(steps)/len(steps), ppo.step)
            writer.add_scalar('Test/winning_rate', sum(wins)/len(wins), ppo.step)
            # if avgstep >= 195.0:
            #     print('Solved!')
            #     #break
            ppo.save(ckpt)
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
