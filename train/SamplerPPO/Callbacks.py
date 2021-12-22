from gym_TD.utils import logger
import numpy as np
import torch
import time

__prob_index = 0
def SamplerPPO_action(ppo, states, infos, model_infos, writer, title, config, determined):
    global __prob_index
    actions = ppo.get_action(states, determined = determined)

    prob = ppo.get_prob(states)
    prob = torch.softmax(prob, -1)
    prob = torch.mean(prob, 0).detach().cpu().numpy()

    for i in range(prob.shape[0]): # for each road
        action_prob_dict = {}
        action_freq_dict = {}
        cnt = [0 for _ in range(5)] # for each kind of actions
        # for j in range(actions.shape[-1]): # for each sample point
        #     for k in range(len(env.env_fns)): # for each envs
        #         cnt[actions[k, i, j]] += 1
        for act in actions:
            if act == 12:
                cnt[4] += 1
            else:
                cnt[act % 4] += 1
        for j in range(4): # for each action
            action_prob_dict['{}'.format(j)] = prob[i*4+j]
        for j in range(5):
            action_freq_dict['{}'.format(j)] = cnt[j] / sum(cnt)

        writer.add_scalars(title+'/ActionProb_{}'.format(i), 
            action_prob_dict, __prob_index)
        writer.add_scalars(title+'/ActionFreq_{}'.format(i),
            action_freq_dict, __prob_index)
    __prob_index += 1
    return actions, None

def SamplerPPO_loop_fin(*args):
    pass

def SamplerPPO_train_single(ppo, state, action, next_state, reward, done, info, writer, title, config):
    # ppo.record_single(state, info['RealAction'], reward, done)
    ppo.record_single(state, action, reward, done)
    if ppo.len_trajectory % config.horizon == 0:
        ppo.flush_single(next_state)
        logger.debug('P', 'SamplerPPO_train_single: flush one trajectory')
        if ppo.num_trajectories == config.num_actors:
            logger.debug('P', 'SamplerPPO_train: start training')
            ts = time.perf_counter()
            losses = ppo.learn()
            te = time.perf_counter()
            logger.verbose('P', 'SamplerPPO_train: finish training, used {} seconds', te-ts)
            return losses
    return None
    
def SamplerPPO_train(ppo, states, actions, next_states, rewards, dones, infos, writer, title, config):
    # actions = np.asarray([info['RealAction'] for info in infos])
    ppo.record(states, actions, rewards, dones)
    if ppo.len_trajectory % config.horizon == 0:
        ppo.flush(next_states)
        logger.debug('P', 'PPO_train: flush trajectories')
        logger.debug('P', 'PPO_train: start training')
        ts = time.perf_counter()
        losses = ppo.learn()
        te = time.perf_counter()
        logger.verbose('P', 'PPO_train: finish training, used {} seconds', te-ts)
        return losses
    return None

def SamplerPPO_loss_parse(losses, writer, title):
    surr, vf, ent, mpent, ls, step = [], [], [], [], [], []
    for loss in losses:
        surr += map(lambda x: x[0], loss)
        vf += map(lambda x: x[1], loss)
        ent += map(lambda x: x[2], loss)
        mpent += map(lambda x: x[3], loss)
        ls += map(lambda x: x[4], loss)
        step += map(lambda x: x[5], loss)
    for i in range(len(surr)):
        writer.add_scalar(title+'/Surrogate', surr[i], step[i])
        writer.add_scalar(title+'/ValueFunction', vf[i], step[i])
        writer.add_scalar(title+'/Entropy', ent[i], step[i])
        writer.add_scalar(title+'/MeanProbEntropy', mpent[i], step[i])
        writer.add_scalar(title+'/Loss', ls[i], step[i])
    dict = {
        'SurrogateLoss': surr,
        'ValueFunctionLoss': vf,
        'Entropy': ent,
        'MeanProbEntropy': mpent,
        'TotalLoss': ls
    }
    return dict

def SamplerPPO_model(env, env_name, map_size, config):
    from SamplerPPO import SamplerPPO
    import Net
    if env_name.startswith("TD-def"):
        net = Net.UNet(
            env.observation_space.shape[0], 64,
            env.observation_space.shape[1], env.observation_space.shape[2],
            5, 1
        ).to(config.device)
        ppo = SamplerPPO(
            None, None, net,
            env.observation_space.shape,
            [env.action_space.n],
            0,
            config
        )
    elif env_name.startswith("TD-atk"):
        policy_shape = [env.action_space.n]
        a = Net.FCN(
            env.observation_space.shape[0],
            env.observation_space.shape[1], env.observation_space.shape[2],
            policy_shape, None, prob_channel=-1
        ).to(config.device)
        c = Net.FCN(
            env.observation_space.shape[0],
            env.observation_space.shape[1], env.observation_space.shape[2],
            None, [1]
        ).to(config.device)
        ppo = SamplerPPO(
            a, c, None,
            env.observation_space.shape,
            policy_shape,
            0,
            config
        )
    else:
        logger.error('P', 'Unknown Environment {} ({})', env_name, type(env))
    return ppo
