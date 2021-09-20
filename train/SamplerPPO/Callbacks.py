from gym_TD.utils import logger
import time

def SamplerPPO_train_single(ppo, state, action, next_state, reward, done, info, writer, title, config):
    if (action != info['RealAction']).any():
        reward -= 0.3
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
    for i, action in enumerate(actions):
        if (action != infos[i]['RealAction']).any():
            rewards[i] -= 0.3
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
            4, 1
        ).to(config.device)
        ppo = SamplerPPO(
            None, None, net,
            env.observation_space.shape,
            [env.action_space.n],
            0,
            config
        )
    elif env_name.startswith("TD-atk"):
        policy_shape = [env.action_space.shape[0], env.action_space.high+1]
        net = Net.FCN(
            env.observation_space.shape[0],
            env.observation_space.shape[1], env.observation_space.shape[2],
            policy_shape, [1], prob_channel=-1
        ).to(config.device)
        ppo = SamplerPPO(
            None, None, net,
            env.observation_space.shape,
            policy_shape,
            env.action_space.shape[1],
            config
        )
    else:
        logger.error('P', 'Unknown Environment {} ({})', env_name, type(env))
    return ppo
