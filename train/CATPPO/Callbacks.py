from gym_TD.utils import logger
import numpy as np
import torch
import time

__prob_index = 0
def CATPPO_action(cat, states, infos, model_infos, writer, title, config, **kwargs):
    global __prob_index
    actions, masks = cat.get_action(states, infos, **kwargs)
    for i in range(cat.nlayers):
        prob = cat.get_prob(i, states)
        prob = torch.softmax(prob, -1)
        prob = torch.mean(prob, 0).detach().cpu().numpy()

        prob_dict = {}
        for j in range(cat.nvec[i]):
            prob_dict['{}'.format(j)] = prob[j]
        writer.add_scalars(title+'/ActionProb_{}'.format(i),
            prob_dict, __prob_index)
    __prob_index += 1

    return actions, masks

def CATPPO_loop_fin(*args):
    pass

def CATPPO_train_single(cat, state, action, next_state, reward, done, info, model_infos, writer, title, config):
    cat.record_single(state, action, reward, done, model_infos)
    if cat.len_trajectory % config.horizon == 0:
        cat.flush_single(next_state)
        logger.debug('CP', 'CATPPO_train_single: flush one trajectory')
        if cat.num_trajectories == config.num_actors:
            logger.debug('CP', 'CATPPO_train: start training')
            ts = time.perf_counter()
            losses = cat.learn()
            te = time.perf_counter()
            logger.verbose('CP', 'CATPPO_train: finish training, used {} seconds', te-ts)
            return losses
    return None
    
def CATPPO_train(cat, states, actions, next_states, rewards, dones, infos, model_infos, writer, title, config):
    cat.record(states, actions, rewards, dones, model_infos)
    if cat.len_trajectory % config.horizon == 0:
        cat.flush(next_states)
        logger.debug('CP', 'PPO_train: flush trajectories & start training')
        ts = time.perf_counter()
        losses = cat.learn()
        te = time.perf_counter()
        logger.verbose('CP', 'PPO_train: finish training, used {} seconds', te-ts)
        return losses
    return None

def CATPPO_loss_parse(losses, writer, title):
    surr, vf, ent, ls, step = [], [], [], [], []
    for loss in losses:
        surr += map(lambda x: x[0], loss)
        vf += map(lambda x: x[1], loss)
        ent += map(lambda x: x[2], loss)
        ls += map(lambda x: x[3], loss)
        step += map(lambda x: x[4], loss)
    for i in range(len(surr)):
        writer.add_scalar(title+'/Surrogate', surr[i], step[i])
        writer.add_scalar(title+'/ValueFunction', vf[i], step[i])
        writer.add_scalar(title+'/Entropy', ent[i], step[i])
        writer.add_scalar(title+'/Loss', ls[i], step[i])
    dict = {
        'SurrogateLoss': surr,
        'ValueFunctionLoss': vf,
        'Entropy': ent,
        'TotalLoss': ls
    }
    return dict

def CATPPO_model(env, env_name, map_size, config):
    from CATPPO import CATPPO
    import Net

    if env_name.startswith("TD-atk"):
        nvec = env.action_space.nvec
        layers = []

        def atk_mask_gen_0(states, last_actions, i, infos):
            nonlocal nvec
            vms = []
            for info in infos:
                if info is None:
                    dummy_mask = np.zeros([nvec[i]], dtype=np.bool)
                    dummy_mask[0] = True
                    vms.append(dummy_mask)
                else:
                    mask = np.empty([nvec[i]], dtype=np.bool)
                    mask[0] = True
                    mask[1:] = np.any(info['ValidMask'], 1)
                    vms.append(mask)
            vms = np.vstack(vms)
            return np.where(vms, np.zeros_like(vms, dtype=np.float32), np.full_like(vms, -1e7, dtype=np.float32))
        def atk_mask_gen_i(states, last_actions, i, infos):
            nonlocal nvec
            vms = []
            for j, info in enumerate(infos):
                if info is None:
                    dummy_mask = np.zeros([nvec[i]], dtype=np.bool)
                    dummy_mask[0] = True
                    vms.append(dummy_mask)
                else:
                    mask = np.zeros([nvec[i]], dtype=np.bool)
                    mask[0] = True
                    if last_actions[j] > 0:
                        mask[1:] = info['ValidMask'][last_actions[j] - 1]
                    vms.append(mask)
            vms = np.vstack(vms)
            return np.where(vms, np.zeros_like(vms, dtype=np.float32), np.full_like(vms, -1e7, dtype=np.float32))

        for i in range(len(nvec)):
            layer = []
            layer.append(
                Net.FCN(
                    env.observation_space.shape[0],
                    env.observation_space.shape[1], env.observation_space.shape[2],
                    [nvec[i]], None, prob_channel=-1
                ).to(config.device)
            )
            if i == 0:
                layer.append(atk_mask_gen_0)
            else:
                layer.append(atk_mask_gen_i)
            layers.append(layer)
        c = Net.FCN(
            env.observation_space.shape[0],
            env.observation_space.shape[1], env.observation_space.shape[2],
            None, [1]
        ).to(config.device)
        cat = CATPPO(
            env.observation_space.shape,
            nvec,
            c,
            layers,
            config
        )
    else:
        logger.error('CP', 'Unknown Environment {} ({})', env_name, type(env))
    return cat
