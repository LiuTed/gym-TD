import torch
from torch._C import Value
import torch.distributions as tdist
import numpy as np
from gym_TD import logger

class CATPPO(object):
    def __init__(
        self,
        state_shape,
        nvec,
        critic,
        layers,
        config
    ):
        assert len(layers) == len(nvec)
        self.__actor = []
        self.__actor_optimizer = []
        self.__mask_gen = []
        self.__nvec = nvec
        self.__policy_shape = [np.sum(nvec)]
        self.__nlayers = len(layers)
        for layer in layers:
            actor, mask_generator = layer
            self.__mask_gen.append(mask_generator)
            self.__actor.append(actor)
            self.__actor_optimizer.append(
                torch.optim.Adam(
                    actor.parameters(),
                    lr = config.actor_learning_rate,
                    weight_decay = 0.001,
                    amsgrad = True
                )
            )
        self.__critic = critic
        self.__critic_optimizer = torch.optim.Adam(
            self.__critic.parameters(),
            lr = config.critic_learning_rate,
            weight_decay = 0.001,
            amsgrad = True
        )
        
        self.__config = config

        self.__states = np.zeros(
            shape = (config.horizon, config.num_actors, *state_shape),
            dtype = np.float32
        )
        self.__actions = np.zeros(
            shape = (config.horizon, config.num_actors, len(layers)),
            dtype = np.int64
        )
        self.__dones = np.zeros(
            shape = (config.horizon, config.num_actors),
            dtype = np.bool
        )
        self.__rewards = np.zeros(
            shape = (config.horizon, config.num_actors),
            dtype = np.float32
        )
        self.__advs = np.zeros(
            shape = (config.horizon, config.num_actors, 1),
            dtype = np.float32
        )
        self.__returns = np.zeros_like(self.__advs)
        self.__logp = np.zeros(
            shape = (config.horizon, config.num_actors, *self.__policy_shape),
            dtype = np.float32
        )
        self.__masks = np.zeros(
            shape = (config.horizon, config.num_actors, *self.__policy_shape),
            dtype = np.float32
        )
        
        self.__storage_ptr = 0
        self.__storage_subptr = 0

        self.__step = 0
    
    @property
    def step(self):
        return self.__step
    
    @property
    def nlayers(self):
        return self.__nlayers
    @property
    def nvec(self):
        return self.__nvec
    
    @property
    def len_trajectory(self):
        return self.__storage_ptr
    
    @property
    def num_trajectories(self):
        return self.__storage_subptr
    
    def restore(self, ckpt):
        sd = torch.load(ckpt+'/model.pkl')
        actors = sd['actors']
        for i in range(self.__nlayers):
            d = actors[i]
            self.__actor[i].load_state_dict(d['actor'])
            self.__actor_optimizer[i].load_state_dict(d['actor_optim'])
        self.__critic.load_state_dict(sd['critic'])
        self.__critic_optimizer.load_state_dict(sd['critic_optim'])
        self.__step = sd['step']
        logger.verbose('CP', 'CATPPO: restored')
    
    def save(self, ckpt):
        actors = []
        for i in range(self.__nlayers):
            d = {
                'actor': self.__actor[i].state_dict(),
                'actor_optim': self.__actor_optimizer[i].state_dict(),
            }
            actors.append(d)
        sd = {
            'actors': actors,
            'critic': self.__critic.state_dict(),
            'critic_optim': self.__critic_optimizer.state_dict(),
            'step': self.__step
        }
        logger.debug('CP', 'Model: {}', sd)
        torch.save(sd, ckpt+'/model.pkl')
        logger.verbose('CP', 'CATPPO: saved')

    def get_action(self, states, infos, determined=False):
        ret = []
        masks = []
        with torch.no_grad():
            last_actions = None
            for i in range(self.__nlayers):
                prob = self.get_prob(i, states)
                mask = self.__mask_gen[i](states, last_actions, i, infos)
                mask = torch.tensor(mask)
                prob += mask.to(prob.device)

                if not determined:
                    dist = tdist.categorical.Categorical(logits=prob)
                    s = dist.sample()
                else:
                    s = torch.max(prob, -1)[1]
                
                last_actions = s
                ret.append(s)
                masks.append(mask)
            return torch.stack(ret, -1).cpu().numpy(), torch.cat(masks, -1).cpu().numpy()
    
    def get_prob(self, i, state):
        return self.__actor[i](state.to(self.__config.device))

    def get_value(self, state):
        return self.__critic(state.to(self.__config.device))
    
    def record_single(self, state, action, reward, done, mask):
        self.__states[self.__storage_ptr, self.__storage_subptr] = state
        self.__rewards[self.__storage_ptr, self.__storage_subptr] = reward
        self.__dones[self.__storage_ptr, self.__storage_subptr] = done
        self.__actions[self.__storage_ptr, self.__storage_subptr] = action
        self.__masks[self.__storage_ptr, self.__storage_subptr] = mask
        self.__storage_ptr += 1
        if self.__storage_ptr == self.__config.horizon:
            self.__storage_subptr += 1
            self.__storage_ptr = 0

    def record(self, states, actions, rewards, dones, masks):
        self.__states[self.__storage_ptr] = states
        self.__rewards[self.__storage_ptr] = rewards
        self.__dones[self.__storage_ptr] = dones
        self.__actions[self.__storage_ptr] = actions
        self.__masks[self.__storage_ptr] = masks
        self.__storage_ptr += 1
        self.__storage_subptr = self.__config.num_actors
    
    def flush_single(self, next_state):
        gamma = self.__config.gamma
        lam = self.__config.lam
        i = self.__storage_subptr - 1
        with torch.no_grad():
            s = torch.tensor(self.__states[:, i])
            dones = self.__dones[:, i]
            r = self.__rewards[:, i]

            v = self.get_value(s).cpu().numpy()
            logp = []
            for j in range(self.__nlayers):
                logpj = self.get_prob(j, s)
                logp.append(logpj)
            self.__logp[:, i] = torch.cat(logp, -1).cpu().numpy()

            last_gae = 0.
            for j in reversed(range(self.__config.horizon)):
                next_nonterminal = 1.0 - dones[j]
                if j == self.__config.horizon - 1:
                    next_value = self.get_value(next_state).item()
                else:
                    next_value = v[j + 1]
                delta = r[j] + gamma * next_value * next_nonterminal - v[j]
                self.__advs[j, i] = last_gae = delta + gamma * lam * next_nonterminal * last_gae
            
            self.__returns[:, i] = self.__advs[:, i] + v

    def flush(self, next_states):
        gamma = self.__config.gamma
        lam = self.__config.lam
        with torch.no_grad():
            for i in range(self.__config.num_actors):
                s = torch.tensor(self.__states[:, i])
                dones = self.__dones[:, i]
                r = self.__rewards[:, i]

                v = self.get_value(s).cpu().numpy()
                logp = []
                for j in range(self.__nlayers):
                    logpi = self.get_prob(j, s)
                    logp.append(logpi)
                self.__logp[:, i] = torch.cat(logp, -1).cpu().numpy()

                last_gae = 0.
                for j in reversed(range(self.__config.horizon)):
                    next_nonterminal = 1.0 - dones[j]
                    if j == self.__config.horizon - 1:
                        next_value = self.get_value(next_states[i].unsqueeze(0)).item()
                    else:
                        next_value = v[j + 1]
                    delta = r[j] + gamma * next_value * next_nonterminal - v[j]
                    self.__advs[j, i] = last_gae = delta + gamma * lam * next_nonterminal * last_gae
                
                self.__returns[:, i] = self.__advs[:, i] + v

    def learn(self):
        n_record = self.__config.horizon * self.__config.num_actors
        index = list(range(n_record))

        losses = []

        states = self.__states.reshape((n_record, *self.__states.shape[2:]))
        actions = self.__actions.reshape((n_record, *self.__actions.shape[2:]))
        advs = self.__advs.reshape((n_record, 1))
        returns = self.__returns.reshape((n_record, 1))
        logp = self.__logp.reshape((n_record, *self.__logp.shape[2:]))
        masks = self.__masks.reshape((n_record, *self.__masks.shape[2:]))

        nvec_shape = tuple(self.__nvec)

        vf_coeff = self.__config.vf_coeff
        ent_coeff = self.__config.ent_coeff
        trunc_eps = self.__config.trunc_eps

        for _ in range(self.__config.train_epoch):
            np.random.shuffle(index)
            for start in range(0, n_record, self.__config.batch_size):
                slice = index[start: min(start+self.__config.batch_size, n_record)]

                s = torch.tensor(states[slice], device=self.__config.device)
                a = torch.tensor(actions[slice], dtype=torch.int64, device=self.__config.device)

                adv = torch.tensor(advs[slice], device=self.__config.device)
                ret = torch.tensor(returns[slice], device=self.__config.device)

                log_prob_old = torch.tensor(logp[slice], device=self.__config.device)
                m = torch.tensor(masks[slice], device=self.__config.device)

                adv = (adv - torch.mean(adv)) / torch.std(adv)

                surr = torch.tensor(0, dtype=torch.float32, device=self.__config.device)
                entropy = torch.tensor(0, dtype=torch.float32, device=self.__config.device)

                log_prob_old_split = torch.split(log_prob_old, nvec_shape, -1)
                mask_split = torch.split(m, nvec_shape, -1)
                
                self.__critic_optimizer.zero_grad()
                for i in range(self.__nlayers):
                    self.__actor_optimizer[i].zero_grad()
                    log_prob = self.get_prob(i, s) + mask_split[i]
                    lpold = log_prob_old_split[i] + mask_split[i]
                    ratio = torch.exp(
                        torch.clip(
                            torch.sum(
                                (log_prob - lpold).gather(-1, a[:, i: i+1]),
                                -1
                            ),
                            None, 10
                        )
                    )
                    surr += torch.mean(
                        torch.minimum(
                            ratio * adv,
                            torch.clip(ratio, 1-trunc_eps, 1+trunc_eps) * adv
                        )
                    )
                    entropy += torch.mean(
                        torch.sum(
                            - torch.exp(log_prob) * log_prob,
                            -1
                        )
                    )

                v = self.get_value(s)
                vf = torch.nn.functional.mse_loss(ret, v)

                loss = - surr + vf * vf_coeff - entropy * ent_coeff
                if torch.isinf(loss).item() or torch.isnan(loss).item():
                    logger.error('CP', 'Loss error {} ({} {} {})', loss.item(), surr.item(), vf.item(), entropy.item())
                
                losses.append((surr.detach(), vf.detach(), entropy.detach(), loss.detach(), self.step))
                loss.backward()

                for i in range(self.__nlayers):
                    self.__actor_optimizer[i].step()
                self.__critic_optimizer.step()

                self.__step += 1

        self.__storage_ptr = 0
        self.__storage_subptr = 0
        return losses
