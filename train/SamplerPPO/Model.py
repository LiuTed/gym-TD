import torch
import torch.distributions as tdist
import numpy as np
from gym_TD import logger

class SamplerPPO(object):
    def __init__(
        self,
        actor, critic, actor_critic,
        state_shape, policy_shape, len_sample,
        config
    ):
        if actor is not None and critic is not None:
            self.__actor = actor
            self.__critic = critic
            self.__actor_optimizer = torch.optim.Adam(
                self.__actor.parameters(),
                lr = config.actor_learning_rate,
                weight_decay = 0.001,
                amsgrad = True
            )
            self.__critic_optimizer = torch.optim.Adam(
                self.__critic.parameters(),
                lr = config.critic_learning_rate,
                weight_decay = 0.001,
                amsgrad = True
            )
            self.__unified = False
        elif actor_critic is not None:
            self.__actor_critic = actor_critic
            self.__optimizer = torch.optim.Adam(
                self.__actor_critic.parameters(),
                lr = config.learning_rate,
                weight_decay = 0.001,
                amsgrad = True
            )
            self.__unified = True
        else:
            raise ValueError('SamplerPPO: one of (actor, critic) or actor_critic must be specified')

        self.__config = config

        action_shape = policy_shape.copy()
        if len_sample > 0:
            action_shape[-1] = len_sample
            self.reduce_dim = False
            self.len_sample = len_sample
        else:
            action_shape = action_shape[:-1]
            self.reduce_dim = True
            self.len_sample = 1

        self.__states = np.zeros(shape=(config.horizon, config.num_actors, *state_shape), dtype=np.float32)
        self.__actions = np.zeros(shape=(config.horizon, config.num_actors, *action_shape), dtype=np.int64)
        self.__dones = np.zeros(shape=(config.horizon, config.num_actors), dtype=np.bool)
        self.__rewards = np.zeros(shape=(config.horizon, config.num_actors), dtype=np.float32)
        self.__advs = np.zeros(shape=(config.horizon, config.num_actors, 1), dtype=np.float32)
        self.__returns = np.zeros_like(self.__advs)
        self.__logp = np.zeros(shape=(config.horizon, config.num_actors, *policy_shape), dtype=np.float32)
        self.__storage_ptr = 0
        self.__storage_subptr = 0

        self.__step = 0

        logger.debug('P',
            'state: {}; policy: {}, action: {}',
            state_shape, policy_shape, action_shape
        )
    
    @property
    def step(self):
        return self.__step
    
    @property
    def len_trajectory(self):
        return self.__storage_ptr
    
    @property
    def num_trajectories(self):
        return self.__storage_subptr
    
    def restore(self, ckpt):
        d = torch.load(ckpt+'/model.pkl')
        if self.__unified:
            self.__actor_critic.load_state_dict(d['actor_critic'])
            self.__optimizer.load_state_dict(d['optim'])
        else:
            self.__actor.load_state_dict(d['actor'])
            self.__critic.load_state_dict(d['critic'])
            self.__actor_optimizer.load_state_dict(d['actor_optim'])
            self.__critic_optimizer.load_state_dict(d['critic_optim'])
        self.__step = d['step']
        logger.verbose('P', 'SamplerPPO: restored')
    
    def save(self, ckpt):
        if self.__unified:
            d = {
                'actor_critic': self.__actor_critic.state_dict(),
                'optim': self.__optimizer.state_dict(),
                'step': self.step
            }
        else:
            d = {
                'actor': self.__actor.state_dict(),
                'critic': self.__critic.state_dict(),
                'actor_optim': self.__actor_optimizer.state_dict(),
                'critic_optim': self.__critic_optimizer.state_dict(),
                'step': self.step
            }
        logger.debug('P', 'Model: {}', d)
        torch.save(d, ckpt+'/model.pkl')
        logger.verbose('P', 'SamplerPPO: saved')

    def get_action(self, state, determined=False):
        with torch.no_grad():
            prob = self.get_prob(state)
            batch_shape = prob.shape[:-1]

            if not determined:
                dist = tdist.categorical.Categorical(logits=prob)

                s = dist.sample([self.len_sample])
                s = s.reshape([s.shape[0], -1]).T

                if self.reduce_dim:
                    s = s.reshape(batch_shape)
                else:
                    s = s.reshape(batch_shape + torch.Size([self.len_sample]))
            else:
                s = torch.max(prob, -1)[1]
                if not self.reduce_dim:
                    s = s.unsqueeze(-1)
                    if self.len_sample > 1:
                        shape = s.shape
                        # logger.warn('P',
                        #     'SamplerPPO: terminated action with length > 1 may generate result unexpectedly'
                        # )
                        s = s.expand(*shape[:-1], self.len_sample)

            return s.cpu().numpy()
    
    def get_prob(self, state):
        if self.__unified:
            p, _ = self.__actor_critic(state.to(self.__config.device))
            return p
        else:
            return self.__actor(state.to(self.__config.device))

    def get_value(self, state):
        if self.__unified:
            _, v = self.__actor_critic(state.to(self.__config.device))
            return v
        else:
            return self.__critic(state.to(self.__config.device))
    
    def get_p_v(self, state):
        if self.__unified:
            return self.__actor_critic(state.to(self.__config.device))
        else:
            return self.get_prob(state), self.get_value(state)
    
    def record_single(self, state, action, reward, done):
        self.__states[self.__storage_ptr, self.__storage_subptr] = state
        self.__actions[self.__storage_ptr, self.__storage_subptr] = action
        self.__rewards[self.__storage_ptr, self.__storage_subptr] = reward
        self.__dones[self.__storage_ptr, self.__storage_subptr] = done
        self.__storage_ptr += 1
        if self.__storage_ptr == self.__config.horizon:
            self.__storage_subptr += 1
            self.__storage_ptr = 0

    def record(self, states, actions, rewards, dones):
        self.__states[self.__storage_ptr] = states
        self.__actions[self.__storage_ptr] = actions
        self.__rewards[self.__storage_ptr] = rewards
        self.__dones[self.__storage_ptr] = dones
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

            logp, v = self.get_p_v(s)
            self.__logp[:, i] = logp.cpu().numpy()
            v = v.cpu().numpy()

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

                logp, v = self.get_p_v(s)
                self.__logp[:, i] = logp.cpu().numpy()
                v = v.cpu().numpy()

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

        vf_coeff = self.__config.vf_coeff
        ent_coeff = self.__config.ent_coeff
        trunc_eps = self.__config.trunc_eps

        for _ in range(self.__config.train_epoch):
            np.random.shuffle(index)
            for start in range(0, n_record, self.__config.batch_size):
                slice = index[start: min(start+self.__config.batch_size, n_record)]

                s = torch.tensor(states[slice], device=self.__config.device)
                a = torch.tensor(actions[slice], dtype=torch.int64, device=self.__config.device)
                # a_soft = torch.empty([len(slice), *self.__actions.shape[2:-1], 4], dtype=torch.int64)
                
                # for i in range(4):
                #     a_soft[:,i] = i

                if self.reduce_dim:
                    a = a.unsqueeze(-1)
                # else:
                #     a = torch.cat([a, a_soft.to(self.__config.device)], -1)
                adv = torch.tensor(advs[slice], device=self.__config.device)
                ret = torch.tensor(returns[slice], device=self.__config.device)
                log_prob_old = torch.tensor(logp[slice], device=self.__config.device)

                adv = (adv - torch.mean(adv)) / torch.std(adv)

                if self.__unified:
                    self.__optimizer.zero_grad()
                else:
                    self.__actor_optimizer.zero_grad()
                    self.__critic_optimizer.zero_grad()

                log_prob, value = self.get_p_v(s)
                logger.debug('P', 'log_prob: {}; a: {}; adv: {}', log_prob.shape, a.shape, adv.shape)
                ratio = torch.exp(
                    torch.clip(
                        torch.sum(
                            (log_prob - log_prob_old).gather(-1, a),
                            -1
                        ),
                        None, 10
                    )
                )

                adv = adv.reshape([-1, *[1 for x in range(1, ratio.ndim)]])
                surr = torch.mean(
                    torch.minimum(
                        ratio * adv,
                        torch.clip(ratio, 1-trunc_eps, 1+trunc_eps) * adv
                    )
                )
                vf = torch.nn.functional.mse_loss(ret, value)
                mean_prob = torch.logsumexp(log_prob, 0)
                mean_prob_ent = torch.mean(
                    torch.sum(
                        - torch.exp(mean_prob) * mean_prob,
                        -1
                    )
                ) / len(slice)
                entropy = torch.mean(
                    torch.sum(
                        - torch.exp(log_prob) * log_prob,
                        -1
                    )
                )
                # log_prob_loss = torch.mean(torch.square(log_prob))

                loss = - surr + vf * vf_coeff - mean_prob_ent * ent_coeff# + 0.01 * log_prob_loss
                
                if torch.isinf(loss).item() or torch.isnan(loss).item():
                    logger.error('P', 'Loss error {} ({} {} {} {})', loss.item(), surr.item(), vf.item(), entropy.item(), mean_prob_ent.item())
                
                losses.append((surr.detach(), vf.detach(), entropy.detach(), mean_prob_ent.detach(), loss.detach(), self.step))
                loss.backward()

                if self.__unified:
                    self.__optimizer.step()
                else:
                    self.__critic_optimizer.step()
                    self.__actor_optimizer.step()

                self.__step += 1

        self.__storage_ptr = 0
        self.__storage_subptr = 0
        return losses
