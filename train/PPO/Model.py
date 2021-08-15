from os import getpgid
import torch
import numpy as np
from gym_TD import logger

class PPO(object):
    def __init__(
        self,
        actor, critic, actor_critic,
        state_shape, action_shape,
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
            raise ValueError('PPO: one of (actor, critic) or actor_critic must be specified')

        self.__config = config

        self.__traj_states = np.zeros(shape=(config.horizon, *state_shape), dtype=np.float32)
        self.__traj_actions = np.zeros(shape=(config.horizon, *action_shape), dtype=np.int64)
        self.__traj_rewards = np.zeros(shape=(config.horizon), dtype=np.float32)
        self.__traj_dones = np.zeros(shape=(config.horizon), dtype=np.bool8)
        self.__traj_ptr = 0

        self.__states = np.zeros(shape=(config.horizon*config.num_actors, *state_shape), dtype=np.float32)
        self.__actions = np.zeros(shape=(config.horizon*config.num_actors, *action_shape), dtype=np.int64)
        self.__advs = np.zeros(shape=(config.horizon*config.num_actors, 1), dtype=np.float32)
        self.__returns = np.zeros_like(self.__advs)
        self.__logp = np.zeros(shape=(config.horizon*config.num_actors, 1, *action_shape), dtype=np.float32)
        self.__storage_ptr = 0

        self.__step = 0

        logger.debug('P', 'state: {}; action: {}', state_shape, action_shape)
    
    @property
    def step(self):
        return self.__step
    
    @property
    def len_trajectory(self):
        return self.__traj_ptr
    
    @property
    def n_record(self):
        return self.__storage_ptr
    
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
        logger.verbose('P', 'PPO: restored')
    
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
        logger.verbose('P', 'PPO: saved')

    def get_action(self, state):
        with torch.no_grad():
            return self.get_prob(state).max(1)[1].cpu().numpy()
    
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

    def record(self, state, action, reward, done):
        self.__traj_states[self.__traj_ptr] = state
        self.__traj_actions[self.__traj_ptr] = action
        self.__traj_rewards[self.__traj_ptr] = reward
        self.__traj_dones[self.__traj_ptr] = done
        self.__traj_ptr += 1
    
    def flush(self, next_state):
        last_gae = 0.
        with torch.no_grad():
            states = torch.tensor(self.__traj_states)
            traj_logp, traj_values = self.get_p_v(states)
            
            acts = torch.tensor(self.__traj_actions).unsqueeze(1)
            traj_logp = traj_logp.cpu().gather(1, acts).numpy()

            traj_values = traj_values.cpu().numpy()
            advs = np.zeros_like(traj_values)
            returns = np.zeros_like(advs)
            for i in reversed(range(self.len_trajectory)):
                next_nonterminal = 1.0 - self.__traj_dones[i]
                if i == self.len_trajectory - 1:
                    next_value = self.get_value(next_state).item()
                else:
                    next_value = traj_values[i + 1]
                delta = self.__traj_rewards[i] + self.__config.gamma * next_value * next_nonterminal - traj_values[i]
                advs[i] = last_gae = delta + self.__config.gamma * self.__config.lam * next_nonterminal * last_gae
            returns = advs + traj_values

        ptr0, ptr1 = self.__storage_ptr, self.__storage_ptr + self.__config.horizon
        self.__states[ptr0: ptr1] = self.__traj_states
        self.__actions[ptr0: ptr1] = self.__traj_actions
        self.__advs[ptr0: ptr1] = advs
        self.__logp[ptr0: ptr1] = traj_logp
        self.__returns[ptr0: ptr1] = returns
        
        self.__storage_ptr = ptr1
        self.__traj_ptr = 0

    def learn(self):
        index = list(range(self.n_record))

        losses = []

        for _ in range(self.__config.train_epoch):
            np.random.shuffle(index)
            for start in range(0, len(index), self.__config.batch_size):
                slice = index[start: min(start+self.__config.batch_size, self.n_record)]

                s = torch.tensor(self.__states[slice], device=self.__config.device)
                a = torch.tensor(self.__actions[slice], dtype=torch.int64, device=self.__config.device)
                a.unsqueeze_(1)
                adv = torch.tensor(self.__advs[slice], device=self.__config.device)
                ret = torch.tensor(self.__returns[slice], device=self.__config.device)
                log_prob_old = torch.tensor(self.__logp[slice], device=self.__config.device)

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
                        log_prob.gather(1, a) - log_prob_old,
                        None, 10
                    )
                )

                adv = adv.reshape([-1, *[1 for x in range(1, ratio.ndim)]])
                surr = torch.mean(
                    torch.minimum(
                        ratio * adv,
                        torch.clip(ratio, 1-self.__config.trunc_eps, 1+self.__config.trunc_eps) * adv
                    )
                )
                vf = torch.nn.functional.mse_loss(ret, value)
                entropy = torch.mean(
                    torch.sum(
                        - torch.exp(log_prob) * log_prob,
                        1
                    )
                )

                loss = - surr + vf * self.__config.vf_coeff - entropy * self.__config.ent_coeff
                
                if torch.isinf(loss).item() or torch.isnan(loss).item():
                    logger.error('P', 'Loss error {} ({} {} {})', loss.item(), surr.item(), vf.item(), entropy.item())
                
                losses.append((surr.detach(), vf.detach(), entropy.detach(), loss.detach(), self.step))
                loss.backward()

                if self.__unified:
                    self.__optimizer.step()
                else:
                    self.__critic_optimizer.step()
                    self.__actor_optimizer.step()

                self.__step += 1

        self.__storage_ptr = 0
        return losses
