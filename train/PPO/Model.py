import torch
import numpy as np
from gym_TD import logger

class PPO(object):
    SAVING_STATES = ['__step']
    def __init__(
        self,
        actor, actor_old, critic,
        state_shape, action_shape,
        config
    ):
        self.__actor = actor
        self.__actor_old = actor_old
        self.__actor_old.train(False)
        self.__actor_old.requires_grad_(False)
        self.__actor_old.load_state_dict(self.__actor.state_dict())
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
        self.__actor.load_state_dict(torch.load(ckpt+'/actor.pkl'))
        self.__actor_old.load_state_dict(torch.load(ckpt+'/actor_old.pkl'))
        self.__critic.load_state_dict(torch.load(ckpt+'/critic.pkl'))
        d = torch.load(ckpt+'/model.pkl')
        logger.debug('P', 'PPO: {} -> {}', d, self.__dict__)
        self.__dict__.update(d)
        logger.verbose('P', 'PPO: restored')
    
    def save(self, ckpt):
        torch.save(self.__actor.state_dict(), ckpt+'/actor.pkl')
        torch.save(self.__actor_old.state_dict(), ckpt+'/actor_old.pkl')
        torch.save(self.__critic.state_dict(), ckpt+'/critic.pkl')
        torch.save({k: self.__dict__[k] for k in self.SAVING_STATES}, ckpt+'/model.pkl')
        logger.verbose('P', 'PPO: saved')

    def get_action(self, state):
        with torch.no_grad():
            return self.get_prob(state).max(1)[1].cpu().numpy()
    
    def get_prob(self, state):
        return self.__actor(state.to(self.__config.device))

    def get_value(self, state):
        return self.__critic(state.to(self.__config.device))

    def record(self, state, action, reward, done):
        self.__traj_states[self.__traj_ptr] = state
        self.__traj_actions[self.__traj_ptr] = action
        self.__traj_rewards[self.__traj_ptr] = reward
        self.__traj_dones[self.__traj_ptr] = done
        self.__traj_ptr += 1
    
    def flush(self, next_state):
        last_gae = 0.
        with torch.no_grad():
            traj_values = self.get_value(torch.Tensor(self.__traj_states)).cpu().numpy()
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

                adv = (adv - torch.mean(adv)) / torch.std(adv)

                with torch.no_grad():
                    prob_old = self.__actor_old(s) + 1e-5

                self.__actor_optimizer.zero_grad()
                self.__critic_optimizer.zero_grad()

                prob = self.__actor(s)
                value = self.__critic(s)
                logger.debug('P', 'prob: {}; a: {}', prob.shape, a.shape)
                ratio = prob.gather(1, a) / prob_old.gather(1, a)
                surr = torch.mean(
                    torch.minimum(
                        ratio * adv,
                        torch.clip(ratio, 1-self.__config.trunc_eps, 1+self.__config.trunc_eps) * adv
                    )
                )
                vf = torch.nn.functional.mse_loss(ret, value)
                entropy = torch.mean(- prob * torch.log(prob))

                loss = - surr + vf * self.__config.vf_coeff - entropy * self.__config.ent_coeff
                
                losses.append((surr.detach(), vf.detach(), entropy.detach(), loss.detach(), self.step))
                loss.backward()

                self.__critic_optimizer.step()
                self.__actor_optimizer.step()

                self.__step += 1

        self.__storage_ptr = 0
        self.__actor_old.load_state_dict(self.__actor.state_dict())
        return losses
