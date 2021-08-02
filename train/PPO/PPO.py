import Param
import torch
import numpy as np

import random

class PPO(object):
    def __init__(
        self, actor, actor_old, critic, num_act,
        lam = .5, kl_target = 0.01,
        trunc_eps = 0.2
    ):
        self.actor = actor
        self.actor_old = actor_old
        self.actor_old.train(False)
        self.actor_old.requires_grad_(False)
        self.critic = critic

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr = Param.ACTOR_LEARNING_RATE,
            weight_decay = 0.001,
            amsgrad = True
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr = Param.CRITIC_LEARNING_RATE,
            weight_decay = 0.001,
            amsgrad = True
        )

        self.lam = lam
        self.kl_target = kl_target

        self.trunc_eps = trunc_eps

        self.num_act = num_act
        self.__states = []
        self.__actions = []
        self.__rewards = []
        self.__step = 0
    
    @property
    def step(self):
        return self.__step
    
    @property
    def n_record(self):
        return len(self.__states)
    
    def restore(self, ckpt):
        self.actor.load_state_dict(torch.load(ckpt+'/actor.pkl'))
        self.actor_old.load_state_dict(torch.load(ckpt+'/actor_old.pkl'))
        self.critic.load_state_dict(torch.load(ckpt+'/critic.pkl'))
    
    def save(self, ckpt):
        torch.save(self.actor.state_dict(), ckpt+'/actor.pkl')
        torch.save(self.actor_old.state_dict(), ckpt+'/actor_old.pkl')
        torch.save(self.critic.state_dict(), ckpt+'/critic.pkl')

    def get_action(self, state):
        with torch.no_grad():
            return self.get_prob(state).max(1)[1]
    
    def get_prob(self, state):
        return self.actor(state.to(Param.device))

    def get_value(self, state):
        return self.critic(state.to(Param.device))

    def record(self, state, action, reward):
        self.__states.append(state)
        self.__actions.append(action)
        self.__rewards.append(reward)

    def learn(self, last):
        if len(self.__states) < 1:
            return None, None
        with torch.no_grad():
            v = self.get_value(last)
            discount_r = []
            for r in reversed(self.__rewards):
                v = r + Param.GAMMA * v
                discount_r.append(v)
            discount_r.reverse()
            s = torch.cat(self.__states, 0).to(device=Param.device)
            a = torch.cat(self.__actions, 0).to(device=Param.device)
            a.unsqueeze_(1)
            r = torch.cat(discount_r, 0).to(device=Param.device)

        self.actor_old.load_state_dict(self.actor.state_dict())

        if Param.PPO_VERSION == 1:
            prob_old = self.actor_old(s).detach() + 1e-3
            for _ in range(Param.ACTOR_UPDATE_LOOP):
                self.actor_optimizer.zero_grad()
                prob = self.actor(s)
                ratio = prob.gather(1, a) / prob_old.gather(1, a)
                kl = torch.nn.functional.kl_div(prob_old, prob)
                JPPO = -torch.mean(ratio * r - self.lam * kl)
                JPPO.backward()
                self.actor_optimizer.step()
                if kl > 4 * self.kl_target:
                    break
            for _ in range(Param.CRITIC_UPDATE_LOOP):
                self.critic_optimizer.zero_grad()
                v = self.critic(s)
                LBL = torch.nn.functional.mse_loss(r, v)
                LBL.backward()
                self.critic_optimizer.step()
            prob = self.actor(s)
            prob_old = self.actor_old(s)
            kl = torch.nn.functional.kl_div(prob_old, prob)
            if kl > 1.5 * self.kl_target:
                self.lam *= 2
            elif kl < self.kl_target / 1.5:
                self.lam /= 2
            self.lam = np.clip(self.lam, 1e-4, 10)

        elif Param.PPO_VERSION == 2:
            prob_old = self.actor_old(s).detach() + 1e-3
            for _ in range(Param.ACTOR_UPDATE_LOOP):
                self.actor_optimizer.zero_grad()
                prob = self.actor(s)
                ratio = prob.gather(1, a) / prob_old.gather(1, a)
                JPPO = -torch.mean(torch.minimum(
                    ratio * r,
                    torch.clip(ratio, 1-self.trunc_eps, 1+self.trunc_eps) * r
                ))
                JPPO.backward()
                self.actor_optimizer.step()
            for _ in range(Param.CRITIC_UPDATE_LOOP):
                self.critic_optimizer.zero_grad()
                v = self.critic(s)
                LBL = torch.nn.functional.mse_loss(r, v)
                LBL.backward()
                self.critic_optimizer.step()
        
        self.__states = []
        self.__actions = []
        self.__rewards = []
        self.__step += 1

        return JPPO, LBL

