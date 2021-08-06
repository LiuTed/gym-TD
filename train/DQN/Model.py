from os import setpgid
from DQN.Memory import Memory
import random
import torch
import numpy as np

class DQN(object):
    def __init__(self, eps_sche, num_act, policy_network, config, target_network=None):
        self.memory = Memory(config.memory_size)
        self.eps_scheduler = eps_sche
        
        self.num_act = num_act

        self.policy = policy_network
        self.target = target_network
        if target_network is not None:
            self.target.load_state_dict(self.policy.state_dict())
            self.target.train(False)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=0.001,
            amsgrad=True
        )

        self.config = config

        self.step = 0

    def get_action(self, state, training=True):
        r = random.random()
        if r < self.eps_scheduler.eps and training:
            return torch.tensor(np.random.randint(0, self.num_act, [1, ]))
        else:
            with torch.no_grad():
                return self.policy(state.to(self.config.device)).max(1)[1].to('cpu')
    
    def push(self, val):
        '''
        val should be [state, action, next_state, reward]
        map: [1, channels, height, width]
        action: [1,]

        net_output: [1, 4]
        rewards: 1
        '''
        self.memory.push(val)
    
    def learn(self):
        if(len(self.memory) < self.config.batch_size):
            return
        batch = self.memory.sample(self.config.batch_size)

        states = torch.cat([v[0] for v in batch], 0).to(device=self.config.device)
        next_states = torch.cat([v[2] for v in batch if v[2] is not None], 0).to(device=Param.device)
        if next_states.shape[0] == 0:
            print('!')
            return

        actions = torch.cat([v[1] for v in batch], 0).to(device=self.config.device)
        actions.unsqueeze_(1)
        # [?, 1]

        # if_nonterm_mask = []
        # for i, v in enumerate(batch):
        #     if v[2] is not None:
        #         if_nonterm_mask.append(i)
        # if_nonterm_mask = torch.tensor(if_nonterm_mask, dtype=torch.long, device=Param.device)
        if_nonterm_mask = [[v[2] is not None] for v in batch] # [?, 1]
        if_nonterm_mask = torch.tensor(if_nonterm_mask, dtype=torch.bool, device=self.config.device)
        rewards = torch.tensor([[v[3]] for v in batch], dtype=torch.float32, device=self.config.device)
        # [?, 1]

        next_action_values = torch.zeros_like(rewards).to(self.config.device)
        if self.target is not None:
            result = self.target(next_states)
        else:
            result = self.policy(next_states)
        # next_action_values.index_copy_(0, if_nonterm_mask, result.max(2)[0]) # [?, 3]
        next_action_values[if_nonterm_mask] = result.max(1)[0]

        y = rewards + next_action_values * self.config.gamma
        y = y.detach() # [?, 1]

        self.optimizer.zero_grad()
        Q_sa = self.policy(states).gather(1, actions) # [?, 1]
        # Q_sa.squeeze_(2)
        loss = torch.nn.functional.mse_loss(Q_sa, y)
        loss.backward()
        self.optimizer.step()

        self.step += 1
        if(self.target is not None and self.step % self.config.update_interval == 0):
            self.target.load_state_dict(self.policy.state_dict())
        self.eps_scheduler.update()
        return loss, self.step
    
    def save(self, ckpt):
        torch.save(self.policy.state_dict(), ckpt+'/policy.pkl')
        save_dict = {
            'step': self.step,
            'eps_step': self.eps_scheduler.step
        }
        torch.save(save_dict, ckpt+'/model.pkl')
    
    def restore(self, ckpt):
        self.policy.load_state_dict(torch.load(ckpt+'/policy.pkl'))
        save_dict = torch.load(ckpt+'/model.pkl')
        self.step = save_dict['step']
        self.eps_scheduler.step = save_dict['eps_step']