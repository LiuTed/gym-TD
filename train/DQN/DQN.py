import Memory
import Param
import random
import torch
import numpy as np

class DQN(object):
    def __init__(self, capacity, eps_sche, num_act, policy_network, target_network=None):
        self.memory = Memory.Memory(capacity)
        self.eps_scheduler = eps_sche
        
        self.num_act = num_act

        self.policy = policy_network
        self.target = target_network
        if target_network is not None:
            self.target.load_state_dict(self.policy.state_dict())
            self.target.train(False)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=Param.LEARNING_RATE,
            weight_decay=0.001,
            amsgrad=True
        )
        self.step = 0

    def get_action(self, state, training=True):
        r = random.random()
        if r < self.eps_scheduler.eps and training:
            return torch.tensor(np.random.randint(0, 4, [1, ]))
        else:
            with torch.no_grad():
                return self.policy(state.to(Param.device)).max(1)[1].to('cpu')
    
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
        if(len(self.memory) < Param.BATCH_SIZE):
            return
        batch = self.memory.sample(Param.BATCH_SIZE)

        states = torch.cat([v[0] for v in batch], 0).to(device=Param.device)
        next_states = torch.cat([v[2] for v in batch if v[2] is not None], 0).to(device=Param.device)
        if next_states.shape[0] == 0:
            print('!')
            return

        actions = torch.cat([v[1] for v in batch], 0).to(device=Param.device)
        actions.unsqueeze_(1)
        # [?, 1]

        # if_nonterm_mask = []
        # for i, v in enumerate(batch):
        #     if v[2] is not None:
        #         if_nonterm_mask.append(i)
        # if_nonterm_mask = torch.tensor(if_nonterm_mask, dtype=torch.long, device=Param.device)
        if_nonterm_mask = [[v[2] is not None] for v in batch] # [?, 1]
        if_nonterm_mask = torch.tensor(if_nonterm_mask, dtype=torch.bool, device=Param.device)
        rewards = torch.tensor([[v[3]] for v in batch], dtype=torch.float32, device=Param.device)
        # [?, 1]

        next_action_values = torch.zeros_like(rewards).to(Param.device)
        if self.target is not None:
            result = self.target(next_states)
        else:
            result = self.policy(next_states)
        # next_action_values.index_copy_(0, if_nonterm_mask, result.max(2)[0]) # [?, 3]
        next_action_values[if_nonterm_mask] = result.max(1)[0]

        y = rewards + next_action_values * Param.GAMMA
        y = y.detach() # [?, 1]

        self.optimizer.zero_grad()
        Q_sa = self.policy(states).gather(1, actions) # [?, 1]
        # Q_sa.squeeze_(2)
        loss = torch.nn.functional.mse_loss(Q_sa, y)
        loss.backward()
        self.optimizer.step()

        self.step += 1
        if(self.target is not None and self.step % Param.UPDATE == 0):
            self.target.load_state_dict(self.policy.state_dict())
        self.eps_scheduler.update()
        return loss
    
    def save(self, pos):
        torch.save(self.policy.state_dict(), pos)
    
    def restore(self, pos):
        self.policy.load_state_dict(torch.load(pos))
