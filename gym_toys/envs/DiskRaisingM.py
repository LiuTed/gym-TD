import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

class DiskRaisingMultiEnv(gym.Env):
    '''
    Description:
        Multiple disks on rods are slowly moving down. You can speed some budget,
        which will grow by the time, to raise one of them a certain distance
        (or do nothing). The goal is to move all the disks to the top of the rod
        (as fast as possible). As long as the disk reaches the bottom or top of
        the rod, it will no longer moves, and will not be moved (will not cost budget).

    Parameters:
        n: int: The number of disks. Must be >= 1. Default to 3.

    Observation:
        Type: Box(shape=(n+1,), dtype=np.float32)
        Num     Observation                     Min     Max
        0       How many budget do you have.    0       1
                Increase 1/40 for each timestep
        [1,n]   Distance from disk to the       0       1
                bottom of the rod
    
    Actions:
        Type: Box(shape=(n,1), low=0, high=4, dtype=np.int64)
        Action[i, 0] is the action to the i-th disk
        Num     Action                  Cost    Raising Distance
        0       Doing nothing           0       0
        1       Slightly pushing        1/40n   0.01
        2       Pushing                 3/40n   0.04
        3       Heavily pushing         10/40n  0.2
        4       Extremely pushing       30/40n  0.8

        If the budget is not enough to use the input action, the action
        will be changed to Doing nothing (0). The actually used action
        could be obtained through info with key='RealAct'.
    
    Reward:
        1 for each disk reaches the top of the rod,
        -1 for each disk falls to the bottom of the rod.
        -0.001 for every step except the above two.

    Start State:
        Each disks will randomly start at [0.05, 0.12].
        The budget will start at 0.
    
    Episode Termination:
        All of the disks have reached either the bottom or the top of the rods.
        Episode length is greater than 1000.

        Solved Requirements:
            When the average return is greater than 0.9n
            (reach top within 100 steps) over 100 consecutive trials.
    
    Note:
        Although it is really easy to write an agent to solve this
        problem (simply using heavily pushing as long as the disk
        will not falls to the bottom), the main idea is to test the
        learning ability of the machine learning agents when the
        reward is discrete and waiting becomes important.
    '''
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frames_per_second': 24
    }
    max_cost = 40
    need = [0, 1, 3, 10, 30]
    gain = [0, 1, 4, 20, 80]
    def __init__(self, n=3) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=0., high=1., shape=(1+n,), dtype=np.float32)
        # self.action_space = spaces.Box(low=0, high=len(self.need), shape=(n,1), dtype=np.int64)
        self.action_space = spaces.Discrete(5*n)
        self.viewer = None
        self.n = n
        self.seed()
        self.reset()
    
    @property
    def state(self):
        obsv = np.empty((1+self.n,), np.float32)
        obsv[0] = self.cost / self.max_cost
        obsv[1:] = self.pos / 100
        return obsv
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        reward = 0
        idx = action // 5
        act = action % 5
        if 0 <= self.pos[idx] < 100:
            if self.cost < self.need[act] / self.n:
                act = 0
            self.cost -= self.need[act] / self.n
            self.pos[idx] += self.gain[act]
        for i in range(self.n):
            if self.dead[i]:
                continue
            self.pos[i] -= 1
            if self.pos[i] <= 0:
                self.pos[i] = 0
                # self.dead[i] = 1
                # reward -= 1
            elif self.pos[i] >= 100:
                self.pos[i] = 100
                self.dead[i] = 1
                reward += 1
        real_action = idx * 5 + act
        # real_action = action.copy()
        # for i in range(self.n):
        #     if not (0 < self.pos[i] < 100):
        #         continue
        #     act = action[i, 0]
        #     if self.cost < self.need[act] / self.n:
        #         act = 0

        #     self.cost -= self.need[act] / self.n
        #     self.pos[i] += self.gain[act] - 1

        #     if self.pos[i] <= 0:
        #         self.pos[i] = 0
        #         reward -= 1
        #     elif self.pos[i] >= 100:
        #         self.pos[i] = 100
        #         reward += 1
        #     real_action[i, 0] = act

        self.cost = min(self.cost+1, self.max_cost)

        self.nstep += 1

        done = np.all(self.pos >= 100)

        return self.state, reward, (done or self.nstep>=1000), {'RealAct': real_action}
    
    def reset(self):
        self.cost = 0
        self.pos = self.np_random.randint(8, size=self.n) + 5
        self.dead = np.zeros((self.n), dtype=np.int64)
        self.nstep = 0
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        return self.state
    
    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode="human"):
        width = 200
        height = 200
        bin_width = width * 0.9 / self.n
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(width, height)
            for i in range(self.n):
                pole = rendering.FilledPolygon([
                    (bin_width*(i+0.4), 0), (bin_width*(i+0.4), height),
                    (bin_width*(i+0.6), height), (bin_width*(i+0.6), 0)
                ])
                pole.set_color(0.8, 0.2, 0.2)
                self.viewer.add_geom(pole)
        
        for i in range(self.n):
            self.viewer.draw_polygon(
                v=[
                    (bin_width*(i+0.2), height*self.pos[i]/100), (bin_width*(i+0.2), height*(self.pos[i]+5)/100),
                    (bin_width*(i+0.8), height*(self.pos[i]+5)/100), (bin_width*(i+0.8), height*self.pos[i]/100)
                ],
                color=(0.6, 0.6, 0.6)
            )
        self.viewer.draw_polygon(
            v=[
                (width*0.9, 0), (width*0.9, height*self.cost/self.max_cost),
                (width, height*self.cost/self.max_cost), (width, 0)
            ],
            color=(0, 0.1, 0.8)
        )

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None