import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

class DiskRaisingEnv(gym.Env):
    '''
    Description:
        A disk on a rod is slowly moving down. You can speed some budget,
        which will grow by the time, to raise it a certain distance
        (or do nothing). The goal is to move the disk to the top of the rod
        (as fast as possible).

    Observation:
        Type: Box(shape=(2,), dtype=np.float32)
        Num     Observation                     Min     Max
        0       Distance from disk to the       0       1
                bottom of the rod
        1       How many budget do you have.    0       1
                Increase 0.05 for each timestep
    
    Actions:
        Type: Discrete(4)
        Num     Action                  Cost    Raising Distance
        0       Doing nothing           0       0
        1       Slightly pushing        0.05    0.01
        2       Pushing                 0.15    0.04
        3       Heavily pushing         0.5     0.2

        If the budget is not enough to use the input action, the action
        will be changed to Doing nothing (0). The actually used action
        could be obtained through info with key='RealAct'.
    
    Reward:
        1 for the step that the disk reaches the top of the rod,
        -1 for the step that the disk falls to the bottom of the rod.
        -0.001 for every step except the above two.

    Start State:
        The disk will randomly start at [0.05, 0.12].
        The budget will start at 0.
    
    Episode Termination:
        Disk reaches the bottom of the rod (height <= 0).
        Disk reaches the top of the rod (height >= 1).
        Episode length is greater than 1000.

        Solved Requirements:
            When the average return is greater than 0.93
            (reach top within 70 steps) over 100 consecutive trials.
    
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
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=0., high=1., shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.viewer = None
        self.seed()
        self.reset()
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        need = [0, 1, 3, 10]
        gain = [0, 1, 4, 20]

        if self.cost < need[action]:
            action = 0

        self.cost -= need[action]
        self.pos += gain[action] - 1

        if self.pos <= 0:
            reward = -1
            done = True
        elif self.pos >= 100:
            reward = 1
            done = True
        else:
            reward = -0.001
            done = False

        self.cost += 1

        obsv = np.empty((2,), np.float32)
        obsv[0] = self.pos / 100
        obsv[1] = self.cost / 20

        self.nstep += 1

        return obsv, reward, (done or self.nstep>=1000), {'RealAct': action}
    
    def reset(self):
        self.cost = 0
        self.pos = self.np_random.randint(8) + 5
        self.nstep = 0
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        obsv = np.empty((2,), np.float32)
        obsv[0] = self.pos / 100
        obsv[1] = self.cost / 20
        return obsv
    
    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode="human"):
        width = 200
        height = 200
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(width, height)
            pole = rendering.FilledPolygon([
                (width*0.4, 0), (width*0.4, height),
                (width*0.6, height), (width*0.6, 0)
            ])
            pole.set_color(0.8, 0.2, 0.2)
            self.viewer.add_geom(pole)
        
        self.viewer.draw_polygon(
            v=[
                (width*0.2, height*self.pos/100), (width*0.2, height*(self.pos+5)/100),
                (width*0.8, height*(self.pos+5)/100), (width*0.8, height*self.pos/100)
            ],
            color=(0.6, 0.6, 0.6)
        )
        self.viewer.draw_polygon(
            v=[
                (width*0.9, 0), (width*0.9, height*self.cost/20),
                (width, height*self.cost/20), (width, 0)
            ],
            color=(0, 0.1, 0.8)
        )

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None