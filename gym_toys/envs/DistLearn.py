import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
from scipy.special import logsumexp

class DistLearnEnv(gym.Env):
    '''
    Description:
        Given a distribution Q, the goal is to generate a new distribution P
        or a sample from the new distribution P, so that the KL divergence
        D_KL(P||Q) is as large as possible.
        The distribution Q and P are all represented using log probability.
    
    Parameters:
        nclass (int = 8):
            The number of classes.
        discrete (bool = False):
            If discrete == True, the environment will accept a sample from
            the distribution instead of the distribution itself.
        nsample (int = 100):
            The number of sample points in each sample. It only works when
            discrete == True

    Observation:
        Type: Box(shape=(nclass,), dtype=np.float32)
        Num     Observation                     Min     Max
        i       Log probability of class i      -inf    0
    
    Actions:
        discrete == True:
            Type: Box(shape=(nsample,), dtype=np.int64)
            Num     Observation                             Min     Max
            i       The class which sample point i          0       nclass-1
                    belongs to
            
            To avoid the situation of probability = 0, the estimated distribution
            will be smoothed with 1 extra sample point for each class.
        
        discrete == False:
            Same as observation.
    
    Reward:
        Reward is the KL divergence for each action, including the termination one

    State Updating:
        For each step, the distribution Q will becomes closer to the (estimated) input
        distribution P, to force the agent learning a different distribution.
        A small disturbance will also be added.
    
    Episode Termination:
        Episode length reached 1000 steps.
    '''
    metadata = {
        "render.modes": ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self, nclass = 8, discrete = False, nsample = 100) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=0., shape=(nclass,), dtype=np.float32)
        if discrete:
            self.action_space = spaces.Box(low=0, high=nclass-1, shape=(nsample,), dtype=np.int64)
        else:
            self.action_space = spaces.Box(low=-np.inf, high=0., shape=(nclass,), dtype=np.float32)
        self.nclass = nclass
        self.discrete = discrete
        self.nsample = nsample
        self.nstep = 0
        self.viewer = None
        self.seed()
        self.reset()

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.state = self.np_random.rand(self.nclass)
        self.state -= logsumexp(self.state)
        self.nstep = 0
        return self.state
    
    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # estimate and/or normalize the distribution
        if self.discrete:
            prob = np.ones((self.nclass), dtype=np.float32)
            for i in action:
                prob[i] += 1
            prob = np.log(prob / np.sum(prob))
        else:
            prob = action - logsumexp(action)
        # compute KL divergence
        KL = np.sum(np.exp(prob) * (prob - self.state))
        # generate new state
        ns = (np.exp(self.state) + np.exp(prob) * 2 + np.random.rand(4)*0.1) / 3
        ns /= np.sum(ns)
        ns = np.log(ns)
        self.state = ns
        # increase step
        self.nstep += 1
        return self.state, KL, self.nstep >= 1000, {}
    
    def render(self, mode="human"):
        width = 200
        height = 200
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(width, height)
        bar_width = width / self.nclass
        prob = np.exp(self.state)
        for i, p in enumerate(prob):
            bar_height = height * p
            left, right = i * bar_width, (i+1) * bar_width
            self.viewer.draw_polygon(
                v=[
                    (left, 0), (left, bar_height),
                    (right, bar_height), (right, 0)
                ],
                color = (0, 0, 1)
            )
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
