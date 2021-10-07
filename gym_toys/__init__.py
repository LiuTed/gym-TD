from gym import register

register(
    id='DistributionLearning-v0',
    entry_point='gym_toys.envs:DistLearnEnv',
    max_episode_steps=1000
)

register(
    id='DiskRaising-v0',
    entry_point='gym_toys.envs:DiskRaisingEnv',
    max_episode_steps=1000
)

