from gym.envs.registration import register

register(
    id='TD-def-small-v0',
    entry_point='gym_TD.envs:TDSingle',
    kwargs={'map_size': 10},
    max_episode_steps=500
)

register(
    id='TD-def-middle-v0',
    entry_point='gym_TD.envs:TDSingle',
    kwargs={'map_size': 20},
    max_episode_steps=500
)

register(
    id='TD-def-large-v0',
    entry_point='gym_TD.envs:TDSingle',
    kwargs={'map_size': 30},
    max_episode_steps=500
)
