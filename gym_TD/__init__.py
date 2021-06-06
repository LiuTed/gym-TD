from gym.envs.registration import register

register(
    id='TD-def-small-v0',
    entry_point='gym_TD.envs:TDSingle',
    kwargs={'map_size': 10},
    max_episode_steps=200
)

register(
    id='TD-def-middle-v0',
    entry_point='gym_TD.envs:TDSingle',
    kwargs={'map_size': 20},
    max_episode_steps=200
)

register(
    id='TD-def-large-v0',
    entry_point='gym_TD.envs:TDSingle',
    kwargs={'map_size': 30},
    max_episode_steps=200
)

register(
    id='TD-atk-small-v0',
    entry_point='gym_TD.envs:TDAttack',
    kwargs={'map_size': 10},
    max_episode_steps=200
)

register(
    id='TD-atk-middle-v0',
    entry_point='gym_TD.envs:TDAttack',
    kwargs={'map_size': 20},
    max_episode_steps=200
)

register(
    id='TD-atk-large-v0',
    entry_point='gym_TD.envs:TDAttack',
    kwargs={'map_size': 30},
    max_episode_steps=200
)

register(
    id='TD-2p-small-v0',
    entry_point='gym_TD.envs:TDMulti',
    kwargs={'map_size': 10},
    max_episode_steps=200
)

register(
    id='TD-2p-middle-v0',
    entry_point='gym_TD.envs:TDMulti',
    kwargs={'map_size': 20},
    max_episode_steps=200
)

register(
    id='TD-2p-large-v0',
    entry_point='gym_TD.envs:TDMulti',
    kwargs={'map_size': 30},
    max_episode_steps=200
)
