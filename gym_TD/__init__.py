from gym.envs.registration import register
from gym_TD.envs.TDParam import hyper_parameters

register(
    id='TD-def-small-v0',
    entry_point='gym_TD.envs:TDSingle',
    kwargs={'map_size': 10},
    max_episode_steps=hyper_parameters.max_episode_steps
)

register(
    id='TD-def-middle-v0',
    entry_point='gym_TD.envs:TDSingle',
    kwargs={'map_size': 20},
    max_episode_steps=hyper_parameters.max_episode_steps
)

register(
    id='TD-def-large-v0',
    entry_point='gym_TD.envs:TDSingle',
    kwargs={'map_size': 30},
    max_episode_steps=hyper_parameters.max_episode_steps
)

register(
    id='TD-def-v0',
    entry_point='gym_TD.envs:TDSingle',
    max_episode_steps=hyper_parameters.max_episode_steps
)

register(
    id='TD-atk-small-v0',
    entry_point='gym_TD.envs:TDAttack',
    kwargs={'map_size': 10},
    max_episode_steps=hyper_parameters.max_episode_steps
)

register(
    id='TD-atk-middle-v0',
    entry_point='gym_TD.envs:TDAttack',
    kwargs={'map_size': 20},
    max_episode_steps=hyper_parameters.max_episode_steps
)

register(
    id='TD-atk-large-v0',
    entry_point='gym_TD.envs:TDAttack',
    kwargs={'map_size': 30},
    max_episode_steps=hyper_parameters.max_episode_steps
)

register(
    id='TD-atk-v0',
    entry_point='gym_TD.envs:TDAttack',
    max_episode_steps=hyper_parameters.max_episode_steps
)

register(
    id='TD-2p-small-v0',
    entry_point='gym_TD.envs:TDMulti',
    kwargs={'map_size': 10},
    max_episode_steps=hyper_parameters.max_episode_steps
)

register(
    id='TD-2p-middle-v0',
    entry_point='gym_TD.envs:TDMulti',
    kwargs={'map_size': 20},
    max_episode_steps=hyper_parameters.max_episode_steps
)

register(
    id='TD-2p-large-v0',
    entry_point='gym_TD.envs:TDMulti',
    kwargs={'map_size': 30},
    max_episode_steps=hyper_parameters.max_episode_steps
)

register(
    id='TD-2p-v0',
    entry_point='gym_TD.envs:TDMulti',
    max_episode_steps=hyper_parameters.max_episode_steps
)
