from gym_TD.utils import logger
import numpy as np

class RetryFailedActionWrapper:
    def __init__(self) -> None:
        self.__saved_actions = {}

    def retry_failed_action_wrapper(self, action_callback):
        def inner(model, states, infos, model_infos, writer, title, config, **kwargs):
            nonlocal self
            actions, model_infos = action_callback(model, states, infos, model_infos, writer, title, config, **kwargs)

            if not title in self.__saved_actions:
                self.__saved_actions[title] = None
            saved_actions = self.__saved_actions[title]
            if saved_actions is None:
                saved_actions = [None for _ in range(states.shape[0])]
                self.__saved_actions[title] = saved_actions

            for i in range(states.shape[0]):
                if saved_actions[i] is not None:
                    logger.debug('M', 'action {}: {} -> {}', i, actions[i], saved_actions[i])
                    actions[i] = saved_actions[i]

            return actions, model_infos
        return inner

    def retry_failed_action_fin_wrapper(self, loop_fin_callback, dummy_env):
        def inner(model, states, actions, next_states, rewards, dones, infos, model_infos, writer, title, config, **kwargs):
            nonlocal self, dummy_env
            loop_fin_callback(model, states, actions, next_states, rewards, dones, infos, model_infos, writer, title, config, **kwargs)

            saved_actions = self.__saved_actions[title]
            for i in range(states.shape[0]):
                if np.isscalar(actions[i]):
                    if actions[i] != infos[i]['RealAction']:
                        logger.debug('M', 'failed {}: {} -> {}', i, actions[i], infos[i]['RealAction'])
                        saved_actions[i] = actions[i]
                    else:
                        logger.debug('M', 'success {}', i)
                        saved_actions[i] = None
                else:
                    mask = (actions[i] != infos[i]['RealAction'])
                    if np.any(mask):
                        logger.debug('M', 'failed {}: {} -> {} {}', i, actions[i], infos[i]['RealAction'], np.where(mask, actions[i], dummy_env.empty_action()))
                        saved_actions[i] = np.where(mask, actions[i], dummy_env.empty_action())
                    else:
                        logger.debug('M', 'success {}', i)
                        saved_actions[i] = None
                if dones[i]:
                    saved_actions[i] = None
            self.__saved_actions[title] = saved_actions
        return inner



