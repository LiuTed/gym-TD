import gym

import gym_TD
from gym_TD import logger
from gym_TD.envs.TDParam import config, hyper_parameters
import gym_TD.utils.fail_code as FC

import tqdm
from tqdm import trange
import numpy as np

def td_atk_random(d, n, render_first=False):
    env = gym.make('TD-atk-v0', map_size=20, difficulty=d)
    action_space = env.action_space
    wins = []
    rwds = []

    if render_first:
        env.reset()
        env.render()
        done = False
        mem = None
        while not done:
            if mem is not None:
                act = mem
            else:
                act = np.random.randint(0, config.enemy_types+1, size=action_space.shape)

            s, r, done, info = env.step(act)
            env.render()

            if FC.IMPOSSIBLE_CLUSTER in info['FailCode']:
                mem = None
            elif FC.COST_SHORTAGE in info['FailCode']:
                mem = act
            else:
                mem = None

    for _ in trange(n, leave=False):
        env.reset()
        done = False
        mem = None
        rwd = []
        while not done:
            if mem is not None:
                act = mem
            else:
                act = np.random.randint(0, config.enemy_types+1, size=action_space.shape)

            s, r, done, info = env.step(act)

            if FC.IMPOSSIBLE_CLUSTER in info['FailCode']:
                mem = None
            elif FC.COST_SHORTAGE in info['FailCode']:
                mem = act
            else:
                mem = None
            
            if done:
                wins.append(info['Win'])
            rwd.append(r)
        rwds.append(sum(rwd))
    return wins, rwds

def td_atk_single_round_road(d, n, t, render_first=False):
    env = gym.make('TD-atk-v0', map_size=20, difficulty=d)
    action_space = env.action_space
    wins = []
    rwds = []
    num_enemy = min(config.max_cost // config.enemy_cost[t][0], hyper_parameters.max_cluster_length)
    
    if render_first:
        env.reset()
        env.render()
        done = False
        mem = None
        road = 0
        while not done:
            if mem is not None:
                act = mem
            else:
                act = np.full(action_space.shape, config.enemy_types, np.int64)
                act[road, :num_enemy] = t
                road += 1
                if road >= env.num_roads:
                    road = 0

            s, r, done, info = env.step(act)
            env.render()

            if FC.COST_SHORTAGE in info['FailCode']:
                mem = act
            else:
                mem = None


    for _ in trange(n, leave=False):
        env.reset()
        done = False
        mem = None
        road = 0
        rwd = []
        while not done:
            if mem is not None:
                act = mem
            else:
                act = np.full(action_space.shape, config.enemy_types, np.int64)
                act[road, :num_enemy] = t
                road += 1
                if road >= env.num_roads:
                    road = 0

            s, r, done, info = env.step(act)

            if FC.COST_SHORTAGE in info['FailCode']:
                mem = act
            else:
                mem = None
            
            if done:
                wins.append(info['Win'])
            rwd.append(r)
        rwds.append(sum(rwd))
    return wins, rwds

def td_multi_cross_round_road(n, et, tt, render_first=False):
    env = gym.make('TD-2p-v0', map_size=20)
    action_space = env.action_space
    wins = []
    rwds = []
    num_enemy = min(config.max_cost // config.enemy_cost[et][0], hyper_parameters.max_cluster_length)

    def atk_act(road):
        act = np.full(action_space['Attacker'].shape, config.enemy_types, np.int64)
        act[road, :num_enemy] = et
        return act
    def def_act(board):
        dp = [[r, c] for r in range(-2, 3) for c in range(-2, 3)]

        import random
        act = random.randint(0, 2)
        empty_action = board.map_size * board.map_size * 6

        if act == 0:
            roads = []

            for r in range(board.map_size):
                for c in range(board.map_size):
                    if board.map[0, r, c] == 1:
                        roads.append([r, c])
            
            random.shuffle(roads)

            for r, c in roads:
                d = dp[random.randint(0, len(dp)-1)]
                pos = [r+d[0], c+d[1]]
                if not board.is_valid_pos(pos):
                    continue
                return (tt * board.map_size + pos[0]) * board.map_size + pos[1]
        elif act == 1:
            if len(board.towers) == 0:
                return empty_action
            id = random.randint(0, len(board.towers)-1)
            loc = board.towers[id].loc
            return (4 * board.map_size + loc[0]) * board.map_size + loc[1]
        elif act == 2:
            if len(board.towers) == 0:
                return empty_action
            if random.random() > 0.01:
                return empty_action
            id = random.randint(0, len(board.towers)-1)
            loc = board.towers[id].loc
            return (5 * board.map_size + loc[0]) * board.map_size + loc[1]
        else:
            return empty_action

    if render_first:
        env.reset()
        env.render()
        done = False
        dmem = None
        amem = None
        road = 0
        while not done:
            if amem is not None:
                aact = amem
            else:
                aact = atk_act(road)
                road += 1
                if road >= env.num_roads:
                    road = 0
            if dmem is not None:
                dact = dmem
            else:
                dact = def_act(env.board)

            s, r, done, info = env.step({'Attacker': aact, 'Defender': dact})
            env.render()

            if FC.COST_SHORTAGE in info['FailCode']['Attacker']:
                amem = aact
            else:
                amem = None
            if FC.COST_SHORTAGE == info['FailCode']['Defender']:
                dmem = dact
            else:
                dmem = None
            

    for _ in trange(n, leave=False):
        env.reset()
        done = False
        dmem = None
        amem = None
        road = 0
        rwd = []
        while not done:
            if amem is not None:
                aact = amem
            else:
                aact = atk_act(road)
                road += 1
                if road >= env.num_roads:
                    road = 0
            if dmem is not None:
                dact = dmem
            else:
                dact = def_act(env.board)

            s, r, done, info = env.step({'Attacker': aact, 'Defender': dact})

            if FC.COST_SHORTAGE in info['FailCode']['Attacker']:
                amem = aact
            else:
                amem = None
            if FC.COST_SHORTAGE == info['FailCode']['Defender']:
                dmem = dact
            else:
                dmem = None
            
            if done:
                wins.append(info['Win']['Attacker'])
            rwd.append(r)
        rwds.append(sum(rwd))
    return wins, rwds


def td_combine_cross_round_road(n, et, tt, render_first=False):
    env = gym.make('TD-2p-v0', map_size=20)
    action_space = env.action_space
    wins = []
    rwds = []
    
    def atk_act(road):
        import random
        p = random.random()
        t = 0
        for i in range(len(et)):
            if p < et[i]:
                t = i
                break
            p -= et[i]
        num_enemy = min(config.max_cost // config.enemy_cost[t][0], hyper_parameters.max_cluster_length)

        act = np.full(action_space['Attacker'].shape, config.enemy_types, np.int64)
        act[road, :num_enemy] = t
        return act
    def def_act(board):
        dp = [[r, c] for r in range(-2, 3) for c in range(-2, 3)]

        import random
        act = random.randint(0, 2)
        empty_action = board.map_size * board.map_size * 6

        if act == 0:
            roads = []

            p = random.random()
            t = 0
            for i in range(len(tt)):
                if p < tt[i]:
                    t = i
                    break
                p -= tt[i]

            for r in range(board.map_size):
                for c in range(board.map_size):
                    if board.map[0, r, c] == 1:
                        roads.append([r, c])
            
            random.shuffle(roads)

            for r, c in roads:
                d = dp[random.randint(0, len(dp)-1)]
                pos = [r+d[0], c+d[1]]
                if not board.is_valid_pos(pos):
                    continue
                return (t * board.map_size + pos[0]) * board.map_size + pos[1]
        elif act == 1:
            if len(board.towers) == 0:
                return empty_action
            id = random.randint(0, len(board.towers)-1)
            loc = board.towers[id].loc
            return (4 * board.map_size + loc[0]) * board.map_size + loc[1]
        elif act == 2:
            if len(board.towers) == 0:
                return empty_action
            if random.random() > 0.01:
                return empty_action
            id = random.randint(0, len(board.towers)-1)
            loc = board.towers[id].loc
            return (5 * board.map_size + loc[0]) * board.map_size + loc[1]
        else:
            return empty_action

    if render_first:
        env.reset()
        env.render()
        done = False
        dmem = None
        amem = None
        road = 0
        while not done:
            if amem is not None:
                aact = amem
            else:
                aact = atk_act(road)
                road += 1
                if road >= env.num_roads:
                    road = 0
            if dmem is not None:
                dact = dmem
            else:
                dact = def_act(env.board)

            s, r, done, info = env.step({'Attacker': aact, 'Defender': dact})
            env.render()

            if FC.COST_SHORTAGE in info['FailCode']['Attacker']:
                amem = aact
            else:
                amem = None
            if FC.COST_SHORTAGE == info['FailCode']['Defender']:
                dmem = dact
            else:
                dmem = None
            

    for _ in trange(n, leave=False):
        env.reset()
        done = False
        dmem = None
        amem = None
        road = 0
        rwd = []
        while not done:
            if amem is not None:
                aact = amem
            else:
                aact = atk_act(road)
                road += 1
                if road >= env.num_roads:
                    road = 0
            if dmem is not None:
                dact = dmem
            else:
                dact = def_act(env.board)

            s, r, done, info = env.step({'Attacker': aact, 'Defender': dact})

            if FC.COST_SHORTAGE in info['FailCode']['Attacker']:
                amem = aact
            else:
                amem = None
            if FC.COST_SHORTAGE == info['FailCode']['Defender']:
                dmem = dact
            else:
                dmem = None
            
            if done:
                wins.append(info['Win']['Attacker'])
            rwd.append(r)
        rwds.append(sum(rwd))
    return wins, rwds

if __name__ == '__main__':
    logger.enable_all_region()
    logger.remove_region('R', 'B')
    logger.set_level(logger.DEBUG)
    logger.set_writer(tqdm.tqdm.write)

    n = 100
    render_first = False
    print(config.__dict__)
    # w, r = td_combine_cross_round_road(n, [0,0,1,0], [0,1,0,0], render_first)
    # logger.verbose('Result', 'combine: {} {}', sum(w)/len(w), sum(r)/len(r))
    # w, r = td_combine_cross_round_road(n, [0,0,1,0], [0,0.8,0,0.2], render_first)
    # logger.verbose('Result', 'combine: {} {}', sum(w)/len(w), sum(r)/len(r))
    # w, r = td_combine_cross_round_road(n, [0,0,1,0], [0,0.6,0,0.4], render_first)
    # logger.verbose('Result', 'combine: {} {}', sum(w)/len(w), sum(r)/len(r))
    for tt in reversed(range(config.tower_types)):
        for et in range(config.enemy_types):
            w, r = td_multi_cross_round_road(n, et, tt, render_first)
            logger.verbose('Result', 'cross t{} e{}: {} {}', tt, et, sum(w)/len(w), sum(r)/len(r))
        logger.verbose('Result', '-----------------')
    for d in reversed(range(3)):
        for t in range(config.enemy_types):
            w, r = td_atk_single_round_road(d, n, t, render_first)
            logger.verbose('Result', 't {} {}: {} {}', t, d, sum(w)/len(w), sum(r)/len(r))
        w, r = td_atk_random(d, n, render_first)
        logger.verbose('Result', 'r {}: {} {}', d, sum(w)/len(w), sum(r)/len(r))
        logger.verbose('Result', '-----------------')

