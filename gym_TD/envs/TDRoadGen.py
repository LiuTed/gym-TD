import numpy as np
from gym_TD import logger 

def create_road_v2(np_random, map_size, num_roads):
    assert 1 <= num_roads <= 3
    def inner_pos(pos):
        return pos[0] > 0 and pos[0] < map_size-1 and pos[1] > 0 and pos[1] < map_size-1
    
    center_range = [map_size // 3, (map_size * 2 + 2) // 3]
    center_point = [
        np_random.randint(low=center_range[0], high=center_range[1]),
        np_random.randint(low=center_range[0], high=center_range[1])
    ]
    
    dp = [[1, 0], [0, -1], [-1, 0], [0, 1]] # up, left, down, right
    field = np.zeros((map_size, map_size), dtype=np.int32)
    is_rotate_point = np.zeros((map_size, map_size), dtype=np.int32)
    field[center_point[0], center_point[1]] = 1

    dir = np_random.randint(4) # start direction

    logger.debug('R', 'create_road_v2: center {}, dir {}', center_point, dir)

    def find_possible_direction(pos):
        res = []
        for i, (dx, dy) in enumerate(dp):
            if field[pos[0] + dx, pos[1] + dy] == 0:
                res.append(i)
        return res

    def generate_road(start, dir):
        logger.debug('R', 'start {} dir {}', start, dir)
        pos = start.copy()
        length = 0
        road = []
        next_rotate = None
        loop = 0
        while inner_pos(pos) and loop < 100:
            loop += 1
            shape = np_random.randint(2)
            seg_length = np_random.randint(low=map_size*3//20, high=map_size//4)
            cross = False
            if shape <= 0: # line 1/2
                logger.debug('R', 'direct {}', seg_length*2)
                for _ in range(seg_length*2):
                    pos[0] += dp[dir][0]
                    pos[1] += dp[dir][1]
                    if field[pos[0], pos[1]] != 0:
                        pos[0] -= dp[dir][0]
                        pos[1] -= dp[dir][1]
                        cross = True
                        logger.debug('R', '\tcross {}', _)
                        break
                    road.append(pos.copy())
                    field[pos[0], pos[1]] = 1
                    length += 1
                    if not inner_pos(pos):
                        logger.debug('R', '\tedge {}', _+1)
                        break
            else: # rotate
                logger.debug('R', 'rotate {}', seg_length)
                for _ in range(seg_length):
                    pos[0] += dp[dir][0]
                    pos[1] += dp[dir][1]
                    if field[pos[0], pos[1]] != 0:
                        pos[0] -= dp[dir][0]
                        pos[1] -= dp[dir][1]
                        cross = True
                        logger.debug('R', '\tcross {}', _)
                        break
                    road.append(pos.copy())
                    field[pos[0], pos[1]] = 1
                    length += 1
                    if not inner_pos(pos):
                        logger.debug('R', '\tedge {}', _+1)
                        break
                if not inner_pos(pos):
                    break
                if next_rotate is not None:
                    rd = next_rotate
                    next_rotate = None
                else:
                    rd = np_random.randint(2) * 2 - 1 # left or right (-1 or 1)
                    next_rotate = - rd
                    
                logger.debug('R', '\tr {}, next {}', rd, next_rotate)
                is_rotate_point[pos[0], pos[1]] = 1
                dir = (dir + 4 + rd) % 4 # new direction
                for _ in range(seg_length):
                    pos[0] += dp[dir][0]
                    pos[1] += dp[dir][1]
                    if field[pos[0], pos[1]] != 0:
                        pos[0] -= dp[dir][0]
                        pos[1] -= dp[dir][1]
                        cross = True
                        logger.debug('R', '\tcross {}', _)
                        break
                    cross = False
                    road.append(pos.copy())
                    field[pos[0], pos[1]] = 1
                    length += 1
                    if not inner_pos(pos):
                        logger.debug('R', '\tedge {}', _+1)
                        break
            if cross:
                pd = find_possible_direction(pos)
                if len(pd) == 0:
                    logger.debug('R', 'no where to go')
                    return road, False
                else:
                    dir = pd[np_random.randint(low=0, high=len(pd))]
                    next_rotate = None
                    is_rotate_point[pos[0], pos[1]] = 1
                    logger.debug('R', 'new direction {}', dir)
                    
        if loop >= 100:
            logger.debug('R', 'failed')
            return road, False
        return road, True
    
    def clean_up(road):
        for p in road:
            field[p[0], p[1]] = 0
            is_rotate_point[p[0], p[1]] = 0

    # center point to end point
    logger.debug('R', 'create_road_v2: generating main road part 1')
    road1, succ = [], False
    while not succ:
        road1, succ = generate_road(center_point, dir)
        if not succ:
            clean_up(road1)
        else:
            if len(road1) >= map_size:
                clean_up(road1)
                logger.debug('R', 'too long')
                succ = False

    # center point to start point
    logger.debug('R', 'create_road_v2: generating main road part 2')
    road2, succ = [], False
    while not succ:
        road2, succ = generate_road(center_point, (dir+2)%4)
        if not succ:
            clean_up(road2)
        else:
            if len(road1) + len(road2) + 1 >= map_size * 2:
                clean_up(road2)
                logger.debug('R', 'too long')
                succ = False
            elif abs(road2[-1][0] - road1[-1][0]) +\
                abs(road2[-1][1] - road1[-1][1]) < map_size * 3 // 4:
                clean_up(road2)
                logger.debug('R', 'too close')
                succ = False

    road2.reverse()
    main_road = road2 + [center_point] + road1

    roads = [main_road]
    
    selectable = []
    i = 0
    while i < len(main_road):
        if not is_rotate_point[main_road[i][0], main_road[i][1]]:
            if i < len(main_road)-1 and not is_rotate_point[main_road[i+1][0], main_road[i+1][1]]:
                selectable.append((main_road[i], i))
            i += 1
        else:
            i += 2
            
    logger.debug('R', 'create_road_v2: selectable {}', selectable)

    for _ in range(1, num_roads):
        logger.debug('R', 'create_road_v2: generating road {}', _+1)
        new_road, succ = [], False
        while not succ:
            dist = np_random.randint(low=len(selectable)*2//5, high=len(selectable)*4//5)
            new_dir = np_random.randint(4)
            new_start, dist = selectable[dist]
            new_road, succ = generate_road(new_start, new_dir)
            if not succ:
                clean_up(new_road)
            else:
                if len(new_road) + len(main_road) - dist >= map_size * 2:
                    clean_up(new_road)
                    logger.debug('R', 'too long')
                    succ = False
                elif abs(new_road[-1][0] - main_road[-1][0]) +\
                    abs(new_road[-1][1] - main_road[-1][1]) < map_size * 3 // 4:
                    clean_up(new_road)
                    logger.debug('R', 'too close')
                    succ = False

        new_road.reverse()
        new_road += main_road[dist:]
        roads.append(new_road)
    
    return roads


def create_road_v1(np_random, map_size, num_roads):
    # Return a random list of points from end point to start point.
    class Point(object):
        def __init__(self, pos, dist, prev=None):
            self.pos = pos
            self.dist = dist
            self.prev = prev
    _dp = [[0,1], [0,-1], [1,0], [-1,0]]

    start_edge = np_random.randint(low=0, high=4) #start point at which edge
    start_idx = np_random.randint(low=0, high=map_size-1) #location of start point
    end_idx = np_random.randint(low=0, high=map_size-1) #location of end point
    idx2point = [
        lambda i: [0, i], #top
        lambda i: [i, map_size-1], #right
        lambda i: [map_size-1-i, 0], #left
        lambda i: [map_size-1, map_size-1-i] #bottom
    ] # map index to coordinate
    start = idx2point[start_edge](start_idx)
    end = idx2point[3-start_edge](end_idx)
    # end point at the other side of start point

    height = np_random.randint(low=1, high=100, size=(map_size, map_size))
    # height map, used for create actual map

    dist = np.full((map_size, map_size), 101*map_size*map_size)
    dist[start[0], start[1]] = 0
    queue = [Point(start, 0)]

    while len(queue) > 0:
        p = queue[0]
        if p.pos == end:
            break
        queue.remove(p)
        for i in range(4):
            cord = [
                p.pos[0] + _dp[i][0],
                p.pos[1] + _dp[i][1]
            ]
            if cord[0] < 0 or cord[0] >= map_size \
                or cord[1] < 0 or  cord[1] >= map_size:
                continue # out of range
            if p.dist != dist[p.pos[0], p.pos[1]]: #updated
                continue
            if p.dist + height[cord[0], cord[1]] \
                    < dist[cord[0], cord[1]]:
                dist[cord[0], cord[1]] = \
                    p.dist + height[cord[0], cord[1]]
                queue.append(Point(cord, dist[cord[0], cord[1]], p))
                queue.sort(key=lambda p: p.dist) #Dijkstra
    p = queue[0]
    road = []
    while p.prev is not None:
        road.append(p.pos)
        p = p.prev
    road.append(start)
    return [road]


create_road = create_road_v2
