import numpy as np

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
    field[center_point[0], center_point[1]] = 1

    dir = np_random.randint(4) # start direction

    def generate_road(start, dir):
        pos = start.copy()
        length = 0
        road = []
        next_rotate = None
        loop = 0
        while inner_pos(pos) and loop < 1000:
            loop += 1
            shape = np_random.randint(2)
            seg_length = np_random.randint(low=map_size*3//20, high=map_size//4)
            if shape <= 0: # line 1/2
                for _ in range(seg_length*2):
                    pos[0] += dp[dir][0]
                    pos[1] += dp[dir][1]
                    if field[pos[0], pos[1]] != 0:
                        pos[0] -= dp[dir][0]
                        pos[1] -= dp[dir][1]
                        break
                    road.append(pos.copy())
                    field[pos[0], pos[1]] = 1
                    length += 1
                    if not inner_pos(pos):
                        break
            else: # rotate
                for _ in range(seg_length):
                    pos[0] += dp[dir][0]
                    pos[1] += dp[dir][1]
                    if field[pos[0], pos[1]] != 0:
                        pos[0] -= dp[dir][0]
                        pos[1] -= dp[dir][1]
                        break
                    road.append(pos.copy())
                    field[pos[0], pos[1]] = 1
                    length += 1
                    if not inner_pos(pos):
                        break
                if not inner_pos(pos):
                    break
                if next_rotate is not None:
                    rd = next_rotate
                    next_rotate = None
                else:
                    rd = np_random.randint(2) * 2 - 1 # left or right (-1 or 1)
                    next_rotate = - rd
                dir = (dir + 4 + rd) % 4 # new direction
                for _ in range(seg_length):
                    pos[0] += dp[dir][0]
                    pos[1] += dp[dir][1]
                    if field[pos[0], pos[1]] != 0:
                        pos[0] -= dp[dir][0]
                        pos[1] -= dp[dir][1]
                        break
                    road.append(pos.copy())
                    field[pos[0], pos[1]] = 1
                    length += 1
                    if not inner_pos(pos):
                        break
        if loop >= 1000:
            return None
        return road

    # center point to end point
    road1 = generate_road(center_point, dir)
    # center point to start point
    road2 = generate_road(center_point, (dir+2)%4)

    road2.reverse()
    main_road = road2 + [center_point] + road1

    roads = [main_road]

    for _ in range(1, num_roads):
        dist = np_random.randint(low=len(main_road)*2//5, high=len(main_road)*4//5)
        new_dir = np_random.randint(4)
        new_road = generate_road(main_road[dist], new_dir)
        while new_road is None:
            dist = np_random.randint(low=len(main_road)*2//5, high=len(main_road)*4//5)
            new_dir = np_random.randint(4)
            new_road = generate_road(main_road[dist], new_dir)
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
