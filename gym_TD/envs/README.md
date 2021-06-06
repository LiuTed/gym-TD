# gym-TD/envs

## Introduction

Here is the implementation of gym-TD. Hope this document could help you read the source code.

## TDParam

Here stores all the parameters that could be modified. Some descriptions and the default values of initial parameters are shown below:
- `enemy_balance_LP`: 10
- `enemy_balance_speed`: 2
- `enemy_balance_cost`: 3
- `enemy_strong_LP`: 20
- `enemy_strong_speed`: 1
- `enemy_strong_cost`: 3
- `enemy_fast_LP`: 5
- `enemy_fast_speed`: 4
- `enemy_fast_cost`: 3
- `attacker_init_cost`: 10
- `defender_init_cost`: 5
- `base_LP`: How many enemies can be leaked. Set to `None` means never ends because of leakage. Default is 10
- `max_cost`: The upper limit of costs. Default is 50
- `tower_basic_cost`: 5
- `tower_destruct_return`: The ratio of return costs. Default is 0.5.
- `tower_basic_ATK`: 5
- `tower_ATKUp_list`: A list of the differences of ATK between levels. Default is [5, 5]
- `tower_ATKUp_cost`: A list of the costs of ATK upgrade. Default is [5, 5]
- `tower_basic_range`: 4
- `tower_rangeUp_list`: A list of the differences of attack range between levels. Default is [2, 2]
- `tower_rangeUp_cost`: A list of the costs of attack range upgrade. Default is [3, 3]
- `reward_kill`: Reward of killing an enemy. Default is 0.1
- `penalty_leak`: Penalty of leaking an enemy. (Enemy enters the base point). Default is 1.
- `reward_time`: Reward of each step. Default is 0.001
- **Note**: The rewards and penalties listed above are given to the defender. The reward value given to the attacker is always the opposite number of the reward given to the defender.

To change the values, you are supposed to use the function `paramConfig()` like `paramConfig(xxx=abc)`, instead of directly assigning values like `config.xxx=abc`.

## TDBoard

This file implements the basic rules of TD game. It contains the following parts:

### Enemy Definition

Enemy is what will follow the road from the start point to the base point. It will be attacked by defense towers, and give penalty to the defender if reaches the base point. An enemy should contains the following properties:

- LP: Life point, indicate how many damage it could bear.
- maxLP: The initial life point of this enemy. This is used to calculate the ratio of rest LP, which will be passed to the agent and used for rendering.
- speed: How many blocks it could pass in one step.
- cost: How much costs needed to be consumed when summoning. This variable is only used for simplify the implementation.
- loc: The current location of this enemy.
- type: Which type this enemy is.

Three types of enemy are implemented.
- EnemyBalance (type=0): Enemy with balanced maxLP and speed.
- EnemyStrong (type=1): Enemy with the highest maxLP and the lowest speed.
- EnemyFast (type=2): Enemy with the lowest maxLP and the highest speed.

Although they are named 'Balance', 'Strong', 'Fast', you could freely adjust the parameters to change their behaviour.

### Tower Definition

Tower is what the defender use to prevent enemies from entering the base point. Tower always attack the enemy closest to the base point (the one summoned earlier if there are 2+ enemies at the same location). Tower is not movable but upgradable. A tower contains the following properties:
- atk: How many damages it could cause.
- rge: How far could it attack. All the points with Manhattan distance less than rge could be attacked.
- atklv: The ATK level, used for deciding the cost and increments of atk when upgrading.
- rangelv: The attack range level, used for deciding the cost and increments of atk when upgrading.
- loc: The location of this tower.
- cost: How many costs have been spent to this tower (build and upgrade). Some part of it will be returned after destruction.

### Board Implementation

TDBoard implements the basic rules of TD game, including map generation, interactions between towers and enemies, summon of enemies, tower building and upgrading. Rendering is also implemented in this class.

- Making environment:
    - Generate start point and base point.
    The start point and base point are always on the opposite sides.
    - Randomly generate one road from the start point to the base point.
    Implemented in method `create_road`.
        - Assign a random value to each point (called height).
        - Use Dijkstra shortest path algorithm to find a road, so that the sum of height in this road is the smallest.
    - Record the road and the distance to the base point from each point in map.
    - Record costs that the defender and attacker have.

- Forward one step:
    - For each tower:
        - Find the enemies that is the closest one to the base point within its attack range.
        - Cause damage.
        - Remove the enemy and record reward if killed.
    - For each living enemy:
        - Move `speed` steps towards the base point.
        - Remove the enemy and record penalty if reached the base point.
    - Increase the costs.

- Rendering:
    - Set the height and width of the window.
    The window contains two parts: the cost bar at the top and the game map.
    - For a new viewer, set the permanent elements.
        - A white background
        - Black grid lines.
        - Dark gray road.
        - Red start point.
        - Blue base point.
    - Then draw changeable elements like enemies, towers.
        - Attacker's cost bar (Red orange)
        - Defender's cost bar (Green blue)
        - Enemies with LP bar
        - Towers with ATK level bar and range level bar.
    - **Note**: The origin point (0, 0) is at the left **bottom** corner.

## TDSingle/TDAttack/TDMulti

These files wrap the `TDBoard` into `gym.Env`.

Random AIs for the opponent are provided:
- random_enemy (available in `TDSingle` and `TDMulti`): Randomly summon 1-3 enemies. Do nothing with probability 1/4.
- random_tower (available in `TDAttack` and `TDMulti`): Build a tower at a random location.

Rule based better AI may be introduced in future work.

Here are some methods for debug, you are not supposed to use it:
- `empty_step`: Doing nothing and forward the game one step.
- `test`: Summon 1 enemy of each type, and build a tower at the center of the map (if possible).
