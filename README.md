# gym-TD (Gym - Tower Defense)

## Introduction
gym-TD is a simplified Tower Defense game (TD game). Your goal is to build towers to prevent enemies entering your base as long as possible. You can also control the attacker to summon enemies to enter the defender's base as many times as possible.

<kbd>![Simple TD game environment](gymTD.png)</kbd>

### Board
Board is where this game happens. It contains blocks of shape `(length, length)`. The elements is described as follow:
- Bars at the top: Cost bar.
    - Orange bar: The cost that the attacker has.
    - Blue bar: The cost that the defender has.
    - Note: Each time the attacker summons a enemy or the defender build/upgrade a tower, some costs will be consumed.
- White square: A empty place where towers could be built on.
- Black square: A part of a path where the enemies will walk.
- Red square: The start point where the enemies will be summoned.
- Blue square: The base point of defender.
- Lake blue square: A defense tower.
    - Green bar on the left side: The level of ATK of this tower.
    - Green bar on the right side: The level of attack range of this tower.
- Small blocks on the road: Enemies.
    - Red block: A enemy that has balanced speed and life point. It will always be at the left top of a road square. (EnemyBalance)
    - Green block: A enemy that has the lowest speed and highest life point. It will always be at the right top of a road square. (EnemyStrong)
    - Blue block: A enemy that has the highest speed and lowest life point. It will always be at the bottom center of a road square. (EnemyFast)
    - Green bars on the bottom of enemies: The life bar of the enemies.
    - **Note**: Because the attacker could only summon one enemy of each type at a single step, there will not be 2+ enemies of the same type at the same location.

### Defender
Defender is the player in most other TD games. Your goal is to build and upgrade your towers, which could automatically attack the enemies, to protect your base point.
- Observation space:
    A Python dict of this form:
    ```
    {
        "Map": "A NumPy Array of shape (length, length, 9) and dtype=np.float32.",
        "Cost": "The cost you have. It is a integer in [0, max_cost]."
    }
    ```
    Channels in Map:
    - 0: If this block is a road. (0 for False and 1 for True)
    - 1: If this block is a start point.
    - 2: If this block is a base point.
    - 3: If this block has a tower.
    - 4: The ATK level of this tower: One of 0, 1, 2
    - 5: The Range level of this tower: One of 0, 1, 2
    - 6: The LP (life point) ratio of EnemyBalance in this block.
    - 7: The LP ratio of EnemyStrong in this block.
    - 8: The LP ratio of EnemyFast in this block.

- Action space:
    A Python dict of this form:
    ```
    {
        "Build": "A NumPy Array of shape (length, length) and dtype=np.int32. It represents whether to build a tower at this place. (1 for True)",
        "ATKUp": "A NumPy Array of shape (length, length) and dtype=np.int32. It represents whether to upgrade the ATK of the tower at this place.",
        "RangeUp": "A NumPy Array of shape (length, length) and dtype=np.int32. It represents whether to upgrade the attack range of the tower at this place.",
        "Destruct": "A NumPy Array of shape (length, length) and dtype=np.int32. It represents whether to destruct the tower at this place. Destruction will return some costs spent on this tower."
    }
    ```
    The four actions are processed in order, which means you could build a tower and upgrade it and then destruct it in a single action, but you could not upgrade the same property twice in a single action.

    **Note**: The illegal actions will be ignored.

### Attacker
In development.

### Config
You could config lots of parameters of this game with the function `paramConfig(**kwargs)`. You should config these parameters before making an environment. The description and default values are shown below:
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
- `tower_basic_cost`: 5
- `tower_destruct_return`: The ratio of return costs. Default is 0.5.
- `tower_basic_ATK`: 5
- `tower_ATKUp_list`: A list of the differences of ATK between levels. Default is [5, 5]
- `tower_ATKUp_cost`: A list of the costs of ATK upgrade. Default is [5, 5]
- `tower_basic_range`: 4
- `tower_rangeUp_List`: A list of the differences of attack range between levels. Default is [2, 2]
- `tower_rangeUp_Cost`: A list of the costs of attack range upgrade. Default is [3, 3]
- `reward_kill`: Reward of killing an enemy. Default is 0.1
- `penalty_leak`: Penalty of leaking an enemy. (Enemy enters the base point). Default is 1.
- `reward_time`: Reward of each step. Default is 0.001

### End condition
This game will run continuously until reaching 500 steps.

### Versions
- TD-def-small-v0: Map size = (10, 10)
- TD-def-middle-v0: Map size = (20, 20)
- TD-def-large-v0: Map size = (30, 30)

## Installation
It should work on Python3. It only requires gym and numpy.

Using the commands following could automatically install this package.
```
cd gym-TD
python setup.py install
```

## Demo
```
cd gym-TD
python test.py
```
You could also use `python test.py -r` to see a random game.
