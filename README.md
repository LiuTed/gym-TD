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
        "Map": "A NumPy Array of shape (length, length, 12) and dtype=np.float32.",
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
    - 9: The distance that EnemyBalance has passed in this block.
    - 10: The distance that EnemyStrong has passed in this block.
    - 11: The distance that EnemyFast has passed in this block.

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
Your goal is to summon the enemies with some strategy so that as many enemies (creepers) could reach the base point as possible.
- Observation space:
    Same as defender.

- Action space:
    A list of len = 3, the element of which should be either 0 or 1.
    Each represents whether summon this type of creeper. (1 for True)

### Multi-player
You could control both the defender and attacker.
- Observation space:
    A Python dict of this form:
    ```
    {
        "Map": "A NumPy Array of shape (length, length, 12) and dtype=np.float32.",
        "Cost_Attacker": "The cost that the attacker has. It is a integer in [0, max_cost].",
        "Cost_Defender": "The cost that the defender has. It is a integer in [0, max_cost].",
    }
    ```
    Channels same as above.

- Action space:
    A Python dict of this form:
    ```
    {
        "Attacker": "A list of len = 3, same as attacker's action space"
        "Defender":
        {
            "Build": "A NumPy Array of shape (length, length) and dtype=np.int32. It represents whether to build a tower at this place. (1 for True)",
            "ATKUp": "A NumPy Array of shape (length, length) and dtype=np.int32. It represents whether to upgrade the ATK of the tower at this place.",
            "RangeUp": "A NumPy Array of shape (length, length) and dtype=np.int32. It represents whether to upgrade the attack range of the tower at this place.",
            "Destruct": "A NumPy Array of shape (length, length) and dtype=np.int32. It represents whether to destruct the tower at this place. Destruction will return some costs spent on this tower."
        }
    }
    ```

- Reward:
    The reward that `step` returns is same as the reward that the defender gets. The reward for the attacker is the opposite of the reward for the defender.

### Config
You could config lots of parameters of this game with the function `paramConfig(**kwargs)` to customize your environment. For example, if you want to set the variable `max_cost=100`, all you need is to simply execute `paramConfig(max_cost=100)` before making the environment. You should config parameters before making an environment. You could read [gym_TD/envs/README.md](gym_TD/envs/README.md) for detailed descriptions.

### End condition
This game will run continuously until reaching 200 steps, or `base_LP` enemies have been leaked (`base_LP=None` means do not end due to leakage). Although you could run even after 200 steps without error, you are not supposed to do so.

### Versions
- TD-def-small-v0: Control the defender. Map size = (10, 10)
- TD-def-middle-v0: Control the defender. Map size = (20, 20)
- TD-def-large-v0: Control the defender. Map size = (30, 30)
- TD-atk-small-v0: Control the attacker. Map size = (10, 10)
- TD-atk-middle-v0: Control the attacker. Map size = (20, 20)
- TD-atk-large-v0: Control the attacker. Map size = (30, 30)
- TD-2p-small-v0: Control both sides. Map size = (10, 10)
- TD-2p-middle-v0: Control both sides. Map size = (20, 20)
- TD-2p-large-v0: Control both sides. Map size = (30, 30)
----
- TD-def-v0: Control the defender. You should pass the parameter `map_size=n` when making the environment, to set the map size to (n, n)
- TD-atk-v0: Control the attacker. You should pass the parameter `map_size=n` when making the environment, to set the map size to (n, n)
- TD-2p-v0: Control both sides. You should pass the parameter `map_size=n` when making the environment, to set the map size to (n, n)

## Installation
It should work on Python3. It only requires gym and numpy.

Using the commands following could automatically install this package.
```
cd gym-TD
python setup.py install
```

Or you could simply use the following commands to install the prerequisites.
```
cd gym-TD
python -m pip install -r requirements.txt
```

## Demo
```
cd gym-TD
python demo.py
```
You could also use `python demo.py -[adm]` to see a random game. (Use `python demo.py -h` for details)

## Training
You could use gym-TD just like use other OpenAI Gym environments like cartpole. You could go [OpenAI Gym](https://github.com/openai/gym) for example codes and documents.
