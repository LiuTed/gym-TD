# TODO List

- different types of tower
    | type | attack | speed | range | costs | special | 
    | - | - | - | - | - | - |
    | basic arrow tower | low | fast | far | cheap | - |
    | strong arrow tower | high | medium | medium | medium-high | - |
    | bomb tower | medium | very slow | near | high | splash damage |
    | frozen tower | low | slow | medium | medium | slow enemy |

- different types of enemy
    | type | speed | LP | defense | costs |
    | - | - | - | - | - |
    | fast | fast | low | low | very cheap |
    | normal | medium | medium | medium | medium |
    | armored | low | high | high | high |
    | strong | low | very high | medium-low | high |

- cross
    | tower | basic arrow | strong arrow | bomb | frozen |
    | - | - | - | - | - |
    | vs enemy |
    | fast | + | + | -+ | + |
    | normal | + | + | - | - |
    | armored | - | + | - | - |
    | strong | + | + | - | + |

- game balancing
    - attacker
        - cost increasing speed increases with game progresses (faster)
        - enemies get stronger with game progresses (stronger)
    - defender
        - /

- tower building
    - cannot build tower at where too close to another tower
    - cannot build tower that is too close to the road

    - level up instead of upgrading each properties

- enemy summon
    - clustered summon
    - do not set interval

- state representation
    - how many enemies of this type could be summoned
    - could build tower
    - reduce channels
        - lowest, highest, average LP of enemies at this position
        - number of enemies at this position
