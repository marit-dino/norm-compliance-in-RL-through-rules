from gymnasium.envs.registration import register

register(
    id='BerkeleyPacmanPO-v0',
    entry_point='gym_pacman.envs:PacmanEnv',
)

register(
    id='BerkeleyPacman-v0',
    entry_point='gym_pacman.envs:PacmanEnv', # TODO turn off partial observability
)
