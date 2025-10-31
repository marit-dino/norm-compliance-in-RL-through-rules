from gymnasium.envs.registration import register

register(
    id='BerkeleyPacmanPO-v0-legible',
    entry_point='legible.gym_pacman_rules.envs:PacmanEnv',
)

register(
    id='BerkeleyPacman-v0-legible',
    entry_point='legible.gym_pacman_rules.envs:PacmanEnv', # TODO turn off partial observability
)
