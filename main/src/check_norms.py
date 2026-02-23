import logging

log = logging.getLogger(__name__)


def violated_vegetarian(state):
    """ Checks whether the given state violates the norm 'vegetarian' (pacman is only allowed to eat the first ghost).

    Args:
        state (GameState): state of the environment which is to be checked for violations

    Returns:
        int: the number of violations caused by the agent in this state
    """    
    violation_count = state.data._eaten[2:].count(True)
    if violation_count > 0:
        log.info("norm 'vegetarian' violated")
    return violation_count

def violated_vegan(state):
    """ Checks whether the given state violates the norm 'vegan' (pacman is not allowed to eat ghosts).

    Args:
        state (GameState): state of the environment which is to be checked for violations

    Returns:
        int: the number of violations caused by the agent in this state
    """
    violation_count = state.data._eaten[1:].count(True)
    if violation_count > 0:
        log.info("norm 'vegan' violated")
    return violation_count


def violated_permissive(state, eaten_ghost):
    """ Checks whether the given state violates the norm 'permissive' (pacman is only allowed to eat ghosts after it has already eaten one).

    Args:
        state (GameState): state of the environment which is to be checked for violations
        eaten_ghost (bool): True if pacman has eaten a ghost before, else False

    Returns:
        int: the number of violations caused by the agent in this state
    """
    current_eaten_count = state.data._eaten[1:].count(True)
    if current_eaten_count > 0 and not eaten_ghost:
        log.info("norm 'permissive' violated")
        return current_eaten_count
    return 0


def violated_ctd(state, moved_north):
    """ Checks whether the given state violates the normative scenario 'ctd' (pacman is vegan, but if he eats a ghost, he should do it when moving north).

    Args:
        state (GameState): state of the environment which is to be checked for violations
        moved_north (bool): True if pacman's previous action was to move north, else False

    Returns:
        int: the number of violations caused by the agent in this state
    """
    viol = violated_vegan(state) 
    if viol > 0 and not moved_north:
        log.info("norm 'ctd' violated")
        return 2 * viol
    return viol


NORM_CHECKS = {
    "vegetarian": violated_vegetarian,
    "vegan": violated_vegan,
    "permissive": violated_permissive,
    "ctd": violated_ctd
}


def num_violations_detected(norm, state, ghost_eaten=False, moved_north=False):
    """ Checks whether the provided state violates any norms, based on the specified norm.

    Args:
        norm (string): norm for which to check whether the state violates it
        state (GameState): state of the environment which is to be checked for violations
        ghost_eaten (bool, optional): whether pacman has eaten a ghost before. Defaults to False.
        moved_north (bool, optional): whether pacman's last action was to move north. Defaults to False.

    Returns:
        int: number of violations detected
    """
    if norm == "permissive":
        return NORM_CHECKS[norm](state, ghost_eaten)
    elif norm == "ctd":
        return NORM_CHECKS[norm](state, moved_north)
    else:
        return NORM_CHECKS[norm](state)
