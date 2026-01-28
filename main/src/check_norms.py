import logging

log = logging.getLogger(__name__)


def violated_vegetarian(state):
    if state.data._eaten[2]:
        log.info("norm 'vegetarian' violated")
        return 1
    return 0

def violated_vegan(state):
    if True in state.data._eaten[1:]:
        log.info("norm 'vegan' violated")
        return 1
    return 0


def violated_permissive(state, eaten_ghost):
    current_eaten = True in state.data._eaten[1:]
    if current_eaten and not eaten_ghost:
        log.info("norm 'permissive' violated")
        return 1
    # TODO remove (currently for debugging purposes)
    if current_eaten and eaten_ghost:
        log.info("eaten ghost, after having already eaten one")
    return 0



NORM_CHECKS = {
    "vegetarian": violated_vegetarian,
    "vegan": violated_vegan,
    "permissive": violated_permissive
}


def num_violations_detected(norms, state, ghost_eaten=False):
    violations = 0
    for norm in norms:
        if norm == "permissive":
            violations += NORM_CHECKS[norm](state, ghost_eaten)
        else:
            violations += NORM_CHECKS[norm](state)
    return violations
