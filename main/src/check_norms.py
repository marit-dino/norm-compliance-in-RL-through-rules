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
    return 0


def violated_ctd(state, moved_north):
    viol = violated_vegan(state) 
    if viol > 0 and not moved_north:
        return viol + 1
    return viol


NORM_CHECKS = {
    "vegetarian": violated_vegetarian,
    "vegan": violated_vegan,
    "permissive": violated_permissive,
    "ctd": violated_ctd
}


def num_violations_detected(norm, state, ghost_eaten=False, moved_north=False):
    if norm == "permissive":
        return NORM_CHECKS[norm](state, ghost_eaten)
    elif norm == "ctd":
        return NORM_CHECKS[norm](state, moved_north)
    else:
        return NORM_CHECKS[norm](state)
