import logging

log = logging.getLogger(__name__)


def violated_vegetarian(state):
    violation_count = state.data._eaten[2:].count(True)
    if violation_count > 0:
        log.info("norm 'vegetarian' violated")
    return violation_count

def violated_vegan(state):
    violation_count = state.data._eaten[1:].count(True)
    if violation_count > 0:
        log.info("norm 'vegan' violated")
    return violation_count


def violated_permissive(state, eaten_ghost):
    current_eaten_count = state.data._eaten[1:].count(True)
    if current_eaten_count > 0 and not eaten_ghost:
        log.info("norm 'permissive' violated")
        return current_eaten_count
    return 0


def violated_ctd(state, moved_north):
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
    if norm == "permissive":
        return NORM_CHECKS[norm](state, ghost_eaten)
    elif norm == "ctd":
        return NORM_CHECKS[norm](state, moved_north)
    else:
        return NORM_CHECKS[norm](state)
