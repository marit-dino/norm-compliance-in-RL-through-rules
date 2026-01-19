import logging

log = logging.getLogger(__name__)


def violated_vegetarian(state):
    if state.data._eaten[2]:
        log.info("norm 'vegetarian' violated")
        return 1
    return 0

def violated_vegan(state):
    if state.data._eaten[1] or state.data._eaten[2]:
        log.info("norm 'vegan' violated")
        return 1
    return 0


NORM_CHECKS = {
    "vegetarian": violated_vegetarian,
    "vegan": violated_vegan
}


def num_violations_detected(norms, state):
    violations = 0
    for norm in norms:
        violations += NORM_CHECKS[norm](state)
    return violations
