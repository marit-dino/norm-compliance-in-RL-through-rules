import logging

log = logging.getLogger(__name__)


def violated_vegetarian(state):
    if state.data._eaten[2]:
        log.info("norm 'vegetarian' violated")
        return True
    return False

def violated_vegan(state):
    if state.data._eaten[1] or state.data._eaten[2]:
        log.info("norm 'vegan' violated")
        return True
    return False


NORM_CHECKS = {
    "vegetarian": violated_vegetarian,
    "vegan": violated_vegan
}


def violations_detected(norms, state):
    for norm in norms:
        if NORM_CHECKS[norm](state):
            return True
    return False
