def violated_vegetarian(state):
    if state.data._eaten[2]:
        print(state.data._eaten)
        print("ghost 2 eaten")
        return True
    return False

def violated_vegan(state):
    if state.data._eaten[1] or state.data._eaten[2]:
        print(state.data._eaten)
        print("ghost eaten")
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
