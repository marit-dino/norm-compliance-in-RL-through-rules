import time
from abc import ABC

class ClingoHelper(ABC):
    def get_action(self, state, actionValuePairs, eaten_ghost_before=False):
        pass

    def get_relevant_states(self, state,obs):
        pass
