from abc import ABC


class LearningInferface(ABC):
    def get_action_rank(self,state,action):
        pass

    def get_action(self,state):
        pass

class GardenerInterface(LearningInferface):
    def __init__(self,policy,envs):
        self.policy = policy
        self.envs = envs
    def get_action_rank(self,state,action):
        pass
    def get_action(self,state):
        pass