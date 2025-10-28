
import gymnasium
import numpy as np
from lime.discretize import BaseDiscretizer


class HwDiscretizer(BaseDiscretizer):
    """
        Class to be used to supply the data stats info when discretize_continuous is true
    """

    def __init__(self, data, categorical_features, feature_names, labels=None, random_state=None,
                 data_stats=None):

        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels,
                                 random_state=random_state,
                                 data_stats=data_stats)

    def bins(self, data, labels):
        bins = []

        for feature in self.to_discretize:
            qts = np.array(np.arange(-10,10,step=2)/10.0)
            bins.append(qts)
        return bins

def create_hw_env(**kwargs):
    if "render_mode" in kwargs:
        env = gymnasium.make(kwargs["id"], render_mode=kwargs["render_mode"])
    else:
        env = gymnasium.make(kwargs["id"])
    env.configure(kwargs["config"])
    env.reset()
    return env


env_kwargs = {
    "id": "highway-v0",
    "config": {
        "lanes_count": 3,
        "vehicles_count": 50,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        },
        "policy_frequency": 2,
        "duration": 600,
    },
}
