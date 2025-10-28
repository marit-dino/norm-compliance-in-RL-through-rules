# from sb3_ext.gardener_helper import Logic
from .utils import get_current_time_ms
from gym_gardener.interface import Interface

from stable_baselines3.common.utils import obs_as_tensor

CLOCK_MS = 500


class Game:

    def __init__(self,
                 learning,
                 show,
                 asp,
                 dqn,
                 instance=None,
                 env=None,
                 horizon=1,
                 radius=1,
                 cache=1,
                 telingo=False):
        self.instance = instance
        self.learning = learning
        self.asp = asp
        self.dqn = dqn
        self.env = env
        self.show = show
        self.telingo = telingo
        self.cache = cache
        if dqn:
            self.obs, _ = self.env.reset()
            self.instance = self.env.unwrapped.instance
        if show:
            self.interface = Interface(instance=self.instance)
        if not dqn:
            if asp:
                raise Exception("Not supported")
                # self.logic = Logic(instance, learning, horizon, radius)

    def next_step(self):
        # call this function again in 1 second
        time_pre_compute = get_current_time_ms()

        if self.instance.player == self.instance.target:
            if self.show:
                self.interface.frame.destroy()
            return

        if self.dqn:
            action = self.learning.predict(self.obs)[0]
        elif not self.asp:
            action = self.learning.get_action(self.instance)
        else:
            if self.logic.ctl is None:
                self.logic.setup()
            ref_action = self.learning.get_action(self.instance)
            # if self.telingo:
            #     action = telingohelper.get_action(self.instance, self.logic.horizon, self.logic.radius, self.learning)
            # else:
            action = self.logic.get_action(self.cache)

            if action != ref_action:
                self.instance.interceptions += 1

        if action is None:
            if self.show:
                self.interface.frame.destroy()
            return

        self.obs = self.env.step(action)[0]
        
        if not self.dqn:
            self.instance.execute(action)
            self.instance.emulate_frogs()
            self.instance.check_violations()

        if self.show:
            self.interface.place_piece("player", self.instance.player)
            if len(self.instance.frogs) > 0:
                self.interface.remove_pieces("frog")
                for c, f in enumerate(self.instance.frogs):
                    self.interface.add_piece("frog_" + str(c),
                                             self.interface.frog_image, f)

        time_post_compute = get_current_time_ms()
        time_diff = time_post_compute - time_pre_compute

        self.instance.times.append(time_diff)

        next_step_delta = CLOCK_MS - time_diff
        if next_step_delta < 0:
            next_step_delta = 1
        if self.show:
            self.interface.after(next_step_delta, self.next_step)
        else:
            self.next_step()

    def run(self):
        self.next_step()
        if self.show:
            self.interface.mainloop()
