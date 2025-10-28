import numpy as np
import random
import pickle
import os
from random import sample

EPISODES = 50000


class Learning:

    def __init__(self, alpha=0.1, gamma=0.8, epsilon=0.6):
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = [0, 1, 2, 3]

        self.q_table = None
        self.states = []
        self.rewards = []
        self.current = None
        self.name = None

    def create_states(self, instance):
        for c in range(1, instance.size + 1):
            for r in range(1, instance.size + 1):
                if (c, r) not in instance.walls:
                    self.states.append((c, r))
        for s in self.states:
            current_idx = self.states.index(s)
            r = {}
            for a in self.actions:
                next_idx = current_idx
                done = False
                reward = -10
                state = None
                if a == 0:
                    state = (s[0], s[1] + 1)
                if a == 1:
                    state = (s[0], s[1] - 1)
                if a == 2:
                    state = (s[0] - 1, s[1])
                if a == 3:
                    state = (s[0] + 1, s[1])
                if state in self.states:
                    next_idx = self.states.index(state)
                    if state != s:
                        reward = -1
                    if state == instance.target:
                        reward = 100
                        done = True
                r[a] = [(1.0, next_idx, reward, done)]
            self.rewards.append(r)
        self.current = self.states.index(instance.player)

    def create_starting_states(self, instance):
        change = True
        reachable = [instance.target]
        actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        while change:
            change = False
            for c in reachable:
                for a in actions:
                    candidate = (c[0] + a[0], c[1] + a[1])
                    if (0 < candidate[0] <= instance.size and 0 < candidate[
                        1] <= instance.size and candidate not in reachable and
                            candidate not in instance.walls):
                        reachable.append(candidate)
                        change = True
        reachable.remove(instance.target)
        return reachable

    def get_action(self, instance):
        if instance.player == instance.target:
            return None
        state_idx = self.states.index(instance.player)
        actions = np.copy(self.q_table[state_idx])
        actions = np.argsort(actions)

        action = actions[len(actions) - 1]
        return action

    def get_action_rank(self, player, action):
        if player not in self.states:
            return -1
        state_idx = self.states.index(player)
        actions = np.copy(self.q_table[state_idx])
        actions = np.argsort(actions)
        actions = list(np.array(actions))

        return actions.index(action)

    def learn(self, instance):
        self.create_states(instance)
        starting_states = self.create_starting_states(instance)
        self.name = instance.name
        self.q_table = np.zeros([len(self.states), len(self.actions)])
        for i in range(1, EPISODES + len(starting_states) * 5):
            if i < EPISODES:
                state_idx = self.states.index(instance.player)
            else:
                state_idx = self.states.index(
                    starting_states[i % len(starting_states)])

            epochs, penalties, reward, = 0, 0, 0
            done = False

            while not done:
                if random.uniform(0, 1) < self.epsilon:
                    action = sample(self.actions, 1)[
                        0]  # Explore action space
                else:
                    action = np.argmax(
                        self.q_table[state_idx])  # Exploit learned values

                done = self.rewards[state_idx][action][0][3]
                reward = self.rewards[state_idx][action][0][2]
                next_state = self.rewards[state_idx][action][0][1]

                old_value = self.q_table[state_idx, action]
                next_max = np.max(self.q_table[next_state])

                new_value = (1 - self.alpha) * old_value + self.alpha * (
                        reward + self.gamma * next_max)
                self.q_table[state_idx, action] = new_value

                if reward == -10:
                    penalties += 1

                state_idx = next_state
                epochs += 1
        os.makedirs(
            os.path.dirname('instances/learning/%s.pkl' % instance.name),
            exist_ok=True)
        pickle.dump(self,
                    open("instances/learning/%s.pkl" % instance.name, 'wb'))
