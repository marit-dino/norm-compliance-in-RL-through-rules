import os

import subprocess

class GardenHelper:
    def write(self,instance, radius, horizon, learning):
        multi = instance.visited[instance.player]
        text = ''
        text += f'#const radius = {radius}.'
        text += f'#const horizon = {horizon}.'
        text += '\n' + f'#program always.'
        text += '\n' + f'multi({multi}).'
        for a in range(4):
            rew = learning.get_action_rank((instance.player[0], instance.player[1]),
                                           a)
            text += '\n' + f'reward({a},{rew},{0},{0}).'
        done = [(0,0)]
        for c in range(radius + 1):
            for r in range(radius + 1):
                position = (c,r)
                if position not in done:
                    absolut = (instance.player[0] + position[0], instance.player[1]  + position[1])
                    if absolut in instance.walls:
                        text += '\n' + f'wall({position[0]},{position[1]}).'
                    if absolut[0] < 1 or absolut[1] < 1 or absolut[0] > instance.size or absolut[1] > instance.size:
                        text += '\n' + f'wall({position[0]},{position[1]}).'
                    if absolut == instance.target:
                        text += '\n' + f'target({position[0]},{position[1]}).'
                    if absolut in instance.plants and absolut not in instance.dead_plants:
                        text += '\n' + f'plant({position[0]},{position[1]}).'
                    for a in range(4):
                        rew = learning.get_action_rank((absolut[0], absolut[1]),
                                                          a)
                        text += '\n' + f'reward({a},{rew},{position[0]},{position[1]}).'
                    done.append(position)
                position = (-c,r)
                if position not in done:
                    absolut = (instance.player[0] + position[0], instance.player[1]  + position[1])
                    if absolut in instance.walls:
                        text += '\n' + f'wall({position[0]},{position[1]}).'
                    if absolut[0] < 1 or absolut[1] < 1 or absolut[0] > instance.size or absolut[1] > instance.size:
                        text += '\n' + f'wall({position[0]},{position[1]}).'
                    if absolut == instance.target:
                        text += '\n' + f'target({position[0]},{position[1]}).'
                    if absolut in instance.plants and absolut not in instance.dead_plants:
                        text += '\n' + f'plant({position[0]},{position[1]}).'
                    for a in range(4):
                        rew = learning.get_action_rank((absolut[0], absolut[1]),
                                                          a)
                        text += '\n' + f'reward({a},{rew},{position[0]},{position[1]}).'
                    done.append(position)
                position = (c,-r)
                if position not in done:
                    absolut = (instance.player[0] + position[0], instance.player[1]  + position[1])
                    if absolut in instance.walls:
                        text += '\n' + f'wall({position[0]},{position[1]}).'
                    if absolut[0] < 1 or absolut[1] < 1 or absolut[0] > instance.size or absolut[1] > instance.size:
                        text += '\n' + f'wall({position[0]},{position[1]}).'
                    if absolut == instance.target:
                        text += '\n' + f'target({position[0]},{position[1]}).'
                    if absolut in instance.plants and absolut not in instance.dead_plants:
                        text += '\n' + f'plant({position[0]},{position[1]}).'
                    for a in range(4):
                        rew = learning.get_action_rank((absolut[0], absolut[1]),
                                                          a)
                        text += '\n' + f'reward({a},{rew},{position[0]},{position[1]}).'
                    done.append(position)
                position = (-c,-r)
                if position not in done:
                    absolut = (instance.player[0] + position[0], instance.player[1]  + position[1])
                    if absolut in instance.walls:
                        text += '\n' + f'wall({position[0]},{position[1]}).'
                    if absolut[0] < 1 or absolut[1] < 1 or absolut[0] > instance.size or absolut[1] > instance.size:
                        text += '\n' + f'wall({position[0]},{position[1]}).'
                    if absolut == instance.target:
                        text += '\n' + f'target({position[0]},{position[1]}).'
                    if absolut in instance.plants and absolut not in instance.dead_plants:
                        text += '\n' + f'plant({position[0]},{position[1]}).'
                    for a in range(4):
                        rew = learning.get_action_rank((absolut[0], absolut[1]),
                                                          a)
                        text += '\n' + f'reward({a},{rew},{position[0]},{position[1]}).'
                    done.append(position)

        os.makedirs(
            os.path.dirname('telingo/' + instance.name + '-' + str(radius) + '-' + str(horizon) + '.lp'),
            exist_ok=True)
        file = open('telingo/' + instance.name + '-' + str(radius) + '-' + str(horizon) + '.lp', 'w')
        file.write(text)
        file.close()


    def get_action(self,instance, horizon, radius, learning):
        if instance.visited[instance.player] > 10:
            return learning.get_action(instance)
        self.write(instance, radius, horizon, learning)
        proc = subprocess.Popen(['telingo', 'telingo.lp', 'telingo/' + instance.name + '-' + str(radius) + '-' + str(horizon) + '.lp',
                                 '0', '--imin=%i' % horizon,
                                 '--imax=%i' % horizon,
                                 '--warn=none'], stdout=subprocess.PIPE)
        models = []
        answer = None
        state = None
        while True:
            line = proc.stdout.readline().rstrip().decode('utf-8')
            if not line:
                break
            #print(line)
            if 'UNSATISFIABLE' in line or 'UNKNOWN' in line:
                break
            if 'SATISFIABLE' in line or 'OPTIMUM FOUND' in line:
                if state is not None:
                    answer.append(state)
                    models.append(answer)
                break
            if 'Answer: 1' == line:
                models = []
                answer = None
                state = None
            if 'Answer:' in line:
                # new answer begins, save cached
                if state is not None:
                    answer.append(state)
                    models.append(answer)
                answer = []
                state = None
            if 'State' in line:
                # new state for same answer
                if state is not None:
                    answer.append(state)
                state = []
            elif state is not None:
                for p in line.split():
                    if 'Optimization' in p:
                        break
                    if 'action' in p:
                        action = int(p.split('(')[1].split(')')[0])
                        state.append(action)
        return models[-1][1][0]
