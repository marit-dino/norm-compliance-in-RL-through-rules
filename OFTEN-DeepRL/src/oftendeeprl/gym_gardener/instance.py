import random


def create_random_instance(size, setting, number,
                           numWalls, numPhenomena,
                           numFrogs, prefix):

    forbidden = []
    player = (random.randint(1, size), random.randint(1, size))
    target = (random.randint(1, size), random.randint(1, size))
    while abs(target[0] - player[0]) + abs(target[1] - player[1]) < 0.5 * size:
        target = (random.randint(1, size), random.randint(1, size))
    forbidden.append(player)
    forbidden.append(target)

    walls = []
    path = None
    while path is None or pow(size, 2) < numWalls * pow(size, 2) + len(path):
        path = generate_random_path(size, player, target)
    while len(walls) < numWalls * pow(size, 2):
        wall = (random.randint(1, size), random.randint(1, size))
        if wall not in forbidden and wall not in path:
            walls.append(wall)
            forbidden.append(wall)

    frogs = []
    plants = []
    if setting == 'nondeterministic':
        while len(frogs) < numFrogs * numPhenomena * pow(size, 2):
            frog = (random.randint(1, size), random.randint(1, size))
            if frog not in forbidden:
                frogs.append(frog)
                forbidden.append(frog)
    while len(frogs) + len(plants) < numPhenomena * pow(size, 2):
        plant = (random.randint(1, size), random.randint(1, size))
        if plant not in forbidden:
            plants.append(plant)
            forbidden.append(plant)

    name = prefix + '-' + ('d' if setting == 'deterministic' else 'nd') + '-' + str(
        size) + '-' + '{:03d}'.format(number + 1)
    instance = Instance(name, size, walls, player, target, plants, frogs)

    return instance


def generate_random_path(size, start, target):
    # Directions for moving in the grid: right, left, down, up
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Set of visited cells to avoid repetition
    visited = [start]

    # Initialize the path with the start cell
    path = [start]

    # Function to check if the move is within the grid bounds and not visited
    def is_valid_move(x, y):
        return 0 < x <= size and 0 < y <= size and (x, y) not in visited

    # Generate the random path
    current = start
    while current != target:
        x, y = current
        random.shuffle(directions)  # Shuffle directions to ensure randomness
        moved = False
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if is_valid_move(new_x, new_y):
                current = (new_x, new_y)
                path.append(current)
                visited.append(current)
                moved = True
                break
        if not moved:  # If no valid move is found, backtrack
            path.pop()
            if not path:
                return None
            current = path[-1]

    return path


def is_path(size, player, target, walls, wall,frogs,plants):
    queue = [(player,0, [])]
    visited = [player]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while len(queue) > 0:
        (current,dist,obs) = queue.pop(0)
        if current == target:
            print(f"Distance to target: {dist} with obs {obs}")
            return True

        for direction in directions:
            r, c = current[0] + direction[0], current[1] + direction[1]
            if (r, c) in frogs:
                new_obs = obs + ["frog"]
            elif (r, c) in plants:
                new_obs = obs + ["plant"]
            else:
                new_obs = obs

            if 1 <= r <= size and 1 <= c <= size and (r, c) not in walls and (
                    r, c) not in visited and (r, c) != wall:
                visited.append((r, c))
                queue.append(((r, c),dist + 1,new_obs))
    return False


def read_from_file(filename):
    file = open(filename, 'r')
    lines = file.readlines()

    name = filename.split('.')[0]
    name = name.split('/')[-1]
    walls = []
    plants = []
    frogs = []

    for line in lines:
        line = line.strip()
        if '#const' in line:
            line = line.replace('#const', '')
            line = line.replace('.', '')
            line_split = line.split('=')
            arg_name = line_split[0].strip()
            arg_value = line_split[1].strip()
            if arg_name == 'size':
                size = int(arg_value)
        if '(' in line:
            sub_lines = line.split('. ')
            for sub_line in sub_lines:
                if sub_line == '':
                    continue
                line_split = sub_line.split('(')
                arg_name = line_split[0]
                arg_value = line_split[1].split(')')[0]
                args = arg_value.split(',')
                if arg_name == 'player':
                    player = (int(args[0]), int(args[1]))
                if arg_name == 'target':
                    target = (int(args[0]), int(args[1]))
                if arg_name == 'wall':
                    wall = (int(args[0]), int(args[1]))
                    walls.append(wall)
                if arg_name == 'frog':
                    frog = (int(args[0]), int(args[1]))
                    frogs.append(frog)
                if arg_name == 'plant':
                    plant = (int(args[0]), int(args[1]))
                    plants.append(plant)
    instance = Instance(name, size, walls, player, target, plants, frogs)

    return instance


class Instance:

    def __init__(self, name, size, walls, player, target, plants, frogs):
        self.name = name
        self.size = size
        self.walls = walls
        self.player = player
        self.target = target
        self.plants = plants
        self.frogs = frogs
        self.dead_frogs = []
        self.dead_plants = []
        self.visited = {player: 1}
        self.times = []
        self.interceptions = 0

    def target_reachable(self):
        return is_path(self.size,self.player,self.target,self.walls,None, self.frogs,self.plants)

    def execute(self, action):
        if action == 0:
            self.player = (self.player[0], self.player[1] + 1)
        if action == 1:
            self.player = (self.player[0], self.player[1] - 1)
        if action == 2:
            self.player = (self.player[0] - 1, self.player[1])
        if action == 3:
            self.player = (self.player[0] + 1, self.player[1])
        if self.player in self.visited:
            self.visited[self.player] = self.visited[self.player] + 1
        else:
            self.visited[self.player] = 1

    def check_violations(self):
        if self.player in self.frogs:
            violations = [i for i, value in enumerate(self.frogs) if value == self.player]
            for i in violations:
                if i not in self.dead_frogs:
                    self.dead_frogs.append(i)
        if self.player in self.plants and self.player not in self.dead_plants:
            self.dead_plants.append(self.player)

    def emulate_frogs(self):
        for c, f in enumerate(self.frogs):
            if c in self.dead_frogs:
                continue
            feasibleActions = []
            for a in range(4):
                if self.check_frog_feasible(c, a):
                    feasibleActions.append(a)
            if len(feasibleActions) > 0:
                frog_action = random.choice(feasibleActions)
                self.execute_frog(c, frog_action)

    def execute_frog(self, frog, action):
        result = None
        match action:
            case 0:
                result = (self.frogs[frog][0], self.frogs[frog][1] + 1)
            case 1:
                result = (self.frogs[frog][0], self.frogs[frog][1] - 1)
            case 2:
                result = (self.frogs[frog][0] - 1, self.frogs[frog][1])
            case 3:
                result = (self.frogs[frog][0] + 1, self.frogs[frog][1])
        self.frogs[frog] = result

    def check_frog_feasible(self, frog, action):
        result = None
        match action:
            case 0:
                result = (self.frogs[frog][0], self.frogs[frog][1] + 1)
            case 1:
                result = (self.frogs[frog][0], self.frogs[frog][1] - 1)
            case 2:
                result = (self.frogs[frog][0] - 1, self.frogs[frog][1])
            case 3:
                result = (self.frogs[frog][0] + 1, self.frogs[frog][1])
        if self.is_feasible(result):
            return True
        else:
            return False

    def is_feasible(self, position):
        if position[0] < 1:
            return False
        if position[0] > self.size:
            return False
        if position[1] < 1:
            return False
        if position[1] > self.size:
            return False
        if position in self.walls:
            return False
        return True

    def print_report(self, folder):
        import os
        text = ''
        text += '{:03d}'.format(
            len(self.dead_plants)) + ' / ' + '{:03d}'.format(
            len(self.plants)) + ' plants killed\n'
        text += '{:03d}'.format(
            len(self.dead_frogs)) + ' / ' + '{:03d}'.format(
            len(self.frogs)) + ' frogs killed\n'
        text += str(len(self.times)) + ' steps taken\n'
        if len(self.times[1:]) > 0:
            text += str(sum(self.times[1:]) / len(self.times[1:])) + ' average time per step\n'
            text += str(max(self.times[1:])) + ' max time per step\n'
            text += str(min(self.times[1:])) + ' min time per step\n'
        else:
            text += str(0) + ' average time per step\n'
            text += str(0) + ' max time per step\n'
            text += str(0) + ' min time per step\n'
        text += str(self.times[0]) + ' startup time\n'
        text += str(self.interceptions) + ' interceptions from ASP'
        #os.makedirs(
        #    os.path.dirname(
        #        'instances/eval/' + folder + '/%s.eval' % self.name),
        #    exist_ok=True)
        #file = open(
        #        'instances/eval/' + folder + '/%s.eval' % self.name, 'w')
        #file.write(text)
        #file.close()
        print(text)

    def save(self):
        import os
        text = ''
        text += '#const size =' + str(self.size) + '.\n'
        text += '\n'
        text += 'player' + str(self.player) + '.\n'
        text += 'target' + str(self.target) + '.\n'
        text += '\n'
        for wall in self.walls:
            text += 'wall' + str(wall) + '. '
        text += '\n'
        text += '\n'
        for plant in self.plants:
            text += 'plant' + str(plant) + '. '
        text += '\n'
        text += '\n'
        for frog in self.frogs:
            text += 'frog' + str(frog) + '. '

        os.makedirs(
            os.path.dirname('instances/%s.lp' % self.name),
            exist_ok=True)
        file = open('instances/%s.lp' % self.name, 'w')
        file.write(text)
        file.close()
