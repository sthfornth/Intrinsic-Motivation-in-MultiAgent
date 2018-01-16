import copy
import numpy as np

class EnvBase:
    def __init__(self, nr_agents):
        self._nr_agents = nr_agents

class SimpleEnv(EnvBase):
    def __init__(self, nr_agents):
        assert nr_agents == 2
        super().__init__(nr_agents)
        self._rewards = [np.array([[-1, 3],[-3, -2]]), np.array([[-1, 3],[-3, -2]])]
        # self._rewards = [np.array([[2, 0],[0, -4]]), np.array([[2, 0],[0, -4]])]

    def reset(self):
        # single state
        self._state = [0, ] * self._nr_agents
        return copy.copy(self._state)

    def get_action_spaces(self):
        return [2, ] * self._nr_agents

    def step(self, actions):
        # prisoner dilemma
        actions = tuple(actions)
        rewards = [self._rewards[i][actions] for i in range(self._nr_agents)]
        for i in range(self._nr_agents):
            self._state[i] += 1
        return copy.copy(self._state), rewards, np.mean(rewards), self._state[0] >= 3


class AxisEnv(EnvBase):
    def __init__(self, nr_agents, nr_rounds):
        super().__init__(nr_agents)
        self._nr_rounds = nr_rounds

    def get_state(self):
        # return list(self._dist)
        # return [self._steps, ] * self._nr_agents
        return [(self._steps, self._dist[i]) for i in range(self._nr_agents)]

    def reset(self):
        self._steps = 0
        self._dist = np.zeros((self._nr_agents,), dtype='int32')
        return self.get_state()

    def get_action_spaces(self):
        return [2, ] * self._nr_agents

    def step(self, actions):
        actions = np.array(actions)
        self._steps += 1
        self._dist += actions * 2 - 1
        if self._steps >= self._nr_rounds:
            rewards, is_over = list(self._dist), True
        else:
            rewards, is_over = [0, ] * self._nr_agents, False
        state = self.get_state()
        return state, rewards, np.mean(rewards), is_over


class FoodEnv(EnvBase):
    def __init__(self, nr_agents, nr_rounds, nr_foods, flag=0):
        super().__init__(nr_agents)
        assert nr_agents == 4 and nr_foods == 4
        self._nr_rounds = nr_rounds
        self._nr_foods = nr_foods
        self._dirs = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]
        self._flag = flag

    def move(self, pos, action):
        d = self._dirs[action]
        res = (pos[0] + d[0], pos[1] + d[1])
        if res[0] < 0 or res[0] >= 4 or res[1] < 0 or res[1] >= 4:
            res = pos
        return res

    def get_state(self):
        # state = (tuple(self._locations), tuple(self._foods))
        # return [state for i in range(self._nr_agents)]
        return [(self._locations[i], ) + tuple(self._foods) for i in range(self._nr_agents)]

    def get_action_spaces(self):
        return [5, ] * self._nr_agents

    def reset(self):
        self._steps = 0
        if self._flag == 0:
            self._foods = [(0, 0), (1, 2), (2, 1), (3, 3)]
            self._locations = [(0, 3), (1, 1), (2, 2), (3, 0)]
        elif self._flag == 1:
            self._foods = [(0, 0), (0, 3), (3, 0), (3, 3)]
            self._locations = [(1, 2), (1, 1), (2, 2), (2, 1)]
        elif self._flag == 2:
            self._foods = [(1, 2), (1, 1), (2, 2), (2, 1)]
            self._locations = [(0, 0), (0, 3), (3, 0), (3, 3)]
        elif self._flag == 3:
            self._foods = [(1, 2), (0, 3), (3, 0), (2, 1)]
            self._locations = [(0, 0), (0, 0), (3, 3), (3, 3)]
        self._scores = [0, 0, 0, 0]
        # self._state = (tuple(self._locations), tuple(self._foods))
        return self.get_state()

    def step(self, actions):
        self._steps += 1
        foods = self._foods
        self._foods = []
        count = {}
        for j in foods:
            count[j] = 0
        for i in range(self._nr_agents):
            self._locations[i] = self.move(self._locations[i], actions[i])
            for j in foods:
                if self._locations[i] == j:
                    count[j] += 1
        for j in foods:
            if count[j] == 0:
                self._foods.append(j)
        if len(self._foods) == 0:
            finish = True
            extra = self._nr_rounds - self._steps + 1
        else:
            finish = False
            extra = 0

        rewards = []
        for i in range(self._nr_agents):
            loc = self._locations[i]
            if loc in count:
                reward = 1.0 / float(count[loc])
            else:
                reward = 0
            self._scores[i] += reward
            rewards.append(reward)
        if len(self._foods) == 0 or self._steps >= self._nr_rounds:
            is_over = True
        else:
            is_over = False
        #self._state = (tuple(self._locations), tuple(self._foods))
        return self.get_state(), rewards, extra, is_over
