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
        return copy.copy(self._state), rewards, self._state[0] >= 3


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
        return state, rewards, is_over


# class FoodEnv(EnvBase):
#     def __init__(self, nr_agents, nr_rounds):
#         super().__init__(nr_agents)
#         self._nr_rounds = nr_rounds

#     def get_state(self):
#         return self._states

#     def reset(self):
#         self._state = None
