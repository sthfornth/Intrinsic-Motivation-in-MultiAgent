import numpy as np

# class EnvBase():
#     def __init__(self, nr_agents):
#         self._nr_agents = nr_agents

class SimpleEnv:
    def __init__(self, nr_agents):
        assert nr_agents == 2
        self._nr_agents = nr_agents
        self._rewards = [np.array([[-1, 3],[-3, -2]]), np.array([[-1, 3],[-3, -2]])]
        # self._rewards = [np.array([[2, 0],[0, -4]]), np.array([[2, 0],[0, -4]])]

    def reset(self):
        # single state
        self._state = [0, ] * self._nr_agents
        return self._state

    def get_action_spaces(self):
        return [2, ] * self._nr_agents

    def step(self, actions):
        # prisoner dilemma
        actions = tuple(actions)
        rewards = [self._rewards[i][actions] for i in range(self._nr_agents)]
        for i in range(self._nr_agents):
            self._state[i] += 1
        return self._state, rewards, self._state[0] >= 3
