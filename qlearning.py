import numpy as np
import numpy.random as random

class QLearningAgent:
    def __init__(self, id, nr_actions, alpha, gamma, eps=0.1):
        self._id = id
        self._nr_actions = nr_actions
        self._alpha = alpha
        self._gamma = gamma
        self._eps = eps
        self.Q = dict()

    def adjust(self, lr_beta, eps_beta):
        # self._alpha *= lr_beta
        # self._eps *= eps_beta
        self._alpha = max(0.0, min(1.0, self._alpha + lr_beta))
        self._eps = max(0.0, min(1.0, self._eps + eps_beta))

    def update(self, state, action, next_state, reward):
        if state not in self.Q:
            self.Q[state] = dict()
        if action not in self.Q[state]:
            self.Q[state][action] = 0
        self.Q[state][action] += self._alpha * (reward + self._gamma \
            * self.get_value(next_state) - self.Q[state][action]) 
    
    def get_Q(self, state, action):
        if state not in self.Q:
            return 0
        if action not in self.Q[state]:
            return 0
        return self.Q[state][action]

    def get_value(self, state):
        max_action = 0
        for action in range(self._nr_actions):
           if self.get_Q(state, action) > self.get_Q(state, max_action):
             max_action = action
        return self.get_Q(state, max_action)

    def get_action(self, state, is_train=True):
        if is_train and random.rand() < self._eps:
            return random.randint(self._nr_actions)
        v = self.get_value(state)
        actions = []
        for action in range(self._nr_actions):
            if self.get_Q(state, action) == v:
                actions.append(action)
        # print(self._id, actions, self.get_Q(state, 0), self.get_Q(state, 1))
        return random.choice(actions)

