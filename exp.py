
from env import SimpleEnv
from qlearning import QLearningAgent

import numpy as np

def main():
    n = 2
    env = SimpleEnv(n)
    action_spaces = env.get_action_spaces()
    alpha = 0.1
    gamma = 0.99
    agents = [QLearningAgent(alpha, gamma, action_spaces[i]) for i in range(n)]

    #Training
    train_rounds = 100
    states = env.reset()
    for _ in range(train_rounds):
        actions = [agent.get_action(state) for agent, state in zip(agents, states)]
        new_states, rewards, is_over = env.step(actions)
        for i in range(n):
            agents[i].update(states[i], actions[i], new_states[i], rewards[i])
        if is_over:
            states = env.reset()
        else:
            states = new_states

    #Testing
    test_rounds = 10
    states = env.reset()
    total_score = np.zeros((n, ))
    for _ in range(test_rounds):
        actions = [agent.get_action(state, False) for agent, state in zip(agents, states)]
        states, rewards, is_over = env.step(actions)
        if is_over:
            states = env.reset()
        else:
            states = new_states
        total_score += np.array(rewards)
    avg_score = total_score / test_rounds
    print(avg_score)

if __name__ == "__main__":
    main()
