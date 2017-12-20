
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
    for _ in range(train_rounds):
        states = env.reset()
        is_over = False
        while not is_over:
            actions = [agent.get_action(state) for agent, state in zip(agents, states)]
            new_states, rewards, is_over = env.step(actions)
            team_reward = np.sum(rewards)
            # print(team_reward)
            for i in range(n):
                agents[i].update(states[i], actions[i], new_states[i], rewards[i] + team_reward * 0)
            states = new_states

    #Testing
    test_rounds = 10
    total_score = np.zeros((n, ))
    for _ in range(test_rounds):
        states = env.reset()
        is_over = False
        scores = np.zeros((n, ))
        while not is_over:
            actions = [agent.get_action(state, False) for agent, state in zip(agents, states)]
            if _ == 0:
                print(actions)
            new_states, rewards, is_over = env.step(actions)
            scores += np.array(rewards)
        total_score += scores
    avg_score = total_score / test_rounds
    print(avg_score)

if __name__ == "__main__":
    main()
