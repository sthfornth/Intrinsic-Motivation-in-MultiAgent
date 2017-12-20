
from env import SimpleEnv
from qlearning import QLearningAgent

import numpy as np

def main():
    n = 2
    env = SimpleEnv(n)
    action_spaces = env.get_action_spaces()
    agents = [QLearningAgent(action_spaces[i]) for i in range(n)]

    #Training
    train_rounds = 100
    states = env.reset()
    for _ in range(train_rounds):
        actions = [agent.get_action(state) for agent, state in zip(agents, states)]
        new_states, rewards = env.step(actions)
        for i in range(n):
            agents[i].update(states[i], actions[i], new_states[i], rewards[i])
        states = new_states

    #Testing
    test_rounds = 10
    states = env.reset()
    total_score = np.zeros((n, ))
    for _ in range(test_rounds):
        actions = [agent.get_action(state) for agent, state in zip(agents, states)]
        states, rewards = env.step(actions)
        total_score += np.array(rewards)
    avg_score = total_score / test_rounds
    print(avg_score)

if __name__ == "__main__":
    main()
