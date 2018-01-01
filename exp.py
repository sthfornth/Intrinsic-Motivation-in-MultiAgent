
from env import SimpleEnv, AxisEnv
from qlearning import QLearningAgent

import numpy as np


def run(adjust=False, ind_reward_weight=1, avg_reward_weigth=0, verbose=True):
    n = 5
    env = AxisEnv(n, 10)
    action_spaces = env.get_action_spaces()
    alpha = 0.1
    gamma = 0.99
    eps = 0.2
    beta = 0.001
    # beta1, beta2 = 1 - beta, 1 + beta
    agents = [QLearningAgent(i, action_spaces[i], alpha, gamma, eps) for i in range(n)]

    #Training
    rounds = 0
    cnt = 0
    train_nr_steps = 2000
    while cnt < train_nr_steps:
        states = env.reset()
        is_over = False
        scores = np.zeros((n, ))
        while not is_over and cnt <= train_nr_steps:
            actions = [agent.get_action(state) for agent, state in zip(agents, states)]
            cnt += 1
            new_states, rewards, is_over = env.step(actions)
            team_reward = np.sum(rewards)
            avg_reward = float(team_reward) / n
            scores += np.array(rewards)
            for i in range(n):
                reward = rewards[i] * ind_reward_weight + avg_reward * avg_reward_weigth
                agents[i].update(states[i], actions[i], new_states[i], reward)
            states = new_states
        if adjust:
            for i in range(n):
                # print(i, rewards[i], avg_reward)
                if rewards[i] >= avg_reward:
                    agents[i].adjust(-beta, -2 * beta)
                else:
                    agents[i].adjust(beta, 2 * beta)
        if verbose:
            str_lr, str_eps = '', ''
            for agent in agents:
                str_lr += '{:.2f} '.format(agent._alpha) 
                str_eps += '{:.2f} '.format(agent._eps) 
            print('Train', rounds, scores, str_lr, str_eps)
        rounds += 1

    #Testing
    test_rounds = 10
    total_score = np.zeros((n, ))
    for _ in range(test_rounds):
        states = env.reset()
        is_over = False
        scores = np.zeros((n, ))
        while not is_over:
            actions = [agent.get_action(state, False) for agent, state in zip(agents, states)]
            new_states, rewards, is_over = env.step(actions)
            states = new_states
            scores += np.array(rewards)
        total_score += scores
        if verbose:
            print('Test', _, scores)
    avg_score = total_score / test_rounds
    if verbose:
        print(avg_score)
    return avg_score

def main():
    m = 50
    st = []
    for i in range(m):
        score = run(adjust=True, verbose=False)
        print('st', i, np.sum(score), score)
        st.append(np.sum(score))
    sf = []
    for i in range(m):
        score = run(adjust=False, verbose=False)
        print('sf', i, np.sum(score), score)
        sf.append(np.sum(score))
    sa = []
    for i in range(m):
        score = run(adjust=False, verbose=False, ind_reward_weight=0, avg_reward_weigth=1)
        print('sa', i, np.sum(score), score)
        sa.append(np.sum(score))
    print(np.mean(st), np.std(st), np.min(st), np.max(st))
    print(np.mean(sf), np.std(sf), np.min(sf), np.max(sf))
    print(np.mean(sa), np.std(sa), np.min(sa), np.max(sa))

if __name__ == "__main__":
    main()
