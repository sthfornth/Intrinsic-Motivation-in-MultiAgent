
from env import SimpleEnv, AxisEnv
from qlearning import QLearningAgent

import numpy as np


def run(train_nr_steps, beta=None, ind_reward_weight=1, avg_reward_weigth=0, verbose=False):
    n = 5
    env = AxisEnv(n, 10)
    action_spaces = env.get_action_spaces()
    alpha = 0.1
    gamma = 0.99
    eps = 0.2
    # beta1, beta2 = 1 - beta, 1 + beta
    agents = [QLearningAgent(i, action_spaces[i], alpha, gamma, eps) for i in range(n)]

    #Training
    rounds = 0
    cnt = 0
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
        if beta is not None:
            for i in range(n):
                # print(i, rewards[i], avg_reward)
                if rewards[i] >= avg_reward:
                    agents[i].adjust(-beta, -beta)
                else:
                    agents[i].adjust(beta, beta)
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
    n = 2000
    m = 500
    st001, st0001, sf, sa, sm = [], [], [], [], []
    for i in range(m):
        score = run(n, beta=0.01)
        print('st001', i, np.sum(score), score)
        st001.append(np.sum(score))
        score = run(n, beta=0.001)
        print('st0001', i, np.sum(score), score)
        st0001.append(np.sum(score))
        score = run(n)
        print('sf', i, np.sum(score), score)
        sf.append(np.sum(score))
        score = run(n, ind_reward_weight=0, avg_reward_weigth=1)
        print('sa', i, np.sum(score), score)
        sa.append(np.sum(score))
        score = run(n, ind_reward_weight=0.5, avg_reward_weigth=0.5)
        print('sm', i, np.sum(score), score)
        sm.append(np.sum(score))

    # print(np.mean(st), np.std(st), np.min(st), np.max(st))
    print(np.mean(st001), np.std(st001), np.min(st001), np.max(st001))
    print(np.mean(st0001), np.std(st0001), np.min(st0001), np.max(st0001))
    print(np.mean(sf), np.std(sf), np.min(sf), np.max(sf))
    print(np.mean(sa), np.std(sa), np.min(sa), np.max(sa))
    print(np.mean(sm), np.std(sm), np.min(sm), np.max(sm))

if __name__ == "__main__":
    main()
