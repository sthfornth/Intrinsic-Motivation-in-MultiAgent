
from env import SimpleEnv, AxisEnv
from qlearning import QLearningAgent

import numpy as np
from numpy import random
from multiprocessing import Pool

def run(train_nr_steps, alpha=0.3, beta=None, eps=0.2, ind_reward_weight=1, avg_reward_weigth=0, seed=None, verbose=False):
    if seed is not None:
        random.seed(seed)
    n = 5
    env = AxisEnv(n, 10)
    action_spaces = env.get_action_spaces()
    gamma = 0.99
    # beta1, beta2 = 1 - beta, 1 + beta
    agents = [QLearningAgent(i, action_spaces[i], alpha, gamma, eps) for i in range(n)]
    norm_alpha = True

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
        if norm_alpha:
            s = 0
            for i in range(n):
                s += agents[i]._alpha
            s = alpha * n / s
            for i in range(n):
                agents[i]._alpha *= s

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
    return np.sum(avg_score)

def run_wrap(args):
    return run(*args)

def main():
    alpha = 0.3
    eps = 0.2
    n = 2000
    m = 50
    pool = Pool(12)
    print('n', n, 'm', m, 'lr', alpha, 'eps', eps)

    def work(n, alpha=alpha, beta=None, eps=eps, ind_reward_weight=1, avg_reward_weigth=0):
        args = (n, alpha, beta, eps, ind_reward_weight, avg_reward_weigth)
        all_args = [args + (i * i, ) for i in range(m)]
        return list(pool.map(run_wrap, all_args))

    st005 = work(n, beta=0.05)
    print('st005', np.mean(st005), np.std(st005), np.min(st005), np.max(st005))

    st001 = work(n, beta=0.01)
    print('st001', np.mean(st001), np.std(st001), np.min(st001), np.max(st001))

    st0001 = work(n, beta=0.001)
    print('st0001', np.mean(st0001), np.std(st0001), np.min(st0001), np.max(st0001))

    sf = work(n)
    print('score individual', np.mean(sf), np.std(sf), np.min(sf), np.max(sf))

    sa = work(n, ind_reward_weight=0, avg_reward_weigth=1)
    print('score team', np.mean(sa), np.std(sa), np.min(sa), np.max(sa))

    # sm = work(n, ind_reward_weight=0.5, avg_reward_weigth=0.5)
    # print(np.mean(sm), np.std(sm), np.min(sm), np.max(sm))

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
