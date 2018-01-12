
from env import SimpleEnv, AxisEnv, FoodEnv
from qlearning import QLearningAgent

import argparse
import numpy as np
from numpy import random
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--alpha', default=0.3, type=float, help='learning rate')
parser.add_argument('-e', '--eps', default=0.2, type=float, help='epsilon greedy')
parser.add_argument('-n', default=2000, type=int, help='num of updates per agent')
parser.add_argument('-m', default=50, type=int, help='num of runs')
parser.add_argument('-env', '--env', default='axis', type=str, help='env name, axis or food[x] for x-th map')
args = parser.parse_args()

def run(train_nr_steps, beta=None, ind_reward_weight=1, team_reward_weigth=0, seed=None, verbose=False):
    if seed is not None:
        random.seed(seed)
    if args.env == 'axis':
        n = 5
        env = AxisEnv(n, 10)
    else:
        n = 4
        case = int(args.env[4:])
        env = FoodEnv(n, 4, n, flag=case)
    action_spaces = env.get_action_spaces()
    alpha = args.alpha
    eps = args.eps
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
            new_states, rewards, team_reward, is_over = env.step(actions)
            scores += np.array(rewards)
            for i in range(n):
                reward = rewards[i] * ind_reward_weight + team_reward * team_reward_weigth
                agents[i].update(states[i], actions[i], new_states[i], reward)
            states = new_states
        if beta is not None:
            avg_scores = np.mean(scores)
            for i in range(n):
                # print(i, scores[i], avg_scores)
                if scores[i] >= avg_scores:
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
    sum_team_reward = 0
    for _ in range(test_rounds):
        states = env.reset()
        is_over = False
        scores = np.zeros((n, ))
        while not is_over:
            actions = [agent.get_action(state, False) for agent, state in zip(agents, states)]
            new_states, rewards, team_reward, is_over = env.step(actions)
            states = new_states
            sum_team_reward += team_reward
            scores += np.array(rewards)
        total_score += scores
        if verbose:
            print('Test', _, scores)
    avg_score = total_score / float(test_rounds)
    if verbose:
        print(avg_score)
    # return np.sum(avg_score)
    return np.sum(avg_score), sum_team_reward / float(test_rounds)

def run_wrap(args):
    return run(*args)

def main():
    n = args.n
    m = args.m
    pool = Pool(12)
    print(args.env, 'n', n, 'm', m, 'lr', args.alpha, 'eps', args.eps)

    def work(n, beta=None, ind_reward_weight=1, team_reward_weigth=0):
        args = (n, beta, ind_reward_weight, team_reward_weigth)
        all_args = [args + (i * i, ) for i in range(m)]
        return list(pool.map(run_wrap, all_args))

    def output(name, s):
        ind_score = []
        team_score = []
        for i in s:
            ind_score.append(i[0])
            team_score.append(i[1])
        s = ind_score
        print(name + ' ind', np.mean(s), np.std(s), np.min(s), np.max(s))
        s = team_score
        print(name + ' team', np.mean(s), np.std(s), np.min(s), np.max(s))

    output('st005', work(n, beta=0.05))
    output('st001', work(n, beta=0.01))
    output('st0001', work(n, beta=0.001))
    output('st0001 mix', work(n, beta=0.001, ind_reward_weight=0.5, team_reward_weigth=0.5))
    output('ind score', work(n))
    output('team score', work(n, ind_reward_weight=0, team_reward_weigth=1))
    output('mixed score', work(n, ind_reward_weight=0.5, team_reward_weigth=0.5))

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
