import random
import time

import gym
import numpy as np

random.seed(1234)
np.random.seed(1234)


def generate_random_policy():
    # Generates a vector of shape (16,) with an action between 0 and 3 (inclusive)
    return np.random.choice(4, size=((16)))


def evaluate_policy(env, policy, n_episodes=100):
    total_rewards = 0.0

    for _ in range(n_episodes):
        total_rewards += run_episode(env, policy)

    return total_rewards / n_episodes


def run_episode(env, policy, n_episodes=100, render=False):
    total_reward = 0
    s = env.reset()

    for i in range(n_episodes):
        if render:
            env.render()
        s, reward, done, _ = env.step(policy[s])
        total_reward += reward
        if done:
            break

    return total_reward


def crossover(policy1, policy2):
    new_policy = policy1.copy()

    for i in range(16):
        rand = np.random.uniform()
        if rand > 0.5:
            new_policy[i] = policy2[i]

    return new_policy


def mutation(policy, p=0.05):
    new_policy = policy.copy()

    for i in range(16):
        rand = np.random.uniform()
        if rand < p:
            new_policy[i] = np.random.choice(4)

    return new_policy


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1')
    env.reset(seed=0)

    ## Policy search
    n_policy = 100
    n_steps = 20
    start = time.time()
    policy_pop = [generate_random_policy() for _ in range(n_policy)]
    for idx in range(n_steps):
        policy_scores = [evaluate_policy(env, p) for p in policy_pop]
        print('Generation %d : max score = %0.2f' % (idx + 1, max(policy_scores)))
        policy_ranks = list(reversed(np.argsort(policy_scores)))
        elite_set = [policy_pop[x] for x in policy_ranks[:5]]
        select_probs = np.array(policy_scores) / np.sum(policy_scores)
        child_set = [crossover(
            policy_pop[np.random.choice(range(n_policy), p=select_probs)],
            policy_pop[np.random.choice(range(n_policy), p=select_probs)])
            for _ in range(n_policy - 5)]
        mutated_list = [mutation(p) for p in child_set]
        policy_pop = elite_set
        policy_pop += mutated_list
    policy_score = [evaluate_policy(env, p) for p in policy_pop]
    best_policy = policy_pop[np.argmax(policy_score)]

    end = time.time()
    print('Best policy score = %0.2f. Time taken = %4.4f' % (np.max(policy_score), (end - start)))

    ## Evaluation
    # env = wrappers.Monitor(env, '/tmp/frozenlake1', force=True)
    for _ in range(200):
        run_episode(env, best_policy)
    env.close()
