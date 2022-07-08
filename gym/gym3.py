import math

import gym
import numpy as np

# 1. Load Environment and Q-table structure
env = gym.make('FrozenLake-v1')

nA = env.env.action_space.n
nS = env.env.observation_space.n

Q = np.zeros([nS, nA])
grid_size = int(math.sqrt(nS))


def setup_invalid_actions(m, grid_size):
    for i in range(grid_size):
        for j in range(grid_size):
            idx = grid_size * i + j
            if j == 0:
                m[idx, 0] = 0
            if j == grid_size - 1:
                m[idx, 2] = 0
            if i == 0:
                m[idx, 3] = 0
            if i == grid_size - 1:
                m[idx, 1] = 0


valid_actions = np.ones([nS, nA])
setup_invalid_actions(valid_actions, grid_size)

# env.observation.n, env.action_space.n gives number of states and action in env loaded
# 2. Parameters of Q-learning
alpha = .8  # learning rate
gamma = .9    # reward discount rate
episodes = 100000
episode_break = int(episodes / 10)

rev_list = []  # rewards per episode calculate
# 3. Q-learning Algorithm
for i in range(episodes+1):
    # Reset environment
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        if i % episode_break == 0:
            env.render()
        j += 1
        # Choose action from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, nA) * (10. / (i + 1)))
        while valid_actions[s, a] == 0.:
            a = np.argmax(Q[s, :] + np.random.randn(1, nA) * (10. / (i + 1)))

        # Get new state & reward from environment
        s1, r, d, _ = env.step(a)
        # if d:
        #     if r == 0:
        #         r = -1.
        #     else:
        #         r = 1.
        # else:
        #     r = -0.01

        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + alpha * (r + gamma * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
        if d:
            break
    rev_list.append(rAll)
    if i % episode_break == 0:
        print(Q)
        print("i is: ", i)
        env.render()
print("Reward Sum on all episodes " + str(sum(rev_list) / episodes))
print("Final Values Q-Table")
print(Q)
