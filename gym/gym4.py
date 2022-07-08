# -*- coding: utf-8 -*-
# ref: http://taotao54321.hatenablog.com/entry/2016/11/08/180245

import numpy as np
import gym

# Q learning params
from gym.wrappers import RecordEpisodeStatistics, TransformReward

ALPHA = 0.1  # learning rate
GAMMA = 0.99  # reward discount
LEARNING_COUNT = 100000
TEST_COUNT = 1000

EPS = 0.1

TURN_LIMIT = 100
IS_MONITOR = False

class Agent:
    def __init__(self, env):
        self.env = env
        self.episode_reward = 0.0
        self.Q = np.zeros(16 * 4).reshape(16, 4).astype(np.float32)

    def learn(self):
        # one episode learning
        state = self.env.reset()
        # self.env.render()

        for t in range(TURN_LIMIT):
            if np.random.rand() < EPS:  # explore
                act = self.env.action_space.sample()  # random
            else:  # exploit
                act = np.argmax(self.Q[state])
            next_state, reward, done, info = self.env.step(act)
            q_next_max = np.max(self.Q[next_state])
            # Q <- Q + a(Q' - Q)
            # <=> Q <- (1-a)Q + a(Q')
            self.Q[state][act] = (1 - ALPHA) * self.Q[state][act] \
                                 + ALPHA * (reward + GAMMA * q_next_max)

            # self.env.render()
            if done:
                return reward
            else:
                state = next_state
        return 0.0  # over limit

    def test(self):
        state = self.env.reset()
        for t in range(TURN_LIMIT):
            act = np.argmax(self.Q[state])
            next_state, reward, done, info = self.env.step(act)
            if done:
                return reward
            else:
                state = next_state
        return 0.0  # over limit

def main():
    env = gym.make("FrozenLake-v1")
    if IS_MONITOR:
        env = RecordEpisodeStatistics(env)
    agent = Agent(env)

    print("###### LEARNING #####")
    reward_total = 0.0
    for i in range(LEARNING_COUNT):
        reward_total += agent.learn()
    print("episodes      : {}".format(LEARNING_COUNT))
    print("total reward  : {}".format(reward_total))
    print("average reward: {:.2f}".format(reward_total / LEARNING_COUNT))
    print("Q Value       :{}".format(agent.Q))

    print("###### TEST #####")
    reward_total = 0.0
    for i in range(TEST_COUNT):
        reward_total += agent.test()
    print("episodes      : {}".format(TEST_COUNT))
    print("total reward  : {}".format(reward_total))
    print("average reward: {:.2f}".format(reward_total / TEST_COUNT))


if __name__ == "__main__":
    main()
