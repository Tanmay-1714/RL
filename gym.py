import gymnasium as gym
import numpy as np

from algorithms.q_learning import q_learning
from algorithms.sarsa import sarsa
from algorithms.expected_sarsa import expected_sarsa
from algorithms.double_q_learning import double_q_learning


env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

# Q-learning
Q, rewards = q_learning(env)

# SARSA
Q_sarsa, rewards_sarsa = sarsa(env)

# Expected SARSA
Q_expected, rewards_expected = expected_sarsa(env)

# Double Q-learning
Q1, Q2, rewards_double = double_q_learning(env)

# Combined Double Q estimate
Q_double = Q1 + Q2


def evaluate(env, Q, episodes=100):

    total_rewards = []

    for _ in range(episodes):

        state, info = env.reset()

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            action = np.argmax(Q[state])

            state, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

        total_rewards.append(total_reward)

    return np.mean(total_rewards)


print("Q-learning:", evaluate(env, Q))
print("SARSA:", evaluate(env, Q_sarsa))
print("Expected SARSA:", evaluate(env, Q_expected))
print("Double Q-learning:", evaluate(env, Q_double))

env.close()
