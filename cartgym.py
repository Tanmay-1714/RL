import gymnasium as gym
import numpy as np

from algorithms.linear_q_learning import (
    linear_q_learning,
    cartpole_features
)


env = gym.make("CartPole-v1")

W, rewards = linear_q_learning(
    env,
    episodes=1000,
    alpha=0.01,
    gamma=0.99
)


# Evaluate the learned policy
def evaluate(env, W, episodes=20):

    total_rewards = []

    for _ in range(episodes):

        state, info = env.reset()

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            features = cartpole_features(state)

            action = np.argmax(W @ features)

            state, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

        total_rewards.append(total_reward)

    return np.mean(total_rewards)


print("Average evaluation reward:", evaluate(env, W))

env.close()
