import numpy as np
from algorithms.utils import epsilon_greedy


def sarsa(
    env,
    episodes=10000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.9995,
    min_epsilon=0.01
):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    rewards_per_episode = []

    for episode in range(episodes):

        state, info = env.reset()

        action = epsilon_greedy(Q, state, epsilon)

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            if terminated:
                target = reward

                Q[state, action] += alpha * (
                    target - Q[state, action]
                )

                total_reward += reward
                break

            # Select next action using the same behaviour policy
            next_action = epsilon_greedy(
                Q, next_state, epsilon
            )

            target = (
                reward + gamma * Q[next_state, next_action]
            )

            Q[state, action] += alpha * (
                target - Q[state, action]
            )

            state = next_state
            action = next_action

            total_reward += reward

        rewards_per_episode.append(total_reward)

        epsilon = max(
            min_epsilon,
            epsilon * epsilon_decay
        )

    return Q, rewards_per_episode
