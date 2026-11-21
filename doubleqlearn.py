import numpy as np
from algorithms.utils import epsilon_greedy


def double_q_learning(
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

    Q1 = np.zeros((n_states, n_actions))
    Q2 = np.zeros((n_states, n_actions))

    rewards_per_episode = []

    for episode in range(episodes):

        state, info = env.reset()

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            # Behaviour policy uses combined estimates
            action = epsilon_greedy(
                Q1 + Q2, state, epsilon
            )

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            if terminated:
                target = reward

            else:
                if np.random.random() < 0.5:

                    # Select using Q1
                    best_action = np.argmax(
                        Q1[next_state]
                    )

                    # Evaluate using Q2
                    target = (
                        reward
                        + gamma * Q2[next_state, best_action]
                    )

                else:

                    # Select using Q2
                    best_action = np.argmax(
                        Q2[next_state]
                    )

                    # Evaluate using Q1
                    target = (
                        reward
                        + gamma * Q1[next_state, best_action]
                    )

            # Randomly update one Q-table
            if np.random.random() < 0.5:

                Q1[state, action] += alpha * (
                    target - Q1[state, action]
                )

            else:

                Q2[state, action] += alpha * (
                    target - Q2[state, action]
                )

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        epsilon = max(
            min_epsilon,
            epsilon * epsilon_decay
        )

    return Q1, Q2, rewards_per_episode
