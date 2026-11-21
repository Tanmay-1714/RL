import numpy as np
from algorithms.utils import epsilon_greedy


def q_learning(
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
        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            action = epsilon_greedy(Q, state, epsilon)

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            # Do not bootstrap from a true terminal state.
            if terminated:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])

            # TD update
            Q[state, action] += alpha * (
                target - Q[state, action]
            )

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        epsilon = max(
            min_epsilon,
            epsilon * epsilon_decay
        )

    return Q, rewards_per_episode
