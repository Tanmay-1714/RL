import numpy as np


def cartpole_features(state):
    """
    Convert CartPole's continuous state into features.

    State:
        [cart_position, cart_velocity,
         pole_angle, pole_angular_velocity]
    """

    scale = np.array([2.4, 3.0, 0.2095, 3.5])

    x = np.clip(
        np.asarray(state, dtype=np.float64) / scale,
        -2.0,
        2.0
    )

    p, v, angle, angular_velocity = x

    features = np.array([
        1.0,
        p, v, angle, angular_velocity,
        p**2, v**2, angle**2, angular_velocity**2,
        p * v,
        p * angle,
        v * angle,
        angle * angular_velocity,
        p * angular_velocity,
        v * angular_velocity
    ])

    return features


def linear_epsilon_greedy(W, features, epsilon):
    """
    W shape: (n_actions, n_features)
    """

    if np.random.random() < epsilon:
        return np.random.randint(W.shape[0])

    q_values = W @ features

    return np.random.choice(
        np.flatnonzero(q_values == np.max(q_values))
    )


def linear_q_learning(
    env,
    episodes=1000,
    alpha=0.01,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.01
):

    n_actions = env.action_space.n

    # Determine number of features
    initial_state, _ = env.reset()
    n_features = len(cartpole_features(initial_state))

    # One weight vector per action
    W = np.zeros((n_actions, n_features))

    rewards_per_episode = []

    for episode in range(episodes):

        state, info = env.reset()

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            features = cartpole_features(state)

            action = linear_epsilon_greedy(
                W, features, epsilon
            )

            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            prediction = W[action] @ features

            if terminated:
                target = reward

            else:
                next_features = cartpole_features(next_state)

                next_q_values = W @ next_features

                target = (
                    reward + gamma * np.max(next_q_values)
                )

            td_error = target - prediction

            # Semi-gradient update
            W[action] += alpha * td_error * features

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        epsilon = max(
            min_epsilon,
            epsilon * epsilon_decay
        )

    return W, rewards_per_episode
