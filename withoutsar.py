

import gymnasium as gym
import random


def create_q_table(n_states, n_actions):
    return [[0.0 for _ in range(n_actions)]
            for _ in range(n_states)]


def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(len(Q[state]))

    max_value = max(Q[state])

    best_actions = [
        a for a in range(len(Q[state]))
        if Q[state][a] == max_value
    ]

    return random.choice(best_actions)


def epsilon_greedy_probs(Q, state, epsilon):
    """
    Calculate probability of selecting each action
    under an epsilon-greedy policy.
    """

    n_actions = len(Q[state])

    probs = [epsilon / n_actions] * n_actions

    max_value = max(Q[state])

    best_actions = [
        a for a in range(n_actions)
        if Q[state][a] == max_value
    ]

    # Distribute exploitation probability among
    # all greedy actions, including ties.
    greedy_probability = (1 - epsilon) / len(best_actions)

    for action in best_actions:
        probs[action] += greedy_probability

    return probs


def expected_sarsa(
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

    Q = create_q_table(n_states, n_actions)

    rewards_per_episode = []

    for episode in range(episodes):

        state, info = env.reset()

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            # 1. Select action
            action = epsilon_greedy(Q, state, epsilon)

            # 2. Take action
            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            # 3. Calculate target
            if terminated:
                target = reward

            else:
                # Probability distribution over next actions
                probs = epsilon_greedy_probs(
                    Q, next_state, epsilon
                )

                # Expected Q-value
                expected_q = 0.0

                for a in range(n_actions):
                    expected_q += probs[a] * Q[next_state][a]

                target = reward + gamma * expected_q

            # 4. TD update
            td_error = target - Q[state][action]

            Q[state][action] += alpha * td_error

            # 5. Move to next state
            state = next_state

            total_reward += reward

        rewards_per_episode.append(total_reward)

        # 6. Decay epsilon
        epsilon = max(
            min_epsilon,
            epsilon * epsilon_decay
        )

    return Q, rewards_per_episode


def evaluate(env, Q, episodes=100):

    total_rewards = []

    for _ in range(episodes):

        state, info = env.reset()

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            action = Q[state].index(max(Q[state]))

            state, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

        total_rewards.append(total_reward)

    return sum(total_rewards) / len(total_rewards)


if __name__ == "__main__":

    env = gym.make(
        "FrozenLake-v1",
        is_slippery=False
    )

    Q, rewards = expected_sarsa(
        env,
        episodes=10000,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0
    )

    print("Learned Q-table:")

    for state, values in enumerate(Q):
        print(f"State {state}: {values}")

    print(
        "\nAverage evaluation reward:",
        evaluate(env, Q)
    )

    env.close()
