import gymnasium as gym
import random


def create_q_table(n_states, n_actions):
    """
    Create a Q-table initialized with zeros.
    Rows represent states.
    Columns represent actions.
    """
    Q = []

    for state in range(n_states):
        Q.append([0.0] * n_actions)

    return Q


def epsilon_greedy(Q, state, epsilon):
    """
    Select an action using epsilon-greedy policy.
    """

    # Exploration
    if random.random() < epsilon:
        return random.randrange(len(Q[state]))

    # Exploitation
    max_value = max(Q[state])

    # Handle ties randomly
    best_actions = []

    for action in range(len(Q[state])):
        if Q[state][action] == max_value:
            best_actions.append(action)

    return random.choice(best_actions)


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
                target = reward + gamma * max(Q[next_state])

            # 4. Calculate TD error
            td_error = target - Q[state][action]

            # 5. Update Q-value
            Q[state][action] += alpha * td_error

            # 6. Move to next state
            state = next_state

            total_reward += reward

        rewards_per_episode.append(total_reward)

        # 7. Decay epsilon
        epsilon = max(
            min_epsilon,
            epsilon * epsilon_decay
        )

    return Q, rewards_per_episode


def evaluate(env, Q, episodes=100):

    total_rewards = []

    for episode in range(episodes):

        state, info = env.reset()

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            # Greedy action selection
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

    Q, rewards = q_learning(
        env,
        episodes=10000,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0
    )

    print("Learned Q-table:")

    for state, values in enumerate(Q):
        print(f"State {state}: {values}")

    avg_reward = evaluate(env, Q)

    print("\nAverage evaluation reward:", avg_reward)

    env.close()
