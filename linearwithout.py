import gymnasium as gym
import random


# --------------------------------------------------
# 1. FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(state):

    # Normalize CartPole observations
    scale = [2.4, 3.0, 0.2095, 3.5]

    x = []

    for i in range(len(state)):
        value = state[i] / scale[i]

        # Clip values
        value = max(-2.0, min(2.0, value))

        x.append(value)

    p, v, angle, angular_velocity = x

    # Feature vector
    features = [
        1.0,

        # Linear terms
        p,
        v,
        angle,
        angular_velocity,

        # Squared terms
        p ** 2,
        v ** 2,
        angle ** 2,
        angular_velocity ** 2,

        # Interaction terms
        p * v,
        p * angle,
        v * angle,
        angle * angular_velocity,
        p * angular_velocity,
        v * angular_velocity
    ]

    return features


# --------------------------------------------------
# 2. DOT PRODUCT
# --------------------------------------------------

def dot_product(w, x):

    result = 0.0

    for i in range(len(w)):
        result += w[i] * x[i]

    return result


# --------------------------------------------------
# 3. Q-VALUE APPROXIMATION
# --------------------------------------------------

def get_q_value(W, state_features, action):

    return dot_product(
        W[action],
        state_features
    )


# --------------------------------------------------
# 4. EPSILON-GREEDY ACTION SELECTION
# --------------------------------------------------

def epsilon_greedy(W, features, epsilon):

    n_actions = len(W)

    # Exploration
    if random.random() < epsilon:
        return random.randrange(n_actions)

    # Calculate Q-values for every action
    q_values = []

    for action in range(n_actions):

        value = get_q_value(
            W, features, action
        )

        q_values.append(value)

    # Exploitation
    max_value = max(q_values)

    best_actions = [
        a for a in range(n_actions)
        if q_values[a] == max_value
    ]

    return random.choice(best_actions)


# --------------------------------------------------
# 5. LINEAR Q-LEARNING
# --------------------------------------------------

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

    # Find number of features
    initial_state, info = env.reset()

    n_features = len(
        extract_features(initial_state)
    )

    # Initialize weights
    # One weight vector per action

    W = [
        [0.0 for _ in range(n_features)]
        for _ in range(n_actions)
    ]

    rewards_per_episode = []

    for episode in range(episodes):

        state, info = env.reset()

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            # STEP 1: Extract features
            features = extract_features(state)

            # STEP 2: Select action
            action = epsilon_greedy(
                W, features, epsilon
            )

            # STEP 3: Perform action
            next_state, reward, terminated, truncated, info = (
                env.step(action)
            )

            # STEP 4: Calculate current Q-value
            current_q = get_q_value(
                W, features, action
            )

            # STEP 5: Calculate TD target
            if terminated:

                target = reward

            else:

                next_features = extract_features(next_state)

                next_q_values = []

                for a in range(n_actions):

                    q = get_q_value(
                        W, next_features, a
                    )

                    next_q_values.append(q)

                target = (
                    reward + gamma * max(next_q_values)
                )

            # STEP 6: Calculate TD error
            td_error = target - current_q

            # STEP 7: Update weights
            for i in range(n_features):

                W[action][i] += (
                    alpha * td_error * features[i]
                )

            # STEP 8: Move to next state
            state = next_state

            total_reward += reward

        rewards_per_episode.append(total_reward)

        # STEP 9: Decay epsilon
        epsilon = max(
            min_epsilon,
            epsilon * epsilon_decay
        )

        if (episode + 1) % 100 == 0:

            avg_reward = sum(
                rewards_per_episode[-100:]
            ) / 100

            print(
                f"Episode {episode + 1}, "
                f"Average reward: {avg_reward:.2f}, "
                f"Epsilon: {epsilon:.3f}"
            )

    return W, rewards_per_episode


# --------------------------------------------------
# 6. EVALUATION
# --------------------------------------------------

def evaluate(env, W, episodes=20):

    total_rewards = []

    for episode in range(episodes):

        state, info = env.reset()

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            features = extract_features(state)

            # Greedy action selection
            action = max(
                range(len(W)),
                key=lambda a: get_q_value(
                    W, features, a
                )
            )

            state, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

        total_rewards.append(total_reward)

    return sum(total_rewards) / len(total_rewards)


# --------------------------------------------------
# 7. MAIN
# --------------------------------------------------

if __name__ == "__main__":

    env = gym.make("CartPole-v1")

    W, rewards = linear_q_learning(
        env,
        episodes=1000,
        alpha=0.01,
        gamma=0.99,
        epsilon=1.0
    )

    print("\nTraining completed!")

    print(
        "Average evaluation reward:",
        evaluate(env, W)
    )

    env.close()
