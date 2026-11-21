import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Select an action using epsilon-greedy exploration.

    With probability epsilon:
        Choose a random action.

    Otherwise:
        Choose a greedy action.
    """
    if np.random.random() < epsilon:
        return np.random.randint(len(Q[state]))

    # Randomly break ties between greedy actions
    greedy_actions = np.flatnonzero(
        Q[state] == np.max(Q[state])
    )

    return np.random.choice(greedy_actions)


def epsilon_greedy_probs(Q, state, epsilon):
    """
    Return the action probability distribution
    for an epsilon-greedy policy.
    """
    n_actions = len(Q[state])

    probs = np.ones(n_actions) * (epsilon / n_actions)

    greedy_actions = np.flatnonzero(
        Q[state] == np.max(Q[state])
    )

    probs[greedy_actions] += (
        (1 - epsilon) / len(greedy_actions)
    )

    return probs
