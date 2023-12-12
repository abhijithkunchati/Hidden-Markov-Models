import numpy as np

def generate_test_case(observation_size, state_size):
    # Generate random probabilities for start, transition, and emission matrices
    start_prob = np.random.dirichlet(np.ones(state_size), size=1)[0]
    transition_prob = np.random.dirichlet(np.ones(state_size), size=state_size)
    emission_prob = np.random.dirichlet(np.ones(observation_size), size=state_size)

    # Generate random sequence of observations
    observations = np.random.randint(low=0, high=observation_size, size=observation_size)

    return observations, start_prob, transition_prob, emission_prob