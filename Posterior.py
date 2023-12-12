import numpy as np

def posterior_decoding(observations, states, start_prob, transition_prob, emission_prob):
    num_states = len(states)
    num_obs = len(observations)
    
    # Initialize the trellis matrix
    trellis = np.zeros((num_states, num_obs))
    backpointers = np.zeros((num_states, num_obs), dtype=int)
    
    # Initialization step
    for i in range(num_states):
        trellis[i, 0] = start_prob[i] * emission_prob[i, observations[0]]
    




    # Recursion step for Posterior
    for t in range(1, num_obs):
        for j in range(num_states):
            trellis[j, t] = np.max([trellis[i, t-1] * transition_prob[i, j] * emission_prob[j, observations[t]] for i in range(num_states)])
            backpointers[j, t] = np.argmax([trellis[i, t-1] * transition_prob[i, j] for i in range(num_states)])


            
    
    # Termination step
    best_path_prob = np.max(trellis[:, num_obs - 1])
    best_path_pointer = np.argmax(trellis[:, num_obs - 1])
    
    # Backtrace to find the best path
    best_path = [best_path_pointer]
    for t in range(num_obs - 1, 0, -1):
        best_path_pointer = backpointers[best_path_pointer, t]
        best_path.insert(0, best_path_pointer)
    
    return best_path, best_path_prob

