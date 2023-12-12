import time
from TestCaseGen import generate_test_case
from Posterior import posterior_decoding
from Viterbi import viterbi_decode


max_state_size = 200
max_observation_size = 200

#default Values
def_observation_size = 2
def_state_size = 2
number_of_iterations = 200


for observation_size in range(def_observation_size, def_observation_size + 1):
    for state_size in range(10, max_state_size + 1,10):
        runtime_viterbi = 0
        runtime_posterior = 0

        for iter in range(1, number_of_iterations) :
            observations, start_prob, transition_prob, emission_prob = generate_test_case(observation_size, state_size)
            states = list(range(state_size))


            start_time_viterbi = time.time()
            best_path, best_path_prob = viterbi_decode(observations, states, start_prob, transition_prob, emission_prob)
            end_time_viterbi = time.time()
            runtime_viterbi += end_time_viterbi - start_time_viterbi


            start_time_posterior = time.time()
            best_path_, best_path_prob_ = posterior_decoding(observations, states, start_prob, transition_prob, emission_prob)
            end_time_posterior = time.time()
            runtime_posterior += end_time_posterior - start_time_posterior

        print("STATE SIZE [",state_size,"] Viterbi => [ {:.2f} ]".format(runtime_viterbi),"[ {:.2f} ] <= Posterior".format(runtime_posterior) )
