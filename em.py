"""
EM Algorithm for Hidden Markov Model Learning
Models emotion trends in Reddit threads with 4 hidden states and 32 observation symbols.
"""

import numpy as np
from numpy.typing import NDArray
import os
from typing import Tuple, List


def initialize_parameters(num_states: int = 4, num_observations: int = 32, random_seed: int = None) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Initialize HMM parameters with random distributions.
    
    Args:
        num_states (int): Number of hidden states (default 4)
        num_observations (int): Number of possible observations (default 32)
        random_seed (int): Random seed for reproducibility (default None)
    
    Returns:
        tuple: (initial_dist, transition_matrix, emission_matrix)
            - initial_dist: shape (S,) - random initial state distribution
            - transition_matrix: shape (S, S) - random transition probabilities
            - emission_matrix: shape (S, M) - random emission probabilities
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Random initial state distribution
    initial_dist = np.random.dirichlet(np.ones(num_states))
    
    # Random transition matrix (each row is a random distribution)
    transition_matrix = np.zeros((num_states, num_states))
    for i in range(num_states):
        transition_matrix[i] = np.random.dirichlet(np.ones(num_states))
    
    # Random emission matrix (each row is a random distribution)
    emission_matrix = np.zeros((num_states, num_observations))
    for i in range(num_states):
        emission_matrix[i] = np.random.dirichlet(np.ones(num_observations))
    
    return initial_dist, transition_matrix, emission_matrix


def save_parameters(initial_dist: NDArray, transition_matrix: NDArray, emission_matrix: NDArray,
                   initial_file: str = "initialStateDistribution.txt",
                   transition_file: str = "transitionMatrix.txt",
                   emission_file: str = "emissionMatrix.txt") -> None:
    """
    Save HMM parameters to text files in the required format.
    
    Args:
        initial_dist: Initial state distribution vector (S,)
        transition_matrix: Transition matrix (S, S)
        emission_matrix: Emission matrix (S, M)
        initial_file: Filepath for initial state distribution
        transition_file: Filepath for transition matrix
        emission_file: Filepath for emission matrix
    """
    # Save initial state distribution (one value per line)
    with open(initial_file, 'w') as f:
        for prob in initial_dist:
            f.write(f"{prob:.12f}\n")
    
    # Save transition matrix (one row per line, values separated by tabs)
    with open(transition_file, 'w') as f:
        for row in transition_matrix:
            f.write('\t'.join([f"{prob:.12f}" for prob in row]) + '\n')
    
    # Save emission matrix (one row per line, values separated by tabs)
    with open(emission_file, 'w') as f:
        for row in emission_matrix:
            f.write('\t'.join([f"{prob:.12f}" for prob in row]) + '\n')
    
    print(f"Parameters saved to {initial_file}, {transition_file}, {emission_file}")


def load_data(emission_file: str = "emissionMatrix.txt",
             initial_file: str = "initialStateDistribution.txt",
             obs_file: str = "observations.txt",
             transition_file: str = "transitionMatrix.txt") -> Tuple[NDArray, NDArray, List[NDArray], NDArray]:
    """
    Load HMM parameters and observation sequences from text files.
    
    Args:
        emission_file (str): Filepath for the Emission Matrix (B).
        initial_file (str): Filepath for the Initial State Distribution (Pi).
        obs_file (str): Filepath for the Observation Sequence.
        transition_file (str): Filepath for the Transition Matrix (A).
    
    Returns:
        tuple[NDArray]: A tuple containing four elements:
            (emissions, initials, observations, transitions)
            - emissions - shape (S, M): Emission matrix (S states, M possible observations).
            - initials - shape (S,): Initial state distribution vector.
            - observations - List[NDArray]: List of observation sequences (each thread is a separate sequence).
            - transitions - shape (S, S): Transition matrix.
    """
    # Load initial state distribution
    initials = np.loadtxt(initial_file)
    
    # Load transition matrix
    transitions = np.loadtxt(transition_file)
    
    # Load emission matrix
    emissions = np.loadtxt(emission_file)
    
    # Load observation sequences (each line is a separate thread/sequence)
    observations = []
    with open(obs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                # Parse space-separated integers
                obs_seq = np.array([int(x) for x in line.split()])
                observations.append(obs_seq)
    
    print(f"Loaded {len(observations)} observation sequences")
    print(f"Initial distribution shape: {initials.shape}")
    print(f"Transition matrix shape: {transitions.shape}")
    print(f"Emission matrix shape: {emissions.shape}")
    
    return emissions, initials, observations, transitions


def forward(observations: NDArray, initial_dist: NDArray, transition_matrix: NDArray, 
           emission_matrix: NDArray) -> Tuple[NDArray, float]:
    """
    Forward algorithm with log-space computation for numerical stability.
    
    Args:
        observations: Observation sequence (T,)
        initial_dist: Initial state distribution (S,)
        transition_matrix: Transition matrix (S, S)
        emission_matrix: Emission matrix (S, M)
    
    Returns:
        tuple: (log_alpha, log_likelihood)
            - log_alpha: shape (T, S) - log forward probabilities
            - log_likelihood: scalar - log probability of observation sequence
    """
    T = len(observations)
    S = len(initial_dist)
    
    # Initialize log_alpha in log space
    log_alpha = np.zeros((T, S))
    
    # Convert to log space
    log_initial = np.log(initial_dist + 1e-10)
    log_transition = np.log(transition_matrix + 1e-10)
    log_emission = np.log(emission_matrix + 1e-10)
    
    # Initialize: alpha[0][i] = pi[i] * B[i][obs[0]]
    log_alpha[0] = log_initial + log_emission[:, observations[0]]
    
    # Recursion: alpha[t][j] = sum_i(alpha[t-1][i] * A[i][j]) * B[j][obs[t]]
    for t in range(1, T):
        for j in range(S):
            # log(sum(exp(log_alpha[t-1, i] + log_transition[i, j])))
            log_sum = log_alpha[t-1, 0] + log_transition[0, j]
            for i in range(1, S):
                log_sum = np.logaddexp(log_sum, log_alpha[t-1, i] + log_transition[i, j])
            log_alpha[t, j] = log_sum + log_emission[j, observations[t]]
    
    # Termination: log P(O) = log(sum_i alpha[T-1][i])
    log_likelihood = log_alpha[T-1, 0]
    for i in range(1, S):
        log_likelihood = np.logaddexp(log_likelihood, log_alpha[T-1, i])
    
    return log_alpha, log_likelihood


def backward(observations: NDArray, transition_matrix: NDArray, emission_matrix: NDArray) -> NDArray:
    """
    Backward algorithm with log-space computation for numerical stability.
    
    Args:
        observations: Observation sequence (T,)
        transition_matrix: Transition matrix (S, S)
        emission_matrix: Emission matrix (S, M)
    
    Returns:
        log_beta: shape (T, S) - log backward probabilities
    """
    T = len(observations)
    S = transition_matrix.shape[0]
    
    # Initialize log_beta
    log_beta = np.zeros((T, S))
    
    # Convert to log space
    log_transition = np.log(transition_matrix + 1e-10)
    log_emission = np.log(emission_matrix + 1e-10)
    
    # Initialize: beta[T-1][i] = 1 (log_beta[T-1][i] = 0)
    # Already initialized to 0
    
    # Recursion (backwards): beta[t][i] = sum_j(A[i][j] * B[j][obs[t+1]] * beta[t+1][j])
    for t in range(T-2, -1, -1):
        for i in range(S):
            # log(sum(exp(log_transition[i, j] + log_emission[j, obs[t+1]] + log_beta[t+1, j])))
            log_sum = log_transition[i, 0] + log_emission[0, observations[t+1]] + log_beta[t+1, 0]
            for j in range(1, S):
                log_sum = np.logaddexp(log_sum, 
                                      log_transition[i, j] + log_emission[j, observations[t+1]] + log_beta[t+1, j])
            log_beta[t, i] = log_sum
    
    return log_beta


def e_step(observation_sequences: List[NDArray], initial_dist: NDArray, 
          transition_matrix: NDArray, emission_matrix: NDArray) -> Tuple[dict, float]:
    """
    E-Step: Compute expected sufficient statistics across all observation sequences.
    
    Args:
        observation_sequences: List of observation sequences
        initial_dist: Initial state distribution (S,)
        transition_matrix: Transition matrix (S, S)
        emission_matrix: Emission matrix (S, M)
    
    Returns:
        tuple: (sufficient_statistics, total_log_likelihood)
            - sufficient_statistics: dict containing expected counts
            - total_log_likelihood: sum of log-likelihoods across all sequences
    """
    S = len(initial_dist)
    M = emission_matrix.shape[1]
    
    # Initialize sufficient statistics
    expected_initial = np.zeros(S)
    expected_transitions = np.zeros((S, S))
    expected_emissions = np.zeros((S, M))
    total_log_likelihood = 0.0
    
    # Process each observation sequence
    for obs_seq in observation_sequences:
        if len(obs_seq) == 0:
            continue
        
        T = len(obs_seq)
        
        # Run forward-backward
        log_alpha, log_likelihood = forward(obs_seq, initial_dist, transition_matrix, emission_matrix)
        log_beta = backward(obs_seq, transition_matrix, emission_matrix)
        
        total_log_likelihood += log_likelihood
        
        # Compute gamma (state posteriors): gamma[t][i] = P(q_t = i | O, params)
        # gamma[t][i] = alpha[t][i] * beta[t][i] / P(O)
        log_gamma = log_alpha + log_beta
        # Normalize at each time step
        for t in range(T):
            log_sum = log_gamma[t, 0]
            for i in range(1, S):
                log_sum = np.logaddexp(log_sum, log_gamma[t, i])
            log_gamma[t] -= log_sum
        
        gamma = np.exp(log_gamma)
        
        # Accumulate expected initial state counts
        expected_initial += gamma[0]
        
        # Compute xi (transition posteriors) and accumulate expected transition counts
        # xi[t][i][j] = P(q_t = i, q_{t+1} = j | O, params)
        log_transition = np.log(transition_matrix + 1e-10)
        log_emission = np.log(emission_matrix + 1e-10)
        
        for t in range(T - 1):
            log_xi = np.zeros((S, S))
            for i in range(S):
                for j in range(S):
                    log_xi[i, j] = (log_alpha[t, i] + log_transition[i, j] + 
                                   log_emission[j, obs_seq[t+1]] + log_beta[t+1, j])
            
            # Normalize
            log_sum = -np.inf
            for i in range(S):
                for j in range(S):
                    log_sum = np.logaddexp(log_sum, log_xi[i, j])
            log_xi -= log_sum
            
            xi = np.exp(log_xi)
            expected_transitions += xi
        
        # Accumulate expected emission counts
        for t in range(T):
            for i in range(S):
                expected_emissions[i, obs_seq[t]] += gamma[t, i]
    
    sufficient_statistics = {
        'expected_initial': expected_initial,
        'expected_transitions': expected_transitions,
        'expected_emissions': expected_emissions
    }
    
    return sufficient_statistics, total_log_likelihood


def m_step(sufficient_statistics: dict, epsilon: float = 1e-10) -> Tuple[NDArray, NDArray, NDArray]:
    """
    M-Step: Update parameters based on expected sufficient statistics.
    
    Args:
        sufficient_statistics: Dictionary containing expected counts from E-step
        epsilon: Small constant to avoid zero probabilities
    
    Returns:
        tuple: (new_initial_dist, new_transition_matrix, new_emission_matrix)
    """
    expected_initial = sufficient_statistics['expected_initial']
    expected_transitions = sufficient_statistics['expected_transitions']
    expected_emissions = sufficient_statistics['expected_emissions']
    
    # Update initial state distribution
    new_initial_dist = expected_initial + epsilon
    new_initial_dist /= new_initial_dist.sum()
    
    # Update transition matrix (normalize each row)
    new_transition_matrix = expected_transitions + epsilon
    row_sums = new_transition_matrix.sum(axis=1, keepdims=True)
    new_transition_matrix /= row_sums
    
    # Update emission matrix (normalize each row)
    new_emission_matrix = expected_emissions + epsilon
    row_sums = new_emission_matrix.sum(axis=1, keepdims=True)
    new_emission_matrix /= row_sums
    
    return new_initial_dist, new_transition_matrix, new_emission_matrix


def em_algorithm(observation_sequences: List[NDArray], num_iterations: int,
                initial_dist: NDArray, transition_matrix: NDArray, 
                emission_matrix: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Run the EM algorithm for HMM parameter learning.
    
    Args:
        observation_sequences: List of observation sequences
        num_iterations: Number of EM iterations to run
        initial_dist: Initial state distribution (S,)
        transition_matrix: Initial transition matrix (S, S)
        emission_matrix: Initial emission matrix (S, M)
    
    Returns:
        tuple: (learned_initial_dist, learned_transition_matrix, learned_emission_matrix)
    """
    current_initial = initial_dist.copy()
    current_transition = transition_matrix.copy()
    current_emission = emission_matrix.copy()
    
    print(f"\nStarting EM algorithm with {num_iterations} iterations...")
    print(f"Number of observation sequences: {len(observation_sequences)}")
    
    for iteration in range(num_iterations):
        # E-step: Compute expected sufficient statistics
        sufficient_stats, log_likelihood = e_step(observation_sequences, current_initial, 
                                                  current_transition, current_emission)
        
        # M-step: Update parameters
        current_initial, current_transition, current_emission = m_step(sufficient_stats)
        
        # Log progress
        print(f"Iteration {iteration + 1}/{num_iterations}: Log-Likelihood = {log_likelihood:.4f}")
    
    print("\nEM algorithm completed!")
    return current_initial, current_transition, current_emission


def main():
    """
    Main execution function for EM HMM learning.
    """
    NUM_STATES = 4
    NUM_OBSERVATIONS = 32
    NUM_ITERATIONS = 50
    
    print("=" * 60)
    print("EM Algorithm for HMM Learning - Reddit Emotion Modeling")
    print("=" * 60)
    
    # Check if parameter files exist, if not create them
    if not os.path.exists("initialStateDistribution.txt"):
        print("\nInitializing parameters with random distributions...")
        initial_dist, transition_matrix, emission_matrix = initialize_parameters(
            NUM_STATES, NUM_OBSERVATIONS, random_seed=42
        )
        save_parameters(initial_dist, transition_matrix, emission_matrix)
    else:
        print("\nParameter files already exist. Skipping initialization.")
    
    # Load data
    print("\nLoading data...")
    emission_matrix, initial_dist, observation_sequences, transition_matrix = load_data()
    
    # Run EM algorithm
    learned_initial, learned_transition, learned_emission = em_algorithm(
        observation_sequences, NUM_ITERATIONS, 
        initial_dist, transition_matrix, emission_matrix
    )
    
    # Save learned parameters
    print("\nSaving learned parameters...")
    save_parameters(learned_initial, learned_transition, learned_emission,
                   initial_file="learned_initialStateDistribution.txt",
                   transition_file="learned_transitionMatrix.txt",
                   emission_file="learned_emissionMatrix.txt")
    
    print("\n" + "=" * 60)
    print("Learning complete! Check learned_*.txt files for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()

