from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray

def load_data_viterbi_multi(
    emission_file: str = "emissionMatrix.txt",
    initial_file: str = "initialStateDistribution.txt",
    obs_file: str = "observations.txt",
    transition_file: str = "transitionMatrix.txt",
) -> Tuple[NDArray, NDArray, List[NDArray], NDArray]:
    """
    Loads the HMM parameters and a *list* of observation sequences
    (one sequence per line in obs_file).

    Returns:
        emissions:   (S, M)  emission matrix B
        initials:    (S,)    initial distribution pi
        sequences:   list of length L; each element is a 1D array (T_l,)
        transitions: (S, S)  transition matrix A
    """
    emissions = np.loadtxt(emission_file)         # (S, M)
    initials = np.loadtxt(initial_file)           # (S,)
    transitions = np.loadtxt(transition_file)     # (S, S)

    sequences: List[NDArray] = []
    with open(obs_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                # Skip empty lines
                continue
            obs = np.fromstring(line, sep=" ", dtype=int)
            if obs.size > 0:
                sequences.append(obs)

    return emissions, initials, sequences, transitions

def viterbi(
    emissions: NDArray,
    initials: NDArray,
    observations: NDArray,
    transitions: NDArray
) -> NDArray:
    """
    Implements the Viterbi algorithm to find the most likely sequence of hidden
    states given a sequence of observations and the HMM parameters.

    Use log-probabilities to avoid underflow issues with small probability values.

    Args:
        emissions (NDArray): The Emission Matrix (B) of shape (S, M).
        initials (NDArray): The Initial State Distribution (Pi) of shape (S,).
        observations (NDArray): The sequence of observations of shape (T,).
        transitions (NDArray): The Transition Matrix (A) of shape (S, S).

    Returns:
        NDArray: The optimal path (most likely sequence of hidden states)
            of shape (T,) as an array of integer indices.
    """
    N = emissions.shape[0]      # number of states S
    T = observations.shape[0]   # length of this sequence

    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T), dtype=int)

    # Use np.clip to avoid log(0)
    log_emissions   = np.log(np.clip(emissions,   1e-12, 1.0))
    log_initials    = np.log(np.clip(initials,    1e-12, 1.0))
    log_transitions = np.log(np.clip(transitions, 1e-12, 1.0))

    # t = 0 initialization
    viterbi[:, 0] = log_initials + log_emissions[:, observations[0]]

    # t >= 1 recursion
    for t in range(1, T):
        # trans_probs[i, j] = viterbi[i, t-1] + log A[i, j]
        trans_probs = viterbi[:, t - 1, np.newaxis] + log_transitions
        backpointer[:, t] = np.argmax(trans_probs, axis=0)      # argmax over previous states
        viterbi[:, t]     = np.max(trans_probs, axis=0) \
                            + log_emissions[:, observations[t]] # Add emission probabilities

    # Backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(viterbi[:, -1])
    for t in range(T - 2, -1, -1):
        path[t] = backpointer[path[t + 1], t + 1]

    return path

def run_viterbi_for_all_sequences(
    emissions: NDArray,
    initials: NDArray,
    sequences: List[NDArray],
    transitions: NDArray,
) -> List[NDArray]:
    """
    Run Viterbi on every observation sequence.

    Returns:
        List of paths; paths[l] corresponds to sequences[l]'s hidden state sequence.
    """
    decoded_paths: List[NDArray] = []
    for obs in sequences:
        path = viterbi(emissions, initials, obs, transitions)
        decoded_paths.append(path)
    return decoded_paths

def save_paths(paths: List[NDArray], outfile: str = "viterbi_states.txt") -> None:
    with open(outfile, "w") as f:
        for path in paths:
            line = " ".join(str(int(s)) for s in path)
            f.write(line + "\n")