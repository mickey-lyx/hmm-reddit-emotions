from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from collections import defaultdict
from inference import viterbi


def load_data_with_labels(
    obs_file: str = "observations.txt",
    labels_file: str = "labels.txt",
    map_file: str = "map.txt",
    emission_file: str = "emissionMatrix.txt",
    initial_file: str = "initialStateDistribution.txt",
    transition_file: str = "transitionMatrix.txt"
) -> Tuple[NDArray, NDArray, NDArray, List[NDArray], List[NDArray]]:
    """
    Load HMM parameters, observations, and labels.
    
    Data format (following em.py and inference.py):
    - observations.txt: each line is one thread's observation sequence (space-separated)
    - labels.txt: each line is one thread's label sequence (space-separated)
    - map.txt: each line is one thread ID
    
    Returns:
        Tuple of (emissions, initials, transitions, obs_by_thread, labels_by_thread)
    """
    # Load HMM parameters
    emissions = np.loadtxt(emission_file)
    initials = np.loadtxt(initial_file)
    transitions = np.loadtxt(transition_file)
    
    # Load observations (each line is a thread's sequence)
    observations = []
    with open(obs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                obs_seq = np.fromstring(line, sep=" ", dtype=int)
                if obs_seq.size > 0:
                    observations.append(obs_seq)
    
    # Load labels (each line is a thread's label sequence)
    labels = []
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                label_seq = np.fromstring(line, sep=" ", dtype=int)
                if label_seq.size > 0:
                    labels.append(label_seq)
    
    # Load thread IDs (each line is a thread ID)
    thread_ids = []
    with open(map_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                thread_ids.append(line)
    
    # Verify counts match
    assert len(observations) == len(labels) == len(thread_ids), \
        f"Count mismatch: observations={len(observations)}, labels={len(labels)}, map={len(thread_ids)}"
    
    print(f"Loaded {len(observations)} threads")
    print(f"Emission matrix shape: {emissions.shape}")
    print(f"Initial distribution shape: {initials.shape}")
    print(f"Transition matrix shape: {transitions.shape}")
    
    return emissions, initials, transitions, observations, labels


def learn_state_label_mapping(
    emissions: NDArray,
    initials: NDArray,
    transitions: NDArray,
    observations: List[NDArray],
    labels: List[NDArray],
    num_states: int,
    num_labels: int,
) -> Tuple[NDArray, NDArray]:
    """
    Learn a mapping from HMM states to emotion labels via majority voting.
    
    Args:
        emissions: Emission matrix (S, M)
        initials: Initial state distribution (S,)
        transitions: Transition matrix (S, S)
        observations: List of observation sequences (one per thread)
        labels: List of label sequences (one per thread)
        num_states: Number of HMM hidden states S
        num_labels: Number of label classes L
    
    Returns:
        state_to_label: Array of length S, where state_to_label[s] is the label id
            most frequently aligned with state s
        counts: State-label co-occurrence counts, shape (S, L)
    """
    counts = np.zeros((num_states, num_labels), dtype=int)
    
    for obs_seq, label_seq in zip(observations, labels):
        if len(obs_seq) == 0 or len(label_seq) == 0:
            continue
        
        # Ensure observation and label sequences have the same length
        if len(obs_seq) != len(label_seq):
            print(f"Warning: observation length {len(obs_seq)} != label length {len(label_seq)}")
            continue
        
        # Infer hidden state sequence using Viterbi algorithm
        state_seq = viterbi(emissions, initials, obs_seq, transitions)
        
        # Count state-label co-occurrences
        for state, label in zip(state_seq, label_seq):
            if label >= 0:  # Ignore negative labels if any
                counts[state, label] += 1
    
    # Map each state to its most frequent label
    state_to_label = counts.argmax(axis=1)
    return state_to_label, counts


def evaluate_hmm_label_accuracy(
    emissions: NDArray,
    initials: NDArray,
    transitions: NDArray,
    observations: List[NDArray],
    labels: List[NDArray],
    state_to_label: NDArray,
) -> float:
    """
    Evaluate label prediction accuracy induced by the HMM + state_to_label mapping.
    
    Args:
        emissions: Emission matrix (S, M)
        initials: Initial state distribution (S,)
        transitions: Transition matrix (S, S)
        observations: List of observation sequences (one per thread)
        labels: List of label sequences (one per thread)
        state_to_label: Mapping from state id to label id, shape (S,)
    
    Returns:
        float: Overall token-level accuracy over all labeled comments
    """
    correct = 0
    total = 0
    
    for obs_seq, label_seq in zip(observations, labels):
        if len(obs_seq) == 0 or len(label_seq) == 0:
            continue
        
        # Ensure same length
        if len(obs_seq) != len(label_seq):
            continue
        
        # Infer state sequence using Viterbi
        state_seq = viterbi(emissions, initials, obs_seq, transitions)
        
        # Map states to labels
        pred_labels = state_to_label[state_seq]
        
        # Calculate accuracy (only consider valid labels)
        mask = (label_seq >= 0)
        if mask.any():
            correct += (pred_labels[mask] == label_seq[mask]).sum()
            total += mask.sum()
    
    return correct / total if total > 0 else float("nan")


def main():
    """
    Main execution function for HMM evaluation.
    """
    print("=" * 60)
    print("HMM State-Label Mapping and Evaluation")
    print("=" * 60)
    
    # Load learned HMM parameters, observations, and labels
    print("\nLoading data...")
    emissions, initials, transitions, observations, labels = load_data_with_labels(
        obs_file="observations.txt",
        labels_file="labels.txt",
        map_file="map.txt",
        emission_file="learned_emissionMatrix.txt",
        initial_file="learned_initialStateDistribution.txt",
        transition_file="learned_transitionMatrix.txt"
    )
    
    num_states = emissions.shape[0]
    # Determine number of label classes
    all_labels = np.concatenate([l for l in labels if len(l) > 0])
    num_labels = int(all_labels.max()) + 1
    print(f"Number of states: {num_states}")
    print(f"Number of label classes: {num_labels}")
    
    # Learn state-to-label mapping
    print("\nLearning state-to-label mapping...")
    state_to_label, counts = learn_state_label_mapping(
        emissions, initials, transitions, 
        observations, labels,
        num_states, num_labels
    )
    
    print("\nState-to-Label Mapping:")
    for state_id, label_id in enumerate(state_to_label):
        print(f"  State {state_id} -> Label {label_id}")
    
    print("\nState-Label Co-occurrence Counts:")
    print(counts)
    
    # Evaluate accuracy
    print("\nEvaluating accuracy...")
    accuracy = evaluate_hmm_label_accuracy(
        emissions, initials, transitions,
        observations, labels,
        state_to_label
    )
    
    print(f"\nToken-level Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

