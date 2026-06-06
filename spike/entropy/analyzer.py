import numpy as np

class EntropyAnalyzer:
    @staticmethod
    def find_optimal_merges(token_array: np.ndarray, min_freq=2, entropy_threshold=0.1):
        """
        Vectorized Entropy-BPE implementation.
        Analyzes a 1D array of token IDs, finds adjacent pairs, calculates 
        Shannon entropy, and returns the mathematically optimal pairs to merge.
        Runs at C-level speeds via numpy.
        """
        if len(token_array) < 2:
            return []
            
        # Create pairs using sliding window
        pairs = np.column_stack((token_array[:-1], token_array[1:]))
        
        # Find unique pairs and their frequencies instantly using numpy
        unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
        
        # Filter by minimum frequency first to save compute
        valid_idx = pair_counts >= min_freq
        if not np.any(valid_idx):
            return []
            
        filtered_pairs = unique_pairs[valid_idx]
        filtered_counts = pair_counts[valid_idx]
        
        # Calculate conditional probabilities: P(B|A) = count(A,B) / count(A)
        unique_A, A_counts = np.unique(token_array[:-1], return_counts=True)
        # Fast lookup dictionary for A base counts
        A_count_map = dict(zip(unique_A, A_counts))
        
        merges_to_make = []
        for i, pair in enumerate(filtered_pairs):
            A, B = pair[0], pair[1]
            p_AB = filtered_counts[i] / A_count_map[A]
            
            # Entropy calculation: -P * log2(P)
            entropy = -(p_AB * np.log2(p_AB)) if p_AB > 0 else 0.0
            
            # If the transition is highly predictable, it fuses them.
            if entropy <= entropy_threshold:
                merges_to_make.append((A, B))
                
        return merges_to_make
