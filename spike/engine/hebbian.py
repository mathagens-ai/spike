import numpy as np

class HebbianTracker:
    """
    Hebbian Co-Activation Tracker.
    Tracks which input dimensions fire together across batches, allowing us to
    deliberately route their hash bits to overlap in the Bloom filter, turning
    random hash collisions into structured co-activation signals.
    """
    def __init__(self, N, decay=0.999):
        self.N = N
        self.decay = decay
        # Co-activation score matrix of shape (N, N)
        self.coactivation_score = np.zeros((N, N), dtype=np.float32)

    def update(self, x_bin):
        """
        Update the running co-activation scores using a batch of binary activations.
        x_bin: numpy array of shape (Batch, N) containing 0 or 1 values.
        """
        Batch = x_bin.shape[0]
        if Batch == 0:
            return
            
        # Ensure input is float32/float64 for matmul
        x_bin_f = x_bin.astype(np.float32)
        
        # Vectorized batch co-activation counts: (N, Batch) @ (Batch, N) -> (N, N)
        coactivations_batch = x_bin_f.T @ x_bin_f
        
        # Normalize by batch size to get co-activation probability in this batch
        coactivations_batch /= float(Batch)
        
        # Update running EMA
        self.coactivation_score = self.decay * self.coactivation_score + (1.0 - self.decay) * coactivations_batch

    def get_coactivation_matrix(self, threshold=0.1):
        """
        Get the co-activation matrix, zeroing out weak connections to save memory/noise.
        """
        matrix = self.coactivation_score.copy()
        # Zero out diagonal (we only want correlation between different features)
        np.fill_diagonal(matrix, 0.0)
        # Apply threshold
        matrix[matrix < threshold] = 0.0
        return matrix

    def reset(self):
        """Reset the tracker's running history."""
        self.coactivation_score.fill(0.0)
