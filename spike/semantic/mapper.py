import numpy as np

class SemanticMapper:
    def __init__(self, d_model: int, num_hashes: int = 2):
        self.d_model = d_model 
        self.num_hashes = num_hashes
        
    def hash_to_fluxbits_vectorized(self, token_hashes: np.ndarray) -> np.ndarray:
        """
        Vectorized mapping of an array of N tokens -> N x d_model FluxBits.
        Achieves production-grade speed using numpy bitwise operations across 
        the entire context window simultaneously.
        """
        N = len(token_hashes)
        # footprint shape: (N, d_model) initialized to zero
        footprints = np.zeros((N, self.d_model), dtype=np.uint8)
        
        # Use a vectorized LCG on the entire array at once
        seeds = token_hashes.astype(np.uint64)
        
        # Precompute the row indices [0, 1, ..., N-1] for advanced numpy indexing
        row_indices = np.arange(N)
        
        for i in range(self.num_hashes):
            # Fast bitwise LCG on the entire array in C-backend
            seeds = (seeds * np.uint64(0x9E3779B97F4A7C15) + np.uint64(0xBF58476D1CE4E5B9))
            
            # Base indices (no position awareness)
            base_indices = seeds % np.uint64(self.d_model)
            
            # BITWISE RoPE: Physically rotate the bits in 3D space based on sequence position!
            # We circular-shift the target dimension index by the sequence row index.
            rotated_col_indices = (base_indices + np.uint64(row_indices)) % np.uint64(self.d_model)
            
            # Fire the rotated bits instantly
            footprints[row_indices, rotated_col_indices] = 1
            
        return footprints
