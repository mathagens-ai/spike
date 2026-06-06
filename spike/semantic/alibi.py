import numpy as np
import math

class ALiBiEngine:
    @staticmethod
    def get_slopes(n_heads: int) -> np.ndarray:
        """
        Generates the ALiBi geometric penalty slopes for N attention heads.
        Example for 8 heads: [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256]
        """
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
        slopes = np.power(base, np.arange(1, closest_power_of_2 + 1))
        
        if closest_power_of_2 < n_heads:
            extra_base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 2)))
            extra_slopes = np.power(extra_base, np.arange(1, 2 * (n_heads - closest_power_of_2) + 1, 2))
            slopes = np.concatenate([slopes, extra_slopes])
            
        return slopes

    @staticmethod
    def build_penalty_matrix(seq_len: int, n_heads: int) -> np.ndarray:
        """
        Builds the ALiBi distance penalty matrix to be added directly to 
        the FluxAttention POPCOUNT scores.
        Shape: (n_heads, seq_len, seq_len)
        """
        slopes = ALiBiEngine.get_slopes(n_heads)
        
        # Calculate distance between tokens: |i - j|
        # For autoregressive, we usually only care about j <= i (causal masking)
        # distances[i, j] = i - j
        i_indices = np.arange(seq_len)[:, None]
        j_indices = np.arange(seq_len)[None, :]
        distances = np.maximum(0, i_indices - j_indices) # Causal distances
        
        # Calculate penalty: -m * distance
        # Shape broadcast: (n_heads, 1, 1) * (1, seq_len, seq_len)
        penalties = -slopes[:, None, None] * distances[None, :, :]
        
        return penalties
