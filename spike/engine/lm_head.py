import numpy as np
import sys
import os

from ..semantic.mapper import SemanticMapper

class HolographicLMHead:
    """
    SNN Native Language Modeling Head.
    Bypasses standard FP32 weight matrices by measuring the exact
    mathematical overlap between the Decoder's output state and 
    the native FluxBits boolean footprints of the vocabulary.
    """
    def __init__(self, mapper: SemanticMapper, vocab_hashes: list):
        """
        mapper: The semantic mapper used during encoding.
        vocab_hashes: A list of the integer hash IDs for the known vocabulary.
        """
        self.d_model = mapper.d_model
        self.vocab_hashes = np.array(vocab_hashes, dtype=np.uint64)
        self.vocab_size = len(self.vocab_hashes)
        
        # 1. Pre-compute the Holographic Matrix (V x d_model)
        # This takes 0 training and minimal memory because it's native FluxBits.
        # It's an array of 0s and 1s representing every word in the dictionary.
        self.vocab_matrix = mapper.hash_to_fluxbits_vectorized(self.vocab_hashes).astype(np.float32)
        
    def compute_logits(self, decoder_state: np.ndarray) -> np.ndarray:
        """
        Generates probability logits based on structural similarity.
        decoder_state: Shape (Batch, d_model)
        Returns: logits of shape (Batch, vocab_size)
        """
        # The dot product perfectly measures the energy overlap.
        # It adds up the decoder's state values only where the vocabulary word has a 1-bit.
        logits = decoder_state @ self.vocab_matrix.T
        return logits
