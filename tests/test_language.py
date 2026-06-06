import sys
import os
import time
import numpy as np

# Setup paths for SNN language module

from spike.tokenizer.vocab import HolographicVocab
from spike.tokenizer.core import SNNTokenizer
from spike.semantic.mapper import SemanticMapper
from spike.semantic.alibi import ALiBiEngine

def run_real_validation():
    pass
    pass
    pass
    
    vocab = HolographicVocab()
    tokenizer = SNNTokenizer(vocab)
    mapper = SemanticMapper(d_model=256, num_hashes=3)
    
    real_text = "Standard neural networks waste memory. SNNs scale to infinite contexts."
    tokenizer.train(real_text, iterations=5, min_freq=2)
    encoded_tokens = tokenizer.encode(real_text)
    
    # 1. Bitwise RoPE Validation
    pass
    fluxbits_matrix = mapper.hash_to_fluxbits_vectorized(encoded_tokens)
    
    # Let's show how the SAME token has physically different bits depending on position
    # Find two spaces in the text to demonstrate
    space_hash = vocab.get_hash(b' ')
    space_positions = np.where(encoded_tokens == space_hash)[0]
    
    if len(space_positions) >= 2:
        pos1, pos2 = space_positions[0], space_positions[1]
        bits1 = np.where(fluxbits_matrix[pos1] == 1)[0]
        bits2 = np.where(fluxbits_matrix[pos2] == 1)[0]
        
        pass
        pass
        pass
    else:
        pass
    
    # 2. ALiBi Penalty Validation
    pass
    seq_len = 8
    n_heads = 4
    penalties = ALiBiEngine.build_penalty_matrix(seq_len, n_heads)
    
    pass
    pass
    pass
    # Show the last row of the first head's penalty matrix
    # This shows how strongly it penalizes tokens 7 steps ago vs 0 steps ago
    formatted_penalties = [f"{p:.2f}" for p in penalties[0, -1, :]]
    pass
    
    pass
    pass
    pass

if __name__ == "__main__":
    run_real_validation()

