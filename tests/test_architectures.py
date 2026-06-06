import sys
import os
import time
import numpy as np

# Setup paths for SNN Python module

from spike.engine.architectures import SNNEncoderDecoder
from spike.engine.fluxbits import FluxCompiler
from spike.tokenizer.vocab import HolographicVocab
from spike.tokenizer.core import SNNTokenizer
from spike.semantic.mapper import SemanticMapper
from spike.engine.lm_head import HolographicLMHead
from spike.engine.generation import DecodingEngine

def run_architecture_benchmark():
    d_model = 256
    batch_size = 1
    prompt_length = 1000 
    
    for prec in [0.45, '1bf16']:
        print(f"Running SNN Architecture Benchmark with precision: {prec}")
        model = SNNEncoderDecoder(d_model=d_model, bits_per_param=prec)
        
        # Instantiate lexical elements for decoding
        vocab = HolographicVocab()
        tokenizer = SNNTokenizer(vocab)
        mapper = SemanticMapper(d_model=d_model, num_hashes=3)
        vocab_hashes = [i for i in range(100)] # dummy vocabulary hashes for simulation
        lm_head = HolographicLMHead(mapper=mapper, vocab_hashes=vocab_hashes)
        decoding_engine = DecodingEngine(vocab_size=len(vocab_hashes))
        
        prompt_dense = np.random.uniform(-1, 1, size=(batch_size, prompt_length, d_model)).astype(np.float32)
        
        # Binarize
        prompt_dense_flat = prompt_dense.reshape(-1, d_model)
        prompt_fluxbits_flat = FluxCompiler.binarize(prompt_dense_flat, model.encoder.attention.compiled_q['K_bits'], model.encoder.attention.compiled_q['d_hashes'])
        prompt_fluxbits = prompt_fluxbits_flat.reshape(batch_size, prompt_length, -1)
        
        start_t = time.perf_counter()
        # Generate 5 tokens autoregressively from the prompt
        outputs, s_enc = model.generate(
            prompt_fluxbits=prompt_fluxbits,
            prompt_dense=prompt_dense,
            lm_head=lm_head,
            decoding_engine=decoding_engine,
            max_new_tokens=5
        )
        total_time = time.perf_counter() - start_t
        print(f"Completed in {total_time:.3f} seconds. Outputs: {outputs}")
    
    pass
    pass
    pass
    
    pass
    pass
    pass
    
    pass
    pass
    for i, out in enumerate(outputs):
        val = out # sample first dimension
        
    pass
    pass
    pass

if __name__ == "__main__":
    run_architecture_benchmark()
