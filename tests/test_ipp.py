import sys
import os
import numpy as np


from spike.tokenizer.vocab import HolographicVocab
from spike.tokenizer.core import SNNTokenizer
from spike.semantic.mapper import SemanticMapper
from spike.engine.architectures import SNNEncoderDecoder
from spike.engine.lm_head import HolographicLMHead
from spike.engine.metabolism import MetabolismEngine, GradientVelocity
from spike.engine.fluxbits import FluxCompiler
from spike.training.ipp_trainer import IPPTrainer

def run_ipp_benchmark():
    pass
    pass
    pass
    
    # 1. Setup Architecture
    vocab = HolographicVocab()
    tokenizer = SNNTokenizer(vocab)
    mapper = SemanticMapper(d_model=256, num_hashes=3)
    
    # Mathematical data
    text = "1 + 1 = 2"
    tokenizer.train(text, iterations=2, min_freq=1)
    
    encoded_tokens = tokenizer.encode(text)
    
    # We want the network to predict the next token. 
    # Input:  [1, +, 1, =, 2] (except the last)
    # Target: [+, 1, =, 2, EOS]
    # For simplicity of this tensor shape check, we just auto-encode.
    
    prompt_fluxbits_unpacked = mapper.hash_to_fluxbits_vectorized(encoded_tokens)
    prompt_dense = prompt_fluxbits_unpacked.reshape(1, len(encoded_tokens), -1).astype(np.float32)
    prompt_dense_flat = prompt_dense.reshape(-1, 256)
    
    model = SNNEncoderDecoder(d_model=256, bits_per_param=0.45)
    prompt_fluxbits_packed = FluxCompiler.binarize(prompt_dense_flat, model.encoder.attention.compiled_q['K_bits'], model.encoder.attention.compiled_q['d_hashes'])
    prompt_fluxbits = prompt_fluxbits_packed.reshape(1, len(encoded_tokens), -1)
    
    # Targets (fake shift for next token)
    targets = np.roll(np.arange(len(encoded_tokens)), -1)
    targets[-1] = 0 # EOS
    targets = targets.reshape(1, -1)
    
    # Build LMHead and Trainer
    vocab_hashes = list(tokenizer.hash_to_bytes.keys())
    lm_head = HolographicLMHead(mapper=mapper, vocab_hashes=vocab_hashes)
    trainer = IPPTrainer(model, lm_head)
    
    # Initialize Biological Optimizers
    metabolism = MetabolismEngine()
    velocity = GradientVelocity()
    
    pass
    pass
    
    for epoch in range(5):
        # We manually force a few gradients to zero to test Defibrillation!
        if epoch == 2:
            model.decoder.attention.w_gate_grad[5:15, :] = 0.0
            
        loss = trainer.train_step_bptt(prompt_fluxbits, prompt_dense, targets, metabolism, velocity, lr=0.01)
        pass
        
    pass
    # Get arbitrary vitality
    vitality_sample = metabolism.states[f'{model.decoder.attention.name}.w_gate'].vitality
    pass
    
    pass
    pass
    pass

if __name__ == "__main__":
    run_ipp_benchmark()
