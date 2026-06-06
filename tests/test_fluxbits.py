import os
import sys
import numpy as np

# Add the current directory to path so we can import engine components easily

try:
    import snn_core
    pass
    pass
except ImportError as e:
    pass

from spike.engine.fluxbits import FluxCompiler, AffineCalibrator
from spike.engine.creature import SNNCreature
from spike.engine.trainer import CreatureTrainer

def test_compiler_backends():
    pass
    pass
    pass
    
    # 2D dummy weights (M=16, N=32)
    weights = np.random.randn(16, 32).astype(np.float32)
    
    precisions = [
        ("0.22-bit Bloom", 0.22, "bloom", 1),
        ("0.45-bit Bloom", 0.45, "bloom", 2),
        ("1-bit Binary", 1.0, "1bf16", 0),
        ("int4 Quant", "int4", "int4", 0),
        ("int8 Quant", "int8", "int8", 0)
    ]
    
    for label, prec, expected_backend, expected_hashes in precisions:
        pass
        
        # Compile
        compiled = FluxCompiler.compile(weights, prec)
        assert compiled is not None, "Compilation returned None"
        assert compiled['backend'] == expected_backend, f"Expected backend {expected_backend}, got {compiled['backend']}"
        
        pass
        pass
        pass
        pass
        pass
        pass
        
        if expected_backend == 'bloom':
            assert compiled['d_hashes'] == expected_hashes, f"Expected {expected_hashes} hashes, got {compiled['d_hashes']}"
        
        # Binarize
        dummy_in = np.random.randn(4, 32).astype(np.float32)
        packed = FluxCompiler.binarize(dummy_in, compiled_dict=compiled)
        pass
        
        if expected_backend in ('int4', 'int8'):
            # Must return the dense input directly
            assert np.array_equal(packed, dummy_in), "For int4/int8, binarize must be identity mapping"
        else:
            assert packed.dtype == np.uint8, "Packed binary representation must be uint8"
            
        # Forward pass
        out = FluxCompiler.forward(packed, compiled)
        pass
        assert out.shape == (4, 16), f"Expected output shape (4, 16), got {out.shape}"
        assert not np.isnan(out).any(), "Forward pass output contains NaNs"


def test_creature_lifecycle():
    pass
    pass
    pass
    
    configs = [
        ("0.22-bit Bloom for all", 0.22, 0.22),
        ("0.45-bit Bloom for all", 0.45, 0.45),
        ("1-bit Binary for all", 1.0, 1.0),
        ("int4 Quant for all", "int4", "int4"),
        ("int8 Quant for all", "int8", "int8"),
        ("Mixed: 0.45-bit Attn, 0.22-bit FFN (Defaults)", 0.45, 0.22),
        ("Mixed: int8 Attn, int4 FFN", "int8", "int4")
    ]
    
    vocab_size = 1000
    d_model = 128
    n_heads = 4
    hidden_dim = 256
    n_layers = 2
    
    for label, attn_prec, ffn_prec in configs:
        pass
        pass
        
        # 1. Initialize creature
        creature = SNNCreature(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            bits_per_param_attn=attn_prec,
            bits_per_param_ffn=ffn_prec
        )
        pass
        
        # Check backend assignments
        pass
        pass
        pass
        pass
        
        # 2. Forward pass with dummy tokens
        dummy_ids = np.random.randint(0, vocab_size, size=(2, 8)).astype(np.int32)
        logits = creature.forward(dummy_ids)
        pass
        assert logits.shape == (2, 8, vocab_size), f"Expected shape (2, 8, {vocab_size}), got {logits.shape}"
        assert not np.isnan(logits).any(), "Forward pass logits contain NaNs"
        
        # 3. Backward pass
        dummy_grad = np.random.randn(*logits.shape).astype(np.float32)
        creature.backward(dummy_grad)
        pass
        
        # 4. Step Optimizer & Recompile
        creature.step_optimizer(lr=0.01)
        pass


if __name__ == "__main__":
    pass
    test_compiler_backends()
    test_creature_lifecycle()
    pass
    pass
    pass
