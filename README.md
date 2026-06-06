<p align="center">
  <h1 align="center">Spike</h1>
  <p align="center"><strong>Super Neural Network (SNN) Core Engine with 0.25-bit AVX2 Quantization</strong></p>
</p>

<p align="center">
  <a href="https://pypi.org/project/spike/"><img src="https://img.shields.io/pypi/v/spike?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/spike/"><img src="https://img.shields.io/pypi/pyversions/spike" alt="Python"></a>
  <a href="https://github.com/mathagens-ai/spike/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
</p>

---

Spike is a high-performance Spiking Neural Network (SNN) core and language modeling framework designed for extreme efficiency. It combines a hardware-accelerated C++ core optimized with AVX2 instruction sets and a biologically-inspired Python layer that models parameters as living, metabolic creatures with lateral inhibition and dynamic rebirth cycles.

## Features

| Subsystem | Description |
|-----------|-------------|
| **0.25-bit Quantization** | Sub-bit Bloom-filter based weight quantization for ultra-low memory footprints. |
| **Metabolic Parameters** | Parameters behave like living cells with energy consumption, regeneration, and hunger. |
| **Holographic LM Head** | Bypasses standard projection layers via holographic token mapping. |
| **Associative Scans** | Lightning-fast parallel recurrent associative state scans for sequence modeling. |
| **IPP Rebirth Protocol** | Intelligence Per Parameter tracking that recycles dead parameters during training. |

## Installation

You can install Spike directly from PyPI:
```bash
pip install spike
```

**From source (requires a C++ compiler supporting C++11/C++14/C++17):**
```bash
git clone https://github.com/mathagens-ai/spike.git
cd spike
pip install -e .
```

## Quick Start

```python
import spike
import numpy as np

# Instantiate standard SNN configuration
config = spike.SNNConfig.SNN_Nano()
print(f"Loaded config: vocab_size={config.vocab_size}, d_model={config.d_model}")

# Instantiate model
model = spike.SNNModel(config)

# Compile model weights down to 0.25-bit/0.45-bit Bloom Tensors
dummy_weights = {
    "encoder_attn": np.random.randn(256, 256).astype(np.float32),
    "ffn": np.random.randn(256, 512).astype(np.float32)
}
model.compile_from_dense(dummy_weights)

# Run a forward pass
input_ids = np.random.randint(0, config.vocab_size, (1, 32))
logits, states = model.forward(input_ids)

print("Forward pass successful. Logits shape:", logits.shape)
```

## Running Tests

To verify the installation:
```bash
pytest tests/
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

Copyright 2026 Mathagens AI
