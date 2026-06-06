import numpy as np
from .fluxbits_bloom import BloomBackend
from .fluxbits_binary import BinaryBackend
from .fluxbits_int4 import Int4Backend
from .fluxbits_int8 import Int8Backend

class AffineCalibrator:
    """
    Learnable Affine Calibrator for Bloom/Binary popcounts.
    Shifts and scales discrete collision counts into a standard, zero-centered FP32 space.
    For int4 and int8 layers, acts as a strict identity mapping.
    """
    def __init__(self, M, K_bits, bloom_density, identity=False):
        self.M = M
        self.K_bits = K_bits
        self.bloom_density = bloom_density
        
        # Auto-detect identity mode for int4/int8 (which have K_bits = 0)
        self.identity = identity or (K_bits == 0)
        
        if self.identity:
            self.mu_expected = 0.0
            self.gamma = np.ones(M, dtype=np.float32)
            self.beta = np.zeros(M, dtype=np.float32)
        else:
            # Expected collision mean under typical ~10% input density
            self.mu_expected = float(K_bits) * bloom_density * 0.1
            
            # Initialize gamma to map variance to target stddev of 1.0
            p = bloom_density * 0.1
            popcount_stddev = np.sqrt(float(K_bits) * p * (1.0 - p))
            if popcount_stddev < 0.01:
                popcount_stddev = 1.0
                
            self.gamma = np.full(M, 1.0 / popcount_stddev, dtype=np.float32)
            self.beta = np.zeros(M, dtype=np.float32)

    def apply(self, output):
        """
        Calibrates values in-place (or returns output directly if identity).
        output: shape (Batch, M)
        """
        if self.identity:
            return output
        return self.gamma * (output - self.mu_expected) + self.beta

    def calibrate(self, raw_output):
        """
        Empirically calibrate the expected mean and variance using actual batch data.
        Runs once before training.
        """
        if self.identity:
            return
            
        global_mean = np.mean(raw_output)
        global_stddev = np.std(raw_output)
        if global_stddev < 0.01:
            global_stddev = 1.0
            
        self.mu_expected = float(global_mean)
        gamma_val = 1.0 / global_stddev
        
        self.gamma.fill(gamma_val)
        self.beta.fill(0.0)


class FluxCompiler:
    """
    Thin router that dispatches compilation, binarization, and forward passes
    to five separate precision backends:
      - 0.22-bit Bloom (1 hash)
      - 0.45-bit Bloom (2 hashes)
      - 1-bit Binary (±1 sign weights)
      - int4 Quantization
      - int8 Quantization
    """
    @staticmethod
    def _parse_precision(bits_per_param):
        """
        Norms various precision representations into one of: 'bloom_0.22', 'bloom_0.45', '1bf16', 'int4', 'int8'.
        """
        if isinstance(bits_per_param, str):
            s = bits_per_param.lower().strip()
            if 'int4' in s or '4-bit' in s or '4bit' in s or s == '4':
                return 'int4'
            elif 'int8' in s or '8-bit' in s or '8bit' in s or s == '8':
                return 'int8'
            elif 'binary' in s or '1-bit' in s or '1bit' in s or '1bf16' in s or s == '1' or s == '1.0':
                return '1bf16'
            elif '0.22' in s or '22' in s:
                return 'bloom_0.22'
            elif '0.45' in s or '45' in s:
                return 'bloom_0.45'
            # Fallback based on float conversion of string if it contains float
            try:
                val = float(s)
                return FluxCompiler._parse_precision(val)
            except ValueError:
                pass
        elif isinstance(bits_per_param, (int, float)):
            val = float(bits_per_param)
            if abs(val - 0.22) < 0.05 or abs(val - 0.25) < 0.05: # map both 0.22 and 0.25 to 0.22-bit Bloom
                return 'bloom_0.22'
            elif abs(val - 0.45) < 0.05:
                return 'bloom_0.45'
            elif abs(val - 1.0) < 0.05:
                return '1bf16'
            elif abs(val - 4.0) < 0.05:
                return 'int4'
            elif abs(val - 8.0) < 0.05:
                return 'int8'
                
        raise ValueError(f"Unsupported bits_per_param / precision format: {bits_per_param}")

    @staticmethod
    def compile(weights, bits_per_param, coactivation_matrix=None, threshold=0.05):
        precision = FluxCompiler._parse_precision(bits_per_param)
        
        if precision == 'bloom_0.22':
            return BloomBackend.compile(weights, 0.22, coactivation_matrix, threshold)
        elif precision == 'bloom_0.45':
            return BloomBackend.compile(weights, 0.45, coactivation_matrix, threshold)
        elif precision == '1bf16':
            return BinaryBackend.compile(weights)
        elif precision == 'int4':
            return Int4Backend.compile(weights)
        elif precision == 'int8':
            return Int8Backend.compile(weights)

    @staticmethod
    def binarize(x_fp32, K_bits=None, d_hashes=None, compiled_dict=None):
        """
        Binarize continuous inputs according to the backend.
        If a compiled_dict is provided, dispatches directly based on its backend.
        Otherwise, infers backend from K_bits/d_hashes.
        """
        if compiled_dict is not None:
            backend = compiled_dict.get('backend')
            if backend == 'bloom':
                return BloomBackend.binarize(x_fp32, compiled_dict['K_bits'], compiled_dict['d_hashes'])
            elif backend in ('1bf16', 'binary'):
                return BinaryBackend.binarize(x_fp32)
            elif backend in ('int4', 'int8'):
                return x_fp32
                
        # Inference fallback based on arguments
        if d_hashes == 0 or K_bits is None:
            if K_bits == 0 or K_bits is None:
                return x_fp32 # int4/int8 dense bypass
            return BinaryBackend.binarize(x_fp32)
        else:
            return BloomBackend.binarize(x_fp32, K_bits, d_hashes)

    @staticmethod
    def forward(x, compiled):
        backend = compiled.get('backend')
        if backend == 'bloom':
            return BloomBackend.forward(x, compiled)
        elif backend in ('1bf16', 'binary'):
            return BinaryBackend.forward(x, compiled)
        elif backend == 'int4':
            return Int4Backend.forward(x, compiled)
        elif backend == 'int8':
            return Int8Backend.forward(x, compiled)
        else:
            raise ValueError(f"Unknown compiled backend type in forward: {backend}")
