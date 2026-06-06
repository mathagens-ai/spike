import numpy as np

try:
    import snn_core
    HAS_SNN_CORE = True
except ImportError:
    HAS_SNN_CORE = False

class BinaryBackend:
    @staticmethod
    def compile(weights):
        """
        Compile FP32 weights into a 1-bit binary representation.
        """
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        M, N = weights.shape
        
        if HAS_SNN_CORE:
            compiled_struct = snn_core.binary_compile(weights)
            return {
                'backend': '1bf16',
                'compiled': compiled_struct,
                'M': compiled_struct.M,
                'N': compiled_struct.N,
                'K_bits': compiled_struct.K_bits,
                'K_bytes': compiled_struct.K_bytes,
                'd_hashes': 0,
                'bits_per_param': 1.0,
                'density': 0.5,
            }
            
        # Python fallback
        K_bits = N
        if K_bits % 8 != 0:
            K_bits += (8 - K_bits % 8)
        K_bytes = K_bits // 8
        
        flux_rows = np.zeros((M, K_bytes), dtype=np.uint8)
        for i in range(M):
            row = weights[i]
            for j in range(N):
                if row[j] >= 0.0:
                    byte_idx = j // 8
                    bit_offset = j % 8
                    flux_rows[i, byte_idx] |= (1 << bit_offset)
                    
        return {
            'backend': '1bf16',
            'compiled': None,
            'flux_rows': flux_rows,
            'M': M,
            'N': N,
            'K_bits': K_bits,
            'K_bytes': K_bytes,
            'd_hashes': 0,
            'bits_per_param': 1.0,
            'density': 0.5,
        }

    @staticmethod
    def binarize(x_fp32, K_bits=None, d_hashes=None):
        """
        Binarize inputs into packed sign bits.
        """
        Batch, N = x_fp32.shape
        K_bits = N
        if K_bits % 8 != 0:
            K_bits += (8 - K_bits % 8)
        K_bytes = K_bits // 8
        
        if HAS_SNN_CORE:
            out = np.zeros((Batch, K_bytes), dtype=np.uint8)
            snn_core.binary_binarize(x_fp32, out)
            return out
            
        # Python fallback
        Q = np.zeros((Batch, K_bytes), dtype=np.uint8)
        for b in range(Batch):
            row = x_fp32[b]
            for j in range(N):
                if row[j] >= 0.0:
                    byte_idx = j // 8
                    bit_offset = j % 8
                    Q[b, byte_idx] |= (1 << bit_offset)
        return Q

    @staticmethod
    def forward(x_packed, compiled):
        """
        Perform 1-bit binary forward pass.
        """
        if HAS_SNN_CORE and compiled.get('compiled') is not None:
            Batch = x_packed.shape[0]
            out = np.zeros((Batch, compiled['M']), dtype=np.float32)
            snn_core.binary_forward(x_packed, compiled['compiled'], out)
            return out
            
        # Python fallback: Popcount(A ^ B)
        flux_rows = compiled['flux_rows']
        M, K_bytes = flux_rows.shape
        Batch = x_packed.shape[0]
        N = compiled['N']
        
        # Unpack to get individual bits
        w_bits = np.unpackbits(flux_rows, axis=-1)[:, :N] # (M, N)
        x_bits = np.unpackbits(x_packed, axis=-1)[:, :N] # (Batch, N)
        
        # XOR and sum
        xor_sum = (x_bits[:, np.newaxis, :] != w_bits[np.newaxis, :, :]).sum(axis=-1)
        scale = np.sqrt(N)
        return (N - 2.0 * xor_sum) / (scale if scale > 0.0 else 1.0)
