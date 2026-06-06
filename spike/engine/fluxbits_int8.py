import numpy as np

try:
    import snn_core
    HAS_SNN_CORE = True
except ImportError:
    HAS_SNN_CORE = False

class Int8Backend:
    @staticmethod
    def compile(weights):
        """
        Compile FP32 weights into symmetric 8-bit integer quantization.
        """
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        M, N = weights.shape
        
        if HAS_SNN_CORE:
            compiled_struct = snn_core.int8_compile(weights)
            return {
                'backend': 'int8',
                'compiled': compiled_struct,
                'M': compiled_struct.M,
                'N': compiled_struct.N,
                'K_bits': 0,
                'K_bytes': N,
                'd_hashes': 0,
                'bits_per_param': 8.0,
                'density': 1.0,
            }
            
        # Python fallback
        data = np.zeros((M, N), dtype=np.int8)
        scales = np.zeros(M, dtype=np.float32)
        
        for i in range(M):
            row = weights[i]
            max_abs = np.max(np.abs(row))
            scale = max_abs / 127.0
            if scale < 1e-5:
                scale = 1.0
            scales[i] = scale
            inv_scale = 1.0 / scale
            
            q = np.round(row * inv_scale)
            q = np.clip(q, -128, 127).astype(np.int8)
            data[i] = q
            
        return {
            'backend': 'int8',
            'compiled': None,
            'data': data,
            'scales': scales,
            'M': M,
            'N': N,
            'K_bits': 0,
            'K_bytes': N,
            'd_hashes': 0,
            'bits_per_param': 8.0,
            'density': 1.0,
        }

    @staticmethod
    def binarize(x_fp32, K_bits=None, d_hashes=None):
        """
        No-op for int8, returns inputs directly.
        """
        return x_fp32

    @staticmethod
    def forward(x_dense, compiled):
        """
        Perform symmetric int8 quantized forward pass.
        """
        x_dense = np.ascontiguousarray(x_dense, dtype=np.float32)
        if HAS_SNN_CORE and compiled.get('compiled') is not None:
            Batch = x_dense.shape[0]
            out = np.zeros((Batch, compiled['M']), dtype=np.float32)
            snn_core.int8_forward(x_dense, compiled['compiled'], out)
            return out
            
        # Python fallback
        data = compiled['data']
        scales = compiled['scales']
        
        w_dequant = data.astype(np.float32) * scales[:, np.newaxis]
        return x_dense @ w_dequant.T
