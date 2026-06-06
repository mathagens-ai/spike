import numpy as np

try:
    import snn_core
    HAS_SNN_CORE = True
except ImportError:
    HAS_SNN_CORE = False

class Int4Backend:
    @staticmethod
    def compile(weights):
        """
        Compile FP32 weights into symmetric packed 4-bit integer quantization.
        """
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        M, N = weights.shape
        
        if HAS_SNN_CORE:
            compiled_struct = snn_core.int4_compile(weights)
            return {
                'backend': 'int4',
                'compiled': compiled_struct,
                'M': compiled_struct.M,
                'N': compiled_struct.N,
                'K_bits': 0,
                'K_bytes': compiled_struct.K_bytes,
                'd_hashes': 0,
                'bits_per_param': 4.0,
                'density': 1.0,
            }
            
        # Python fallback
        K_bytes = (N + 1) // 2
        data = np.zeros((M, K_bytes), dtype=np.uint8)
        scales = np.zeros(M, dtype=np.float32)
        
        for i in range(M):
            row = weights[i]
            max_abs = np.max(np.abs(row))
            scale = max_abs / 7.0
            if scale < 1e-5:
                scale = 1.0
            scales[i] = scale
            inv_scale = 1.0 / scale
            
            for k in range(K_bytes):
                idx0 = k * 2
                q0 = 0
                if idx0 < N:
                    q0 = int(np.round(row[idx0] * inv_scale))
                    q0 = max(-8, min(7, q0))
                    
                idx1 = k * 2 + 1
                q1 = 0
                if idx1 < N:
                    q1 = int(np.round(row[idx1] * inv_scale))
                    q1 = max(-8, min(7, q1))
                    
                packed = (q0 & 0x0F) | ((q1 & 0x0F) << 4)
                data[i, k] = packed
                
        return {
            'backend': 'int4',
            'compiled': None,
            'data': data,
            'scales': scales,
            'M': M,
            'N': N,
            'K_bits': 0,
            'K_bytes': K_bytes,
            'd_hashes': 0,
            'bits_per_param': 4.0,
            'density': 1.0,
        }

    @staticmethod
    def binarize(x_fp32, K_bits=None, d_hashes=None):
        """
        No-op for int4, returns inputs directly.
        """
        return x_fp32

    @staticmethod
    def forward(x_dense, compiled):
        """
        Perform symmetric int4 quantized forward pass.
        """
        x_dense = np.ascontiguousarray(x_dense, dtype=np.float32)
        if HAS_SNN_CORE and compiled.get('compiled') is not None:
            Batch = x_dense.shape[0]
            out = np.zeros((Batch, compiled['M']), dtype=np.float32)
            snn_core.int4_forward(x_dense, compiled['compiled'], out)
            return out
            
        # Python fallback
        data = compiled['data']
        scales = compiled['scales']
        M, K_bytes = data.shape
        N = compiled['N']
        Batch = x_dense.shape[0]
        
        w_dequant = np.zeros((M, N), dtype=np.float32)
        for i in range(M):
            scale = scales[i]
            for k in range(K_bytes):
                packed = data[i, k]
                q0 = packed & 0x0F
                if q0 & 0x08:
                    q0 -= 16
                q1 = (packed >> 4) & 0x0F
                if q1 & 0x08:
                    q1 -= 16
                    
                idx0 = k * 2
                if idx0 < N:
                    w_dequant[i, idx0] = q0 * scale
                idx1 = k * 2 + 1
                if idx1 < N:
                    w_dequant[i, idx1] = q1 * scale
                    
        return x_dense @ w_dequant.T
