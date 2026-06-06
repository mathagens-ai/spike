import numpy as np

try:
    import snn_core
    HAS_SNN_CORE = True
except ImportError:
    HAS_SNN_CORE = False

def bloom_hash(j, hash_idx, max_bits):
    """
    64-bit FNV-1a Hash with SplitMix-like post-mixing for perfect avalanche.
    Eliminates row-coherence and rank collapse in Bloom filters.
    """
    hash_val = 14695981039346656037
    hash_val ^= int(j)
    hash_val = (hash_val * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    hash_val ^= int(hash_idx)
    hash_val = (hash_val * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    hash_val ^= hash_val >> 33
    hash_val = (hash_val * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF
    hash_val ^= hash_val >> 33
    hash_val = (hash_val * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    hash_val ^= hash_val >> 33
    return int(hash_val % max_bits)

POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

class BloomBackend:
    @staticmethod
    def compile(weights, bits_per_param, coactivation_matrix=None, threshold=0.05):
        """
        Compile FP32 weights into a Bloom filter quantized representation.
        """
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        M, N = weights.shape
        bits_per_param = float(bits_per_param)
        
        if HAS_SNN_CORE and coactivation_matrix is None:
            compiled_struct = snn_core.bloom_compile(weights, bits_per_param)
            return {
                'backend': 'bloom',
                'compiled': compiled_struct,
                'M': compiled_struct.M,
                'N': compiled_struct.N,
                'K_bits': compiled_struct.K_bits,
                'K_bytes': compiled_struct.K_bytes,
                'd_hashes': compiled_struct.d_hashes,
                'bits_per_param': bits_per_param,
                'density': compiled_struct.expected_density,
            }
        
        # Python fallback or Hebbian-coactivation compile
        K_bits = int(max(8, N * bits_per_param))
        if K_bits % 8 != 0:
            K_bits += (8 - K_bits % 8)
        K_bytes = K_bits // 8
        d_hashes = int(max(1, np.floor(2.0 * bits_per_param / 0.45)))
        
        flux_rows = np.zeros((M, K_bytes), dtype=np.uint8)
        hash_map = np.zeros((N, d_hashes), dtype=np.int32)
        for j in range(N):
            for h in range(d_hashes):
                hash_map[j, h] = bloom_hash(j, h, K_bits)
                
        active_synapses = 0
        for i in range(M):
            row = weights[i]
            row_thresh = np.mean(np.abs(row))
            active_j = np.where(np.abs(row) > row_thresh)[0]
            active_synapses += len(active_j)
            
            for j in active_j:
                for h in range(d_hashes):
                    bit_idx = hash_map[j, h]
                    if coactivation_matrix is not None and coactivation_matrix.shape == (N, N) and len(active_j) > 1:
                        other_j = active_j[active_j != j]
                        co_scores = coactivation_matrix[j, other_j]
                        if len(co_scores) > 0 and np.max(co_scores) > 0.5:
                            partner = other_j[np.argmax(co_scores)]
                            bit_idx = hash_map[partner, h]
                    byte_idx = bit_idx // 8
                    bit_offset = bit_idx % 8
                    flux_rows[i, byte_idx] |= (1 << bit_offset)
                    
        bloom_density = float(np.unpackbits(flux_rows).mean())
        
        return {
            'backend': 'bloom',
            'compiled': None,
            'flux_rows': flux_rows,
            'M': M,
            'N': N,
            'K_bits': K_bits,
            'K_bytes': K_bytes,
            'd_hashes': d_hashes,
            'bits_per_param': bits_per_param,
            'density': bloom_density,
            'hash_map': hash_map,
        }

    @staticmethod
    def binarize(x_fp32, K_bits, d_hashes, input_hash_map=None):
        """
        Binarize and pack inputs.
        """
        Batch, N = x_fp32.shape
        K_bytes = K_bits // 8
        
        if HAS_SNN_CORE:
            out = np.zeros((Batch, K_bytes), dtype=np.uint8)
            snn_core.bloom_binarize(x_fp32, K_bits, d_hashes, out)
            return out
            
        # Python fallback
        if input_hash_map is None:
            input_hash_map = np.zeros((N, d_hashes), dtype=np.int32)
            for j in range(N):
                for h in range(d_hashes):
                    input_hash_map[j, h] = bloom_hash(j, h, K_bits)
                    
        thresh = np.mean(np.abs(x_fp32), axis=-1, keepdims=True)
        x_active = np.abs(x_fp32) > thresh
        
        Q = np.zeros((Batch, K_bytes), dtype=np.uint8)
        for h in range(d_hashes):
            bit_indices = input_hash_map[:, h]
            for b in range(Batch):
                active_j = np.where(x_active[b])[0]
                if len(active_j) > 0:
                    bits = bit_indices[active_j]
                    bytes_idx = bits // 8
                    bit_offsets = bits % 8
                    np.bitwise_or.at(Q[b], bytes_idx, 1 << bit_offsets)
        return Q

    @staticmethod
    def forward(x_packed, compiled):
        """
        Perform forward pass.
        """
        if HAS_SNN_CORE and compiled.get('compiled') is not None:
            Batch = x_packed.shape[0]
            out = np.zeros((Batch, compiled['M']), dtype=np.float32)
            snn_core.bloom_forward(x_packed, compiled['compiled'], out)
            return out
            
        # Python fallback
        flux_rows = compiled['flux_rows']
        and_res = x_packed[:, np.newaxis, :] & flux_rows[np.newaxis, :, :]
        popcounts = POPCOUNT_TABLE[and_res].sum(axis=-1).astype(np.float32)
        return popcounts
