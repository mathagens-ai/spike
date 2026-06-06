#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace snn {
namespace core {
namespace fluxbits {

inline float popcnt_xor_avx2(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
    size_t count = 0;
    size_t i = 0;
    
#if defined(__AVX2__)
    for (; i + 31 < num_bytes; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        __m256i vxor = _mm256_xor_si256(va, vb);
        
        uint64_t chunks[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(chunks), vxor);
#ifdef _MSC_VER
        count += __popcnt64(chunks[0]);
        count += __popcnt64(chunks[1]);
        count += __popcnt64(chunks[2]);
        count += __popcnt64(chunks[3]);
#else
        count += __builtin_popcountll(chunks[0]);
        count += __builtin_popcountll(chunks[1]);
        count += __builtin_popcountll(chunks[2]);
        count += __builtin_popcountll(chunks[3]);
#endif
    }
#endif

    // Scalar fallback
    for (; i + 7 < num_bytes; i += 8) {
        uint64_t va = *reinterpret_cast<const uint64_t*>(a + i);
        uint64_t vb = *reinterpret_cast<const uint64_t*>(b + i);
#ifdef _MSC_VER
        count += __popcnt64(va ^ vb);
#else
        count += __builtin_popcountll(va ^ vb);
#endif
    }

    for (; i < num_bytes; ++i) {
        uint8_t vxor = a[i] ^ b[i];
#ifdef _MSC_VER
        count += __popcnt16(vxor);
#else
        count += __builtin_popcount(vxor);
#endif
    }

    return static_cast<float>(count);
}

struct CompiledFluxBinary {
    std::vector<uint8_t> data;
    size_t M;          // Output features
    size_t N;          // Input features
    size_t K_bits;     // Number of weight bits (same as N aligned)
    size_t K_bytes;    // Alignment in bytes
};

inline CompiledFluxBinary binary_compile(const float* dense_weights, size_t M, size_t N) {
    CompiledFluxBinary out;
    out.M = M;
    out.N = N;
    out.K_bits = N;
    
    if (out.K_bits % 8 != 0) {
        out.K_bits += (8 - out.K_bits % 8);
    }
    out.K_bytes = out.K_bits / 8;
    out.data.assign(M * out.K_bytes, 0);

    for (size_t i = 0; i < M; ++i) {
        size_t row_offset = i * out.K_bytes;
        for (size_t j = 0; j < N; ++j) {
            if (dense_weights[i * N + j] >= 0.0f) {
                size_t byte_idx = j / 8;
                size_t bit_idx = j % 8;
                out.data[row_offset + byte_idx] |= (1 << bit_idx);
            }
        }
    }
    return out;
}

inline void binary_binarize(const float* inputs, size_t Batch, size_t N, uint8_t* out) {
    size_t K_bits = N;
    if (K_bits % 8 != 0) {
        K_bits += (8 - K_bits % 8);
    }
    size_t K_bytes = K_bits / 8;
    std::fill(out, out + Batch * K_bytes, 0);

    for (size_t b = 0; b < Batch; ++b) {
        size_t row_offset = b * K_bytes;
        for (size_t j = 0; j < N; ++j) {
            if (inputs[b * N + j] >= 0.0f) {
                size_t byte_idx = j / 8;
                size_t bit_idx = j % 8;
                out[row_offset + byte_idx] |= (1 << bit_idx);
            }
        }
    }
}

inline void binary_forward(const uint8_t* x_packed, size_t Batch, const CompiledFluxBinary& compiled, float* out) {
    size_t K_bytes = compiled.K_bytes;
    size_t M = compiled.M;
    float N_float = static_cast<float>(compiled.N);
    float scale = (N_float > 0.0f) ? std::sqrt(N_float) : 1.0f;

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < static_cast<int>(Batch); ++b) {
        for (int i = 0; i < static_cast<int>(M); ++i) {
            const uint8_t* x_ptr = x_packed + (b * K_bytes);
            const uint8_t* w_ptr = compiled.data.data() + (i * K_bytes);
            
            float pop_xor = popcnt_xor_avx2(w_ptr, x_ptr, K_bytes);
            out[b * M + i] = (N_float - 2.0f * pop_xor) / scale;
        }
    }
}

} // namespace fluxbits
} // namespace core
} // namespace snn
