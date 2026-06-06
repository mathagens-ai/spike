#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "../simd/simd_popcount.h"

namespace snn {
namespace core {
namespace fluxbits {

inline uint64_t splitmix64_bloom(uint64_t seed) {
    uint64_t z = (seed + 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct CompiledFluxBloom {
    std::vector<uint8_t> data;
    size_t M;          // Output features
    size_t N;          // Input features
    size_t K_bits;     // Bloom filter size in bits
    size_t K_bytes;    // Bloom filter size in bytes
    size_t d_hashes;   // Number of hash functions
    float compression_ratio;
    float expected_density;
};

inline CompiledFluxBloom bloom_compile(const float* dense_weights, size_t M, size_t N, float bits_per_param) {
    CompiledFluxBloom out;
    out.M = M;
    out.N = N;
    out.compression_ratio = bits_per_param;
    
    // Ensure 1 bit minimum per Bloom row
    out.K_bits = static_cast<size_t>(std::max(1.0f, N * bits_per_param));
    
    // Pad to byte boundary
    if (out.K_bits % 8 != 0) {
        out.K_bits += (8 - out.K_bits % 8);
    }
    out.K_bytes = out.K_bits / 8;
    
    // Determine hash counts (d_hashes=1 for 0.22-bit, d_hashes=2 for 0.45-bit)
    out.d_hashes = static_cast<size_t>(std::max(1.0f, std::floor(2.0f * bits_per_param / 0.45f)));
    
    out.data.assign(M * out.K_bytes, 0);

    size_t total_set_bits = 0;

    for (size_t i = 0; i < M; ++i) {
        const float* row = dense_weights + (i * N);
        
        // Calculate dynamic threshold (Mean Absolute Deviation)
        float sum_abs = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            sum_abs += std::abs(row[j]);
        }
        float threshold = sum_abs / static_cast<float>(N);
        
        size_t row_offset = i * out.K_bytes;
        
        for (size_t j = 0; j < N; ++j) {
            if (std::abs(row[j]) > threshold) {
                for (size_t h = 0; h < out.d_hashes; ++h) {
                    uint64_t seed = (static_cast<uint64_t>(j) << 32) | static_cast<uint64_t>(h);
                    uint64_t hash_val = splitmix64_bloom(seed);
                    
                    size_t bit_pos = hash_val % out.K_bits;
                    size_t byte_idx = bit_pos / 8;
                    size_t bit_idx = bit_pos % 8;
                    
                    out.data[row_offset + byte_idx] |= (1 << bit_idx);
                }
            }
        }
        
        // Calculate density for affine recovery
        for(size_t b = 0; b < out.K_bytes; ++b) {
#ifdef _MSC_VER
            total_set_bits += __popcnt16(out.data[row_offset + b]);
#else
            total_set_bits += __builtin_popcount(out.data[row_offset + b]);
#endif
        }
    }
    
    out.expected_density = static_cast<float>(total_set_bits) / static_cast<float>(M * out.K_bits);
    return out;
}

inline void bloom_binarize(const float* inputs, size_t Batch, size_t N, size_t K_bits, size_t d_hashes, uint8_t* out) {
    size_t K_bytes = K_bits / 8;
    std::fill(out, out + Batch * K_bytes, 0);

    for (size_t b = 0; b < Batch; ++b) {
        const float* row = inputs + (b * N);
        
        float sum_abs = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            sum_abs += std::abs(row[j]);
        }
        float threshold = sum_abs / static_cast<float>(N);
        
        size_t row_offset = b * K_bytes;
        for (size_t j = 0; j < N; ++j) {
            if (std::abs(row[j]) > threshold) {
                for (size_t h = 0; h < d_hashes; ++h) {
                    uint64_t seed = (static_cast<uint64_t>(j) << 32) | static_cast<uint64_t>(h);
                    uint64_t hash_val = splitmix64_bloom(seed);
                    
                    size_t bit_pos = hash_val % K_bits;
                    size_t byte_idx = bit_pos / 8;
                    size_t bit_idx = bit_pos % 8;
                    
                    out[row_offset + byte_idx] |= (1 << bit_idx);
                }
            }
        }
    }
}

inline void bloom_forward(const uint8_t* x_packed, size_t Batch, const CompiledFluxBloom& compiled, float* out) {
    size_t K_bytes = compiled.K_bytes;
    size_t M = compiled.M;

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < static_cast<int>(Batch); ++b) {
        for (int i = 0; i < static_cast<int>(M); ++i) {
            const uint8_t* x_ptr = x_packed + (b * K_bytes);
            const uint8_t* w_ptr = compiled.data.data() + (i * K_bytes);
            
            out[b * M + i] = snn::core::simd::popcnt_dot_avx2(w_ptr, x_ptr, K_bytes);
        }
    }
}

} // namespace fluxbits
} // namespace core
} // namespace snn
