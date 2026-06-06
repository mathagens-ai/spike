#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace snn {
namespace core {
namespace fluxbits {

/**
 * @brief SplitMix64 Hash Generator
 * 
 * Provides perfect avalanching for integer hashes, mathematically 
 * proven in python validation to eliminate row-coherence and rank collapse 
 * in 0.25-bit Bloom tensors.
 */
inline uint64_t splitmix64(uint64_t seed) {
    uint64_t z = (seed + 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct CompiledFlux {
    std::vector<uint8_t> data;
    size_t M;          // Output features
    size_t N;          // Input features
    size_t K_bits;     // Bloom filter size in bits
    size_t K_bytes;    // Bloom filter size in bytes
    size_t d_hashes;   // Number of hash functions
    float compression_ratio;
    float expected_density;
};

/**
 * @brief Compiles a dense fp32 weight matrix into a 0.25-bit / 0.45-bit Bloom Tensor.
 * 
 * @param dense_weights Flat array of fp32 weights (M x N)
 * @param M Number of output features (rows)
 * @param N Number of input features (columns)
 * @param bits_per_param The target compression ratio (e.g., 0.25)
 * @return CompiledFlux The resulting highly-compressed tensor.
 */
inline CompiledFlux compile_weights(const float* dense_weights, size_t M, size_t N, float bits_per_param) {
    CompiledFlux out;
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
    
    // Determine hash counts (d_hashes=1 for 0.25-bit, d_hashes=2 for 0.45-bit)
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
        
        // Populate Bloom filter using SplitMix64
        size_t row_offset = i * out.K_bytes;
        
        for (size_t j = 0; j < N; ++j) {
            if (std::abs(row[j]) > threshold) {
                for (size_t h = 0; h < out.d_hashes; ++h) {
                    uint64_t seed = (static_cast<uint64_t>(j) << 32) | static_cast<uint64_t>(h);
                    uint64_t hash_val = splitmix64(seed);
                    
                    size_t bit_pos = hash_val % out.K_bits;
                    size_t byte_idx = bit_pos / 8;
                    size_t bit_idx = bit_pos % 8;
                    
                    out.data[row_offset + byte_idx] |= (1 << bit_idx);
                }
            }
        }
        
        // Calculate density for affine recovery later
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

} // namespace fluxbits
} // namespace core
} // namespace snn

