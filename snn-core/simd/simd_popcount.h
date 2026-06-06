#pragma once
#include <cstdint>
#include <cstddef>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace snn {
namespace core {
namespace simd {

/**
 * @brief Computes the dot product of two binary vectors using SIMD AND + POPCOUNT.
 * 
 * This is the absolute core of the FluxAttention and FluxFFN execution engine.
 * By using AVX2 (256-bit) or AVX-512, we compute 256/512 boolean multiplications 
 * and additions in a single CPU clock cycle.
 * 
 * @param a Pointer to the first binary vector (e.g., compressed weights).
 * @param b Pointer to the second binary vector (e.g., binarized inputs).
 * @param num_bytes Length of the vectors in bytes. Must be a multiple of 32 for AVX2.
 * @return float The popcount result.
 */
inline float popcnt_dot_avx2(const uint8_t* a, const uint8_t* b, size_t num_bytes) {
    size_t count = 0;
    size_t i = 0;
    
#if defined(__AVX2__)
    // Process 32 bytes (256 bits) at a time
    __m256i sum = _mm256_setzero_si256();
    for (; i + 31 < num_bytes; i += 32) {
        __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        
        // Bitwise AND
        __m256i vand = _mm256_and_si256(va, vb);
        
        // We need to popcount the 256 bits. AVX2 does not have a native _mm256_popcnt.
        // We use the AVX512 popcnt if available, or fallback to 64-bit hardware popcnt.
        // For broad compatibility (AVX2), we fallback to 64-bit _popcnt64.
        
        uint64_t* chunks = reinterpret_cast<uint64_t*>(&vand);
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

    // Scalar fallback for remaining bytes
    for (; i + 7 < num_bytes; i += 8) {
        uint64_t va = *reinterpret_cast<const uint64_t*>(a + i);
        uint64_t vb = *reinterpret_cast<const uint64_t*>(b + i);
#ifdef _MSC_VER
        count += __popcnt64(va & vb);
#else
        count += __builtin_popcountll(va & vb);
#endif
    }

    for (; i < num_bytes; ++i) {
        uint8_t vand = a[i] & b[i];
#ifdef _MSC_VER
        count += __popcnt16(vand);
#else
        count += __builtin_popcount(vand);
#endif
    }

    return static_cast<float>(count);
}

} // namespace simd
} // namespace core
} // namespace snn

