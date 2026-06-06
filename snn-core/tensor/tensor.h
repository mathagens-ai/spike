#pragma once
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include "../memory/aligned_allocator.h"

namespace snn {
namespace core {
namespace tensor {

using namespace snn::core::memory;

/**
 * @brief FluxTensor - The core data structure for 0.25-bit representations.
 * 
 * Uses the AlignedAllocator to guarantee SIMD-friendly memory layouts.
 * Stores raw bytes, but logically represents K_bits.
 */
struct FluxTensor {
    std::vector<uint8_t, AlignedAllocator<uint8_t, 32>> data;
    size_t rows;
    size_t cols_bytes;
    size_t original_in_features;

    FluxTensor() : rows(0), cols_bytes(0), original_in_features(0) {}

    FluxTensor(size_t r, size_t c_bytes, size_t in_feat) 
        : rows(r), cols_bytes(c_bytes), original_in_features(in_feat) {
        data.resize(r * c_bytes, 0);
    }

    // Direct access to aligned row pointer
    const uint8_t* row_ptr(size_t i) const {
        return data.data() + (i * cols_bytes);
    }
    
    uint8_t* row_ptr(size_t i) {
        return data.data() + (i * cols_bytes);
    }
};

/**
 * @brief Dense FP32 Tensor for Latent State and Gradients.
 */
struct DenseTensor {
    std::vector<float, AlignedAllocator<float, 32>> data;
    size_t rows;
    size_t cols;

    DenseTensor() : rows(0), cols(0) {}

    DenseTensor(size_t r, size_t c) : rows(r), cols(c) {
        data.resize(r * c, 0.0f);
    }

    const float* row_ptr(size_t i) const {
        return data.data() + (i * cols);
    }

    float* row_ptr(size_t i) {
        return data.data() + (i * cols);
    }
};

} // namespace tensor
} // namespace core
} // namespace snn

