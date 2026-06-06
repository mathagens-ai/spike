#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace snn {
namespace core {
namespace fluxbits {

struct CompiledFluxInt4 {
    std::vector<uint8_t> data; // Packed 4-bit weights
    std::vector<float> scales; // Per-row scales
    size_t M;          // Output features
    size_t N;          // Input features
    size_t K_bytes;    // Number of bytes per row
};

inline CompiledFluxInt4 int4_compile(const float* dense_weights, size_t M, size_t N) {
    CompiledFluxInt4 out;
    out.M = M;
    out.N = N;
    out.K_bytes = (N + 1) / 2;
    out.data.assign(M * out.K_bytes, 0);
    out.scales.assign(M, 1.0f);

    for (size_t i = 0; i < M; ++i) {
        const float* row = dense_weights + (i * N);
        
        // Find maximum absolute value
        float max_abs = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            max_abs = std::max(max_abs, std::abs(row[j]));
        }
        
        float scale = max_abs / 7.0f;
        if (scale < 1e-5f) {
            scale = 1.0f;
        }
        out.scales[i] = scale;
        float inv_scale = 1.0f / scale;

        size_t row_offset = i * out.K_bytes;
        for (size_t k = 0; k < out.K_bytes; ++k) {
            size_t idx0 = k * 2;
            int8_t q0 = 0;
            if (idx0 < N) {
                float val = row[idx0] * inv_scale;
                q0 = static_cast<int8_t>(std::round(val));
                q0 = std::max<int8_t>(-8, std::min<int8_t>(7, q0));
            }

            size_t idx1 = k * 2 + 1;
            int8_t q1 = 0;
            if (idx1 < N) {
                float val = row[idx1] * inv_scale;
                q1 = static_cast<int8_t>(std::round(val));
                q1 = std::max<int8_t>(-8, std::min<int8_t>(7, q1));
            }

            uint8_t packed = (static_cast<uint8_t>(q0) & 0x0F) | ((static_cast<uint8_t>(q1) & 0x0F) << 4);
            out.data[row_offset + k] = packed;
        }
    }
    return out;
}

inline void int4_forward(const float* x_dense, size_t Batch, const CompiledFluxInt4& compiled, float* out) {
    size_t K_bytes = compiled.K_bytes;
    size_t M = compiled.M;
    size_t N = compiled.N;

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < static_cast<int>(Batch); ++b) {
        for (int i = 0; i < static_cast<int>(M); ++i) {
            float sum = 0.0f;
            float scale = compiled.scales[i];
            const uint8_t* w_row = compiled.data.data() + (i * K_bytes);
            const float* x_row = x_dense + (b * N);

            for (size_t k = 0; k < K_bytes; ++k) {
                uint8_t packed = w_row[k];
                
                int8_t q0 = packed & 0x0F;
                if (q0 & 0x08) q0 |= 0xF0; // sign extension

                int8_t q1 = (packed >> 4) & 0x0F;
                if (q1 & 0x08) q1 |= 0xF0; // sign extension

                size_t idx0 = k * 2;
                if (idx0 < N) {
                    sum += x_row[idx0] * static_cast<float>(q0);
                }
                size_t idx1 = k * 2 + 1;
                if (idx1 < N) {
                    sum += x_row[idx1] * static_cast<float>(q1);
                }
            }
            out[b * M + i] = sum * scale;
        }
    }
}

} // namespace fluxbits
} // namespace core
} // namespace snn
