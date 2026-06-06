#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace snn {
namespace core {
namespace fluxbits {

struct CompiledFluxInt8 {
    std::vector<int8_t> data;  // 8-bit weights (M x N)
    std::vector<float> scales; // Per-row scales
    size_t M;          // Output features
    size_t N;          // Input features
};

inline CompiledFluxInt8 int8_compile(const float* dense_weights, size_t M, size_t N) {
    CompiledFluxInt8 out;
    out.M = M;
    out.N = N;
    out.data.assign(M * N, 0);
    out.scales.assign(M, 1.0f);

    for (size_t i = 0; i < M; ++i) {
        const float* row = dense_weights + (i * N);
        
        // Find maximum absolute value
        float max_abs = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            max_abs = std::max(max_abs, std::abs(row[j]));
        }
        
        float scale = max_abs / 127.0f;
        if (scale < 1e-5f) {
            scale = 1.0f;
        }
        out.scales[i] = scale;
        float inv_scale = 1.0f / scale;

        size_t row_offset = i * N;
        for (size_t j = 0; j < N; ++j) {
            float val = row[j] * inv_scale;
            int8_t q = static_cast<int8_t>(std::round(val));
            q = std::max<int8_t>(-128, std::min<int8_t>(127, q));
            out.data[row_offset + j] = q;
        }
    }
    return out;
}

inline void int8_forward(const float* x_dense, size_t Batch, const CompiledFluxInt8& compiled, float* out) {
    size_t M = compiled.M;
    size_t N = compiled.N;

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < static_cast<int>(Batch); ++b) {
        for (int i = 0; i < static_cast<int>(M); ++i) {
            float sum = 0.0f;
            float scale = compiled.scales[i];
            const int8_t* w_row = compiled.data.data() + (i * N);
            const float* x_row = x_dense + (b * N);

            for (size_t j = 0; j < N; ++j) {
                sum += x_row[j] * static_cast<float>(w_row[j]);
            }
            out[b * M + i] = sum * scale;
        }
    }
}

} // namespace fluxbits
} // namespace core
} // namespace snn
