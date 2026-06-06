#pragma once
#include "tensor.h"
#include "../simd/simd_popcount.h"
#include <vector>

namespace snn {
namespace core {
namespace runtime {

using namespace snn::core::tensor;
using namespace snn::core::simd;

/**
 * @brief LinearFlux Kernel
 * 
 * Executes the forward pass of a 0.25-bit or 0.45-bit Linear layer.
 * 
 * Computes: Y = POPCNT(W_flux AND X_flux) * scale + bias
 */
class LinearFluxKernel {
private:
    FluxTensor W_flux;
    DenseTensor scale;
    DenseTensor bias;

public:
    LinearFluxKernel(const FluxTensor& w, const DenseTensor& s, const DenseTensor& b) 
        : W_flux(w), scale(s), bias(b) {}

    /**
     * @brief Forward pass for a batch of binary inputs.
     * 
     * @param X_flux Binary packed inputs (Batch x K_bytes)
     * @param out Dense FP32 output (Batch x M)
     */
    void forward(const FluxTensor& X_flux, DenseTensor& out) const {
        size_t batch_size = X_flux.rows;
        size_t M = W_flux.rows;
        size_t K_bytes = W_flux.cols_bytes;

        // Ensure output is sized correctly
        if (out.rows != batch_size || out.cols != M) {
            out = DenseTensor(batch_size, M);
        }

        // Extremely fast nested loop: Batch x Output Features
        // In a production engine, this would be multi-threaded via OpenMP or a ThreadPool.
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < M; ++i) {
                const uint8_t* x_ptr = X_flux.row_ptr(b);
                const uint8_t* w_ptr = W_flux.row_ptr(i);

                // AVX2 SIMD AND+POPCOUNT (Calculates ~256 synapses per clock cycle)
                float raw_popcnt = popcnt_dot_avx2(w_ptr, x_ptr, K_bytes);

                // Affine recovery to restore floating point scale
                out.row_ptr(b)[i] = raw_popcnt * scale.data[i] + bias.data[i];
            }
        }
    }
};

} // namespace runtime
} // namespace core
} // namespace snn

