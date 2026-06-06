#pragma once
#include "../../snn-core/tensor/tensor.h"
#include "../../snn-core/tensor/linear_flux.h"
#include <cmath>

namespace snn {
namespace neural {
namespace sparse {

using namespace snn::core::tensor;
using namespace snn::core::runtime;

/**
 * @brief FluxFFN - 0.25-bit SwiGLU Feed-Forward Network
 * 
 * Contains the bulk of the neural parameters. It utilizes 0.25-bit FluxBits
 * to process massive hidden dimensions (8/3 * d_model) in incredibly small
 * memory footprints (4 parameters per bit).
 */
class FluxFFN {
private:
    size_t d_model;
    size_t hidden_dim;

    // Extreme 0.25-bit Projections
    LinearFluxKernel gate_proj;
    LinearFluxKernel up_proj;
    LinearFluxKernel down_proj;

public:
    FluxFFN(size_t d, size_t hd, 
            const LinearFluxKernel& gp, const LinearFluxKernel& up, const LinearFluxKernel& dp)
        : d_model(d), hidden_dim(hd), gate_proj(gp), up_proj(up), down_proj(dp) {}

    /**
     * @brief Forward pass of SwiGLU
     * 
     * @param x_bin Binary input packed tensor (Batch x d_model bits)
     * @param out Output FP32 tensor (Batch x d_model)
     */
    void forward(const FluxTensor& x_bin, DenseTensor& out) const {
        size_t batch = x_bin.rows;
        
        // 1. POPCOUNT Projections from 0.25-bit Bloom Tensors
        DenseTensor gate(batch, hidden_dim);
        DenseTensor up(batch, hidden_dim);
        
        gate_proj.forward(x_bin, gate);
        up_proj.forward(x_bin, up);

        // 2. Swish Activation (SiLU) and Element-wise Multiply
        // Executed in extreme FP32 precision to prevent numerical decay
        DenseTensor hidden(batch, hidden_dim);
        
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < hidden_dim; ++i) {
                float g = gate.row_ptr(b)[i];
                float u = up.row_ptr(b)[i];
                
                // SiLU = x * sigmoid(x)
                float sigmoid = 1.0f / (1.0f + std::exp(-std::max(-15.0f, std::min(g, 15.0f))));
                float silu = g * sigmoid;
                
                // Multiply gate and up paths
                hidden.row_ptr(b)[i] = silu * u;
            }
        }

        // 3. Binarize Hidden State for Down Projection
        // Since down_proj is a 0.25-bit LinearFluxKernel, it takes a binary input.
        size_t K_bytes = (hidden_dim + 7) / 8;
        FluxTensor hidden_bin(batch, K_bytes, hidden_dim);
        
        #pragma omp parallel for
        for (int b = 0; b < batch; ++b) {
            uint8_t* bin_ptr = hidden_bin.row_ptr(b);
            const float* h_ptr = hidden.row_ptr(b);
            
            for (size_t i = 0; i < hidden_dim; ++i) {
                if (h_ptr[i] > 0.0f) {
                    bin_ptr[i / 8] |= (1 << (i % 8));
                }
            }
        }

        // 4. Final Projection back to d_model
        down_proj.forward(hidden_bin, out);
    }
};

} // namespace sparse
} // namespace neural
} // namespace snn

