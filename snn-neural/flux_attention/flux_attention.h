#pragma once
#include "../../snn-core/tensor/tensor.h"
#include "../../snn-core/tensor/linear_flux.h"
#include <cmath>
#include <algorithm>

namespace snn {
namespace neural {
namespace recursive {

using namespace snn::core::tensor;
using namespace snn::core::runtime;

/**
 * @brief FluxAttention - KV-Free Recursive Attention
 * 
 * Replaces the O(N) KV cache with an O(d_model) recurrent latent state (s_t).
 * Uses extreme precision FP32 for the learned temporal gating mechanism,
 * while utilizing 0.45-bit POPCOUNT for the massive Q, K, V projections.
 */
class FluxAttention {
private:
    size_t d_model;
    size_t n_heads;
    
    // 0.45-bit AND+POPCOUNT Projections
    LinearFluxKernel q_proj;
    LinearFluxKernel k_proj;
    LinearFluxKernel v_proj;
    LinearFluxKernel o_proj;
    
    // FP32 Temporal Gates (Strictly aligned with SNN Architecture Docs)
    // s_t = gate * s_{t-1} + (1 - gate) * current_context
    DenseTensor w_gate;
    DenseTensor w_update;

    // Helper for FP32 Matrix Multiplication (Token x Weights)
    void dense_matmul(const DenseTensor& X, const DenseTensor& W, DenseTensor& out) const {
        size_t batch = X.rows;
        size_t out_dim = W.rows;
        size_t in_dim = X.cols;
        
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch; ++b) {
            for (size_t i = 0; i < out_dim; ++i) {
                float sum = 0.0f;
                const float* x_ptr = X.row_ptr(b);
                const float* w_ptr = W.row_ptr(i);
                
                // AVX FMA would be used here in the core tensor runtime
                // For now, explicit loop which auto-vectorizes in -O3
                for (size_t j = 0; j < in_dim; ++j) {
                    sum += x_ptr[j] * w_ptr[j];
                }
                out.row_ptr(b)[i] = sum;
            }
        }
    }

public:
    FluxAttention(size_t d, size_t heads, 
                  const LinearFluxKernel& q, const LinearFluxKernel& k, 
                  const LinearFluxKernel& v, const LinearFluxKernel& o,
                  const DenseTensor& wg, const DenseTensor& wu)
        : d_model(d), n_heads(heads), q_proj(q), k_proj(k), v_proj(v), o_proj(o),
          w_gate(wg), w_update(wu) {}

    /**
     * @brief Forward Pass (KV-Free)
     * 
     * @param x_bin Packed binary input token (Batch x K_bytes)
     * @param x_dense Full FP32 input token for gate computation (Batch x d_model)
     * @param s_prev Latent state from previous timestep (Batch x d_model)
     * @param out Output attention vector (Batch x d_model)
     * @param s_next Updated latent state (Batch x d_model)
     */
    void forward(const FluxTensor& x_bin, const DenseTensor& x_dense, const DenseTensor& s_prev,
                 DenseTensor& out, DenseTensor& s_next) const {
        
        int batch = x_bin.rows;
        
        // Ensure outputs are sized
        if (s_next.rows != batch || s_next.cols != d_model) {
            s_next = DenseTensor(batch, d_model);
        }
        
        // 1. Extreme Compression Q, K, V Projections
        DenseTensor q, k, v;
        q_proj.forward(x_bin, q);
        k_proj.forward(x_bin, k);
        v_proj.forward(x_bin, v);
        
        // 2. Compute Latent State Update Gates (Using v as the local context)
        DenseTensor gate_logits(batch, d_model);
        dense_matmul(v, w_gate, gate_logits);
        
        DenseTensor update_cand(batch, d_model);
        dense_matmul(v, w_update, update_cand);
        
        // 3. Apply Multi-Scale Recurrent Latent Update (s_t)
        #pragma omp parallel for
        for (int b = 0; b < batch; ++b) {
            float* s_next_ptr = s_next.row_ptr(b);
            const float* s_prev_ptr = s_prev.row_ptr(b);
            const float* gl_ptr = gate_logits.row_ptr(b);
            const float* uc_ptr = update_cand.row_ptr(b);
            
            for (size_t i = 0; i < d_model; ++i) {
                // Sigmoid gate with extreme precision float clipping to prevent NaN
                float g = 1.0f / (1.0f + std::exp(-std::max(-15.0f, std::min(gl_ptr[i], 15.0f))));
                float update = std::tanh(uc_ptr[i]);
                
                // Working memory update!
                s_next_ptr[i] = g * s_prev_ptr[i] + (1.0f - g) * update;
            }
        }
        
        // 4. Binarize s_next for the O_proj (Output projection)
        // In actual implementation, this requires packing s_next into a FluxTensor.
        // For demonstration, we assume o_proj takes binary. We must pack it.
        size_t K_bytes = o_proj_k_bytes(); // Helper to get byte width
        FluxTensor s_next_bin(batch, K_bytes, d_model);
        
        #pragma omp parallel for
        for (int b = 0; b < batch; ++b) {
            uint8_t* bin_ptr = s_next_bin.row_ptr(b);
            const float* s_ptr = s_next.row_ptr(b);
            
            for (size_t i = 0; i < d_model; ++i) {
                if (s_ptr[i] > 0.0f) {
                    bin_ptr[i / 8] |= (1 << (i % 8));
                }
            }
        }
        
        // 5. Output Projection
        o_proj.forward(s_next_bin, out);
    }
    
private:
    // Helper to extract K_bytes size from o_proj (would be an accessor in real implementation)
    size_t o_proj_k_bytes() const { return d_model / 8; }
};

} // namespace recursive
} // namespace neural
} // namespace snn

