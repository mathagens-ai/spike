#pragma once
#include "../../snn-core/tensor/tensor.h"
#include "../../snn-neural/flux_attention/flux_attention.h"
#include "../../snn-neural/flux_ffn/flux_ffn.h"
#include <vector>
#include <memory>
#include <iostream>

namespace snn {
namespace runtime {
namespace engine {

using namespace snn::core::tensor;
using namespace snn::neural::recursive;
using namespace snn::neural::sparse;

struct SNNBlock {
    FluxAttention attention;
    FluxFFN ffn;
    
    // RMSNorm weights
    DenseTensor norm1_w;
    DenseTensor norm2_w;

    SNNBlock(const FluxAttention& a, const FluxFFN& f, size_t d_model)
        : attention(a), ffn(f), norm1_w(d_model, 1), norm2_w(d_model, 1) {
        // Initialize norm weights to 1.0
        for(size_t i=0; i<d_model; ++i) {
            norm1_w.data[i] = 1.0f;
            norm2_w.data[i] = 1.0f;
        }
    }

    void rms_norm(const DenseTensor& x, const DenseTensor& w, DenseTensor& out) const {
        size_t batch = x.rows;
        size_t d_model = x.cols;
        const float eps = 1e-5f;

        #pragma omp parallel for
        for (int b = 0; b < batch; ++b) {
            const float* x_ptr = x.row_ptr(b);
            float* out_ptr = out.row_ptr(b);
            
            float sum_sq = 0.0f;
            for (size_t i = 0; i < d_model; ++i) {
                sum_sq += x_ptr[i] * x_ptr[i];
            }
            float var = sum_sq / static_cast<float>(d_model);
            float inv_std = 1.0f / std::sqrt(var + eps);
            
            for (size_t i = 0; i < d_model; ++i) {
                out_ptr[i] = x_ptr[i] * inv_std * w.data[i];
            }
        }
    }
};

/**
 * @brief SNN Inference Engine
 * 
 * Orchestrates the full SNN network execution autoregressively 
 * without a KV cache. Optimized for extreme low latency.
 */
class SNNInferenceEngine {
private:
    size_t d_model;
    size_t vocab_size;
    
    std::vector<SNNBlock> layers;
    DenseTensor embed_tokens; // Simulated dense embeddings for now
    LinearFluxKernel lm_head;
    DenseTensor final_norm_w;

public:
    SNNInferenceEngine(size_t d, size_t v, const LinearFluxKernel& head)
        : d_model(d), vocab_size(v), lm_head(head), final_norm_w(d, 1) {
        
        embed_tokens = DenseTensor(vocab_size, d_model);
        for(size_t i=0; i<vocab_size; ++i) {
            for(size_t j=0; j<d_model; ++j) {
                embed_tokens.row_ptr(i)[j] = ((float)rand() / RAND_MAX) * 0.04f - 0.02f;
            }
        }
        for(size_t i=0; i<d_model; ++i) final_norm_w.data[i] = 1.0f;
    }

    void add_layer(const SNNBlock& block) {
        layers.push_back(block);
    }

    /**
     * @brief Forward pass over a sequence of tokens
     * 
     * @param input_ids Array of token IDs (Batch x SeqLen)
     * @param batch_size Batch dimension
     * @param seq_len Sequence dimension
     * @param states Vector of latent states (one per layer). Will be updated.
     * @return std::vector<float> Output logits (Batch x SeqLen x Vocab)
     */
    std::vector<float> forward(const int* input_ids, size_t batch_size, size_t seq_len, 
                               std::vector<DenseTensor>& states) {
        
        if (states.empty()) {
            for (size_t i = 0; i < layers.size(); ++i) {
                states.emplace_back(batch_size, d_model); // Zero initialized
            }
        }

        std::vector<float> all_logits(batch_size * seq_len * vocab_size, 0.0f);

        DenseTensor x(batch_size, d_model);
        DenseTensor x_norm(batch_size, d_model);
        DenseTensor attn_out(batch_size, d_model);
        DenseTensor ffn_out(batch_size, d_model);
        DenseTensor logits(batch_size, vocab_size);
        
        size_t K_bytes = d_model / 8;
        FluxTensor x_bin(batch_size, K_bytes, d_model);

        for (size_t t = 0; t < seq_len; ++t) {
            // 1. Embedding lookup
            for (size_t b = 0; b < batch_size; ++b) {
                int token = input_ids[b * seq_len + t];
                const float* emb_ptr = embed_tokens.row_ptr(token);
                float* x_ptr = x.row_ptr(b);
                for (size_t i = 0; i < d_model; ++i) {
                    x_ptr[i] = emb_ptr[i];
                }
            }

            // 2. Process layers
            for (size_t l = 0; l < layers.size(); ++l) {
                SNNBlock& layer = layers[l];
                
                // Norm 1
                layer.rms_norm(x, layer.norm1_w, x_norm);
                
                // Binarize input for Flux Attention
                for(size_t b=0; b<batch_size; ++b) {
                    uint8_t* bin = x_bin.row_ptr(b);
                    const float* x_n = x_norm.row_ptr(b);
                    for(size_t i=0; i<K_bytes; ++i) bin[i] = 0;
                    for(size_t i=0; i<d_model; ++i) {
                        if (x_n[i] > 0.0f) bin[i/8] |= (1 << (i%8));
                    }
                }
                
                // Attention
                layer.attention.forward(x_bin, x_norm, states[l], attn_out, states[l]); // state is updated in-place
                
                // Residual
                for(size_t b=0; b<batch_size; ++b) {
                    float* x_p = x.row_ptr(b);
                    const float* a_p = attn_out.row_ptr(b);
                    for(size_t i=0; i<d_model; ++i) x_p[i] += a_p[i];
                }
                
                // Norm 2
                layer.rms_norm(x, layer.norm2_w, x_norm);
                
                // Binarize for Flux FFN
                for(size_t b=0; b<batch_size; ++b) {
                    uint8_t* bin = x_bin.row_ptr(b);
                    const float* x_n = x_norm.row_ptr(b);
                    for(size_t i=0; i<K_bytes; ++i) bin[i] = 0;
                    for(size_t i=0; i<d_model; ++i) {
                        if (x_n[i] > 0.0f) bin[i/8] |= (1 << (i%8));
                    }
                }
                
                // FFN
                layer.ffn.forward(x_bin, ffn_out);
                
                // Residual
                for(size_t b=0; b<batch_size; ++b) {
                    float* x_p = x.row_ptr(b);
                    const float* f_p = ffn_out.row_ptr(b);
                    for(size_t i=0; i<d_model; ++i) x_p[i] += f_p[i];
                }
            }

            // 3. LM Head
            layers[0].rms_norm(x, final_norm_w, x_norm);
            
            // Binarize for LM Head
            for(size_t b=0; b<batch_size; ++b) {
                uint8_t* bin = x_bin.row_ptr(b);
                const float* x_n = x_norm.row_ptr(b);
                for(size_t i=0; i<K_bytes; ++i) bin[i] = 0;
                for(size_t i=0; i<d_model; ++i) {
                    if (x_n[i] > 0.0f) bin[i/8] |= (1 << (i%8));
                }
            }
            
            lm_head.forward(x_bin, logits);
            
            // Write to output tensor
            for(size_t b=0; b<batch_size; ++b) {
                const float* l_p = logits.row_ptr(b);
                size_t offset = (b * seq_len + t) * vocab_size;
                for(size_t v=0; v<vocab_size; ++v) {
                    all_logits[offset + v] = l_p[v];
                }
            }
        }

        return all_logits;
    }
};

} // namespace engine
} // namespace runtime
} // namespace snn

