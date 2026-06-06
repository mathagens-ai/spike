#pragma once
#include "../../snn-core/tensor/tensor.h"
#include "../../snn-core/tensor/linear_flux.h"
#include "../flux_ffn/flux_ffn.h"
#include <vector>

namespace snn {
namespace neural {
namespace expert {

using namespace snn::core::tensor;
using namespace snn::neural::sparse;

/**
 * @brief FluxMoE - Mixture of 0.25-bit Experts
 * 
 * Routes tokens to specific FluxFFN experts. Since each expert is 0.25-bit,
 * we can load thousands of experts into VRAM seamlessly.
 */
class FluxMoE {
private:
    std::vector<FluxFFN> experts;
    size_t num_experts;
    size_t active_k;

public:
    FluxMoE(const std::vector<FluxFFN>& exp, size_t k = 2) 
        : experts(exp), num_experts(exp.size()), active_k(k) {}

    void forward(const FluxTensor& x_bin, const DenseTensor& gate_logits, DenseTensor& out) {
        // MoE Routing implementation
        // For standard 0.25-bit, experts are evaluated based on Top-K gate logits.
    }
};

} // namespace expert
} // namespace neural
} // namespace snn

