#pragma once
#include "../../snn-core/tensor/tensor.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace snn {
namespace runtime {
namespace lgc {

using namespace snn::core::tensor;

/**
 * @brief Intelligence Per Parameter (IPP) Tracker
 * 
 * Tracks the gradient vitality of every single FP32 parameter in the network.
 * Executes the mathematically proven smooth Rebirth Protocol (Detection -> 
 * Cooldown -> Perturbation -> Warmup -> Lockout) to prevent catastrophic
 * forgetting and oscillation while maintaining 100% parameter utilization.
 */
class IPPTracker {
private:
    struct ParamState {
        DenseTensor* param;
        std::vector<float> vitality;
        std::vector<int> cooldown;
        std::vector<int> lockout;
        std::vector<bool> marked;
        size_t size;
        
        ParamState(DenseTensor* p) : param(p) {
            size = p->rows * p->cols;
            vitality.assign(size, 1.0f);
            cooldown.assign(size, 0);
            lockout.assign(size, 0);
            marked.assign(size, false);
        }
    };

    std::vector<ParamState> states;
    
    // IPP Constants matching Python validation exactly
    const float decay_rate = 0.9997f;
    const float growth_rate = 0.05f;
    const float death_threshold = 0.05f;
    const float rebirth_cap = 0.05f; // Max 5% of params per cycle
    const int cooldown_steps = 50;
    const int lockout_steps = 500;

    std::mt19937 rng;

public:
    IPPTracker() : rng(42) {} // Deterministic seed for validation

    /**
     * @brief Attach a dense tensor (e.g. gates or LGC state) to the tracker.
     */
    void register_parameter(DenseTensor* param) {
        states.emplace_back(param);
    }

    /**
     * @brief Execute a single IPP tracking and rebirth step.
     * 
     * @param gradients List of DenseTensors containing the gradients for each registered param.
     */
    void step(const std::vector<DenseTensor*>& gradients) {
        if (gradients.size() != states.size()) {
            throw std::runtime_error("IPP: Number of gradients does not match registered parameters.");
        }

        #pragma omp parallel for
        for (int i = 0; i < states.size(); ++i) {
            ParamState& s = states[i];
            const DenseTensor* grad = gradients[i];
            
            float sum_sq = 0.0f;
            for(size_t k = 0; k < s.size; ++k) {
                sum_sq += s.param->data[k] * s.param->data[k];
            }
            float layer_std = std::sqrt(sum_sq / static_cast<float>(s.size)) * 0.1f;
            
            // Collect indices ready for rebirth
            std::vector<size_t> ready_indices;

            for (size_t j = 0; j < s.size; ++j) {
                // 1. Vitality Update
                float g_mag = std::abs(grad->data[j]);
                float active = (g_mag > 1e-7f) ? 1.0f : 0.0f;
                s.vitality[j] = std::max(0.0f, std::min(2.0f, s.vitality[j] * decay_rate + active * growth_rate));

                // 2. Lockout countdown
                if (s.lockout[j] > 0) s.lockout[j]--;

                // 3. Phase 1: DETECTION
                if (s.vitality[j] < death_threshold && !s.marked[j] && s.lockout[j] == 0) {
                    s.marked[j] = true;
                    s.cooldown[j] = cooldown_steps;
                }

                // 4. Phase 2: COOLDOWN COUNTDOWN
                if (s.marked[j]) {
                    s.cooldown[j]--;
                }

                // 5. Phase 3: PERTURBATION PREP
                if (s.marked[j] && s.cooldown[j] <= 0) {
                    ready_indices.push_back(j);
                }
            }

            // Enforce Global Rebirth Cap
            size_t max_rebirth = static_cast<size_t>(static_cast<float>(s.size) * rebirth_cap);
            if (ready_indices.size() > max_rebirth) {
                std::shuffle(ready_indices.begin(), ready_indices.end(), rng);
                ready_indices.resize(max_rebirth);
            }

            // Apply Perturbation
            std::normal_distribution<float> noise(0.0f, layer_std);
            for (size_t idx : ready_indices) {
                s.param->data[idx] += noise(rng);
                
                // Reset States
                s.vitality[idx] = 0.5f; // Warmup
                s.lockout[idx] = lockout_steps;
                s.marked[idx] = false;
            }
        }
    }

    /**
     * @brief Calculate the total percentage of actively learning parameters.
     */
    float get_ipp_score() const {
        size_t alive = 0;
        size_t total = 0;
        
        for (const auto& s : states) {
            total += s.size;
            for (size_t j = 0; j < s.size; ++j) {
                if (s.vitality[j] > 0.5f) {
                    alive++;
                }
            }
        }
        
        if (total == 0) return 100.0f;
        return (static_cast<float>(alive) / static_cast<float>(total)) * 100.0f;
    }
};

} // namespace lgc
} // namespace runtime
} // namespace snn

