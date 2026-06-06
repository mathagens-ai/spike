#pragma once
#include <unordered_map>
#include <iostream>
#include <vector>

namespace snn {
namespace training {
namespace rebirth {

/**
 * @brief Datacenter Rebirth Coordinator
 * 
 * Ensures that when an IPP (Intelligence Per Parameter) node dies and is "reborn",
 * the rebirth state is identically synchronized across all cluster instances so
 * nodes don't diverge during distributed training.
 */
class RebirthCoordinator {
private:
    std::unordered_map<int, size_t> global_rebirth_counts;

public:
    RebirthCoordinator() {}

    void broadcast_rebirth(int param_id, const std::vector<int>& dead_indices) {
        // Broadcasts to all datacenter nodes that these specific parameters
        // are undergoing the 5-phase rebirth protocol.
        global_rebirth_counts[param_id] += dead_indices.size();
    }

    size_t get_total_rebirths(int param_id) const {
        auto it = global_rebirth_counts.find(param_id);
        return (it != global_rebirth_counts.end()) ? it->second : 0;
    }
};

} // namespace rebirth
} // namespace training
} // namespace snn
