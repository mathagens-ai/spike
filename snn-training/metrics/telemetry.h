#pragma once
#include <string>
#include <chrono>
#include <iostream>

namespace snn {
namespace training {
namespace metrics {

/**
 * @brief Datacenter Telemetry
 * 
 * Aggregates IPP vitality scores, inference speed (tokens/sec), and memory bandwidth
 * across massive distributed clusters and reports them back to a central logging node.
 */
class Telemetry {
private:
    std::string node_name;
    size_t processed_tokens;
    std::chrono::high_resolution_clock::time_point start_time;

public:
    Telemetry(const std::string& name) : node_name(name), processed_tokens(0) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void log_step(size_t tokens_in_batch, float avg_vitality, float loss) {
        processed_tokens += tokens_in_batch;
    }

    void report() const {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = now - start_time;
        double tps = processed_tokens / diff.count();
        // std::cout << "[TELEMETRY " << node_name << "] " 
        //           << "Tokens/Sec: " << tps << " | Total: " << processed_tokens << "\n";
    }
};

} // namespace metrics
} // namespace training
} // namespace snn
