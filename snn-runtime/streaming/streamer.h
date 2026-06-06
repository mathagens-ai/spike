#pragma once
#include "../inference/engine.h"
#include <queue>
#include <mutex>
#include <condition_variable>

namespace snn {
namespace runtime {
namespace streaming {

using namespace snn::runtime::engine;

/**
 * @brief Streaming Execution Engine
 * 
 * SNN does not use a KV cache, meaning it can process infinite token streams.
 * This class provides a non-blocking FIFO queue for tokens, running the Engine
 * continuously in a background thread and emitting logits as they are computed.
 */
class Streamer {
private:
    SNNInferenceEngine& engine;
    std::queue<int> token_queue;
    std::mutex q_mtx;
    std::condition_variable q_cv;
    bool stop_flag;
    std::vector<snn::core::tensor::DenseTensor> recurrent_states;

public:
    Streamer(SNNInferenceEngine& eng) : engine(eng), stop_flag(false) {}

    void push_token(int token_id) {
        {
            std::lock_guard<std::mutex> lock(q_mtx);
            token_queue.push(token_id);
        }
        q_cv.notify_one();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(q_mtx);
            stop_flag = true;
        }
        q_cv.notify_all();
    }

    // Function to be run in a dedicated thread
    void run_loop() {
        while (true) {
            int token;
            {
                std::unique_lock<std::mutex> lock(q_mtx);
                q_cv.wait(lock, [this] { return !token_queue.empty() || stop_flag; });
                if (stop_flag && token_queue.empty()) break;
                
                token = token_queue.front();
                token_queue.pop();
            }

            // Execute SNN without KV cache. State evolves naturally.
            engine.forward(&token, 1, 1, recurrent_states);
        }
    }
};

} // namespace streaming
} // namespace runtime
} // namespace snn

