#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "rebirth/coordinator.h"
#include "optimizer/distributed_lgc.h"

namespace py = pybind11;

// Fast Associative Scan for TTFT Acceleration
// s_t = g_t * s_{t-1} + (1 - g_t) * u_t
py::array_t<float> fast_associative_scan(py::array_t<float> g_seq_np, py::array_t<float> u_seq_np, py::array_t<float> s_init_np, bool reverse = false) {
    auto g_seq = g_seq_np.unchecked<3>();
    auto u_seq = u_seq_np.unchecked<3>();
    auto s_init = s_init_np.unchecked<2>();

    py::ssize_t batch = g_seq.shape(0);
    py::ssize_t L = g_seq.shape(1);
    py::ssize_t d_model = g_seq.shape(2);

    auto result = py::array_t<float>({batch, L, d_model});
    auto res = result.mutable_unchecked<3>();

    for (py::ssize_t b = 0; b < batch; ++b) {
        // Local state
        std::vector<float> s_curr(d_model);
        for (py::ssize_t d = 0; d < d_model; ++d) {
            s_curr[d] = s_init(b, d);
        }

        if (!reverse) {
            for (py::ssize_t t = 0; t < L; ++t) {
                for (py::ssize_t d = 0; d < d_model; ++d) {
                    float g = g_seq(b, t, d);
                    float u = u_seq(b, t, d);
                    s_curr[d] = g * s_curr[d] + (1.0f - g) * u;
                    res(b, t, d) = s_curr[d];
                }
            }
        } else {
            for (py::ssize_t t = L - 1; t >= 0; --t) {
                for (py::ssize_t d = 0; d < d_model; ++d) {
                    float g = g_seq(b, t, d);
                    float u = u_seq(b, t, d);
                    s_curr[d] = g * s_curr[d] + (1.0f - g) * u;
                    res(b, t, d) = s_curr[d];
                }
            }
        }
    }
    return result;
}

PYBIND11_MODULE(snn_training_cpp, m) {
    m.doc() = "C++ Bindings for SNN IPP Training Engine and Accelerated Scans";

    m.def("fast_associative_scan", &fast_associative_scan,
          py::arg("g_seq"), py::arg("u_seq"), py::arg("s_init"), py::arg("reverse") = false,
          "Lightning-fast parallel associative scan for SNN Recurrent State TTFT.");

    py::class_<snn::training::rebirth::RebirthCoordinator>(m, "RebirthCoordinator")
        .def(py::init<>())
        .def("broadcast_rebirth", &snn::training::rebirth::RebirthCoordinator::broadcast_rebirth,
             py::arg("param_id"), py::arg("dead_indices"),
             "Broadcast dead parameters undergoing rebirth to the cluster.")
        .def("get_total_rebirths", &snn::training::rebirth::RebirthCoordinator::get_total_rebirths,
             py::arg("param_id"),
             "Get the total number of historical rebirths for a specific parameter block.");

    py::class_<snn::training::optimizer::DistributedLGC>(m, "DistributedLGC")
        .def(py::init<float, float, const std::string&, size_t>(),
             py::arg("lr"), py::arg("wd"), py::arg("cache_dir"), py::arg("budget_gb"))
        .def("offload_state_to_ssd", &snn::training::optimizer::DistributedLGC::offload_state_to_ssd,
             py::arg("param_id"))
        .def("prefetch_state_from_ssd", &snn::training::optimizer::DistributedLGC::prefetch_state_from_ssd,
             py::arg("param_id"));
}
