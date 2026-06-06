#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "snn-core/quant/flux_compiler.h"
#include "snn-core/quant/fluxbits_bloom.h"
#include "snn-core/quant/fluxbits_binary.h"
#include "snn-core/quant/fluxbits_int4.h"
#include "snn-core/quant/fluxbits_int8.h"
#include "snn-neural/ipp/ipp_tracker.h"
#include "snn-runtime/inference/engine.h"

namespace py = pybind11;
using namespace snn::core::fluxbits;
using namespace snn::runtime::engine;
using namespace snn::runtime::lgc;
using namespace snn::core::tensor;

// Wrapper struct to hold the compiled state
struct PyCompiledFlux {
    CompiledFlux data;
};

// Python binding module
PYBIND11_MODULE(snn_core, m) {
    m.doc() = "Super Neural Network (SNN) Core Engine with Bloom, Binary, Int4, and Int8 backends.";

    // 1. Legacy FluxBits Compiler (kept for compatibility)
    py::class_<PyCompiledFlux>(m, "CompiledFlux")
        .def_property_readonly("M", [](const PyCompiledFlux& c) { return c.data.M; })
        .def_property_readonly("N", [](const PyCompiledFlux& c) { return c.data.N; })
        .def_property_readonly("K_bits", [](const PyCompiledFlux& c) { return c.data.K_bits; })
        .def_property_readonly("density", [](const PyCompiledFlux& c) { return c.data.expected_density; });

    m.def("compile_weights", [](py::array_t<float> weights, float bits_per_param) {
        auto buf = weights.request();
        if (buf.ndim != 2) throw std::runtime_error("Weights must be 2D");
        
        size_t M = buf.shape[0];
        size_t N = buf.shape[1];
        
        PyCompiledFlux result;
        result.data = compile_weights(static_cast<float*>(buf.ptr), M, N, bits_per_param);
        return result;
    }, "Compile FP32 weights into Bloom Tensor");

    // 2. 0.22/0.45-bit Bloom Quantization
    py::class_<CompiledFluxBloom>(m, "CompiledFluxBloom")
        .def_property_readonly("M", [](const CompiledFluxBloom& c) { return c.M; })
        .def_property_readonly("N", [](const CompiledFluxBloom& c) { return c.N; })
        .def_property_readonly("K_bits", [](const CompiledFluxBloom& c) { return c.K_bits; })
        .def_property_readonly("K_bytes", [](const CompiledFluxBloom& c) { return c.K_bytes; })
        .def_property_readonly("d_hashes", [](const CompiledFluxBloom& c) { return c.d_hashes; })
        .def_property_readonly("compression_ratio", [](const CompiledFluxBloom& c) { return c.compression_ratio; })
        .def_property_readonly("expected_density", [](const CompiledFluxBloom& c) { return c.expected_density; })
        .def_property_readonly("data", [](const CompiledFluxBloom& c) {
            return py::bytes(reinterpret_cast<const char*>(c.data.data()), c.data.size());
        });

    m.def("bloom_compile", [](py::array_t<float> weights, float bits_per_param) {
        auto buf = weights.request();
        if (buf.ndim != 2) throw std::runtime_error("Weights must be 2D");
        size_t M = buf.shape[0];
        size_t N = buf.shape[1];
        return bloom_compile(static_cast<const float*>(buf.ptr), M, N, bits_per_param);
    });

    m.def("bloom_binarize", [](py::array_t<float> inputs, size_t K_bits, size_t d_hashes, py::array_t<uint8_t> out) {
        auto buf_in = inputs.request();
        auto buf_out = out.request();
        if (buf_in.ndim != 2) throw std::runtime_error("Inputs must be 2D");
        if (buf_out.ndim != 2) throw std::runtime_error("Output buffer must be 2D");
        size_t Batch = buf_in.shape[0];
        size_t N = buf_in.shape[1];
        bloom_binarize(static_cast<const float*>(buf_in.ptr), Batch, N, K_bits, d_hashes, static_cast<uint8_t*>(buf_out.ptr));
    });

    m.def("bloom_forward", [](py::array_t<uint8_t> x_packed, const CompiledFluxBloom& compiled, py::array_t<float> out) {
        auto buf_x = x_packed.request();
        auto buf_out = out.request();
        if (buf_x.ndim != 2) throw std::runtime_error("x_packed must be 2D");
        if (buf_out.ndim != 2) throw std::runtime_error("Output must be 2D");
        size_t Batch = buf_x.shape[0];
        bloom_forward(static_cast<const uint8_t*>(buf_x.ptr), Batch, compiled, static_cast<float*>(buf_out.ptr));
    });

    // 3. 1-bit Binary Quantization
    py::class_<CompiledFluxBinary>(m, "CompiledFluxBinary")
        .def_property_readonly("M", [](const CompiledFluxBinary& c) { return c.M; })
        .def_property_readonly("N", [](const CompiledFluxBinary& c) { return c.N; })
        .def_property_readonly("K_bits", [](const CompiledFluxBinary& c) { return c.K_bits; })
        .def_property_readonly("K_bytes", [](const CompiledFluxBinary& c) { return c.K_bytes; })
        .def_property_readonly("data", [](const CompiledFluxBinary& c) {
            return py::bytes(reinterpret_cast<const char*>(c.data.data()), c.data.size());
        });

    m.def("binary_compile", [](py::array_t<float> weights) {
        auto buf = weights.request();
        if (buf.ndim != 2) throw std::runtime_error("Weights must be 2D");
        size_t M = buf.shape[0];
        size_t N = buf.shape[1];
        return binary_compile(static_cast<const float*>(buf.ptr), M, N);
    });

    m.def("binary_binarize", [](py::array_t<float> inputs, py::array_t<uint8_t> out) {
        auto buf_in = inputs.request();
        auto buf_out = out.request();
        if (buf_in.ndim != 2) throw std::runtime_error("Inputs must be 2D");
        if (buf_out.ndim != 2) throw std::runtime_error("Output buffer must be 2D");
        size_t Batch = buf_in.shape[0];
        size_t N = buf_in.shape[1];
        binary_binarize(static_cast<const float*>(buf_in.ptr), Batch, N, static_cast<uint8_t*>(buf_out.ptr));
    });

    m.def("binary_forward", [](py::array_t<uint8_t> x_packed, const CompiledFluxBinary& compiled, py::array_t<float> out) {
        auto buf_x = x_packed.request();
        auto buf_out = out.request();
        if (buf_x.ndim != 2) throw std::runtime_error("x_packed must be 2D");
        if (buf_out.ndim != 2) throw std::runtime_error("Output must be 2D");
        size_t Batch = buf_x.shape[0];
        binary_forward(static_cast<const uint8_t*>(buf_x.ptr), Batch, compiled, static_cast<float*>(buf_out.ptr));
    });

    // 4. int4 Quantization
    py::class_<CompiledFluxInt4>(m, "CompiledFluxInt4")
        .def_property_readonly("M", [](const CompiledFluxInt4& c) { return c.M; })
        .def_property_readonly("N", [](const CompiledFluxInt4& c) { return c.N; })
        .def_property_readonly("K_bytes", [](const CompiledFluxInt4& c) { return c.K_bytes; })
        .def_property_readonly("data", [](const CompiledFluxInt4& c) {
            return py::bytes(reinterpret_cast<const char*>(c.data.data()), c.data.size());
        })
        .def_property_readonly("scales", [](const CompiledFluxInt4& c) {
            py::array_t<float> arr(c.scales.size());
            auto buf = arr.request();
            std::copy(c.scales.begin(), c.scales.end(), static_cast<float*>(buf.ptr));
            return arr;
        });

    m.def("int4_compile", [](py::array_t<float> weights) {
        auto buf = weights.request();
        if (buf.ndim != 2) throw std::runtime_error("Weights must be 2D");
        size_t M = buf.shape[0];
        size_t N = buf.shape[1];
        return int4_compile(static_cast<const float*>(buf.ptr), M, N);
    });

    m.def("int4_forward", [](py::array_t<float> x_dense, const CompiledFluxInt4& compiled, py::array_t<float> out) {
        auto buf_x = x_dense.request();
        auto buf_out = out.request();
        if (buf_x.ndim != 2) throw std::runtime_error("x_dense must be 2D");
        if (buf_out.ndim != 2) throw std::runtime_error("Output must be 2D");
        size_t Batch = buf_x.shape[0];
        int4_forward(static_cast<const float*>(buf_x.ptr), Batch, compiled, static_cast<float*>(buf_out.ptr));
    });

    // 5. int8 Quantization
    py::class_<CompiledFluxInt8>(m, "CompiledFluxInt8")
        .def_property_readonly("M", [](const CompiledFluxInt8& c) { return c.M; })
        .def_property_readonly("N", [](const CompiledFluxInt8& c) { return c.N; })
        .def_property_readonly("data", [](const CompiledFluxInt8& c) {
            return py::bytes(reinterpret_cast<const char*>(c.data.data()), c.data.size());
        })
        .def_property_readonly("scales", [](const CompiledFluxInt8& c) {
            py::array_t<float> arr(c.scales.size());
            auto buf = arr.request();
            std::copy(c.scales.begin(), c.scales.end(), static_cast<float*>(buf.ptr));
            return arr;
        });

    m.def("int8_compile", [](py::array_t<float> weights) {
        auto buf = weights.request();
        if (buf.ndim != 2) throw std::runtime_error("Weights must be 2D");
        size_t M = buf.shape[0];
        size_t N = buf.shape[1];
        return int8_compile(static_cast<const float*>(buf.ptr), M, N);
    });

    m.def("int8_forward", [](py::array_t<float> x_dense, const CompiledFluxInt8& compiled, py::array_t<float> out) {
        auto buf_x = x_dense.request();
        auto buf_out = out.request();
        if (buf_x.ndim != 2) throw std::runtime_error("x_dense must be 2D");
        if (buf_out.ndim != 2) throw std::runtime_error("Output must be 2D");
        size_t Batch = buf_x.shape[0];
        int8_forward(static_cast<const float*>(buf_x.ptr), Batch, compiled, static_cast<float*>(buf_out.ptr));
    });

    // 6. IPP Tracker
    py::class_<IPPTracker>(m, "IPPTracker")
        .def(py::init<>())
        .def("get_ipp_score", &IPPTracker::get_ipp_score);

    // 7. Engine (Mock wrapper to prove compilation)
    m.def("version", []() { return "SNN Engine v1.1 (AVX2/AVX-512)"; });
}

