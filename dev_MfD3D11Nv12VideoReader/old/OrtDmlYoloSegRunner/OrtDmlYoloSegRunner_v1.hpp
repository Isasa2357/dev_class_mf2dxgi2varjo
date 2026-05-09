#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

#pragma comment(lib, "onnxruntime.lib")

class OrtDmlYoloSegRunner {
public:
    struct TensorInfo {
        std::string name;
        ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        std::vector<int64_t> shape;
    };

    struct OutputTensor {
        std::string name;
        std::vector<int64_t> shape;
        std::vector<float> data;
    };

public:
    OrtDmlYoloSegRunner();

    OrtDmlYoloSegRunner(const OrtDmlYoloSegRunner&) = delete;
    OrtDmlYoloSegRunner& operator=(const OrtDmlYoloSegRunner&) = delete;

    void initialize(
        const wchar_t* model_path,
        bool use_directml = true,
        int dml_device_id = 0
    );

    std::vector<OutputTensor> run_cpu_input(
        const std::vector<float>& input_tensor,
        int64_t batch,
        int64_t channels,
        int64_t height,
        int64_t width
    );

    void print_model_info() const;

    const std::vector<TensorInfo>& input_infos() const;
    const std::vector<TensorInfo>& output_infos() const;

private:
    void collect_model_io_info();

    static std::string shape_to_string(
        const std::vector<int64_t>& shape
    );

    static size_t element_count_from_shape(
        const std::vector<int64_t>& shape
    );

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<TensorInfo> input_infos_;
    std::vector<TensorInfo> output_infos_;

    // Ort::AllocatedStringPtr の寿命問題を避けるため、
    // node名は std::string で保持し、Run時に c_str() 配列を作る。
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    bool initialized_ = false;
};