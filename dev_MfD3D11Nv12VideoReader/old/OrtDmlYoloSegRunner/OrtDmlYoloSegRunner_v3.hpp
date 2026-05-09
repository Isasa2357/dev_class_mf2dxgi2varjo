#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <wrl/client.h>

#include <d3d12.h>

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

#include "D3D12Core.hpp"

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

    // Optional. Call this before run_cpu_input_and_upload_outputs()
    // if you want this runner to expose ORT outputs as D3D12 buffers.
    void set_d3d12_core(D3D12Core& d3d12_core);

    // Existing CPU-input / CPU-output path.
    std::vector<OutputTensor> run_cpu_input(
        const std::vector<float>& input_tensor,
        int64_t batch,
        int64_t channels,
        int64_t height,
        int64_t width
    );



    // Transitional D3D12-input path.
    // Input is received as a D3D12 buffer, internally read back to CPU,
    // then the existing CPU-input ORT path is used. ORT outputs are uploaded
    // to D3D12 output buffers, so the caller can pass output0/output1 to
    // YoloSegPostProcessorD3D12::process_external_outputs_and_wait().
    // This removes the explicit readback code from main(), but it is not yet
    // true ORT/DML GPU input binding.
    std::vector<OutputTensor> run_d3d12_input_and_upload_outputs(
        ID3D12Resource* input_buffer,
        D3D12_RESOURCE_STATES& input_buffer_state,
        int64_t batch,
        int64_t channels,
        int64_t height,
        int64_t width
    );

    // Transitional path.
    // Input is still CPU std::vector<float>, but ORT CPU outputs are uploaded
    // into D3D12 default buffers owned by this runner.
    // This lets YoloSegPostProcessorD3D12 consume the outputs through
    // process_external_outputs_and_wait().
    std::vector<OutputTensor> run_cpu_input_and_upload_outputs(
        const std::vector<float>& input_tensor,
        int64_t batch,
        int64_t channels,
        int64_t height,
        int64_t width
    );

    void upload_outputs_to_d3d12(
        const std::vector<OutputTensor>& outputs
    );

    void print_model_info() const;

    const std::vector<TensorInfo>& input_infos() const;
    const std::vector<TensorInfo>& output_infos() const;

    size_t gpu_output_count() const;

    ID3D12Resource* output_buffer(size_t index) const;
    D3D12_RESOURCE_STATES& output_state_ref(size_t index);

    ID3D12Resource* output0_buffer() const;
    ID3D12Resource* output1_buffer() const;

    D3D12_RESOURCE_STATES& output0_state_ref();
    D3D12_RESOURCE_STATES& output1_state_ref();

    const std::vector<int64_t>& gpu_output_shape(size_t index) const;
    size_t gpu_output_element_count(size_t index) const;
    size_t gpu_output_byte_size(size_t index) const;

private:
    struct GpuOutputBuffer {
        Microsoft::WRL::ComPtr<ID3D12Resource> default_buffer;
        Microsoft::WRL::ComPtr<ID3D12Resource> upload_buffer;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
        std::vector<int64_t> shape;
        size_t element_count = 0;
        size_t byte_size = 0;
    };

private:
    void collect_model_io_info();

    static std::string shape_to_string(
        const std::vector<int64_t>& shape
    );

    static size_t element_count_from_shape(
        const std::vector<int64_t>& shape
    );

    void ensure_gpu_output_buffer(
        size_t index,
        const std::vector<int64_t>& shape,
        size_t element_count
    );



    std::vector<float> readback_d3d12_float_buffer(
        ID3D12Resource* buffer,
        D3D12_RESOURCE_STATES& buffer_state,
        size_t element_count,
        D3D12_RESOURCE_STATES restore_state
    );

    void upload_float_vector_to_gpu_output(
        size_t index,
        const std::vector<float>& data
    );

    void transition_resource(
        ID3D12GraphicsCommandList* command_list,
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES& current_state,
        D3D12_RESOURCE_STATES new_state
    );

    static D3D12_RESOURCE_DESC make_buffer_desc(
        UINT64 byte_size,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE
    );

    static D3D12_HEAP_PROPERTIES make_heap_props(
        D3D12_HEAP_TYPE type
    );

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    D3D12Core* d3d12_core_ = nullptr;

    std::vector<TensorInfo> input_infos_;
    std::vector<TensorInfo> output_infos_;

    // Ort::AllocatedStringPtr の寿命問題を避けるため、
    // node名は std::string で保持し、Run時に c_str() 配列を作る。
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    std::vector<GpuOutputBuffer> gpu_outputs_;

    bool initialized_ = false;
};
