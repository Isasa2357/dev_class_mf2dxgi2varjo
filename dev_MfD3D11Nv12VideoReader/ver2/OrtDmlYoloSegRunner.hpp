#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <wrl/client.h>

#include <d3d12.h>
#include <DirectML.h>

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

#include "D3D12Core.hpp"

#pragma comment(lib, "onnxruntime.lib")
#pragma comment(lib, "DirectML.lib")
#pragma comment(lib, "d3d12.lib")

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
    ~OrtDmlYoloSegRunner() = default;

    OrtDmlYoloSegRunner(const OrtDmlYoloSegRunner&) = delete;
    OrtDmlYoloSegRunner& operator=(const OrtDmlYoloSegRunner&) = delete;

    // Legacy initializer.
    // This uses DML device_id mode when use_directml=true. It is kept for CPU-input tests.
    // True D3D12-resource I/O binding requires initialize_with_d3d12().
    void initialize(
        const wchar_t* model_path,
        bool use_directml = true,
        int dml_device_id = 0
    );

    // Preferred initializer for true D3D12 buffer input/output binding.
    // It creates a DirectML device from the same ID3D12Device as D3D12Core and
    // registers DML1 EP with D3D12Core's command queue.
    void initialize_with_d3d12(
        D3D12Core& d3d12_core,
        const wchar_t* model_path
    );

    // Optional for legacy CPU-input + output-upload path.
    void set_d3d12_core(D3D12Core& d3d12_core);

    std::vector<OutputTensor> run_cpu_input(
        const std::vector<float>& input_tensor,
        int64_t batch,
        int64_t channels,
        int64_t height,
        int64_t width
    );

    // True D3D12-input path when initialized with initialize_with_d3d12().
    // If the runner was initialized with the legacy initialize(), this function
    // falls back to the old transitional readback path so existing tests still run.
    // In true DML1 mode, the returned OutputTensor entries contain name/shape only;
    // data is intentionally empty to avoid CPU readback.
    std::vector<OutputTensor> run_d3d12_input_and_upload_outputs(
        ID3D12Resource* input_buffer,
        D3D12_RESOURCE_STATES& input_buffer_state,
        int64_t batch,
        int64_t channels,
        int64_t height,
        int64_t width
    );

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

    bool is_dml1_io_binding_enabled() const;

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
    void initialize_impl(
        const wchar_t* model_path,
        bool use_directml,
        int dml_device_id,
        bool use_dml1_with_d3d12
    );

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

    void ensure_gpu_output_buffers_from_model_info();

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

    std::vector<OutputTensor> run_d3d12_input_with_iobinding(
        ID3D12Resource* input_buffer,
        D3D12_RESOURCE_STATES& input_buffer_state,
        const std::vector<int64_t>& input_shape
    );

    Ort::Value create_dml_tensor_from_d3d12_resource(
        ID3D12Resource* resource,
        size_t byte_size,
        const std::vector<int64_t>& shape,
        void** out_dml_allocation
    );

    void free_dml_allocation(void* allocation) noexcept;

    void transition_resource(
        ID3D12GraphicsCommandList* command_list,
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES& current_state,
        D3D12_RESOURCE_STATES new_state
    );

    void transition_resource_and_wait(
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

    const OrtDmlApi* dml_api_ = nullptr;
    Microsoft::WRL::ComPtr<IDMLDevice> dml_device_;
    bool dml1_io_binding_enabled_ = false;

    std::vector<TensorInfo> input_infos_;
    std::vector<TensorInfo> output_infos_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    std::vector<GpuOutputBuffer> gpu_outputs_;

    bool initialized_ = false;
};
