#include "OrtDmlYoloSegRunner.hpp"

#include <stdexcept>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <cstring>

#include "HResultUtil.hpp"

OrtDmlYoloSegRunner::OrtDmlYoloSegRunner()
    : env_(ORT_LOGGING_LEVEL_WARNING, "OrtDmlYoloSegRunner")
{
}

void OrtDmlYoloSegRunner::initialize(
    const wchar_t* model_path,
    bool use_directml,
    int dml_device_id
)
{
    this->initialize_impl(
        model_path,
        use_directml,
        dml_device_id,
        false
    );
}

void OrtDmlYoloSegRunner::initialize_with_d3d12(
    D3D12Core& d3d12_core,
    const wchar_t* model_path
)
{
    this->set_d3d12_core(d3d12_core);

    this->initialize_impl(
        model_path,
        true,
        0,
        true
    );
}

void OrtDmlYoloSegRunner::initialize_impl(
    const wchar_t* model_path,
    bool use_directml,
    int dml_device_id,
    bool use_dml1_with_d3d12
)
{
    if (!model_path) {
        throw std::runtime_error("OrtDmlYoloSegRunner::initialize: model_path is null");
    }

    if (use_dml1_with_d3d12 && !this->d3d12_core_) {
        throw std::runtime_error("OrtDmlYoloSegRunner::initialize_with_d3d12: D3D12Core is not set");
    }

    session_options_ = Ort::SessionOptions{};

    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED
    );

    session_options_.SetIntraOpNumThreads(1);

    // DirectML EP does not benefit from ORT parallel execution here and memory
    // pattern can interfere with externally bound outputs. Keep this explicit.
    session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    session_options_.DisableMemPattern();

    dml_api_ = nullptr;
    dml_device_.Reset();
    dml1_io_binding_enabled_ = false;

    if (use_directml) {
        const OrtApi& ort_api = Ort::GetApi();

        Ort::ThrowOnError(
            ort_api.GetExecutionProviderApi(
                "DML",
                ORT_API_VERSION,
                reinterpret_cast<const void**>(&dml_api_)
            )
        );

        if (!dml_api_) {
            throw std::runtime_error("Failed to get OrtDmlApi");
        }

        if (use_dml1_with_d3d12) {
            HRESULT hr = DMLCreateDevice1(
                this->d3d12_core_->device(),
                DML_CREATE_DEVICE_FLAG_NONE,
                DML_FEATURE_LEVEL_1_0,
                IID_PPV_ARGS(this->dml_device_.GetAddressOf())
            );

            win_util::ThrowIfFailed(
                hr,
                "DMLCreateDevice1 failed"
            );

            Ort::ThrowOnError(
                dml_api_->SessionOptionsAppendExecutionProvider_DML1(
                    session_options_,
                    this->dml_device_.Get(),
                    this->d3d12_core_->command_queue()
                )
            );

            dml1_io_binding_enabled_ = true;
        }
        else {
            Ort::ThrowOnError(
                dml_api_->SessionOptionsAppendExecutionProvider_DML(
                    session_options_,
                    dml_device_id
                )
            );
        }
    }

    session_ = std::make_unique<Ort::Session>(
        env_,
        model_path,
        session_options_
    );

    collect_model_io_info();

    initialized_ = true;

    if (dml1_io_binding_enabled_) {
        this->set_output_slot_count(1);
    }
}

void OrtDmlYoloSegRunner::set_d3d12_core(
    D3D12Core& d3d12_core
)
{
    if (!d3d12_core.device()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::set_d3d12_core: D3D12Core device is null");
    }

    this->d3d12_core_ = &d3d12_core;
}

std::vector<OrtDmlYoloSegRunner::OutputTensor>
OrtDmlYoloSegRunner::run_cpu_input(
    const std::vector<float>& input_tensor,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width
)
{
    if (!initialized_ || !session_) {
        throw std::runtime_error("OrtDmlYoloSegRunner is not initialized");
    }

    if (input_infos_.empty()) {
        throw std::runtime_error("Model has no input");
    }

    if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0) {
        throw std::runtime_error("Invalid input shape");
    }

    const std::vector<int64_t> input_shape = {
        batch,
        channels,
        height,
        width
    };

    const size_t expected_elements =
        static_cast<size_t>(batch) *
        static_cast<size_t>(channels) *
        static_cast<size_t>(height) *
        static_cast<size_t>(width);

    if (input_tensor.size() != expected_elements) {
        std::stringstream ss;
        ss << "Input tensor size mismatch. expected="
            << expected_elements
            << " actual="
            << input_tensor.size();

        throw std::runtime_error(ss.str());
    }

    if (input_infos_[0].element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("First model input is not float32");
    }

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

    Ort::Value input_value =
        Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input_tensor.data()),
            input_tensor.size(),
            input_shape.data(),
            input_shape.size()
        );

    std::vector<const char*> input_name_ptrs;
    input_name_ptrs.reserve(input_names_.size());
    for (const auto& name : input_names_) {
        input_name_ptrs.push_back(name.c_str());
    }

    std::vector<const char*> output_name_ptrs;
    output_name_ptrs.reserve(output_names_.size());
    for (const auto& name : output_names_) {
        output_name_ptrs.push_back(name.c_str());
    }

    std::vector<Ort::Value> input_values;
    input_values.emplace_back(std::move(input_value));

    std::vector<Ort::Value> output_values =
        session_->Run(
            Ort::RunOptions{ nullptr },
            input_name_ptrs.data(),
            input_values.data(),
            input_values.size(),
            output_name_ptrs.data(),
            output_name_ptrs.size()
        );

    std::vector<OutputTensor> results;
    results.reserve(output_values.size());

    for (size_t i = 0; i < output_values.size(); ++i) {
        if (!output_values[i].IsTensor()) {
            throw std::runtime_error("Output is not a tensor");
        }

        auto shape_info =
            output_values[i].GetTensorTypeAndShapeInfo();

        const ONNXTensorElementDataType elem_type =
            shape_info.GetElementType();

        if (elem_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            std::stringstream ss;
            ss << "Output "
                << i
                << " is not float32. element_type="
                << static_cast<int>(elem_type);

            throw std::runtime_error(ss.str());
        }

        std::vector<int64_t> output_shape =
            shape_info.GetShape();

        const size_t output_elements =
            shape_info.GetElementCount();

        const float* output_data =
            output_values[i].GetTensorData<float>();

        OutputTensor out{};
        out.name = output_names_[i];
        out.shape = std::move(output_shape);
        out.data.assign(
            output_data,
            output_data + output_elements
        );

        results.emplace_back(std::move(out));
    }

    return results;
}

std::vector<OrtDmlYoloSegRunner::OutputTensor>
OrtDmlYoloSegRunner::run_cpu_input_and_upload_outputs(
    const std::vector<float>& input_tensor,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width
)
{
    std::vector<OutputTensor> outputs =
        this->run_cpu_input(
            input_tensor,
            batch,
            channels,
            height,
            width
        );

    this->upload_outputs_to_d3d12(outputs);

    return outputs;
}

std::vector<OrtDmlYoloSegRunner::OutputTensor>
OrtDmlYoloSegRunner::run_d3d12_input_and_upload_outputs(
    ID3D12Resource* input_buffer,
    D3D12_RESOURCE_STATES& input_buffer_state,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width
)
{
    if (!this->d3d12_core_) {
        throw std::runtime_error(
            "OrtDmlYoloSegRunner::run_d3d12_input_and_upload_outputs: D3D12Core is not set"
        );
    }

    if (!input_buffer) {
        throw std::runtime_error(
            "OrtDmlYoloSegRunner::run_d3d12_input_and_upload_outputs: input_buffer is null"
        );
    }

    if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0) {
        throw std::runtime_error(
            "OrtDmlYoloSegRunner::run_d3d12_input_and_upload_outputs: invalid input shape"
        );
    }

    const std::vector<int64_t> input_shape = {
        batch,
        channels,
        height,
        width
    };

    if (this->dml1_io_binding_enabled_) {
        return this->run_d3d12_input_and_upload_outputs_to_slot(
            this->active_output_slot_index_,
            input_buffer,
            input_buffer_state,
            batch,
            channels,
            height,
            width
        );
    }

    // Fallback path for legacy initialize(). This is intentionally slower.
    const size_t expected_elements =
        static_cast<size_t>(batch) *
        static_cast<size_t>(channels) *
        static_cast<size_t>(height) *
        static_cast<size_t>(width);

    std::vector<float> input_tensor =
        this->readback_d3d12_float_buffer(
            input_buffer,
            input_buffer_state,
            expected_elements,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );

    return this->run_cpu_input_and_upload_outputs(
        input_tensor,
        batch,
        channels,
        height,
        width
    );
}


std::vector<OrtDmlYoloSegRunner::OutputTensor>
OrtDmlYoloSegRunner::run_d3d12_input_and_upload_outputs_to_slot(
    size_t output_slot_index,
    ID3D12Resource* input_buffer,
    D3D12_RESOURCE_STATES& input_buffer_state,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width
)
{
    if (!this->dml1_io_binding_enabled_) {
        if (output_slot_index != 0) {
            throw std::runtime_error(
                "run_d3d12_input_and_upload_outputs_to_slot: legacy mode only supports output slot 0"
            );
        }
        return this->run_d3d12_input_and_upload_outputs(
            input_buffer,
            input_buffer_state,
            batch,
            channels,
            height,
            width
        );
    }

    if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0) {
        throw std::runtime_error(
            "run_d3d12_input_and_upload_outputs_to_slot: invalid input shape"
        );
    }

    this->set_output_slot_count(
        std::max(this->output_slot_count(), output_slot_index + 1)
    );

    const std::vector<int64_t> input_shape = {
        batch,
        channels,
        height,
        width
    };

    return this->run_d3d12_input_with_iobinding(
        output_slot_index,
        input_buffer,
        input_buffer_state,
        input_shape
    );
}

std::vector<OrtDmlYoloSegRunner::OutputTensor>
OrtDmlYoloSegRunner::run_d3d12_input_with_iobinding(
    size_t output_slot_index,
    ID3D12Resource* input_buffer,
    D3D12_RESOURCE_STATES& input_buffer_state,
    const std::vector<int64_t>& input_shape
)
{
    if (!initialized_ || !session_) {
        throw std::runtime_error("OrtDmlYoloSegRunner is not initialized");
    }

    if (!dml_api_) {
        throw std::runtime_error("OrtDmlYoloSegRunner: DML API is not available");
    }

    if (input_infos_.empty() || output_infos_.empty()) {
        throw std::runtime_error("OrtDmlYoloSegRunner: model IO info is empty");
    }

    if (input_infos_[0].element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("First model input is not float32");
    }

    const size_t input_elements =
        element_count_from_shape(input_shape);

    if (input_elements == 0) {
        throw std::runtime_error("run_d3d12_input_with_iobinding: invalid input element count");
    }

    const size_t input_byte_size =
        input_elements * sizeof(float);

    const D3D12_RESOURCE_DESC input_desc =
        input_buffer->GetDesc();

    if (input_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
        throw std::runtime_error("run_d3d12_input_with_iobinding: input resource is not a buffer");
    }

    if (input_desc.Width < input_byte_size) {
        std::stringstream ss;
        ss << "run_d3d12_input_with_iobinding: input buffer too small. required="
           << input_byte_size
           << " actual="
           << input_desc.Width;
        throw std::runtime_error(ss.str());
    }

    this->set_output_slot_count(
        std::max(this->output_slot_count(), output_slot_index + 1)
    );

    std::vector<GpuOutputBuffer>& slot_outputs =
        this->gpu_output_slots_.at(output_slot_index).outputs;

    // DirectML EP owns D3D12 work on the same command queue. Use COMMON as the
    // handoff state for external resources. After Run, keep state tracking as
    // COMMON so downstream postprocess can transition from COMMON to SRV.
    this->transition_resource_and_wait(
        input_buffer,
        input_buffer_state,
        D3D12_RESOURCE_STATE_COMMON
    );

    for (auto& out : slot_outputs) {
        this->transition_resource_and_wait(
            out.default_buffer.Get(),
            out.state,
            D3D12_RESOURCE_STATE_COMMON
        );
    }

    void* input_dml_allocation = nullptr;
    Ort::Value input_value =
        this->create_dml_tensor_from_d3d12_resource(
            input_buffer,
            input_byte_size,
            input_shape,
            &input_dml_allocation
        );

    std::vector<void*> output_dml_allocations(
        slot_outputs.size(),
        nullptr
    );

    std::vector<Ort::Value> output_values;
    output_values.reserve(slot_outputs.size());

    try {
        for (size_t i = 0; i < slot_outputs.size(); ++i) {
            output_values.emplace_back(
                this->create_dml_tensor_from_d3d12_resource(
                    slot_outputs[i].default_buffer.Get(),
                    slot_outputs[i].byte_size,
                    slot_outputs[i].shape,
                    &output_dml_allocations[i]
                )
            );
        }

        Ort::IoBinding binding(*session_);

        binding.BindInput(
            this->input_names_[0].c_str(),
            input_value
        );

        for (size_t i = 0; i < output_values.size(); ++i) {
            binding.BindOutput(
                this->output_names_[i].c_str(),
                output_values[i]
            );
        }

        binding.SynchronizeInputs();

        session_->Run(
            Ort::RunOptions{ nullptr },
            binding
        );

        binding.SynchronizeOutputs();
    }
    catch (...) {
        for (void* p : output_dml_allocations) {
            this->free_dml_allocation(p);
        }
        this->free_dml_allocation(input_dml_allocation);
        throw;
    }

    for (void* p : output_dml_allocations) {
        this->free_dml_allocation(p);
    }
    this->free_dml_allocation(input_dml_allocation);

    input_buffer_state = D3D12_RESOURCE_STATE_COMMON;
    for (auto& out : slot_outputs) {
        out.state = D3D12_RESOURCE_STATE_COMMON;
    }

    std::vector<OutputTensor> metadata;
    metadata.reserve(slot_outputs.size());

    for (size_t i = 0; i < slot_outputs.size(); ++i) {
        OutputTensor out{};
        out.name = this->output_names_[i];
        out.shape = slot_outputs[i].shape;
        // Intentionally empty. Do not read back GPU outputs here.
        metadata.emplace_back(std::move(out));
    }

    return metadata;
}

Ort::Value OrtDmlYoloSegRunner::create_dml_tensor_from_d3d12_resource(
    ID3D12Resource* resource,
    size_t byte_size,
    const std::vector<int64_t>& shape,
    void** out_dml_allocation
)
{
    if (!resource) {
        throw std::runtime_error("create_dml_tensor_from_d3d12_resource: resource is null");
    }

    if (!out_dml_allocation) {
        throw std::runtime_error("create_dml_tensor_from_d3d12_resource: out_dml_allocation is null");
    }

    if (!dml_api_) {
        throw std::runtime_error("create_dml_tensor_from_d3d12_resource: DML API is not available");
    }

    *out_dml_allocation = nullptr;

    Ort::ThrowOnError(
        dml_api_->CreateGPUAllocationFromD3DResource(
            resource,
            out_dml_allocation
        )
    );

    if (!*out_dml_allocation) {
        throw std::runtime_error("CreateGPUAllocationFromD3DResource returned null allocation");
    }

    Ort::MemoryInfo memory_info(
        "DML",
        OrtAllocatorType::OrtDeviceAllocator,
        0,
        OrtMemTypeDefault
    );

    OrtValue* raw_value = nullptr;

    Ort::ThrowOnError(
        Ort::GetApi().CreateTensorWithDataAsOrtValue(
            memory_info,
            *out_dml_allocation,
            byte_size,
            shape.data(),
            shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &raw_value
        )
    );

    if (!raw_value) {
        throw std::runtime_error("CreateTensorWithDataAsOrtValue returned null OrtValue");
    }

    return Ort::Value(raw_value);
}

void OrtDmlYoloSegRunner::free_dml_allocation(void* allocation) noexcept
{
    if (!allocation || !dml_api_) {
        return;
    }

    dml_api_->FreeGPUAllocation(allocation);
}

void OrtDmlYoloSegRunner::upload_outputs_to_d3d12(
    const std::vector<OutputTensor>& outputs
)
{
    if (!this->d3d12_core_) {
        throw std::runtime_error(
            "OrtDmlYoloSegRunner::upload_outputs_to_d3d12: D3D12Core is not set. Call set_d3d12_core(d3d12) first."
        );
    }

    if (!this->d3d12_core_->device()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::upload_outputs_to_d3d12: D3D12Core device is null");
    }

    if (outputs.empty()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::upload_outputs_to_d3d12: outputs is empty");
    }

    this->set_output_slot_count(
        std::max<size_t>(1, this->output_slot_count())
    );

    std::vector<GpuOutputBuffer>& gpu_outputs =
        this->gpu_output_slots_.at(this->active_output_slot_index_).outputs;

    if (gpu_outputs.size() < outputs.size()) {
        gpu_outputs.resize(outputs.size());
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        this->ensure_gpu_output_buffer(
            i,
            outputs[i].shape,
            outputs[i].data.size()
        );

        this->upload_float_vector_to_gpu_output(
            i,
            outputs[i].data
        );
    }
}

void OrtDmlYoloSegRunner::print_model_info() const
{
    std::cout << "Inputs:\n";

    for (const auto& input : input_infos_) {
        std::cout
            << "  name=" << input.name
            << " type=" << static_cast<int>(input.element_type)
            << " shape=" << shape_to_string(input.shape)
            << "\n";
    }

    std::cout << "Outputs:\n";

    for (const auto& output : output_infos_) {
        std::cout
            << "  name=" << output.name
            << " type=" << static_cast<int>(output.element_type)
            << " shape=" << shape_to_string(output.shape)
            << "\n";
    }
}

const std::vector<OrtDmlYoloSegRunner::TensorInfo>&
OrtDmlYoloSegRunner::input_infos() const
{
    return input_infos_;
}

const std::vector<OrtDmlYoloSegRunner::TensorInfo>&
OrtDmlYoloSegRunner::output_infos() const
{
    return output_infos_;
}

bool OrtDmlYoloSegRunner::is_dml1_io_binding_enabled() const
{
    return this->dml1_io_binding_enabled_;
}

void OrtDmlYoloSegRunner::set_output_slot_count(size_t slot_count)
{
    if (slot_count == 0) {
        slot_count = 1;
    }

    const size_t old_count = this->gpu_output_slots_.size();
    if (old_count == slot_count) {
        return;
    }

    this->gpu_output_slots_.resize(slot_count);

    if (this->active_output_slot_index_ >= slot_count) {
        this->active_output_slot_index_ = 0;
    }

    if (this->initialized_ && !this->output_infos_.empty()) {
        const size_t saved_active = this->active_output_slot_index_;
        for (size_t slot = old_count; slot < slot_count; ++slot) {
            this->active_output_slot_index_ = slot;
            this->ensure_gpu_output_buffers_from_model_info();
        }
        this->active_output_slot_index_ = saved_active;
    }
}

size_t OrtDmlYoloSegRunner::output_slot_count() const
{
    return this->gpu_output_slots_.size();
}

size_t OrtDmlYoloSegRunner::gpu_output_count(size_t output_slot_index) const
{
    if (output_slot_index >= this->gpu_output_slots_.size()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::gpu_output_count: invalid output slot index");
    }
    return this->gpu_output_slots_[output_slot_index].outputs.size();
}

ID3D12Resource* OrtDmlYoloSegRunner::output_buffer(
    size_t output_slot_index,
    size_t output_index
) const
{
    if (output_slot_index >= this->gpu_output_slots_.size()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::output_buffer: invalid output slot index");
    }

    const auto& outputs = this->gpu_output_slots_[output_slot_index].outputs;
    if (output_index >= outputs.size() || !outputs[output_index].default_buffer) {
        throw std::runtime_error("OrtDmlYoloSegRunner::output_buffer: output buffer is not available");
    }

    return outputs[output_index].default_buffer.Get();
}

D3D12_RESOURCE_STATES& OrtDmlYoloSegRunner::output_state_ref(
    size_t output_slot_index,
    size_t output_index
)
{
    if (output_slot_index >= this->gpu_output_slots_.size()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::output_state_ref: invalid output slot index");
    }

    auto& outputs = this->gpu_output_slots_[output_slot_index].outputs;
    if (output_index >= outputs.size() || !outputs[output_index].default_buffer) {
        throw std::runtime_error("OrtDmlYoloSegRunner::output_state_ref: output buffer is not available");
    }

    return outputs[output_index].state;
}

ID3D12Resource* OrtDmlYoloSegRunner::output0_buffer(size_t output_slot_index) const
{
    return this->output_buffer(output_slot_index, 0);
}

ID3D12Resource* OrtDmlYoloSegRunner::output1_buffer(size_t output_slot_index) const
{
    return this->output_buffer(output_slot_index, 1);
}

D3D12_RESOURCE_STATES& OrtDmlYoloSegRunner::output0_state_ref(size_t output_slot_index)
{
    return this->output_state_ref(output_slot_index, 0);
}

D3D12_RESOURCE_STATES& OrtDmlYoloSegRunner::output1_state_ref(size_t output_slot_index)
{
    return this->output_state_ref(output_slot_index, 1);
}

size_t OrtDmlYoloSegRunner::gpu_output_count() const
{
    return this->gpu_output_count(this->active_output_slot_index_);
}

ID3D12Resource* OrtDmlYoloSegRunner::output_buffer(size_t index) const
{
    return this->output_buffer(this->active_output_slot_index_, index);
}

D3D12_RESOURCE_STATES& OrtDmlYoloSegRunner::output_state_ref(size_t index)
{
    return this->output_state_ref(this->active_output_slot_index_, index);
}

ID3D12Resource* OrtDmlYoloSegRunner::output0_buffer() const
{
    return this->output0_buffer(this->active_output_slot_index_);
}

ID3D12Resource* OrtDmlYoloSegRunner::output1_buffer() const
{
    return this->output1_buffer(this->active_output_slot_index_);
}

D3D12_RESOURCE_STATES& OrtDmlYoloSegRunner::output0_state_ref()
{
    return this->output0_state_ref(this->active_output_slot_index_);
}

D3D12_RESOURCE_STATES& OrtDmlYoloSegRunner::output1_state_ref()
{
    return this->output1_state_ref(this->active_output_slot_index_);
}

const std::vector<int64_t>& OrtDmlYoloSegRunner::gpu_output_shape(size_t index) const
{
    if (this->active_output_slot_index_ >= this->gpu_output_slots_.size()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::gpu_output_shape: invalid active output slot");
    }
    const auto& outputs = this->gpu_output_slots_[this->active_output_slot_index_].outputs;
    if (index >= outputs.size()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::gpu_output_shape: invalid index");
    }

    return outputs[index].shape;
}

size_t OrtDmlYoloSegRunner::gpu_output_element_count(size_t index) const
{
    if (this->active_output_slot_index_ >= this->gpu_output_slots_.size()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::gpu_output_element_count: invalid active output slot");
    }
    const auto& outputs = this->gpu_output_slots_[this->active_output_slot_index_].outputs;
    if (index >= outputs.size()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::gpu_output_element_count: invalid index");
    }

    return outputs[index].element_count;
}

size_t OrtDmlYoloSegRunner::gpu_output_byte_size(size_t index) const
{
    if (this->active_output_slot_index_ >= this->gpu_output_slots_.size()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::gpu_output_byte_size: invalid active output slot");
    }
    const auto& outputs = this->gpu_output_slots_[this->active_output_slot_index_].outputs;
    if (index >= outputs.size()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::gpu_output_byte_size: invalid index");
    }

    return outputs[index].byte_size;
}

void OrtDmlYoloSegRunner::collect_model_io_info()
{
    input_infos_.clear();
    output_infos_.clear();
    input_names_.clear();
    output_names_.clear();

    Ort::AllocatorWithDefaultOptions allocator;

    const size_t input_count =
        session_->GetInputCount();

    const size_t output_count =
        session_->GetOutputCount();

    for (size_t i = 0; i < input_count; ++i) {
        Ort::AllocatedStringPtr name_ptr =
            session_->GetInputNameAllocated(
                i,
                allocator
            );

        std::string name =
            name_ptr.get();

        Ort::TypeInfo type_info =
            session_->GetInputTypeInfo(i);

        Ort::ConstTensorTypeAndShapeInfo tensor_info =
            type_info.GetTensorTypeAndShapeInfo();

        TensorInfo info{};
        info.name = name;
        info.element_type = tensor_info.GetElementType();
        info.shape = tensor_info.GetShape();

        input_names_.push_back(name);
        input_infos_.push_back(std::move(info));
    }

    for (size_t i = 0; i < output_count; ++i) {
        Ort::AllocatedStringPtr name_ptr =
            session_->GetOutputNameAllocated(
                i,
                allocator
            );

        std::string name =
            name_ptr.get();

        Ort::TypeInfo type_info =
            session_->GetOutputTypeInfo(i);

        Ort::ConstTensorTypeAndShapeInfo tensor_info =
            type_info.GetTensorTypeAndShapeInfo();

        TensorInfo info{};
        info.name = name;
        info.element_type = tensor_info.GetElementType();
        info.shape = tensor_info.GetShape();

        output_names_.push_back(name);
        output_infos_.push_back(std::move(info));
    }
}

std::string OrtDmlYoloSegRunner::shape_to_string(
    const std::vector<int64_t>& shape
)
{
    std::stringstream ss;
    ss << "[";

    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            ss << ", ";
        }

        ss << shape[i];
    }

    ss << "]";
    return ss.str();
}

size_t OrtDmlYoloSegRunner::element_count_from_shape(
    const std::vector<int64_t>& shape
)
{
    if (shape.empty()) {
        return 0;
    }

    size_t count = 1;

    for (int64_t dim : shape) {
        if (dim <= 0) {
            return 0;
        }

        count *= static_cast<size_t>(dim);
    }

    return count;
}

std::vector<float> OrtDmlYoloSegRunner::readback_d3d12_float_buffer(
    ID3D12Resource* buffer,
    D3D12_RESOURCE_STATES& buffer_state,
    size_t element_count,
    D3D12_RESOURCE_STATES restore_state
)
{
    if (!this->d3d12_core_) {
        throw std::runtime_error("readback_d3d12_float_buffer: D3D12Core is not set");
    }

    if (!buffer) {
        throw std::runtime_error("readback_d3d12_float_buffer: buffer is null");
    }

    if (element_count == 0) {
        throw std::runtime_error("readback_d3d12_float_buffer: element_count is zero");
    }

    const UINT64 byte_size =
        static_cast<UINT64>(element_count) * sizeof(float);

    D3D12_RESOURCE_DESC src_desc = buffer->GetDesc();

    if (src_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
        throw std::runtime_error("readback_d3d12_float_buffer: input resource is not a buffer");
    }

    if (src_desc.Width < byte_size) {
        std::stringstream ss;
        ss << "readback_d3d12_float_buffer: input buffer too small. required="
           << byte_size
           << " actual="
           << src_desc.Width;
        throw std::runtime_error(ss.str());
    }

    auto readback_heap =
        make_heap_props(D3D12_HEAP_TYPE_READBACK);

    Microsoft::WRL::ComPtr<ID3D12Resource> readback_buffer =
        this->d3d12_core_->create_committed_resource(
            readback_heap,
            D3D12_HEAP_FLAG_NONE,
            make_buffer_desc(byte_size),
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr
        );

    this->d3d12_core_->reset_command_list();

    ID3D12GraphicsCommandList* command_list =
        this->d3d12_core_->command_list();

    this->transition_resource(
        command_list,
        buffer,
        buffer_state,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );

    command_list->CopyBufferRegion(
        readback_buffer.Get(),
        0,
        buffer,
        0,
        byte_size
    );

    this->transition_resource(
        command_list,
        buffer,
        buffer_state,
        restore_state
    );

    this->d3d12_core_->close_execute_and_wait();

    D3D12_RANGE read_range{};
    read_range.Begin = 0;
    read_range.End = static_cast<SIZE_T>(byte_size);

    void* mapped = nullptr;

    HRESULT hr = readback_buffer->Map(
        0,
        &read_range,
        &mapped
    );

    win_util::ThrowIfFailed(
        hr,
        "OrtDmlYoloSegRunner: Map D3D12 input readback buffer failed"
    );

    std::vector<float> data(element_count);

    std::memcpy(
        data.data(),
        mapped,
        static_cast<size_t>(byte_size)
    );

    D3D12_RANGE written_range{};
    written_range.Begin = 0;
    written_range.End = 0;

    readback_buffer->Unmap(
        0,
        &written_range
    );

    return data;
}

void OrtDmlYoloSegRunner::ensure_gpu_output_buffer(
    size_t index,
    const std::vector<int64_t>& shape,
    size_t element_count
)
{
    if (!this->d3d12_core_) {
        throw std::runtime_error("ensure_gpu_output_buffer: D3D12Core is not set");
    }

    if (element_count == 0) {
        throw std::runtime_error("ensure_gpu_output_buffer: element_count is zero");
    }

    this->set_output_slot_count(
        std::max<size_t>(1, this->output_slot_count())
    );

    std::vector<GpuOutputBuffer>& outputs =
        this->gpu_output_slots_.at(this->active_output_slot_index_).outputs;

    if (index >= outputs.size()) {
        outputs.resize(index + 1);
    }

    GpuOutputBuffer& gpu_output =
        outputs[index];

    const size_t byte_size =
        element_count * sizeof(float);

    const bool need_recreate =
        !gpu_output.default_buffer ||
        gpu_output.byte_size < byte_size;

    if (!need_recreate) {
        gpu_output.shape = shape;
        gpu_output.element_count = element_count;
        gpu_output.byte_size = byte_size;
        return;
    }

    auto default_heap =
        make_heap_props(D3D12_HEAP_TYPE_DEFAULT);

    auto upload_heap =
        make_heap_props(D3D12_HEAP_TYPE_UPLOAD);

    gpu_output.default_buffer =
        this->d3d12_core_->create_committed_resource(
            default_heap,
            D3D12_HEAP_FLAG_NONE,
            make_buffer_desc(
                static_cast<UINT64>(byte_size),
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            ),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr
        );

    gpu_output.upload_buffer =
        this->d3d12_core_->create_committed_resource(
            upload_heap,
            D3D12_HEAP_FLAG_NONE,
            make_buffer_desc(static_cast<UINT64>(byte_size)),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr
        );

    gpu_output.state = D3D12_RESOURCE_STATE_COMMON;
    gpu_output.shape = shape;
    gpu_output.element_count = element_count;
    gpu_output.byte_size = byte_size;
}

void OrtDmlYoloSegRunner::ensure_gpu_output_buffers_from_model_info()
{
    if (this->output_infos_.empty()) {
        throw std::runtime_error("ensure_gpu_output_buffers_from_model_info: model has no outputs");
    }

    this->set_output_slot_count(
        std::max<size_t>(1, this->output_slot_count())
    );

    std::vector<GpuOutputBuffer>& gpu_outputs =
        this->gpu_output_slots_.at(this->active_output_slot_index_).outputs;

    if (gpu_outputs.size() < this->output_infos_.size()) {
        gpu_outputs.resize(this->output_infos_.size());
    }

    for (size_t i = 0; i < this->output_infos_.size(); ++i) {
        const TensorInfo& info =
            this->output_infos_[i];

        if (info.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            std::stringstream ss;
            ss << "ensure_gpu_output_buffers_from_model_info: output "
               << i
               << " is not float32";
            throw std::runtime_error(ss.str());
        }

        const size_t elements =
            element_count_from_shape(info.shape);

        if (elements == 0) {
            std::stringstream ss;
            ss << "ensure_gpu_output_buffers_from_model_info: output "
               << i
               << " has dynamic or invalid shape "
               << shape_to_string(info.shape);
            throw std::runtime_error(ss.str());
        }

        this->ensure_gpu_output_buffer(
            i,
            info.shape,
            elements
        );
    }
}

void OrtDmlYoloSegRunner::upload_float_vector_to_gpu_output(
    size_t index,
    const std::vector<float>& data
)
{
    if (!this->d3d12_core_) {
        throw std::runtime_error("upload_float_vector_to_gpu_output: D3D12Core is not set");
    }

    if (this->active_output_slot_index_ >= this->gpu_output_slots_.size()) {
        throw std::runtime_error("upload_float_vector_to_gpu_output: invalid active output slot");
    }

    std::vector<GpuOutputBuffer>& outputs =
        this->gpu_output_slots_[this->active_output_slot_index_].outputs;

    if (index >= outputs.size()) {
        throw std::runtime_error("upload_float_vector_to_gpu_output: invalid output index");
    }

    GpuOutputBuffer& gpu_output =
        outputs[index];

    if (!gpu_output.default_buffer || !gpu_output.upload_buffer) {
        throw std::runtime_error("upload_float_vector_to_gpu_output: GPU buffers are not created");
    }

    const size_t byte_size =
        data.size() * sizeof(float);

    if (byte_size > gpu_output.byte_size) {
        throw std::runtime_error("upload_float_vector_to_gpu_output: data is larger than GPU output buffer");
    }

    void* mapped = nullptr;

    D3D12_RANGE read_range{};
    read_range.Begin = 0;
    read_range.End = 0;

    HRESULT hr = gpu_output.upload_buffer->Map(
        0,
        &read_range,
        &mapped
    );

    win_util::ThrowIfFailed(
        hr,
        "OrtDmlYoloSegRunner: Map output upload buffer failed"
    );

    std::memcpy(
        mapped,
        data.data(),
        byte_size
    );

    D3D12_RANGE written_range{};
    written_range.Begin = 0;
    written_range.End = static_cast<SIZE_T>(byte_size);

    gpu_output.upload_buffer->Unmap(
        0,
        &written_range
    );

    this->d3d12_core_->reset_command_list();

    ID3D12GraphicsCommandList* command_list =
        this->d3d12_core_->command_list();

    this->transition_resource(
        command_list,
        gpu_output.default_buffer.Get(),
        gpu_output.state,
        D3D12_RESOURCE_STATE_COPY_DEST
    );

    command_list->CopyBufferRegion(
        gpu_output.default_buffer.Get(),
        0,
        gpu_output.upload_buffer.Get(),
        0,
        byte_size
    );

    this->transition_resource(
        command_list,
        gpu_output.default_buffer.Get(),
        gpu_output.state,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    this->d3d12_core_->close_execute_and_wait();
}

void OrtDmlYoloSegRunner::transition_resource(
    ID3D12GraphicsCommandList* command_list,
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES& current_state,
    D3D12_RESOURCE_STATES new_state
)
{
    if (!command_list) {
        throw std::runtime_error("OrtDmlYoloSegRunner::transition_resource: command_list is null");
    }

    if (!resource) {
        throw std::runtime_error("OrtDmlYoloSegRunner::transition_resource: resource is null");
    }

    if (current_state == new_state) {
        return;
    }

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = current_state;
    barrier.Transition.StateAfter = new_state;

    command_list->ResourceBarrier(
        1,
        &barrier
    );

    current_state = new_state;
}

void OrtDmlYoloSegRunner::transition_resource_and_wait(
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES& current_state,
    D3D12_RESOURCE_STATES new_state
)
{
    if (!this->d3d12_core_) {
        throw std::runtime_error("transition_resource_and_wait: D3D12Core is not set");
    }

    if (current_state == new_state) {
        return;
    }

    this->d3d12_core_->reset_command_list();

    this->transition_resource(
        this->d3d12_core_->command_list(),
        resource,
        current_state,
        new_state
    );

    this->d3d12_core_->close_execute_and_wait();
}

D3D12_RESOURCE_DESC OrtDmlYoloSegRunner::make_buffer_desc(
    UINT64 byte_size,
    D3D12_RESOURCE_FLAGS flags
)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = byte_size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

D3D12_HEAP_PROPERTIES OrtDmlYoloSegRunner::make_heap_props(
    D3D12_HEAP_TYPE type
)
{
    D3D12_HEAP_PROPERTIES props{};
    props.Type = type;
    props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask = 1;
    props.VisibleNodeMask = 1;
    return props;
}
