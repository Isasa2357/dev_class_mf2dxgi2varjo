#include "OrtDmlYoloSegRunner.hpp"

#include <stdexcept>
#include <numeric>
#include <sstream>
#include <algorithm>
#include <cstring>

#include "HResultUtil.hpp"

OrtDmlYoloSegRunner::OrtDmlYoloSegRunner()
    : env_(
        ORT_LOGGING_LEVEL_WARNING,
        "OrtDmlYoloSegRunner"
    )
{
}

void OrtDmlYoloSegRunner::initialize(
    const wchar_t* model_path,
    bool use_directml,
    int dml_device_id
)
{
    if (!model_path) {
        throw std::runtime_error("OrtDmlYoloSegRunner::initialize: model_path is null");
    }

    session_options_ = Ort::SessionOptions{};

    // 最初はデバッグしやすい設定。
    // DirectML EP 使用時は、必要に応じて ORT_SEQUENTIAL の方が安全な場合があります。
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED
    );

    session_options_.SetIntraOpNumThreads(1);

    if (use_directml) {
        // Microsoft.ML.OnnxRuntime.DirectML NuGet 等を使う場合、
        // dml_provider_factory.h と onnxruntime_providers_dml.lib が必要です。
        const OrtApi& ort_api = Ort::GetApi();

        const OrtDmlApi* dml_api = nullptr;

        Ort::ThrowOnError(
            ort_api.GetExecutionProviderApi(
                "DML",
                ORT_API_VERSION,
                reinterpret_cast<const void**>(&dml_api)
            )
        );

        if (!dml_api) {
            throw std::runtime_error("Failed to get OrtDmlApi");
        }

        Ort::ThrowOnError(
            dml_api->SessionOptionsAppendExecutionProvider_DML(
                session_options_,
                dml_device_id
            )
        );
    }

    session_ = std::make_unique<Ort::Session>(
        env_,
        model_path,
        session_options_
    );

    collect_model_io_info();

    initialized_ = true;
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

    // CreateTensor は input_tensor のメモリを参照するため、
    // session.Run が終わるまで input_tensor は生存している必要があります。
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

    if (gpu_outputs_.size() < outputs.size()) {
        gpu_outputs_.resize(outputs.size());
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

size_t OrtDmlYoloSegRunner::gpu_output_count() const
{
    return this->gpu_outputs_.size();
}

ID3D12Resource* OrtDmlYoloSegRunner::output_buffer(size_t index) const
{
    if (index >= this->gpu_outputs_.size() || !this->gpu_outputs_[index].default_buffer) {
        throw std::runtime_error("OrtDmlYoloSegRunner::output_buffer: output buffer is not available");
    }

    return this->gpu_outputs_[index].default_buffer.Get();
}

D3D12_RESOURCE_STATES& OrtDmlYoloSegRunner::output_state_ref(size_t index)
{
    if (index >= this->gpu_outputs_.size() || !this->gpu_outputs_[index].default_buffer) {
        throw std::runtime_error("OrtDmlYoloSegRunner::output_state_ref: output buffer is not available");
    }

    return this->gpu_outputs_[index].state;
}

ID3D12Resource* OrtDmlYoloSegRunner::output0_buffer() const
{
    return this->output_buffer(0);
}

ID3D12Resource* OrtDmlYoloSegRunner::output1_buffer() const
{
    return this->output_buffer(1);
}

D3D12_RESOURCE_STATES& OrtDmlYoloSegRunner::output0_state_ref()
{
    return this->output_state_ref(0);
}

D3D12_RESOURCE_STATES& OrtDmlYoloSegRunner::output1_state_ref()
{
    return this->output_state_ref(1);
}

const std::vector<int64_t>& OrtDmlYoloSegRunner::gpu_output_shape(size_t index) const
{
    if (index >= this->gpu_outputs_.size()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::gpu_output_shape: invalid index");
    }

    return this->gpu_outputs_[index].shape;
}

size_t OrtDmlYoloSegRunner::gpu_output_element_count(size_t index) const
{
    if (index >= this->gpu_outputs_.size()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::gpu_output_element_count: invalid index");
    }

    return this->gpu_outputs_[index].element_count;
}

size_t OrtDmlYoloSegRunner::gpu_output_byte_size(size_t index) const
{
    if (index >= this->gpu_outputs_.size()) {
        throw std::runtime_error("OrtDmlYoloSegRunner::gpu_output_byte_size: invalid index");
    }

    return this->gpu_outputs_[index].byte_size;
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

    if (index >= this->gpu_outputs_.size()) {
        this->gpu_outputs_.resize(index + 1);
    }

    GpuOutputBuffer& gpu_output =
        this->gpu_outputs_[index];

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
            make_buffer_desc(static_cast<UINT64>(byte_size)),
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

void OrtDmlYoloSegRunner::upload_float_vector_to_gpu_output(
    size_t index,
    const std::vector<float>& data
)
{
    if (!this->d3d12_core_) {
        throw std::runtime_error("upload_float_vector_to_gpu_output: D3D12Core is not set");
    }

    if (index >= this->gpu_outputs_.size()) {
        throw std::runtime_error("upload_float_vector_to_gpu_output: invalid output index");
    }

    GpuOutputBuffer& gpu_output =
        this->gpu_outputs_[index];

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
