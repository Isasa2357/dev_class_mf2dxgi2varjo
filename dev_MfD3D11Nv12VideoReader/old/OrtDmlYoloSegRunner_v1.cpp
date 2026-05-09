#include "OrtDmlYoloSegRunner.hpp"

#include <stdexcept>
#include <numeric>
#include <sstream>
#include <algorithm>

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
            Ort::RunOptions{nullptr},
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