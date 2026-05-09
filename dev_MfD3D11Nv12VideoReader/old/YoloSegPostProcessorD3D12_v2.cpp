#include "YoloSegPostProcessorD3D12.hpp"

#include <stdexcept>
#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>

#include <d3dcompiler.h>

#include "HResultUtil.hpp"

#pragma comment(lib, "d3dcompiler.lib")

namespace {

    constexpr UINT SRV_OUTPUT0 = 0;
    constexpr UINT SRV_OUTPUT1 = 1;
    constexpr UINT SRV_CANDIDATES = 2;
    constexpr UINT SRV_CANDIDATE_COUNTER = 3;
    constexpr UINT SRV_KEEP_FLAGS = 4;
    constexpr UINT SRV_SELECTED = 5;
    constexpr UINT SRV_SELECTED_COUNTER = 6;

    constexpr UINT UAV_CANDIDATES = 7;
    constexpr UINT UAV_CANDIDATE_COUNTER = 8;
    constexpr UINT UAV_KEEP_FLAGS = 9;
    constexpr UINT UAV_SELECTED = 10;
    constexpr UINT UAV_SELECTED_COUNTER = 11;
    constexpr UINT UAV_SELECTED_MASKS = 12;

    constexpr UINT DESCRIPTOR_COUNT = 13;

    static D3D12_RESOURCE_DESC MakeBufferDesc(
        UINT64 byte_size,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE
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

    static D3D12_HEAP_PROPERTIES MakeHeapProps(D3D12_HEAP_TYPE type)
    {
        D3D12_HEAP_PROPERTIES props{};
        props.Type = type;
        props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        props.CreationNodeMask = 1;
        props.VisibleNodeMask = 1;
        return props;
    }

} // namespace

YoloSegPostProcessorD3D12::YoloSegPostProcessorD3D12(
    D3D12Core& d3d12_core,
    const Config& config
)
    : d3d12_core_(d3d12_core)
    , config_(config)
{
    if (!this->d3d12_core_.device()) {
        throw std::runtime_error("YoloSegPostProcessorD3D12: D3D12Core device is null");
    }

    if (config_.num_attrs < 5 ||
        config_.num_candidates == 0 ||
        config_.num_classes == 0 ||
        config_.num_mask_coeffs == 0 ||
        config_.mask_width == 0 ||
        config_.mask_height == 0 ||
        config_.max_candidates == 0 ||
        config_.max_detections == 0)
    {
        throw std::runtime_error("YoloSegPostProcessorD3D12: invalid config");
    }

    if (config_.num_attrs != 4 + config_.num_classes + config_.num_mask_coeffs) {
        throw std::runtime_error("YoloSegPostProcessorD3D12: num_attrs != 4 + num_classes + num_mask_coeffs");
    }

    this->create_root_signature();
    this->create_pipeline_states();
    this->create_descriptor_heap();
    this->create_buffers();
    this->create_descriptors();
    this->create_constant_buffer();
    this->update_params();
}

YoloSegPostProcessorD3D12::~YoloSegPostProcessorD3D12()
{
    if (this->constant_buffer_ && this->mapped_constants_) {
        D3D12_RANGE written_range{};
        written_range.Begin = 0;
        written_range.End = 0;

        this->constant_buffer_->Unmap(
            0,
            &written_range
        );

        this->mapped_constants_ = nullptr;
    }
}

void YoloSegPostProcessorD3D12::upload_outputs_from_cpu(
    const std::vector<float>& output0,
    const std::vector<float>& output1
)
{
    const size_t expected_output0 =
        static_cast<size_t>(config_.num_attrs) *
        static_cast<size_t>(config_.num_candidates);

    const size_t expected_output1 =
        static_cast<size_t>(config_.num_mask_coeffs) *
        static_cast<size_t>(config_.mask_width) *
        static_cast<size_t>(config_.mask_height);

    if (output0.size() != expected_output0) {
        std::stringstream ss;
        ss << "output0 size mismatch. expected="
            << expected_output0
            << " actual="
            << output0.size();
        throw std::runtime_error(ss.str());
    }

    if (output1.size() != expected_output1) {
        std::stringstream ss;
        ss << "output1 size mismatch. expected="
            << expected_output1
            << " actual="
            << output1.size();
        throw std::runtime_error(ss.str());
    }

    this->upload_buffer_from_cpu(
        this->output0_buffer_.Get(),
        this->output0_state_,
        this->output0_upload_buffer_.Get(),
        output0.data(),
        static_cast<UINT64>(output0.size()) * sizeof(float)
    );

    this->upload_buffer_from_cpu(
        this->output1_buffer_.Get(),
        this->output1_state_,
        this->output1_upload_buffer_.Get(),
        output1.data(),
        static_cast<UINT64>(output1.size()) * sizeof(float)
    );
}

void YoloSegPostProcessorD3D12::process_uploaded_outputs_and_wait()
{
    this->d3d12_core_.reset_command_list();

    this->record_process(
        this->d3d12_core_.command_list()
    );

    this->d3d12_core_.close_execute_and_wait();
}

std::vector<YoloSegPostProcessorD3D12::DetectionWithMask>
YoloSegPostProcessorD3D12::readback_results()
{
    const UINT64 detection_bytes =
        static_cast<UINT64>(config_.max_detections) *
        sizeof(Detection);

    const UINT64 mask_pixels =
        static_cast<UINT64>(config_.mask_width) *
        static_cast<UINT64>(config_.mask_height);

    const UINT64 mask_bytes =
        static_cast<UINT64>(config_.max_detections) *
        mask_pixels *
        sizeof(float);

    auto readback_heap = MakeHeapProps(D3D12_HEAP_TYPE_READBACK);

    auto counter_desc = MakeBufferDesc(sizeof(uint32_t));
    auto detection_desc = MakeBufferDesc(detection_bytes);
    auto mask_desc = MakeBufferDesc(mask_bytes);

    auto counter_readback =
        this->d3d12_core_.create_committed_resource(
            readback_heap,
            D3D12_HEAP_FLAG_NONE,
            counter_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr
        );

    auto detection_readback =
        this->d3d12_core_.create_committed_resource(
            readback_heap,
            D3D12_HEAP_FLAG_NONE,
            detection_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr
        );

    auto mask_readback =
        this->d3d12_core_.create_committed_resource(
            readback_heap,
            D3D12_HEAP_FLAG_NONE,
            mask_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr
        );

    this->d3d12_core_.reset_command_list();

    ID3D12GraphicsCommandList* command_list =
        this->d3d12_core_.command_list();

    this->transition_resource(
        command_list,
        this->selected_counter_buffer_.Get(),
        this->selected_counter_state_,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );

    this->transition_resource(
        command_list,
        this->selected_detection_buffer_.Get(),
        this->selected_detection_state_,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );

    this->transition_resource(
        command_list,
        this->selected_mask_buffer_.Get(),
        this->selected_mask_state_,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );

    command_list->CopyBufferRegion(
        counter_readback.Get(),
        0,
        this->selected_counter_buffer_.Get(),
        0,
        sizeof(uint32_t)
    );

    command_list->CopyBufferRegion(
        detection_readback.Get(),
        0,
        this->selected_detection_buffer_.Get(),
        0,
        detection_bytes
    );

    command_list->CopyBufferRegion(
        mask_readback.Get(),
        0,
        this->selected_mask_buffer_.Get(),
        0,
        mask_bytes
    );

    this->transition_resource(
        command_list,
        this->selected_counter_buffer_.Get(),
        this->selected_counter_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    this->transition_resource(
        command_list,
        this->selected_detection_buffer_.Get(),
        this->selected_detection_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    this->transition_resource(
        command_list,
        this->selected_mask_buffer_.Get(),
        this->selected_mask_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    this->d3d12_core_.close_execute_and_wait();

    uint32_t* mapped_counter = nullptr;

    D3D12_RANGE counter_range{};
    counter_range.Begin = 0;
    counter_range.End = sizeof(uint32_t);

    HRESULT hr = counter_readback->Map(
        0,
        &counter_range,
        reinterpret_cast<void**>(&mapped_counter)
    );

    win_util::ThrowIfFailed(
        hr,
        "Map selected counter readback failed"
    );

    uint32_t count = *mapped_counter;

    counter_readback->Unmap(
        0,
        nullptr
    );

    count = std::min(count, config_.max_detections);

    std::vector<DetectionWithMask> results;
    results.resize(count);

    if (count == 0) {
        return results;
    }

    Detection* mapped_detections = nullptr;

    D3D12_RANGE detection_range{};
    detection_range.Begin = 0;
    detection_range.End =
        static_cast<SIZE_T>(
            static_cast<UINT64>(count) * sizeof(Detection)
            );

    hr = detection_readback->Map(
        0,
        &detection_range,
        reinterpret_cast<void**>(&mapped_detections)
    );

    win_util::ThrowIfFailed(
        hr,
        "Map selected detections readback failed"
    );

    float* mapped_masks = nullptr;

    D3D12_RANGE mask_range{};
    mask_range.Begin = 0;
    mask_range.End =
        static_cast<SIZE_T>(
            static_cast<UINT64>(count) *
            mask_pixels *
            sizeof(float)
            );

    hr = mask_readback->Map(
        0,
        &mask_range,
        reinterpret_cast<void**>(&mapped_masks)
    );

    win_util::ThrowIfFailed(
        hr,
        "Map selected masks readback failed"
    );

    for (uint32_t i = 0; i < count; ++i) {
        results[i].detection = mapped_detections[i];
        results[i].mask_width = config_.mask_width;
        results[i].mask_height = config_.mask_height;

        results[i].mask.resize(
            static_cast<size_t>(mask_pixels)
        );

        std::memcpy(
            results[i].mask.data(),
            mapped_masks + static_cast<size_t>(i) * static_cast<size_t>(mask_pixels),
            static_cast<size_t>(mask_pixels) * sizeof(float)
        );
    }

    detection_readback->Unmap(
        0,
        nullptr
    );

    mask_readback->Unmap(
        0,
        nullptr
    );

    return results;
}

ID3D12Resource* YoloSegPostProcessorD3D12::output0_buffer() const
{
    return this->output0_buffer_.Get();
}

ID3D12Resource* YoloSegPostProcessorD3D12::output1_buffer() const
{
    return this->output1_buffer_.Get();
}

ID3D12Resource* YoloSegPostProcessorD3D12::selected_detection_buffer() const
{
    return this->selected_detection_buffer_.Get();
}

ID3D12Resource* YoloSegPostProcessorD3D12::selected_counter_buffer() const
{
    return this->selected_counter_buffer_.Get();
}

ID3D12Resource* YoloSegPostProcessorD3D12::selected_mask_buffer() const
{
    return this->selected_mask_buffer_.Get();
}

const YoloSegPostProcessorD3D12::Config&
YoloSegPostProcessorD3D12::config() const
{
    return this->config_;
}

void YoloSegPostProcessorD3D12::create_root_signature()
{
    D3D12_DESCRIPTOR_RANGE ranges[2]{};

    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 7;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].RegisterSpace = 0;
    ranges[0].OffsetInDescriptorsFromTableStart =
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 6;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].RegisterSpace = 0;
    ranges[1].OffsetInDescriptorsFromTableStart =
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER params[3]{};

    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &ranges[1];
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[2].Descriptor.ShaderRegister = 0;
    params[2].Descriptor.RegisterSpace = 0;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = ARRAYSIZE(params);
    desc.pParameters = params;
    desc.NumStaticSamplers = 0;
    desc.pStaticSamplers = nullptr;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    Microsoft::WRL::ComPtr<ID3DBlob> blob;
    Microsoft::WRL::ComPtr<ID3DBlob> error_blob;

    HRESULT hr = D3D12SerializeRootSignature(
        &desc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        blob.GetAddressOf(),
        error_blob.GetAddressOf()
    );

    if (FAILED(hr)) {
        if (error_blob) {
            std::string msg(
                static_cast<const char*>(error_blob->GetBufferPointer()),
                error_blob->GetBufferSize()
            );

            throw std::runtime_error(
                "D3D12SerializeRootSignature failed: " + msg
            );
        }

        win_util::ThrowIfFailed(
            hr,
            "D3D12SerializeRootSignature failed"
        );
    }

    hr = this->d3d12_core_.device()->CreateRootSignature(
        0,
        blob->GetBufferPointer(),
        blob->GetBufferSize(),
        IID_PPV_ARGS(this->root_signature_.GetAddressOf())
    );

    win_util::ThrowIfFailed(
        hr,
        "CreateRootSignature failed"
    );
}

void YoloSegPostProcessorD3D12::create_pipeline_states()
{
    static const char* hlsl = R"(

struct Detection
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;

    uint classId;
    uint candidateIndex;
    uint reserved;
};

StructuredBuffer<float> g_output0 : register(t0);
StructuredBuffer<float> g_output1 : register(t1);
StructuredBuffer<Detection> g_candidates_in : register(t2);
StructuredBuffer<uint> g_candidate_counter_in : register(t3);
StructuredBuffer<uint> g_keep_flags_in : register(t4);
StructuredBuffer<Detection> g_selected_in : register(t5);
StructuredBuffer<uint> g_selected_counter_in : register(t6);

RWStructuredBuffer<Detection> g_candidates_out : register(u0);
RWStructuredBuffer<uint> g_candidate_counter_out : register(u1);
RWStructuredBuffer<uint> g_keep_flags_out : register(u2);
RWStructuredBuffer<Detection> g_selected_out : register(u3);
RWStructuredBuffer<uint> g_selected_counter_out : register(u4);
RWStructuredBuffer<float> g_selected_masks_out : register(u5);

cbuffer Params : register(b0)
{
    uint numAttrs;
    uint numCandidates;
    uint numClasses;
    uint numMaskCoeffs;

    uint maskWidth;
    uint maskHeight;
    uint maskPixels;
    uint maxCandidates;

    uint maxDetections;
    float confThreshold;
    float iouThreshold;
    float inputWidth;

    float inputHeight;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

float ReadOutput0(uint attr, uint candidate)
{
    return g_output0[attr * numCandidates + candidate];
}

float ReadProto(uint coeffIndex, uint pixelIndex)
{
    return g_output1[coeffIndex * maskPixels + pixelIndex];
}

float Sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float IoU(Detection a, Detection b)
{
    float ix1 = max(a.x1, b.x1);
    float iy1 = max(a.y1, b.y1);
    float ix2 = min(a.x2, b.x2);
    float iy2 = min(a.y2, b.y2);

    float iw = max(0.0f, ix2 - ix1);
    float ih = max(0.0f, iy2 - iy1);

    float interArea = iw * ih;

    float areaA =
        max(0.0f, a.x2 - a.x1) *
        max(0.0f, a.y2 - a.y1);

    float areaB =
        max(0.0f, b.x2 - b.x1) *
        max(0.0f, b.y2 - b.y1);

    float denom = areaA + areaB - interArea + 1e-6f;

    return interArea / denom;
}

[numthreads(256, 1, 1)]
void DecodeCS(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;

    if (i >= numCandidates)
    {
        return;
    }

    float cx = ReadOutput0(0, i);
    float cy = ReadOutput0(1, i);
    float w  = ReadOutput0(2, i);
    float h  = ReadOutput0(3, i);

    float bestScore = -1.0f;
    uint bestClass = 0;

    [loop]
    for (uint c = 0; c < numClasses; ++c)
    {
        float score = ReadOutput0(4 + c, i);

        if (score > bestScore)
        {
            bestScore = score;
            bestClass = c;
        }
    }

    if (bestScore < confThreshold)
    {
        return;
    }

    uint outIndex = 0;
    InterlockedAdd(g_candidate_counter_out[0], 1, outIndex);

    if (outIndex >= maxCandidates)
    {
        return;
    }

    Detection det;
    det.x1 = clamp(cx - w * 0.5f, 0.0f, inputWidth);
    det.y1 = clamp(cy - h * 0.5f, 0.0f, inputHeight);
    det.x2 = clamp(cx + w * 0.5f, 0.0f, inputWidth);
    det.y2 = clamp(cy + h * 0.5f, 0.0f, inputHeight);
    det.score = bestScore;
    det.classId = bestClass;
    det.candidateIndex = i;
    det.reserved = 0;

    g_candidates_out[outIndex] = det;
}

[numthreads(256, 1, 1)]
void NmsCS(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;

    uint candidateCount =
        min(g_candidate_counter_in[0], maxCandidates);

    if (i >= candidateCount)
    {
        return;
    }

    Detection a = g_candidates_in[i];

    uint keep = 1;

    [loop]
    for (uint j = 0; j < candidateCount; ++j)
    {
        if (i == j)
        {
            continue;
        }

        Detection b = g_candidates_in[j];

        if (a.classId != b.classId)
        {
            continue;
        }

        bool higher =
            (b.score > a.score) ||
            ((b.score == a.score) && (j < i));

        if (!higher)
        {
            continue;
        }

        float iou = IoU(a, b);

        if (iou > iouThreshold)
        {
            keep = 0;
            break;
        }
    }

    g_keep_flags_out[i] = keep;
}

[numthreads(256, 1, 1)]
void CompactCS(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;

    uint candidateCount =
        min(g_candidate_counter_in[0], maxCandidates);

    if (i >= candidateCount)
    {
        return;
    }

    if (g_keep_flags_in[i] == 0)
    {
        return;
    }

    uint outIndex = 0;
    InterlockedAdd(g_selected_counter_out[0], 1, outIndex);

    if (outIndex >= maxDetections)
    {
        return;
    }

    g_selected_out[outIndex] = g_candidates_in[i];
}

[numthreads(256, 1, 1)]
void MaskCS(uint3 tid : SV_DispatchThreadID)
{
    uint pixelIndex = tid.x;
    uint detIndex = tid.y;

    if (pixelIndex >= maskPixels)
    {
        return;
    }

    uint selectedCount =
        min(g_selected_counter_in[0], maxDetections);

    if (detIndex >= selectedCount)
    {
        return;
    }

    Detection det = g_selected_in[detIndex];
    uint cand = det.candidateIndex;

    uint mx = pixelIndex % maskWidth;
    uint my = pixelIndex / maskWidth;

    // det.x1/y1/x2/y2 は 640x640 入力空間の bbox。
    // prototype mask は 160x160 なので、bbox を prototype 空間へ縮小する。
    float scaleX = (float)maskWidth / inputWidth;
    float scaleY = (float)maskHeight / inputHeight;

    float boxX1 = det.x1 * scaleX;
    float boxY1 = det.y1 * scaleY;
    float boxX2 = det.x2 * scaleX;
    float boxY2 = det.y2 * scaleY;

    // Ultralytics crop_mask と同様に bbox 外を 0 にする。
    // Python実装では r >= x1, r < x2, c >= y1, c < y2 に近い。
    if ((float)mx < boxX1 || (float)mx >= boxX2 ||
        (float)my < boxY1 || (float)my >= boxY2)
    {
        g_selected_masks_out[detIndex * maskPixels + pixelIndex] = 0.0f;
        return;
    }

    float logit = 0.0f;

    [loop]
    for (uint k = 0; k < numMaskCoeffs; ++k)
    {
        uint coeffAttr = 4 + numClasses + k;

        float coeff = ReadOutput0(coeffAttr, cand);
        float proto = ReadProto(k, pixelIndex);

        logit += coeff * proto;
    }

    float maskValue = (logit > 0.0f) ? 1.0f : 0.0f;

    g_selected_masks_out[detIndex * maskPixels + pixelIndex] = maskValue;
}

)";

    auto compile_cs = [&](const char* entry) {
        Microsoft::WRL::ComPtr<ID3DBlob> blob;
        Microsoft::WRL::ComPtr<ID3DBlob> err;

        HRESULT hr = D3DCompile(
            hlsl,
            std::strlen(hlsl),
            nullptr,
            nullptr,
            nullptr,
            entry,
            "cs_5_0",
            0,
            0,
            blob.GetAddressOf(),
            err.GetAddressOf()
        );

        if (FAILED(hr)) {
            if (err) {
                std::string msg(
                    static_cast<const char*>(err->GetBufferPointer()),
                    err->GetBufferSize()
                );

                throw std::runtime_error(
                    std::string("D3DCompile failed: ") + msg
                );
            }

            win_util::ThrowIfFailed(
                hr,
                "D3DCompile failed"
            );
        }

        return blob;
        };

    auto create_pso = [&](
        ID3DBlob* cs_blob,
        Microsoft::WRL::ComPtr<ID3D12PipelineState>& pso
        ) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = this->root_signature_.Get();
            desc.CS.pShaderBytecode = cs_blob->GetBufferPointer();
            desc.CS.BytecodeLength = cs_blob->GetBufferSize();

            HRESULT hr = this->d3d12_core_.device()->CreateComputePipelineState(
                &desc,
                IID_PPV_ARGS(pso.GetAddressOf())
            );

            win_util::ThrowIfFailed(
                hr,
                "CreateComputePipelineState failed"
            );
        };

    auto decode_blob = compile_cs("DecodeCS");
    auto nms_blob = compile_cs("NmsCS");
    auto compact_blob = compile_cs("CompactCS");
    auto mask_blob = compile_cs("MaskCS");

    create_pso(decode_blob.Get(), this->decode_pso_);
    create_pso(nms_blob.Get(), this->nms_pso_);
    create_pso(compact_blob.Get(), this->compact_pso_);
    create_pso(mask_blob.Get(), this->mask_pso_);
}

void YoloSegPostProcessorD3D12::create_descriptor_heap()
{
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = DESCRIPTOR_COUNT;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    desc.NodeMask = 0;

    HRESULT hr = this->d3d12_core_.device()->CreateDescriptorHeap(
        &desc,
        IID_PPV_ARGS(this->descriptor_heap_.GetAddressOf())
    );

    win_util::ThrowIfFailed(
        hr,
        "CreateDescriptorHeap failed"
    );

    this->descriptor_size_ =
        this->d3d12_core_.device()->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        );
}

void YoloSegPostProcessorD3D12::create_buffers()
{
    const UINT64 output0_bytes =
        static_cast<UINT64>(config_.num_attrs) *
        static_cast<UINT64>(config_.num_candidates) *
        sizeof(float);

    const UINT64 output1_bytes =
        static_cast<UINT64>(config_.num_mask_coeffs) *
        static_cast<UINT64>(config_.mask_width) *
        static_cast<UINT64>(config_.mask_height) *
        sizeof(float);

    const UINT64 candidate_bytes =
        static_cast<UINT64>(config_.max_candidates) *
        sizeof(Detection);

    const UINT64 selected_bytes =
        static_cast<UINT64>(config_.max_detections) *
        sizeof(Detection);

    const UINT64 keep_bytes =
        static_cast<UINT64>(config_.max_candidates) *
        sizeof(uint32_t);

    const UINT64 selected_mask_bytes =
        static_cast<UINT64>(config_.max_detections) *
        static_cast<UINT64>(config_.mask_width) *
        static_cast<UINT64>(config_.mask_height) *
        sizeof(float);

    auto default_heap = MakeHeapProps(D3D12_HEAP_TYPE_DEFAULT);
    auto upload_heap = MakeHeapProps(D3D12_HEAP_TYPE_UPLOAD);

    this->output0_buffer_ =
        this->d3d12_core_.create_committed_resource(
            default_heap,
            D3D12_HEAP_FLAG_NONE,
            MakeBufferDesc(output0_bytes),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr
        );

    this->output0_upload_buffer_ =
        this->d3d12_core_.create_committed_resource(
            upload_heap,
            D3D12_HEAP_FLAG_NONE,
            MakeBufferDesc(output0_bytes),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr
        );

    this->output1_buffer_ =
        this->d3d12_core_.create_committed_resource(
            default_heap,
            D3D12_HEAP_FLAG_NONE,
            MakeBufferDesc(output1_bytes),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr
        );

    this->output1_upload_buffer_ =
        this->d3d12_core_.create_committed_resource(
            upload_heap,
            D3D12_HEAP_FLAG_NONE,
            MakeBufferDesc(output1_bytes),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr
        );

    this->candidate_buffer_ =
        this->d3d12_core_.create_committed_resource(
            default_heap,
            D3D12_HEAP_FLAG_NONE,
            MakeBufferDesc(
                candidate_bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            ),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr
        );

    this->candidate_counter_buffer_ =
        this->d3d12_core_.create_committed_resource(
            default_heap,
            D3D12_HEAP_FLAG_NONE,
            MakeBufferDesc(
                sizeof(uint32_t),
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            ),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr
        );

    this->keep_flag_buffer_ =
        this->d3d12_core_.create_committed_resource(
            default_heap,
            D3D12_HEAP_FLAG_NONE,
            MakeBufferDesc(
                keep_bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            ),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr
        );

    this->selected_detection_buffer_ =
        this->d3d12_core_.create_committed_resource(
            default_heap,
            D3D12_HEAP_FLAG_NONE,
            MakeBufferDesc(
                selected_bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            ),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr
        );

    this->selected_counter_buffer_ =
        this->d3d12_core_.create_committed_resource(
            default_heap,
            D3D12_HEAP_FLAG_NONE,
            MakeBufferDesc(
                sizeof(uint32_t),
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            ),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr
        );

    this->selected_mask_buffer_ =
        this->d3d12_core_.create_committed_resource(
            default_heap,
            D3D12_HEAP_FLAG_NONE,
            MakeBufferDesc(
                selected_mask_bytes,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            ),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr
        );

    this->zero_upload_buffer_ =
        this->d3d12_core_.create_committed_resource(
            upload_heap,
            D3D12_HEAP_FLAG_NONE,
            MakeBufferDesc(sizeof(uint32_t) * 2),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr
        );

    uint32_t* mapped = nullptr;

    D3D12_RANGE read_range{};
    read_range.Begin = 0;
    read_range.End = 0;

    HRESULT hr = this->zero_upload_buffer_->Map(
        0,
        &read_range,
        reinterpret_cast<void**>(&mapped)
    );

    win_util::ThrowIfFailed(
        hr,
        "Map zero upload buffer failed"
    );

    mapped[0] = 0;
    mapped[1] = 0;

    D3D12_RANGE written{};
    written.Begin = 0;
    written.End = sizeof(uint32_t) * 2;

    this->zero_upload_buffer_->Unmap(
        0,
        &written
    );
}

void YoloSegPostProcessorD3D12::create_descriptors()
{
    auto cpu_base =
        this->descriptor_heap_->GetCPUDescriptorHandleForHeapStart();

    auto handle_at = [&](UINT index) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = cpu_base;
        h.ptr += static_cast<SIZE_T>(index) * this->descriptor_size_;
        return h;
        };

    auto create_structured_srv = [&](
        ID3D12Resource* res,
        UINT num_elements,
        UINT stride,
        UINT slot
        ) {
            D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
            desc.Shader4ComponentMapping =
                D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            desc.Buffer.FirstElement = 0;
            desc.Buffer.NumElements = num_elements;
            desc.Buffer.StructureByteStride = stride;
            desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

            this->d3d12_core_.device()->CreateShaderResourceView(
                res,
                &desc,
                handle_at(slot)
            );
        };

    auto create_structured_uav = [&](
        ID3D12Resource* res,
        UINT num_elements,
        UINT stride,
        UINT slot
        ) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
            desc.Format = DXGI_FORMAT_UNKNOWN;
            desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            desc.Buffer.FirstElement = 0;
            desc.Buffer.NumElements = num_elements;
            desc.Buffer.StructureByteStride = stride;
            desc.Buffer.CounterOffsetInBytes = 0;
            desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

            this->d3d12_core_.device()->CreateUnorderedAccessView(
                res,
                nullptr,
                &desc,
                handle_at(slot)
            );
        };

    const UINT mask_pixels =
        config_.mask_width * config_.mask_height;

    create_structured_srv(
        this->output0_buffer_.Get(),
        config_.num_attrs * config_.num_candidates,
        sizeof(float),
        SRV_OUTPUT0
    );

    create_structured_srv(
        this->output1_buffer_.Get(),
        config_.num_mask_coeffs * mask_pixels,
        sizeof(float),
        SRV_OUTPUT1
    );

    create_structured_srv(
        this->candidate_buffer_.Get(),
        config_.max_candidates,
        sizeof(Detection),
        SRV_CANDIDATES
    );

    create_structured_srv(
        this->candidate_counter_buffer_.Get(),
        1,
        sizeof(uint32_t),
        SRV_CANDIDATE_COUNTER
    );

    create_structured_srv(
        this->keep_flag_buffer_.Get(),
        config_.max_candidates,
        sizeof(uint32_t),
        SRV_KEEP_FLAGS
    );

    create_structured_srv(
        this->selected_detection_buffer_.Get(),
        config_.max_detections,
        sizeof(Detection),
        SRV_SELECTED
    );

    create_structured_srv(
        this->selected_counter_buffer_.Get(),
        1,
        sizeof(uint32_t),
        SRV_SELECTED_COUNTER
    );

    create_structured_uav(
        this->candidate_buffer_.Get(),
        config_.max_candidates,
        sizeof(Detection),
        UAV_CANDIDATES
    );

    create_structured_uav(
        this->candidate_counter_buffer_.Get(),
        1,
        sizeof(uint32_t),
        UAV_CANDIDATE_COUNTER
    );

    create_structured_uav(
        this->keep_flag_buffer_.Get(),
        config_.max_candidates,
        sizeof(uint32_t),
        UAV_KEEP_FLAGS
    );

    create_structured_uav(
        this->selected_detection_buffer_.Get(),
        config_.max_detections,
        sizeof(Detection),
        UAV_SELECTED
    );

    create_structured_uav(
        this->selected_counter_buffer_.Get(),
        1,
        sizeof(uint32_t),
        UAV_SELECTED_COUNTER
    );

    create_structured_uav(
        this->selected_mask_buffer_.Get(),
        config_.max_detections * mask_pixels,
        sizeof(float),
        UAV_SELECTED_MASKS
    );
}

void YoloSegPostProcessorD3D12::create_constant_buffer()
{
    const UINT cb_size =
        align256(sizeof(ShaderParams));

    auto upload_heap =
        MakeHeapProps(D3D12_HEAP_TYPE_UPLOAD);

    this->constant_buffer_ =
        this->d3d12_core_.create_committed_resource(
            upload_heap,
            D3D12_HEAP_FLAG_NONE,
            MakeBufferDesc(cb_size),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr
        );

    D3D12_RANGE read_range{};
    read_range.Begin = 0;
    read_range.End = 0;

    HRESULT hr = this->constant_buffer_->Map(
        0,
        &read_range,
        reinterpret_cast<void**>(&this->mapped_constants_)
    );

    win_util::ThrowIfFailed(
        hr,
        "Map postprocess constant buffer failed"
    );
}

void YoloSegPostProcessorD3D12::update_params()
{
    ShaderParams params{};
    params.num_attrs = config_.num_attrs;
    params.num_candidates = config_.num_candidates;
    params.num_classes = config_.num_classes;
    params.num_mask_coeffs = config_.num_mask_coeffs;

    params.mask_width = config_.mask_width;
    params.mask_height = config_.mask_height;
    params.mask_pixels = config_.mask_width * config_.mask_height;
    params.max_candidates = config_.max_candidates;

    params.max_detections = config_.max_detections;
    params.conf_threshold = config_.conf_threshold;
    params.iou_threshold = config_.iou_threshold;
    params.input_width = config_.input_width;

    params.input_height = config_.input_height;

    std::memcpy(
        this->mapped_constants_,
        &params,
        sizeof(params)
    );
}

void YoloSegPostProcessorD3D12::upload_buffer_from_cpu(
    ID3D12Resource* dst_buffer,
    D3D12_RESOURCE_STATES& dst_state,
    ID3D12Resource* upload_buffer,
    const void* data,
    UINT64 byte_size
)
{
    if (!dst_buffer || !upload_buffer || !data || byte_size == 0) {
        throw std::runtime_error("upload_buffer_from_cpu: invalid argument");
    }

    void* mapped = nullptr;

    D3D12_RANGE read_range{};
    read_range.Begin = 0;
    read_range.End = 0;

    HRESULT hr = upload_buffer->Map(
        0,
        &read_range,
        &mapped
    );

    win_util::ThrowIfFailed(
        hr,
        "Map upload buffer failed"
    );

    std::memcpy(
        mapped,
        data,
        static_cast<size_t>(byte_size)
    );

    D3D12_RANGE written_range{};
    written_range.Begin = 0;
    written_range.End = static_cast<SIZE_T>(byte_size);

    upload_buffer->Unmap(
        0,
        &written_range
    );

    this->d3d12_core_.reset_command_list();

    ID3D12GraphicsCommandList* command_list =
        this->d3d12_core_.command_list();

    this->transition_resource(
        command_list,
        dst_buffer,
        dst_state,
        D3D12_RESOURCE_STATE_COPY_DEST
    );

    command_list->CopyBufferRegion(
        dst_buffer,
        0,
        upload_buffer,
        0,
        byte_size
    );

    this->transition_resource(
        command_list,
        dst_buffer,
        dst_state,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    this->d3d12_core_.close_execute_and_wait();
}

void YoloSegPostProcessorD3D12::reset_counters(
    ID3D12GraphicsCommandList* command_list
)
{
    this->transition_resource(
        command_list,
        this->candidate_counter_buffer_.Get(),
        this->candidate_counter_state_,
        D3D12_RESOURCE_STATE_COPY_DEST
    );

    this->transition_resource(
        command_list,
        this->selected_counter_buffer_.Get(),
        this->selected_counter_state_,
        D3D12_RESOURCE_STATE_COPY_DEST
    );

    command_list->CopyBufferRegion(
        this->candidate_counter_buffer_.Get(),
        0,
        this->zero_upload_buffer_.Get(),
        0,
        sizeof(uint32_t)
    );

    command_list->CopyBufferRegion(
        this->selected_counter_buffer_.Get(),
        0,
        this->zero_upload_buffer_.Get(),
        sizeof(uint32_t),
        sizeof(uint32_t)
    );
}

void YoloSegPostProcessorD3D12::record_process(
    ID3D12GraphicsCommandList* command_list
)
{
    this->update_params();

    this->reset_counters(command_list);

    this->transition_resource(
        command_list,
        this->output0_buffer_.Get(),
        this->output0_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    this->transition_resource(
        command_list,
        this->output1_buffer_.Get(),
        this->output1_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    this->transition_resource(
        command_list,
        this->candidate_buffer_.Get(),
        this->candidate_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    this->transition_resource(
        command_list,
        this->candidate_counter_buffer_.Get(),
        this->candidate_counter_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    this->transition_resource(
        command_list,
        this->keep_flag_buffer_.Get(),
        this->keep_flag_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    this->transition_resource(
        command_list,
        this->selected_detection_buffer_.Get(),
        this->selected_detection_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    this->transition_resource(
        command_list,
        this->selected_counter_buffer_.Get(),
        this->selected_counter_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    this->transition_resource(
        command_list,
        this->selected_mask_buffer_.Get(),
        this->selected_mask_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    ID3D12DescriptorHeap* heaps[] = {
        this->descriptor_heap_.Get()
    };

    command_list->SetDescriptorHeaps(
        1,
        heaps
    );

    command_list->SetComputeRootSignature(
        this->root_signature_.Get()
    );

    D3D12_GPU_DESCRIPTOR_HANDLE base_gpu =
        this->descriptor_heap_->GetGPUDescriptorHandleForHeapStart();

    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = base_gpu;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = base_gpu;
    uav_gpu.ptr +=
        static_cast<SIZE_T>(UAV_CANDIDATES) *
        this->descriptor_size_;

    command_list->SetComputeRootDescriptorTable(
        0,
        srv_gpu
    );

    command_list->SetComputeRootDescriptorTable(
        1,
        uav_gpu
    );

    command_list->SetComputeRootConstantBufferView(
        2,
        this->constant_buffer_->GetGPUVirtualAddress()
    );

    D3D12_RESOURCE_BARRIER uav_barrier{};
    uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    uav_barrier.UAV.pResource = nullptr;

    const UINT decode_groups =
        (config_.num_candidates + 255) / 256;

    const UINT nms_groups =
        (config_.max_candidates + 255) / 256;

    const UINT mask_pixels =
        config_.mask_width * config_.mask_height;

    const UINT mask_groups_x =
        (mask_pixels + 255) / 256;

    // Pass 1: Decode candidates
    command_list->SetPipelineState(
        this->decode_pso_.Get()
    );

    command_list->Dispatch(
        decode_groups,
        1,
        1
    );

    command_list->ResourceBarrier(
        1,
        &uav_barrier
    );

    // Pass 2: NMS
    this->transition_resource(
        command_list,
        this->candidate_buffer_.Get(),
        this->candidate_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    this->transition_resource(
        command_list,
        this->candidate_counter_buffer_.Get(),
        this->candidate_counter_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    this->transition_resource(
        command_list,
        this->keep_flag_buffer_.Get(),
        this->keep_flag_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    command_list->SetPipelineState(
        this->nms_pso_.Get()
    );

    command_list->Dispatch(
        nms_groups,
        1,
        1
    );

    command_list->ResourceBarrier(
        1,
        &uav_barrier
    );

    // Pass 3: Compact
    this->transition_resource(
        command_list,
        this->keep_flag_buffer_.Get(),
        this->keep_flag_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    this->transition_resource(
        command_list,
        this->selected_detection_buffer_.Get(),
        this->selected_detection_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    this->transition_resource(
        command_list,
        this->selected_counter_buffer_.Get(),
        this->selected_counter_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    command_list->SetPipelineState(
        this->compact_pso_.Get()
    );

    command_list->Dispatch(
        nms_groups,
        1,
        1
    );

    command_list->ResourceBarrier(
        1,
        &uav_barrier
    );

    // Pass 4: Mask generation
    this->transition_resource(
        command_list,
        this->selected_detection_buffer_.Get(),
        this->selected_detection_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    this->transition_resource(
        command_list,
        this->selected_counter_buffer_.Get(),
        this->selected_counter_state_,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    this->transition_resource(
        command_list,
        this->selected_mask_buffer_.Get(),
        this->selected_mask_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    command_list->SetPipelineState(
        this->mask_pso_.Get()
    );

    command_list->Dispatch(
        mask_groups_x,
        config_.max_detections,
        1
    );

    command_list->ResourceBarrier(
        1,
        &uav_barrier
    );
}

void YoloSegPostProcessorD3D12::transition_resource(
    ID3D12GraphicsCommandList* command_list,
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES& current_state,
    D3D12_RESOURCE_STATES new_state
)
{
    if (!command_list) {
        throw std::runtime_error("transition_resource: command_list is null");
    }

    if (!resource) {
        throw std::runtime_error("transition_resource: resource is null");
    }

    if (current_state == new_state) {
        return;
    }

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource =
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = current_state;
    barrier.Transition.StateAfter = new_state;

    command_list->ResourceBarrier(
        1,
        &barrier
    );

    current_state = new_state;
}

UINT YoloSegPostProcessorD3D12::align256(UINT value)
{
    return (value + 255u) & ~255u;
}

void YoloSegPostProcessorD3D12::create_output_input_descriptors(
    ID3D12Resource* output0_buffer,
    ID3D12Resource* output1_buffer
)
{
    if (!output0_buffer || !output1_buffer) {
        throw std::runtime_error("create_output_input_descriptors: output buffer is null");
    }

    const UINT mask_pixels =
        this->config_.mask_width * this->config_.mask_height;

    D3D12_CPU_DESCRIPTOR_HANDLE base_cpu =
        this->descriptor_heap_->GetCPUDescriptorHandleForHeapStart();

    auto handle_at = [&](UINT index) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = base_cpu;
        h.ptr += static_cast<SIZE_T>(index) * this->descriptor_size_;
        return h;
    };

    auto create_structured_srv = [&](
        ID3D12Resource* res,
        UINT num_elements,
        UINT stride,
        UINT slot
    ) {
        D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
        desc.Shader4ComponentMapping =
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        desc.Buffer.FirstElement = 0;
        desc.Buffer.NumElements = num_elements;
        desc.Buffer.StructureByteStride = stride;
        desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

        this->d3d12_core_.device()->CreateShaderResourceView(
            res,
            &desc,
            handle_at(slot)
        );
    };

    create_structured_srv(
        output0_buffer,
        this->config_.num_attrs * this->config_.num_candidates,
        sizeof(float),
        SRV_OUTPUT0
    );

    create_structured_srv(
        output1_buffer,
        this->config_.num_mask_coeffs * mask_pixels,
        sizeof(float),
        SRV_OUTPUT1
    );
}