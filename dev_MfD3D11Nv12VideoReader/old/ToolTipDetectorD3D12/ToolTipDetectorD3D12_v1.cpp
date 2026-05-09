#include "ToolTipDetectorD3D12.hpp"

#include <stdexcept>
#include <sstream>
#include <string>
#include <cstring>
#include <algorithm>

#include <d3dcompiler.h>

#include "HResultUtil.hpp"

#pragma comment(lib, "d3dcompiler.lib")

namespace {

    constexpr UINT SRV_SELECTED_DETECTIONS = 0;
    constexpr UINT SRV_SELECTED_COUNTER = 1;
    constexpr UINT SRV_SELECTED_MASKS = 2;

    constexpr UINT UAV_RESULTS = 3;

    constexpr UINT DESCRIPTOR_COUNT = 4;

    struct DetectionLayout {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;

        uint32_t class_id;
        uint32_t candidate_index;
        uint32_t reserved;
    };

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

ToolTipDetectorD3D12::ToolTipDetectorD3D12(
    D3D12Core& d3d12_core,
    const Config& config
)
    : d3d12_core_(d3d12_core)
    , config_(config)
{
    if (!this->d3d12_core_.device()) {
        throw std::runtime_error("ToolTipDetectorD3D12: D3D12Core device is null");
    }

    if (this->config_.max_detections == 0 ||
        this->config_.mask_width == 0 ||
        this->config_.mask_height == 0)
    {
        throw std::runtime_error("ToolTipDetectorD3D12: invalid config");
    }

    this->create_root_signature();
    this->create_pipeline_state();
    this->create_descriptor_heap();
    this->create_result_buffer();
    this->create_constant_buffer();
    this->update_params();
}

void ToolTipDetectorD3D12::detect_and_wait(
    ID3D12Resource* selected_detection_buffer,
    D3D12_RESOURCE_STATES& selected_detection_state,
    ID3D12Resource* selected_counter_buffer,
    D3D12_RESOURCE_STATES& selected_counter_state,
    ID3D12Resource* selected_mask_buffer,
    D3D12_RESOURCE_STATES& selected_mask_state
)
{
    if (!selected_detection_buffer ||
        !selected_counter_buffer ||
        !selected_mask_buffer)
    {
        throw std::runtime_error("ToolTipDetectorD3D12::detect_and_wait: input buffer is null");
    }

    this->create_input_descriptors(
        selected_detection_buffer,
        selected_counter_buffer,
        selected_mask_buffer
    );

    this->d3d12_core_.reset_command_list();

    this->record_detect(
        this->d3d12_core_.command_list(),
        selected_detection_buffer,
        selected_detection_state,
        selected_counter_buffer,
        selected_counter_state,
        selected_mask_buffer,
        selected_mask_state
    );

    this->d3d12_core_.close_execute_and_wait();
}

std::vector<ToolTipDetectorD3D12::TipResult>
ToolTipDetectorD3D12::readback_results()
{
    const UINT64 result_bytes =
        static_cast<UINT64>(this->config_.max_detections) *
        sizeof(TipResult);

    auto readback_heap =
        MakeHeapProps(D3D12_HEAP_TYPE_READBACK);

    auto readback_desc =
        MakeBufferDesc(result_bytes);

    auto readback_buffer =
        this->d3d12_core_.create_committed_resource(
            readback_heap,
            D3D12_HEAP_FLAG_NONE,
            readback_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr
        );

    this->d3d12_core_.reset_command_list();

    ID3D12GraphicsCommandList* command_list =
        this->d3d12_core_.command_list();

    this->transition_resource(
        command_list,
        this->result_buffer_.Get(),
        this->result_buffer_state_,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );

    command_list->CopyBufferRegion(
        readback_buffer.Get(),
        0,
        this->result_buffer_.Get(),
        0,
        result_bytes
    );

    this->transition_resource(
        command_list,
        this->result_buffer_.Get(),
        this->result_buffer_state_,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    this->d3d12_core_.close_execute_and_wait();

    D3D12_RANGE read_range{};
    read_range.Begin = 0;
    read_range.End = static_cast<SIZE_T>(result_bytes);

    TipResult* mapped = nullptr;

    HRESULT hr = readback_buffer->Map(
        0,
        &read_range,
        reinterpret_cast<void**>(&mapped)
    );

    win_util::ThrowIfFailed(
        hr,
        "ToolTipDetectorD3D12::readback_results: Map failed"
    );

    std::vector<TipResult> all_results(
        this->config_.max_detections
    );

    std::memcpy(
        all_results.data(),
        mapped,
        static_cast<size_t>(result_bytes)
    );

    readback_buffer->Unmap(
        0,
        nullptr
    );

    std::vector<TipResult> valid_results;

    for (const auto& r : all_results) {
        if (r.valid != 0) {
            valid_results.push_back(r);
        }
    }

    return valid_results;
}

ID3D12Resource* ToolTipDetectorD3D12::result_buffer() const
{
    return this->result_buffer_.Get();
}

D3D12_RESOURCE_STATES& ToolTipDetectorD3D12::result_buffer_state_ref()
{
    return this->result_buffer_state_;
}

const ToolTipDetectorD3D12::Config&
ToolTipDetectorD3D12::config() const
{
    return this->config_;
}

void ToolTipDetectorD3D12::create_root_signature()
{
    D3D12_DESCRIPTOR_RANGE ranges[2]{};

    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 3;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].RegisterSpace = 0;
    ranges[0].OffsetInDescriptorsFromTableStart =
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].RegisterSpace = 0;
    ranges[1].OffsetInDescriptorsFromTableStart =
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER params[3]{};

    params[0].ParameterType =
        D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    params[0].ShaderVisibility =
        D3D12_SHADER_VISIBILITY_ALL;

    params[1].ParameterType =
        D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &ranges[1];
    params[1].ShaderVisibility =
        D3D12_SHADER_VISIBILITY_ALL;

    params[2].ParameterType =
        D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[2].Descriptor.ShaderRegister = 0;
    params[2].Descriptor.RegisterSpace = 0;
    params[2].ShaderVisibility =
        D3D12_SHADER_VISIBILITY_ALL;

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
        "ToolTipDetectorD3D12::CreateRootSignature failed"
    );
}

void ToolTipDetectorD3D12::create_pipeline_state()
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

struct TipResult
{
    uint valid;
    uint detectionIndex;
    uint classId;
    uint selectedCandidate;

    float tipXMask;
    float tipYMask;

    float tipXInput;
    float tipYInput;

    float candidate1XMask;
    float candidate1YMask;

    float candidate2XMask;
    float candidate2YMask;

    float candidate1Width;
    float candidate2Width;

    float area;

    float axisX;
    float axisY;
};

StructuredBuffer<Detection> g_detections : register(t0);
StructuredBuffer<uint> g_detection_counter : register(t1);
StructuredBuffer<float> g_masks : register(t2);

RWStructuredBuffer<TipResult> g_results : register(u0);

cbuffer Params : register(b0)
{
    uint maxDetections;
    uint maskWidth;
    uint maskHeight;
    uint maskPixels;

    float inputWidth;
    float inputHeight;
    uint targetClassId;
    uint minAreaPixels;

    float maskThreshold;
    float endRegionRatio;
    float topEdgeRatio;
    float reserved0;
};

float ReadMask(uint detIndex, uint pixelIndex)
{
    return g_masks[detIndex * maskPixels + pixelIndex];
}

void WriteInvalid(uint detIndex)
{
    TipResult r;
    r.valid = 0;
    r.detectionIndex = detIndex;
    r.classId = 0;
    r.selectedCandidate = 0;

    r.tipXMask = 0.0f;
    r.tipYMask = 0.0f;
    r.tipXInput = 0.0f;
    r.tipYInput = 0.0f;

    r.candidate1XMask = 0.0f;
    r.candidate1YMask = 0.0f;
    r.candidate2XMask = 0.0f;
    r.candidate2YMask = 0.0f;

    r.candidate1Width = 0.0f;
    r.candidate2Width = 0.0f;

    r.area = 0.0f;

    r.axisX = 0.0f;
    r.axisY = 0.0f;

    g_results[detIndex] = r;
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint detIndex = tid.x;

    uint detectionCount =
        min(g_detection_counter[0], maxDetections);

    if (detIndex >= maxDetections)
    {
        return;
    }

    if (detIndex >= detectionCount)
    {
        WriteInvalid(detIndex);
        return;
    }

    Detection det = g_detections[detIndex];

    if (det.classId != targetClassId)
    {
        WriteInvalid(detIndex);
        return;
    }

    float count = 0.0f;
    float sumX = 0.0f;
    float sumY = 0.0f;
    float sumXX = 0.0f;
    float sumYY = 0.0f;
    float sumXY = 0.0f;

    // ------------------------------------------------------------
    // Pass A: moments
    // ------------------------------------------------------------
    [loop]
    for (uint y = 0; y < maskHeight; ++y)
    {
        [loop]
        for (uint x = 0; x < maskWidth; ++x)
        {
            uint idx = y * maskWidth + x;
            float m = ReadMask(detIndex, idx);

            if (m < maskThreshold)
            {
                continue;
            }

            float fx = (float)x + 0.5f;
            float fy = (float)y + 0.5f;

            count += 1.0f;
            sumX += fx;
            sumY += fy;
            sumXX += fx * fx;
            sumYY += fy * fy;
            sumXY += fx * fy;
        }
    }

    if (count < (float)minAreaPixels)
    {
        WriteInvalid(detIndex);
        return;
    }

    float cx = sumX / count;
    float cy = sumY / count;

    float covXX = sumXX / count - cx * cx;
    float covYY = sumYY / count - cy * cy;
    float covXY = sumXY / count - cx * cy;

    // covariance から主軸方向を求める
    float angle = 0.5f * atan2(2.0f * covXY, covXX - covYY);

    float2 axis = float2(cos(angle), sin(angle));

    float axisLen = length(axis);

    if (axisLen < 1e-6f)
    {
        WriteInvalid(detIndex);
        return;
    }

    axis /= axisLen;

    float2 normal = float2(-axis.y, axis.x);

    // ------------------------------------------------------------
    // Pass B: projection range
    // ------------------------------------------------------------

    float minProj =  1e30f;
    float maxProj = -1e30f;

    [loop]
    for (uint y2 = 0; y2 < maskHeight; ++y2)
    {
        [loop]
        for (uint x2 = 0; x2 < maskWidth; ++x2)
        {
            uint idx2 = y2 * maskWidth + x2;
            float m2 = ReadMask(detIndex, idx2);

            if (m2 < maskThreshold)
            {
                continue;
            }

            float2 p = float2((float)x2 + 0.5f, (float)y2 + 0.5f);
            float2 q = p - float2(cx, cy);

            float proj = dot(q, axis);

            minProj = min(minProj, proj);
            maxProj = max(maxProj, proj);
        }
    }

    float span = maxProj - minProj;

    if (span < 1e-4f)
    {
        WriteInvalid(detIndex);
        return;
    }

    float endRegion = max(1.0f, span * endRegionRatio);

    // ------------------------------------------------------------
    // Pass C: end region width and center
    // ------------------------------------------------------------

    float minSideCount = 0.0f;
    float maxSideCount = 0.0f;

    float minSideSumX = 0.0f;
    float minSideSumY = 0.0f;
    float maxSideSumX = 0.0f;
    float maxSideSumY = 0.0f;

    float minSidePerpMin =  1e30f;
    float minSidePerpMax = -1e30f;
    float maxSidePerpMin =  1e30f;
    float maxSidePerpMax = -1e30f;

    [loop]
    for (uint y3 = 0; y3 < maskHeight; ++y3)
    {
        [loop]
        for (uint x3 = 0; x3 < maskWidth; ++x3)
        {
            uint idx3 = y3 * maskWidth + x3;
            float m3 = ReadMask(detIndex, idx3);

            if (m3 < maskThreshold)
            {
                continue;
            }

            float2 p3 = float2((float)x3 + 0.5f, (float)y3 + 0.5f);
            float2 q3 = p3 - float2(cx, cy);

            float proj3 = dot(q3, axis);
            float perp3 = dot(q3, normal);

            if (proj3 <= minProj + endRegion)
            {
                minSideCount += 1.0f;
                minSideSumX += p3.x;
                minSideSumY += p3.y;
                minSidePerpMin = min(minSidePerpMin, perp3);
                minSidePerpMax = max(minSidePerpMax, perp3);
            }

            if (proj3 >= maxProj - endRegion)
            {
                maxSideCount += 1.0f;
                maxSideSumX += p3.x;
                maxSideSumY += p3.y;
                maxSidePerpMin = min(maxSidePerpMin, perp3);
                maxSidePerpMax = max(maxSidePerpMax, perp3);
            }
        }
    }

    if (minSideCount < 1.0f || maxSideCount < 1.0f)
    {
        WriteInvalid(detIndex);
        return;
    }

    float minCenterX = minSideSumX / minSideCount;
    float minCenterY = minSideSumY / minSideCount;
    float maxCenterX = maxSideSumX / maxSideCount;
    float maxCenterY = maxSideSumY / maxSideCount;

    float minWidth = max(0.0f, minSidePerpMax - minSidePerpMin);
    float maxWidth = max(0.0f, maxSidePerpMax - maxSidePerpMin);

    // 幅が細い側を candidate1、反対側を candidate2
    float candidate1X;
    float candidate1Y;
    float candidate1Width;

    float candidate2X;
    float candidate2Y;
    float candidate2Width;

    if (minWidth <= maxWidth)
    {
        candidate1X = minCenterX;
        candidate1Y = minCenterY;
        candidate1Width = minWidth;

        candidate2X = maxCenterX;
        candidate2Y = maxCenterY;
        candidate2Width = maxWidth;
    }
    else
    {
        candidate1X = maxCenterX;
        candidate1Y = maxCenterY;
        candidate1Width = maxWidth;

        candidate2X = minCenterX;
        candidate2Y = minCenterY;
        candidate2Width = minWidth;
    }

    // candidate1 がフレーム上端5%以内なら candidate2 を採用
    float topEdgeY = topEdgeRatio * (float)maskHeight;

    uint selectedCandidate = 1;
    float tipX = candidate1X;
    float tipY = candidate1Y;

    if (candidate1Y <= topEdgeY)
    {
        selectedCandidate = 2;
        tipX = candidate2X;
        tipY = candidate2Y;
    }

    float tipXInput = tipX * inputWidth / (float)maskWidth;
    float tipYInput = tipY * inputHeight / (float)maskHeight;

    TipResult outR;
    outR.valid = 1;
    outR.detectionIndex = detIndex;
    outR.classId = det.classId;
    outR.selectedCandidate = selectedCandidate;

    outR.tipXMask = tipX;
    outR.tipYMask = tipY;

    outR.tipXInput = tipXInput;
    outR.tipYInput = tipYInput;

    outR.candidate1XMask = candidate1X;
    outR.candidate1YMask = candidate1Y;

    outR.candidate2XMask = candidate2X;
    outR.candidate2YMask = candidate2Y;

    outR.candidate1Width = candidate1Width;
    outR.candidate2Width = candidate2Width;

    outR.area = count;

    outR.axisX = axis.x;
    outR.axisY = axis.y;

    g_results[detIndex] = outR;
}

)";

    Microsoft::WRL::ComPtr<ID3DBlob> cs_blob;
    Microsoft::WRL::ComPtr<ID3DBlob> error_blob;

    HRESULT hr = D3DCompile(
        hlsl,
        std::strlen(hlsl),
        nullptr,
        nullptr,
        nullptr,
        "main",
        "cs_5_0",
        0,
        0,
        cs_blob.GetAddressOf(),
        error_blob.GetAddressOf()
    );

    if (FAILED(hr)) {
        if (error_blob) {
            std::string msg(
                static_cast<const char*>(error_blob->GetBufferPointer()),
                error_blob->GetBufferSize()
            );

            throw std::runtime_error(
                "ToolTipDetectorD3D12 D3DCompile failed: " + msg
            );
        }

        win_util::ThrowIfFailed(
            hr,
            "ToolTipDetectorD3D12 D3DCompile failed"
        );
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = this->root_signature_.Get();
    pso_desc.CS.pShaderBytecode = cs_blob->GetBufferPointer();
    pso_desc.CS.BytecodeLength = cs_blob->GetBufferSize();

    hr = this->d3d12_core_.device()->CreateComputePipelineState(
        &pso_desc,
        IID_PPV_ARGS(this->pipeline_state_.GetAddressOf())
    );

    win_util::ThrowIfFailed(
        hr,
        "ToolTipDetectorD3D12 CreateComputePipelineState failed"
    );
}

void ToolTipDetectorD3D12::create_descriptor_heap()
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
        "ToolTipDetectorD3D12 CreateDescriptorHeap failed"
    );

    this->descriptor_size_ =
        this->d3d12_core_.device()->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        );
}

void ToolTipDetectorD3D12::create_result_buffer()
{
    const UINT64 byte_size =
        static_cast<UINT64>(this->config_.max_detections) *
        sizeof(TipResult);

    auto heap_props =
        MakeHeapProps(D3D12_HEAP_TYPE_DEFAULT);

    auto desc =
        MakeBufferDesc(
            byte_size,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        );

    this->result_buffer_ =
        this->d3d12_core_.create_committed_resource(
            heap_props,
            D3D12_HEAP_FLAG_NONE,
            desc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr
        );

    this->result_buffer_state_ =
        D3D12_RESOURCE_STATE_COMMON;
}

void ToolTipDetectorD3D12::create_constant_buffer()
{
    const UINT cb_size =
        align256(sizeof(ShaderParams));

    auto heap_props =
        MakeHeapProps(D3D12_HEAP_TYPE_UPLOAD);

    auto desc =
        MakeBufferDesc(cb_size);

    this->constant_buffer_ =
        this->d3d12_core_.create_committed_resource(
            heap_props,
            D3D12_HEAP_FLAG_NONE,
            desc,
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
        "ToolTipDetectorD3D12 Map constant buffer failed"
    );
}

void ToolTipDetectorD3D12::update_params()
{
    ShaderParams params{};
    params.max_detections = this->config_.max_detections;
    params.mask_width = this->config_.mask_width;
    params.mask_height = this->config_.mask_height;
    params.mask_pixels =
        this->config_.mask_width *
        this->config_.mask_height;

    params.input_width = this->config_.input_width;
    params.input_height = this->config_.input_height;
    params.target_class_id = this->config_.target_class_id;
    params.min_area_pixels = this->config_.min_area_pixels;

    params.mask_threshold = this->config_.mask_threshold;
    params.end_region_ratio = this->config_.end_region_ratio;
    params.top_edge_ratio = this->config_.top_edge_ratio;

    std::memcpy(
        this->mapped_constants_,
        &params,
        sizeof(params)
    );
}

void ToolTipDetectorD3D12::create_input_descriptors(
    ID3D12Resource* selected_detection_buffer,
    ID3D12Resource* selected_counter_buffer,
    ID3D12Resource* selected_mask_buffer
)
{
    D3D12_CPU_DESCRIPTOR_HANDLE base_cpu =
        this->descriptor_heap_->GetCPUDescriptorHandleForHeapStart();

    auto handle_at = [&](UINT index) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = base_cpu;
        h.ptr +=
            static_cast<SIZE_T>(index) *
            this->descriptor_size_;
        return h;
        };

    auto create_structured_srv = [&](
        ID3D12Resource* resource,
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
                resource,
                &desc,
                handle_at(slot)
            );
        };

    auto create_structured_uav = [&](
        ID3D12Resource* resource,
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
                resource,
                nullptr,
                &desc,
                handle_at(slot)
            );
        };

    create_structured_srv(
        selected_detection_buffer,
        this->config_.max_detections,
        sizeof(DetectionLayout),
        SRV_SELECTED_DETECTIONS
    );

    create_structured_srv(
        selected_counter_buffer,
        1,
        sizeof(uint32_t),
        SRV_SELECTED_COUNTER
    );

    create_structured_srv(
        selected_mask_buffer,
        this->config_.max_detections *
        this->config_.mask_width *
        this->config_.mask_height,
        sizeof(float),
        SRV_SELECTED_MASKS
    );

    create_structured_uav(
        this->result_buffer_.Get(),
        this->config_.max_detections,
        sizeof(TipResult),
        UAV_RESULTS
    );
}

void ToolTipDetectorD3D12::record_detect(
    ID3D12GraphicsCommandList* command_list,
    ID3D12Resource* selected_detection_buffer,
    D3D12_RESOURCE_STATES& selected_detection_state,
    ID3D12Resource* selected_counter_buffer,
    D3D12_RESOURCE_STATES& selected_counter_state,
    ID3D12Resource* selected_mask_buffer,
    D3D12_RESOURCE_STATES& selected_mask_state
)
{
    this->update_params();

    this->transition_resource(
        command_list,
        selected_detection_buffer,
        selected_detection_state,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    this->transition_resource(
        command_list,
        selected_counter_buffer,
        selected_counter_state,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    this->transition_resource(
        command_list,
        selected_mask_buffer,
        selected_mask_state,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );

    this->transition_resource(
        command_list,
        this->result_buffer_.Get(),
        this->result_buffer_state_,
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

    command_list->SetPipelineState(
        this->pipeline_state_.Get()
    );

    D3D12_GPU_DESCRIPTOR_HANDLE base_gpu =
        this->descriptor_heap_->GetGPUDescriptorHandleForHeapStart();

    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = base_gpu;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = base_gpu;
    uav_gpu.ptr +=
        static_cast<SIZE_T>(UAV_RESULTS) *
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

    const UINT groups =
        (this->config_.max_detections + 63) / 64;

    command_list->Dispatch(
        groups,
        1,
        1
    );

    D3D12_RESOURCE_BARRIER uav_barrier{};
    uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    uav_barrier.UAV.pResource = this->result_buffer_.Get();

    command_list->ResourceBarrier(
        1,
        &uav_barrier
    );
}

void ToolTipDetectorD3D12::transition_resource(
    ID3D12GraphicsCommandList* command_list,
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES& current_state,
    D3D12_RESOURCE_STATES new_state
)
{
    if (!command_list) {
        throw std::runtime_error("ToolTipDetectorD3D12::transition_resource: command_list is null");
    }

    if (!resource) {
        throw std::runtime_error("ToolTipDetectorD3D12::transition_resource: resource is null");
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

UINT ToolTipDetectorD3D12::align256(UINT value)
{
    return (value + 255u) & ~255u;
}