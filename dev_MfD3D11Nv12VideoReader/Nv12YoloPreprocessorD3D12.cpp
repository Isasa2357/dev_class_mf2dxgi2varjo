#include "Nv12YoloPreprocessorD3D12.hpp"

#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>

#include <d3dcompiler.h>

#include "HResultUtil.hpp"

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "d3d12.lib")

Nv12YoloPreprocessorD3D12::Nv12YoloPreprocessorD3D12(
    D3D12Core& d3d12_core,
    const Config& config
)
    : d3d12_core_(d3d12_core)
    , config_(config)
{
    if (!this->d3d12_core_.device()) {
        throw std::runtime_error("Nv12YoloPreprocessorD3D12: D3D12Core device is null");
    }

    if (this->config_.input_width == 0 || this->config_.input_height == 0) {
        throw std::runtime_error("Nv12YoloPreprocessorD3D12: invalid input size");
    }

    this->create_root_signature();
    this->create_pipeline_state();
    this->create_descriptor_heap();
    this->create_output_tensor_buffer();
    this->create_constant_buffer();
    this->create_output_tensor_uav();
}

Nv12YoloPreprocessorD3D12::~Nv12YoloPreprocessorD3D12()
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

void Nv12YoloPreprocessorD3D12::record_preprocess(
    ID3D12Resource* nv12_texture,
    UINT src_width,
    UINT src_height
)
{
    if (!nv12_texture) {
        throw std::runtime_error("Nv12YoloPreprocessorD3D12::record_preprocess: nv12_texture is null");
    }

    ID3D12GraphicsCommandList* command_list =
        this->d3d12_core_.command_list();

    if (!command_list) {
        throw std::runtime_error("Nv12YoloPreprocessorD3D12::record_preprocess: command_list is null");
    }

    D3D12_RESOURCE_DESC nv12_desc = nv12_texture->GetDesc();

    if (nv12_desc.Format != DXGI_FORMAT_NV12) {
        throw std::runtime_error("Nv12YoloPreprocessorD3D12::record_preprocess: input texture is not DXGI_FORMAT_NV12");
    }

    if (src_width == 0 || src_height == 0) {
        src_width = static_cast<UINT>(nv12_desc.Width);
        src_height = nv12_desc.Height;
    }

    this->create_nv12_srvs(nv12_texture);
    this->update_params(src_width, src_height);

    // D3D11 shared textureをD3D12で読む場合、ここではNV12入力textureのtransitionを入れない。
    // keyed mutex側で所有権を管理し、D3D12内ではSRVとして読むだけにする。
    //
    // 出力bufferはD3D12で作ったbufferなので、状態を明示的に管理する。
    // BufferはCreateCommittedResource時のUNORDERED_ACCESS指定がCOMMON扱いになることがあるため、
    // 初回は COMMON -> UNORDERED_ACCESS が必要。
    this->transition_output_tensor(
        command_list,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    command_list->SetComputeRootSignature(
        this->root_signature_.Get()
    );

    command_list->SetPipelineState(
        this->pipeline_state_.Get()
    );

    ID3D12DescriptorHeap* heaps[] = {
        this->srv_uav_heap_.Get()
    };

    command_list->SetDescriptorHeaps(
        1,
        heaps
    );

    D3D12_GPU_DESCRIPTOR_HANDLE base_gpu =
        this->srv_uav_heap_->GetGPUDescriptorHandleForHeapStart();

    D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = base_gpu;

    D3D12_GPU_DESCRIPTOR_HANDLE uav_gpu = base_gpu;
    uav_gpu.ptr += static_cast<SIZE_T>(2) * this->descriptor_size_;

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

    const UINT group_x = (this->config_.input_width + 15) / 16;
    const UINT group_y = (this->config_.input_height + 15) / 16;

    command_list->Dispatch(
        group_x,
        group_y,
        1
    );

    D3D12_RESOURCE_BARRIER uav_barrier{};
    uav_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    uav_barrier.UAV.pResource = this->output_tensor_buffer_.Get();

    command_list->ResourceBarrier(
        1,
        &uav_barrier
    );
}

void Nv12YoloPreprocessorD3D12::preprocess_and_wait(
    ID3D12Resource* nv12_texture,
    UINT src_width,
    UINT src_height
)
{
    this->d3d12_core_.reset_command_list();

    this->record_preprocess(
        nv12_texture,
        src_width,
        src_height
    );

    this->d3d12_core_.close_execute_and_wait();
}

ID3D12Resource* Nv12YoloPreprocessorD3D12::output_tensor_buffer() const
{
    return this->output_tensor_buffer_.Get();
}

UINT Nv12YoloPreprocessorD3D12::input_width() const
{
    return this->config_.input_width;
}

UINT Nv12YoloPreprocessorD3D12::input_height() const
{
    return this->config_.input_height;
}

size_t Nv12YoloPreprocessorD3D12::tensor_element_count() const
{
    return static_cast<size_t>(3) *
        static_cast<size_t>(this->config_.input_width) *
        static_cast<size_t>(this->config_.input_height);
}

size_t Nv12YoloPreprocessorD3D12::tensor_byte_size() const
{
    return this->tensor_element_count() * sizeof(float);
}

D3D12_RESOURCE_STATES Nv12YoloPreprocessorD3D12::output_tensor_state() const
{
    return this->output_tensor_state_;
}

void Nv12YoloPreprocessorD3D12::transition_output_tensor(
    ID3D12GraphicsCommandList* command_list,
    D3D12_RESOURCE_STATES new_state
)
{
    if (!command_list) {
        throw std::runtime_error("Nv12YoloPreprocessorD3D12::transition_output_tensor: command_list is null");
    }

    if (this->output_tensor_state_ == new_state) {
        return;
    }

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = this->output_tensor_buffer_.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = this->output_tensor_state_;
    barrier.Transition.StateAfter = new_state;

    command_list->ResourceBarrier(
        1,
        &barrier
    );

    this->output_tensor_state_ = new_state;
}

UINT Nv12YoloPreprocessorD3D12::align256(UINT value)
{
    return (value + 255u) & ~255u;
}

void Nv12YoloPreprocessorD3D12::create_root_signature()
{
    D3D12_DESCRIPTOR_RANGE ranges[2]{};

    // t0-t1: Y SRV, UV SRV
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 2;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].RegisterSpace = 0;
    ranges[0].OffsetInDescriptorsFromTableStart =
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    // u0: output tensor UAV
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
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

    Microsoft::WRL::ComPtr<ID3DBlob> signature_blob;
    Microsoft::WRL::ComPtr<ID3DBlob> error_blob;

    HRESULT hr = D3D12SerializeRootSignature(
        &desc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        signature_blob.GetAddressOf(),
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
        signature_blob->GetBufferPointer(),
        signature_blob->GetBufferSize(),
        IID_PPV_ARGS(this->root_signature_.GetAddressOf())
    );

    win_util::ThrowIfFailed(
        hr,
        "CreateRootSignature failed"
    );
}

void Nv12YoloPreprocessorD3D12::create_pipeline_state()
{
    static const char* hlsl = R"(

Texture2D<float>  g_yTex  : register(t0);
Texture2D<float2> g_uvTex : register(t1);

RWStructuredBuffer<float> g_output : register(u0);

cbuffer Params : register(b0)
{
    uint  srcWidth;
    uint  srcHeight;
    uint  dstWidth;
    uint  dstHeight;

    float scale;
    float padX;
    float padY;
    float padValue;

    uint  limitedRangeYuv;
    uint  reserved0;
    uint  reserved1;
    uint  reserved2;
};

float ReadY(int2 p)
{
    p.x = clamp(p.x, 0, int(srcWidth) - 1);
    p.y = clamp(p.y, 0, int(srcHeight) - 1);
    return g_yTex.Load(int3(p, 0));
}

float2 ReadUV(int2 p)
{
    int uvWidth  = int(srcWidth / 2);
    int uvHeight = int(srcHeight / 2);

    p.x = clamp(p.x, 0, uvWidth - 1);
    p.y = clamp(p.y, 0, uvHeight - 1);

    return g_uvTex.Load(int3(p, 0));
}

float BilinearY(float2 src)
{
    float2 p0f = floor(src);
    float2 f = src - p0f;

    int2 p0 = int2(p0f);
    int2 p1 = p0 + int2(1, 0);
    int2 p2 = p0 + int2(0, 1);
    int2 p3 = p0 + int2(1, 1);

    float y0 = ReadY(p0);
    float y1 = ReadY(p1);
    float y2 = ReadY(p2);
    float y3 = ReadY(p3);

    float ya = lerp(y0, y1, f.x);
    float yb = lerp(y2, y3, f.x);

    return lerp(ya, yb, f.y);
}

float2 BilinearUV(float2 src)
{
    float2 uvSrc = src * 0.5f;

    float2 p0f = floor(uvSrc);
    float2 f = uvSrc - p0f;

    int2 p0 = int2(p0f);
    int2 p1 = p0 + int2(1, 0);
    int2 p2 = p0 + int2(0, 1);
    int2 p3 = p0 + int2(1, 1);

    float2 uv0 = ReadUV(p0);
    float2 uv1 = ReadUV(p1);
    float2 uv2 = ReadUV(p2);
    float2 uv3 = ReadUV(p3);

    float2 uva = lerp(uv0, uv1, f.x);
    float2 uvb = lerp(uv2, uv3, f.x);

    return lerp(uva, uvb, f.y);
}

float3 Nv12ToRgb(float y, float2 uv)
{
    float u = uv.x - 0.5f;
    float v = uv.y - 0.5f;

    float r;
    float g;
    float b;

    if (limitedRangeYuv != 0)
    {
        // BT.601 limited range
        float yy = 1.164383f * (y - 16.0f / 255.0f);

        r = yy + 1.596027f * v;
        g = yy - 0.391762f * u - 0.812968f * v;
        b = yy + 2.017232f * u;
    }
    else
    {
        // full range
        r = y + 1.402000f * v;
        g = y - 0.344136f * u - 0.714136f * v;
        b = y + 1.772000f * u;
    }

    return saturate(float3(r, g, b));
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint x = tid.x;
    uint y = tid.y;

    if (x >= dstWidth || y >= dstHeight)
    {
        return;
    }

    float3 rgb = float3(padValue, padValue, padValue);

    float srcX = ((float)x - padX + 0.5f) / scale - 0.5f;
    float srcY = ((float)y - padY + 0.5f) / scale - 0.5f;

    if (srcX >= 0.0f && srcX <= (float)(srcWidth - 1) &&
        srcY >= 0.0f && srcY <= (float)(srcHeight - 1))
    {
        float yv = BilinearY(float2(srcX, srcY));
        float2 uv = BilinearUV(float2(srcX, srcY));
        rgb = Nv12ToRgb(yv, uv);
    }

    uint hw = dstWidth * dstHeight;
    uint index = y * dstWidth + x;

    // NCHW: R plane, G plane, B plane
    g_output[0 * hw + index] = rgb.r;
    g_output[1 * hw + index] = rgb.g;
    g_output[2 * hw + index] = rgb.b;
}

)";

    Microsoft::WRL::ComPtr<ID3DBlob> compute_shader_blob;
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
        compute_shader_blob.GetAddressOf(),
        error_blob.GetAddressOf()
    );

    if (FAILED(hr)) {
        if (error_blob) {
            std::string msg(
                static_cast<const char*>(error_blob->GetBufferPointer()),
                error_blob->GetBufferSize()
            );

            throw std::runtime_error(
                "D3DCompile failed: " + msg
            );
        }

        win_util::ThrowIfFailed(
            hr,
            "D3DCompile failed"
        );
    }

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = this->root_signature_.Get();
    pso_desc.CS.pShaderBytecode =
        compute_shader_blob->GetBufferPointer();
    pso_desc.CS.BytecodeLength =
        compute_shader_blob->GetBufferSize();

    hr = this->d3d12_core_.device()->CreateComputePipelineState(
        &pso_desc,
        IID_PPV_ARGS(this->pipeline_state_.GetAddressOf())
    );

    win_util::ThrowIfFailed(
        hr,
        "CreateComputePipelineState failed"
    );
}

void Nv12YoloPreprocessorD3D12::create_descriptor_heap()
{
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 3; // Y SRV, UV SRV, output UAV
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    desc.NodeMask = 0;

    HRESULT hr = this->d3d12_core_.device()->CreateDescriptorHeap(
        &desc,
        IID_PPV_ARGS(this->srv_uav_heap_.GetAddressOf())
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

void Nv12YoloPreprocessorD3D12::create_output_tensor_buffer()
{
    const UINT64 byte_size =
        static_cast<UINT64>(this->tensor_element_count()) *
        sizeof(float);

    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

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
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    // Bufferは初期状態にUNORDERED_ACCESSを指定してもCOMMON扱いになる環境があるため、
    // 状態メンバはCOMMONで開始する。
    this->output_tensor_buffer_ =
        this->d3d12_core_.create_committed_resource(
            heap_props,
            D3D12_HEAP_FLAG_NONE,
            desc,
            D3D12_RESOURCE_STATE_COMMON,
            nullptr
        );

    this->output_tensor_state_ = D3D12_RESOURCE_STATE_COMMON;
}

void Nv12YoloPreprocessorD3D12::create_constant_buffer()
{
    const UINT cb_size = align256(
        static_cast<UINT>(sizeof(ShaderParams))
    );

    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_UPLOAD;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = cb_size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_NONE;

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
        "Map constant buffer failed"
    );
}

void Nv12YoloPreprocessorD3D12::create_output_tensor_uav()
{
    D3D12_CPU_DESCRIPTOR_HANDLE base_cpu =
        this->srv_uav_heap_->GetCPUDescriptorHandleForHeapStart();

    D3D12_CPU_DESCRIPTOR_HANDLE uav_cpu = base_cpu;
    uav_cpu.ptr += static_cast<SIZE_T>(2) * this->descriptor_size_;

    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc{};
    uav_desc.Format = DXGI_FORMAT_UNKNOWN;
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.FirstElement = 0;
    uav_desc.Buffer.NumElements =
        static_cast<UINT>(this->tensor_element_count());
    uav_desc.Buffer.StructureByteStride = sizeof(float);
    uav_desc.Buffer.CounterOffsetInBytes = 0;
    uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

    this->d3d12_core_.device()->CreateUnorderedAccessView(
        this->output_tensor_buffer_.Get(),
        nullptr,
        &uav_desc,
        uav_cpu
    );
}

void Nv12YoloPreprocessorD3D12::create_nv12_srvs(
    ID3D12Resource* nv12_texture
)
{
    D3D12_CPU_DESCRIPTOR_HANDLE base_cpu =
        this->srv_uav_heap_->GetCPUDescriptorHandleForHeapStart();

    D3D12_CPU_DESCRIPTOR_HANDLE y_cpu = base_cpu;

    D3D12_CPU_DESCRIPTOR_HANDLE uv_cpu = base_cpu;
    uv_cpu.ptr += this->descriptor_size_;

    D3D12_SHADER_RESOURCE_VIEW_DESC y_desc{};
    y_desc.Shader4ComponentMapping =
        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    y_desc.Format = DXGI_FORMAT_R8_UNORM;
    y_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    y_desc.Texture2D.MostDetailedMip = 0;
    y_desc.Texture2D.MipLevels = 1;
    y_desc.Texture2D.PlaneSlice = 0;
    y_desc.Texture2D.ResourceMinLODClamp = 0.0f;

    D3D12_SHADER_RESOURCE_VIEW_DESC uv_desc{};
    uv_desc.Shader4ComponentMapping =
        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    uv_desc.Format = DXGI_FORMAT_R8G8_UNORM;
    uv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    uv_desc.Texture2D.MostDetailedMip = 0;
    uv_desc.Texture2D.MipLevels = 1;
    uv_desc.Texture2D.PlaneSlice = 1;
    uv_desc.Texture2D.ResourceMinLODClamp = 0.0f;

    this->d3d12_core_.device()->CreateShaderResourceView(
        nv12_texture,
        &y_desc,
        y_cpu
    );

    this->d3d12_core_.device()->CreateShaderResourceView(
        nv12_texture,
        &uv_desc,
        uv_cpu
    );
}

void Nv12YoloPreprocessorD3D12::update_params(
    UINT src_width,
    UINT src_height
)
{
    const float dst_w =
        static_cast<float>(this->config_.input_width);
    const float dst_h =
        static_cast<float>(this->config_.input_height);
    const float src_w =
        static_cast<float>(src_width);
    const float src_h =
        static_cast<float>(src_height);

    const float scale =
        std::min(dst_w / src_w, dst_h / src_h);

    const float resized_w =
        std::round(src_w * scale);
    const float resized_h =
        std::round(src_h * scale);

    const float pad_x =
        (dst_w - resized_w) * 0.5f;
    const float pad_y =
        (dst_h - resized_h) * 0.5f;

    ShaderParams params{};
    params.src_width = src_width;
    params.src_height = src_height;
    params.dst_width = this->config_.input_width;
    params.dst_height = this->config_.input_height;
    params.scale = scale;
    params.pad_x = pad_x;
    params.pad_y = pad_y;
    params.pad_value = this->config_.pad_value;
    params.limited_range_yuv =
        this->config_.limited_range_yuv ? 1u : 0u;

    std::memcpy(
        this->mapped_constants_,
        &params,
        sizeof(params)
    );
}

D3D12_RESOURCE_STATES& Nv12YoloPreprocessorD3D12::output_tensor_state_ref()
{
    return this->output_tensor_state_;
}
