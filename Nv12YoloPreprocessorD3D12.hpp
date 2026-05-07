#pragma once

#define NOMINMAX

#include <windows.h>
#include <wrl/client.h>
#include <comdef.h>

#include <d3d11.h>
#include <d3d10.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <thread>

#include "util.hpp"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "d3d12.lib")

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

class Nv12YoloPreprocessorD3D12 {
public:
    struct Config {
        UINT inputWidth = 640;
        UINT inputHeight = 640;
        bool limitedRangeYuv = true;
        float padValue = 114.0f / 255.0f;
    };

    Nv12YoloPreprocessorD3D12(
        ID3D12Device* device12,
        const Config& config = Config()
    )
        : m_device12(device12)
        , m_config(config)
    {
        if (!m_device12) {
            throw std::runtime_error("Nv12YoloPreprocessorD3D12: device12 is null");
        }

        CreateRootSignature();
        CreatePipelineState();
        CreateDescriptorHeap();
        CreateOutputTensorBuffer();
        CreateConstantBuffer();
        CreateOutputTensorUav();
    }

    void Preprocess(
        ID3D12GraphicsCommandList* commandList,
        ID3D12Resource* nv12Texture12,
        UINT srcWidth,
        UINT srcHeight
    )
    {
        if (!commandList || !nv12Texture12) {
            throw std::runtime_error("Nv12YoloPreprocessorD3D12::Preprocess: null argument");
        }

        D3D12_RESOURCE_DESC nv12Desc = nv12Texture12->GetDesc();

        if (nv12Desc.Format != DXGI_FORMAT_NV12) {
            throw std::runtime_error("Nv12YoloPreprocessorD3D12::Preprocess: input is not NV12");
        }

        if (srcWidth == 0 || srcHeight == 0) {
            srcWidth = static_cast<UINT>(nv12Desc.Width);
            srcHeight = nv12Desc.Height;
        }

        CreateNv12Srvs(nv12Texture12);
        UpdateParams(srcWidth, srcHeight);

        // D3D11側からの共有resourceは、D3D12側ではCOMMONから読む状態へ遷移させる。
        //Transition(
        //    commandList,
        //    nv12Texture12,
        //    D3D12_RESOURCE_STATE_COMMON,
        //    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        //);

        commandList->SetComputeRootSignature(m_rootSignature.Get());
        commandList->SetPipelineState(m_pipelineState.Get());

        ID3D12DescriptorHeap* heaps[] = {m_srvUavHeap.Get()};
        commandList->SetDescriptorHeaps(1, heaps);

        D3D12_GPU_DESCRIPTOR_HANDLE baseGpu =
            m_srvUavHeap->GetGPUDescriptorHandleForHeapStart();

        D3D12_GPU_DESCRIPTOR_HANDLE srvGpu = baseGpu;

        D3D12_GPU_DESCRIPTOR_HANDLE uavGpu = baseGpu;
        uavGpu.ptr += static_cast<SIZE_T>(2) * m_descriptorSize;

        commandList->SetComputeRootDescriptorTable(0, srvGpu);
        commandList->SetComputeRootDescriptorTable(1, uavGpu);
        commandList->SetComputeRootConstantBufferView(
            2,
            m_constantBuffer->GetGPUVirtualAddress()
        );

        const UINT groupX = (m_config.inputWidth + 15) / 16;
        const UINT groupY = (m_config.inputHeight + 15) / 16;

        commandList->Dispatch(groupX, groupY, 1);

        // UAV書き込み完了順序を保証
        D3D12_RESOURCE_BARRIER uavBarrier{};
        uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uavBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        uavBarrier.UAV.pResource = m_outputTensorBuffer.Get();
        commandList->ResourceBarrier(1, &uavBarrier);

        // 次にD3D11側がまた書けるようにCOMMONへ戻しておく。
        //Transition(
        //    commandList,
        //    nv12Texture12,
        //    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        //    D3D12_RESOURCE_STATE_COMMON
        //);
    }

    ID3D12Resource* GetOutputTensorBuffer() const
    {
        return m_outputTensorBuffer.Get();
    }

    UINT GetInputWidth() const
    {
        return m_config.inputWidth;
    }

    UINT GetInputHeight() const
    {
        return m_config.inputHeight;
    }

    size_t GetTensorElementCount() const
    {
        return static_cast<size_t>(3) *
            static_cast<size_t>(m_config.inputWidth) *
            static_cast<size_t>(m_config.inputHeight);
    }

    size_t GetTensorByteSize() const
    {
        return GetTensorElementCount() * sizeof(float);
    }

private:
    struct ShaderParams {
        UINT srcWidth;
        UINT srcHeight;
        UINT dstWidth;
        UINT dstHeight;

        float scale;
        float padX;
        float padY;
        float padValue;

        UINT limitedRangeYuv;
        UINT reserved0;
        UINT reserved1;
        UINT reserved2;
    };

    static UINT Align256(UINT size)
    {
        return (size + 255u) & ~255u;
    }

    static void Transition(
        ID3D12GraphicsCommandList* commandList,
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES before,
        D3D12_RESOURCE_STATES after
    )
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;

        commandList->ResourceBarrier(1, &barrier);
    }

    void CreateRootSignature()
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

        ComPtr<ID3DBlob> sigBlob;
        ComPtr<ID3DBlob> errBlob;

        HRESULT hr = D3D12SerializeRootSignature(
            &desc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            sigBlob.GetAddressOf(),
            errBlob.GetAddressOf()
        );

        if (FAILED(hr)) {
            if (errBlob) {
                std::string msg(
                    static_cast<const char*>(errBlob->GetBufferPointer()),
                    errBlob->GetBufferSize()
                );
                throw std::runtime_error("D3D12SerializeRootSignature failed: " + msg);
            }
            ThrowIfFailed(hr, "D3D12SerializeRootSignature failed");
        }

        hr = m_device12->CreateRootSignature(
            0,
            sigBlob->GetBufferPointer(),
            sigBlob->GetBufferSize(),
            IID_PPV_ARGS(m_rootSignature.GetAddressOf())
        );
        ThrowIfFailed(hr, "CreateRootSignature failed");
    }

    void CreatePipelineState()
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
        //rgb=float3(yv, yv, yv);
    }

    uint hw = dstWidth * dstHeight;
    uint index = y * dstWidth + x;

    g_output[0 * hw + index] = rgb.r;
    g_output[1 * hw + index] = rgb.g;
    g_output[2 * hw + index] = rgb.b;
}

)";

        ComPtr<ID3DBlob> csBlob;
        ComPtr<ID3DBlob> errorBlob;

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
            csBlob.GetAddressOf(),
            errorBlob.GetAddressOf()
        );

        if (FAILED(hr)) {
            if (errorBlob) {
                std::string msg(
                    static_cast<const char*>(errorBlob->GetBufferPointer()),
                    errorBlob->GetBufferSize()
                );
                throw std::runtime_error("D3DCompile failed: " + msg);
            }
            ThrowIfFailed(hr, "D3DCompile failed");
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = m_rootSignature.Get();
        psoDesc.CS.pShaderBytecode = csBlob->GetBufferPointer();
        psoDesc.CS.BytecodeLength = csBlob->GetBufferSize();

        hr = m_device12->CreateComputePipelineState(
            &psoDesc,
            IID_PPV_ARGS(m_pipelineState.GetAddressOf())
        );
        ThrowIfFailed(hr, "CreateComputePipelineState failed");
    }

    void CreateDescriptorHeap()
    {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.NumDescriptors = 3; // Y SRV, UV SRV, output UAV
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        desc.NodeMask = 0;

        HRESULT hr = m_device12->CreateDescriptorHeap(
            &desc,
            IID_PPV_ARGS(m_srvUavHeap.GetAddressOf())
        );
        ThrowIfFailed(hr, "CreateDescriptorHeap failed");

        m_descriptorSize = m_device12->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        );
    }

    void CreateOutputTensorBuffer()
    {
        const UINT64 byteSize =
            static_cast<UINT64>(GetTensorElementCount()) * sizeof(float);

        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapProps.CreationNodeMask = 1;
        heapProps.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Alignment = 0;
        desc.Width = byteSize;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        HRESULT hr = m_device12->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(m_outputTensorBuffer.GetAddressOf())
        );
        ThrowIfFailed(hr, "Create output tensor buffer D3D12 failed");
    }

    void CreateConstantBuffer()
    {
        const UINT cbSize = Align256(sizeof(ShaderParams));

        D3D12_HEAP_PROPERTIES heapProps{};
        heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
        heapProps.CreationNodeMask = 1;
        heapProps.VisibleNodeMask = 1;

        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Alignment = 0;
        desc.Width = cbSize;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_NONE;

        HRESULT hr = m_device12->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(m_constantBuffer.GetAddressOf())
        );
        ThrowIfFailed(hr, "Create constant buffer D3D12 failed");

        D3D12_RANGE range{};
        range.Begin = 0;
        range.End = 0;

        hr = m_constantBuffer->Map(
            0,
            &range,
            reinterpret_cast<void**>(&m_mappedConstants)
        );
        ThrowIfFailed(hr, "Map constant buffer D3D12 failed");
    }

    void CreateNv12Srvs(ID3D12Resource* nv12Texture12)
    {
        D3D12_CPU_DESCRIPTOR_HANDLE baseCpu =
            m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();

        D3D12_CPU_DESCRIPTOR_HANDLE yCpu = baseCpu;

        D3D12_CPU_DESCRIPTOR_HANDLE uvCpu = baseCpu;
        uvCpu.ptr += m_descriptorSize;

        D3D12_SHADER_RESOURCE_VIEW_DESC yDesc{};
        yDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        yDesc.Format = DXGI_FORMAT_R8_UNORM;
        yDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        yDesc.Texture2D.MostDetailedMip = 0;
        yDesc.Texture2D.MipLevels = 1;
        yDesc.Texture2D.PlaneSlice = 0;
        yDesc.Texture2D.ResourceMinLODClamp = 0.0f;

        D3D12_SHADER_RESOURCE_VIEW_DESC uvDesc{};
        uvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        uvDesc.Format = DXGI_FORMAT_R8G8_UNORM;
        uvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        uvDesc.Texture2D.MostDetailedMip = 0;
        uvDesc.Texture2D.MipLevels = 1;
        uvDesc.Texture2D.PlaneSlice = 1;
        uvDesc.Texture2D.ResourceMinLODClamp = 0.0f;

        m_device12->CreateShaderResourceView(
            nv12Texture12,
            &yDesc,
            yCpu
        );

        m_device12->CreateShaderResourceView(
            nv12Texture12,
            &uvDesc,
            uvCpu
        );
    }

    void CreateOutputTensorUav()
    {
        D3D12_CPU_DESCRIPTOR_HANDLE baseCpu =
            m_srvUavHeap->GetCPUDescriptorHandleForHeapStart();

        D3D12_CPU_DESCRIPTOR_HANDLE uavCpu = baseCpu;
        uavCpu.ptr += static_cast<SIZE_T>(2) * m_descriptorSize;

        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.FirstElement = 0;
        uavDesc.Buffer.NumElements = static_cast<UINT>(GetTensorElementCount());
        uavDesc.Buffer.StructureByteStride = sizeof(float);
        uavDesc.Buffer.CounterOffsetInBytes = 0;
        uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;

        m_device12->CreateUnorderedAccessView(
            m_outputTensorBuffer.Get(),
            nullptr,
            &uavDesc,
            uavCpu
        );
    }

    void UpdateParams(UINT srcWidth, UINT srcHeight)
    {
        const float dstW = static_cast<float>(m_config.inputWidth);
        const float dstH = static_cast<float>(m_config.inputHeight);
        const float srcW = static_cast<float>(srcWidth);
        const float srcH = static_cast<float>(srcHeight);

        const float scale = std::min(dstW / srcW, dstH / srcH);
        const float resizedW = std::round(srcW * scale);
        const float resizedH = std::round(srcH * scale);

        const float padX = (dstW - resizedW) * 0.5f;
        const float padY = (dstH - resizedH) * 0.5f;

        ShaderParams params{};
        params.srcWidth = srcWidth;
        params.srcHeight = srcHeight;
        params.dstWidth = m_config.inputWidth;
        params.dstHeight = m_config.inputHeight;
        params.scale = scale;
        params.padX = padX;
        params.padY = padY;
        params.padValue = m_config.padValue;
        params.limitedRangeYuv = m_config.limitedRangeYuv ? 1u : 0u;

        std::memcpy(m_mappedConstants, &params, sizeof(params));
    }

private:
    ComPtr<ID3D12Device> m_device12;
    Config m_config;

    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pipelineState;

    ComPtr<ID3D12DescriptorHeap> m_srvUavHeap;
    UINT m_descriptorSize = 0;

    ComPtr<ID3D12Resource> m_outputTensorBuffer;
    ComPtr<ID3D12Resource> m_constantBuffer;
    uint8_t* m_mappedConstants = nullptr;
};