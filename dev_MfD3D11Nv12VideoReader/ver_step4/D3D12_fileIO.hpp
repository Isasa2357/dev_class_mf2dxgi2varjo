#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

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

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "d3d12.lib")

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

static UINT64 AlignTo256(UINT64 v)
{
    return (v + 255ull) & ~255ull;
}

static void SaveNchwFloatTensorD3D12BufferAsBmp(
    ID3D12Device* device12,
    ID3D12CommandQueue* queue12,
    ID3D12CommandAllocator* allocator12,
    ID3D12GraphicsCommandList* commandList12,
    ID3D12Resource* tensorBuffer,
    UINT width,
    UINT height,
    const wchar_t* outputPath
)
{
    if (!device12) {
        throw std::runtime_error("device12 is null");
    }
    if (!queue12) {
        throw std::runtime_error("queue12 is null");
    }
    if (!allocator12) {
        throw std::runtime_error("allocator12 is null");
    }
    if (!commandList12) {
        throw std::runtime_error("commandList12 is null");
    }
    if (!tensorBuffer) {
        throw std::runtime_error("tensorBuffer is null");
    }
    if (width == 0 || height == 0) {
        throw std::runtime_error("invalid tensor image size");
    }
    if (!outputPath) {
        throw std::runtime_error("outputPath is null");
    }

    const UINT64 hw = static_cast<UINT64>(width) * static_cast<UINT64>(height);
    const UINT64 tensorBytes = hw * 3ull * sizeof(float);

    D3D12_RESOURCE_DESC srcDesc = tensorBuffer->GetDesc();
    if (srcDesc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
        throw std::runtime_error("tensorBuffer is not a buffer");
    }
    if (srcDesc.Width < tensorBytes) {
        throw std::runtime_error("tensorBuffer is smaller than 3 * H * W * sizeof(float)");
    }

    // readback buffer
    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC readbackDesc{};
    readbackDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    readbackDesc.Alignment = 0;
    readbackDesc.Width = tensorBytes;
    readbackDesc.Height = 1;
    readbackDesc.DepthOrArraySize = 1;
    readbackDesc.MipLevels = 1;
    readbackDesc.Format = DXGI_FORMAT_UNKNOWN;
    readbackDesc.SampleDesc.Count = 1;
    readbackDesc.SampleDesc.Quality = 0;
    readbackDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    readbackDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    ComPtr<ID3D12Resource> readbackBuffer;

    HRESULT hr = device12->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &readbackDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(readbackBuffer.GetAddressOf())
    );
    win_util::ThrowIfFailed(hr, "Create readback buffer failed");

    // command list reset
    hr = allocator12->Reset();
    win_util::ThrowIfFailed(hr, "Save BMP: command allocator Reset failed");

    hr = commandList12->Reset(allocator12, nullptr);
    win_util::ThrowIfFailed(hr, "Save BMP: command list Reset failed");

    // tensorBuffer は compute shader の UAV出力直後なら UNORDERED_ACCESS 状態の想定。
    // CopySource に遷移させる。
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = tensorBuffer;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        commandList12->ResourceBarrier(1, &barrier);
    }

    commandList12->CopyBufferRegion(
        readbackBuffer.Get(),
        0,
        tensorBuffer,
        0,
        tensorBytes
    );

    // 後続の ORT/DML などで UAV として再利用する想定なら戻しておく
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = tensorBuffer;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        commandList12->ResourceBarrier(1, &barrier);
    }

    hr = commandList12->Close();
    win_util::ThrowIfFailed(hr, "Save BMP: command list Close failed");

    ID3D12CommandList* lists[] = {commandList12};
    queue12->ExecuteCommandLists(1, lists);

    // WaitForD3D12Queue(device12, queue12);

    // Map readback buffer
    D3D12_RANGE readRange{};
    readRange.Begin = 0;
    readRange.End = static_cast<SIZE_T>(tensorBytes);

    void* mapped = nullptr;
    hr = readbackBuffer->Map(0, &readRange, &mapped);
    win_util::ThrowIfFailed(hr, "Map readback buffer failed");

    const float* tensor = static_cast<const float*>(mapped);

    // BMP 24bit BGR, top-down
    const UINT bmpRowStride = width * 3;
    const UINT bmpRowStridePadded = (bmpRowStride + 3u) & ~3u;
    const UINT imageSize = bmpRowStridePadded * height;

    constexpr UINT fileHeaderSize = 14;
    constexpr UINT infoHeaderSize = 40;
    const UINT pixelOffset = fileHeaderSize + infoHeaderSize;
    const UINT fileSize = pixelOffset + imageSize;

    std::vector<uint8_t> bmp(fileSize, 0);

    auto write16 = [&] (size_t offset, uint16_t v) {
        bmp[offset + 0] = static_cast<uint8_t>(v & 0xff);
        bmp[offset + 1] = static_cast<uint8_t>((v >> 8) & 0xff);
        };

    auto write32 = [&] (size_t offset, uint32_t v) {
        bmp[offset + 0] = static_cast<uint8_t>(v & 0xff);
        bmp[offset + 1] = static_cast<uint8_t>((v >> 8) & 0xff);
        bmp[offset + 2] = static_cast<uint8_t>((v >> 16) & 0xff);
        bmp[offset + 3] = static_cast<uint8_t>((v >> 24) & 0xff);
        };

    auto writeI32 = [&] (size_t offset, int32_t v) {
        write32(offset, static_cast<uint32_t>(v));
        };

    // BITMAPFILEHEADER
    bmp[0] = 'B';
    bmp[1] = 'M';
    write32(2, fileSize);
    write16(6, 0);
    write16(8, 0);
    write32(10, pixelOffset);

    // BITMAPINFOHEADER
    write32(14, infoHeaderSize);
    writeI32(18, static_cast<int32_t>(width));

    // 負のheightでtop-down BMP。tensorの y=0 が画像上端になる。
    writeI32(22, -static_cast<int32_t>(height));

    write16(26, 1);     // planes
    write16(28, 24);    // bpp
    write32(30, 0);     // BI_RGB
    write32(34, imageSize);
    writeI32(38, 0);
    writeI32(42, 0);
    write32(46, 0);
    write32(50, 0);

    uint8_t* dstBase = bmp.data() + pixelOffset;

    const float* rPlane = tensor + 0 * hw;
    const float* gPlane = tensor + 1 * hw;
    const float* bPlane = tensor + 2 * hw;

    for (UINT y = 0; y < height; ++y) {
        uint8_t* dstRow =
            dstBase + static_cast<size_t>(y) * bmpRowStridePadded;

        for (UINT x = 0; x < width; ++x) {
            const UINT64 idx =
                static_cast<UINT64>(y) * static_cast<UINT64>(width) +
                static_cast<UINT64>(x);

            const uint8_t r = Float01ToU8(rPlane[idx]);
            const uint8_t g = Float01ToU8(gPlane[idx]);
            const uint8_t b = Float01ToU8(bPlane[idx]);

            // BMPはBGR順
            dstRow[x * 3 + 0] = b;
            dstRow[x * 3 + 1] = g;
            dstRow[x * 3 + 2] = r;
        }
    }

    D3D12_RANGE writtenRange{};
    writtenRange.Begin = 0;
    writtenRange.End = 0;
    readbackBuffer->Unmap(0, &writtenRange);

    FILE* fp = nullptr;
    errno_t e = _wfopen_s(&fp, outputPath, L"wb");
    if (e != 0 || !fp) {
        throw std::runtime_error("failed to open output BMP file");
    }

    fwrite(bmp.data(), 1, bmp.size(), fp);
    fclose(fp);
}