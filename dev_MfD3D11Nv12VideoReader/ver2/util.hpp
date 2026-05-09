#pragma once

#pragma once

#define NOMINMAX
#include <windows.h>
#include <wrl/client.h>
#include <d3d11.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cstdint>
#include <algorithm>

#include "HResultUtil.hpp"

using Microsoft::WRL::ComPtr;

inline uint8_t Float01ToU8(float v)
{
    v = std::clamp(v, 0.0f, 1.0f);
    return static_cast<uint8_t>(v * 255.0f + 0.5f);
}

// ID3D11Buffer に入っている YOLO 入力テンソル NCHW float32 を BMP として保存する。
// buffer layout:
//   float tensor[3 * height * width]
//   R: tensor[0 * H * W + y * W + x]
//   G: tensor[1 * H * W + y * W + x]
//   B: tensor[2 * H * W + y * W + x]
inline void SaveNchwFloatTensorBufferAsBmp(
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    ID3D11Buffer* tensorBuffer,
    UINT width,
    UINT height,
    const wchar_t* outputPath
) {
    if (!device) {
        throw std::runtime_error("device is null");
    }
    if (!context) {
        throw std::runtime_error("context is null");
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

    const size_t hw = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t requiredBytes = hw * 3 * sizeof(float);

    D3D11_BUFFER_DESC srcDesc{};
    tensorBuffer->GetDesc(&srcDesc);

    if (srcDesc.ByteWidth < requiredBytes) {
        throw std::runtime_error("tensorBuffer is smaller than 3 * H * W * sizeof(float)");
    }

    // CopyResource は buffer サイズが一致している方が安全なので、
    // staging buffer は source buffer と同じ ByteWidth で作る。
    D3D11_BUFFER_DESC stagingDesc{};
    stagingDesc.ByteWidth = srcDesc.ByteWidth;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.BindFlags = 0;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    stagingDesc.MiscFlags = 0;
    stagingDesc.StructureByteStride = 0;

    ComPtr<ID3D11Buffer> stagingBuffer;

    HRESULT hr = device->CreateBuffer(
        &stagingDesc,
        nullptr,
        stagingBuffer.GetAddressOf()
    );
    win_util::ThrowIfFailed(hr, "Create staging buffer failed");

    context->CopyResource(
        stagingBuffer.Get(),
        tensorBuffer
    );

    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = context->Map(
        stagingBuffer.Get(),
        0,
        D3D11_MAP_READ,
        0,
        &mapped
    );
    win_util::ThrowIfFailed(hr, "Map staging buffer failed");

    const float* tensor = static_cast<const float*>(mapped.pData);

    // BMP: 24-bit BGR
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

    // height を負にすると top-down BMP になる。
    // これにより y=0 が画像上端として保存される。
    writeI32(22, -static_cast<int32_t>(height));

    write16(26, 1);    // planes
    write16(28, 24);   // bits per pixel
    write32(30, 0);    // BI_RGB
    write32(34, imageSize);
    writeI32(38, 0);   // x pixels per meter
    writeI32(42, 0);   // y pixels per meter
    write32(46, 0);    // colors used
    write32(50, 0);    // important colors

    uint8_t* dstBase = bmp.data() + pixelOffset;

    const float* rPlane = tensor + 0 * hw;
    const float* gPlane = tensor + 1 * hw;
    const float* bPlane = tensor + 2 * hw;

    for (UINT y = 0; y < height; ++y) {
        uint8_t* dstRow = dstBase + static_cast<size_t>(y) * bmpRowStridePadded;

        for (UINT x = 0; x < width; ++x) {
            const size_t idx =
                static_cast<size_t>(y) * static_cast<size_t>(width) +
                static_cast<size_t>(x);

            const uint8_t r = Float01ToU8(rPlane[idx]);
            const uint8_t g = Float01ToU8(gPlane[idx]);
            const uint8_t b = Float01ToU8(bPlane[idx]);

            // BMP は BGR 順
            dstRow[x * 3 + 0] = b;
            dstRow[x * 3 + 1] = g;
            dstRow[x * 3 + 2] = r;
        }
    }

    context->Unmap(stagingBuffer.Get(), 0);

    FILE* fp = nullptr;
    errno_t e = _wfopen_s(&fp, outputPath, L"wb");
    if (e != 0 || !fp) {
        throw std::runtime_error("failed to open output BMP file");
    }

    fwrite(bmp.data(), 1, bmp.size(), fp);
    fclose(fp);
}