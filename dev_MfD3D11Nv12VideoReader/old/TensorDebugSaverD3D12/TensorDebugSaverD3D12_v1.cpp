#include "TensorDebugSaverD3D12.hpp"

#include <vector>
#include <stdexcept>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cwchar>

#include "HResultUtil.hpp"

namespace {

    uint8_t Float01ToU8(float v)
    {
        v = std::clamp(v, 0.0f, 1.0f);
        return static_cast<uint8_t>(v * 255.0f + 0.5f);
    }

    void TransitionResource(
        ID3D12GraphicsCommandList* command_list,
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES before,
        D3D12_RESOURCE_STATES after
    )
    {
        if (before == after) {
            return;
        }

        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;

        command_list->ResourceBarrier(1, &barrier);
    }

    void SetPixelRgb(
        std::vector<uint8_t>& rgb,
        UINT width,
        UINT height,
        int x,
        int y,
        uint8_t r,
        uint8_t g,
        uint8_t b
    )
    {
        if (x < 0 || y < 0) {
            return;
        }

        if (x >= static_cast<int>(width) ||
            y >= static_cast<int>(height))
        {
            return;
        }

        const size_t idx =
            (static_cast<size_t>(y) * static_cast<size_t>(width) +
                static_cast<size_t>(x)) * 3;

        rgb[idx + 0] = r;
        rgb[idx + 1] = g;
        rgb[idx + 2] = b;
    }

    void DrawRectRgb(
        std::vector<uint8_t>& rgb,
        UINT width,
        UINT height,
        int x1,
        int y1,
        int x2,
        int y2,
        uint8_t r,
        uint8_t g,
        uint8_t b,
        int thickness
    )
    {
        if (x1 > x2) {
            std::swap(x1, x2);
        }
        if (y1 > y2) {
            std::swap(y1, y2);
        }

        x1 = std::clamp(x1, 0, static_cast<int>(width) - 1);
        x2 = std::clamp(x2, 0, static_cast<int>(width) - 1);
        y1 = std::clamp(y1, 0, static_cast<int>(height) - 1);
        y2 = std::clamp(y2, 0, static_cast<int>(height) - 1);

        for (int t = 0; t < thickness; ++t) {
            for (int x = x1; x <= x2; ++x) {
                SetPixelRgb(rgb, width, height, x, y1 + t, r, g, b);
                SetPixelRgb(rgb, width, height, x, y2 - t, r, g, b);
            }

            for (int y = y1; y <= y2; ++y) {
                SetPixelRgb(rgb, width, height, x1 + t, y, r, g, b);
                SetPixelRgb(rgb, width, height, x2 - t, y, r, g, b);
            }
        }
    }

    void DrawSmallMarkerRgb(
        std::vector<uint8_t>& rgb,
        UINT width,
        UINT height,
        int cx,
        int cy,
        uint8_t r,
        uint8_t g,
        uint8_t b
    )
    {
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                SetPixelRgb(rgb, width, height, cx + dx, cy + dy, r, g, b);
            }
        }
    }

    void WriteRgbAsTopDownBmp(
        const std::vector<uint8_t>& rgb,
        UINT width,
        UINT height,
        const wchar_t* output_path
    )
    {
        if (!output_path) {
            throw std::runtime_error("output_path is null");
        }

        const UINT bmp_row_stride = width * 3;
        const UINT bmp_row_stride_padded = (bmp_row_stride + 3u) & ~3u;
        const UINT image_size = bmp_row_stride_padded * height;

        constexpr UINT file_header_size = 14;
        constexpr UINT info_header_size = 40;
        constexpr UINT pixel_offset = file_header_size + info_header_size;
        const UINT file_size = pixel_offset + image_size;

        std::vector<uint8_t> bmp(file_size, 0);

        auto write16 = [&](size_t offset, uint16_t v) {
            bmp[offset + 0] = static_cast<uint8_t>(v & 0xff);
            bmp[offset + 1] = static_cast<uint8_t>((v >> 8) & 0xff);
            };

        auto write32 = [&](size_t offset, uint32_t v) {
            bmp[offset + 0] = static_cast<uint8_t>(v & 0xff);
            bmp[offset + 1] = static_cast<uint8_t>((v >> 8) & 0xff);
            bmp[offset + 2] = static_cast<uint8_t>((v >> 16) & 0xff);
            bmp[offset + 3] = static_cast<uint8_t>((v >> 24) & 0xff);
            };

        auto write_i32 = [&](size_t offset, int32_t v) {
            write32(offset, static_cast<uint32_t>(v));
            };

        bmp[0] = 'B';
        bmp[1] = 'M';
        write32(2, file_size);
        write16(6, 0);
        write16(8, 0);
        write32(10, pixel_offset);

        write32(14, info_header_size);
        write_i32(18, static_cast<int32_t>(width));
        write_i32(22, -static_cast<int32_t>(height)); // top-down
        write16(26, 1);
        write16(28, 24);
        write32(30, 0);
        write32(34, image_size);
        write_i32(38, 0);
        write_i32(42, 0);
        write32(46, 0);
        write32(50, 0);

        uint8_t* dst_base = bmp.data() + pixel_offset;

        for (UINT y = 0; y < height; ++y) {
            uint8_t* dst_row =
                dst_base + static_cast<size_t>(y) * bmp_row_stride_padded;

            const uint8_t* src_row =
                rgb.data() + static_cast<size_t>(y) * static_cast<size_t>(width) * 3;

            for (UINT x = 0; x < width; ++x) {
                const uint8_t r = src_row[x * 3 + 0];
                const uint8_t g = src_row[x * 3 + 1];
                const uint8_t b = src_row[x * 3 + 2];

                // BMP は BGR
                dst_row[x * 3 + 0] = b;
                dst_row[x * 3 + 1] = g;
                dst_row[x * 3 + 2] = r;
            }
        }

        FILE* fp = nullptr;

        errno_t e = _wfopen_s(
            &fp,
            output_path,
            L"wb"
        );

        if (e != 0 || !fp) {
            throw std::runtime_error("failed to open output BMP file");
        }

        fwrite(bmp.data(), 1, bmp.size(), fp);
        fclose(fp);
    }

    void GetClassColorRgb(
        uint32_t class_id,
        uint8_t& r,
        uint8_t& g,
        uint8_t& b
    )
    {
        // 見分けやすい固定パレット。
        // class_id が増えても modulo で繰り返す。
        static constexpr uint8_t palette[][3] = {
            {255,   0,   0}, // red
            {  0, 255,   0}, // green
            {  0, 128, 255}, // blue/cyan
            {255, 255,   0}, // yellow
            {255,   0, 255}, // magenta
            {  0, 255, 255}, // cyan
            {255, 128,   0}, // orange
            {128,   0, 255}, // purple
            {255, 128, 128},
            {128, 255, 128},
            {128, 128, 255},
            {255, 255, 128},
        };

        constexpr size_t palette_size =
            sizeof(palette) / sizeof(palette[0]);

        const size_t idx =
            static_cast<size_t>(class_id) % palette_size;

        r = palette[idx][0];
        g = palette[idx][1];
        b = palette[idx][2];
    }

    void BlendPixelRgb(
        std::vector<uint8_t>& rgb,
        UINT width,
        UINT height,
        int x,
        int y,
        uint8_t overlay_r,
        uint8_t overlay_g,
        uint8_t overlay_b,
        float alpha
    )
    {
        if (x < 0 || y < 0) {
            return;
        }

        if (x >= static_cast<int>(width) ||
            y >= static_cast<int>(height))
        {
            return;
        }

        alpha = std::clamp(alpha, 0.0f, 1.0f);

        const size_t idx =
            (static_cast<size_t>(y) * static_cast<size_t>(width) +
                static_cast<size_t>(x)) * 3;

        const float inv_alpha = 1.0f - alpha;

        const float src_r = static_cast<float>(rgb[idx + 0]);
        const float src_g = static_cast<float>(rgb[idx + 1]);
        const float src_b = static_cast<float>(rgb[idx + 2]);

        rgb[idx + 0] =
            static_cast<uint8_t>(
                std::clamp(
                    src_r * inv_alpha + static_cast<float>(overlay_r) * alpha,
                    0.0f,
                    255.0f
                )
                );

        rgb[idx + 1] =
            static_cast<uint8_t>(
                std::clamp(
                    src_g * inv_alpha + static_cast<float>(overlay_g) * alpha,
                    0.0f,
                    255.0f
                )
                );

        rgb[idx + 2] =
            static_cast<uint8_t>(
                std::clamp(
                    src_b * inv_alpha + static_cast<float>(overlay_b) * alpha,
                    0.0f,
                    255.0f
                )
                );
    }

    void GetTipClassColorRgb(
        uint32_t class_id,
        uint8_t& r,
        uint8_t& g,
        uint8_t& b
    )
    {
        static constexpr uint8_t palette[][3] = {
            {255,   0,   0}, // red
            {  0, 255,   0}, // green
            {  0, 128, 255}, // blue/cyan
            {255, 255,   0}, // yellow
            {255,   0, 255}, // magenta
            {  0, 255, 255}, // cyan
            {255, 128,   0}, // orange
            {128,   0, 255}, // purple
        };

        constexpr size_t palette_size =
            sizeof(palette) / sizeof(palette[0]);

        const size_t idx =
            static_cast<size_t>(class_id) % palette_size;

        r = palette[idx][0];
        g = palette[idx][1];
        b = palette[idx][2];
    }

    void DrawCrossRgb(
        std::vector<uint8_t>& rgb,
        UINT width,
        UINT height,
        int cx,
        int cy,
        uint8_t r,
        uint8_t g,
        uint8_t b,
        int radius
    )
    {
        for (int d = -radius; d <= radius; ++d) {
            SetPixelRgb(rgb, width, height, cx + d, cy, r, g, b);
            SetPixelRgb(rgb, width, height, cx, cy + d, r, g, b);
        }
    }

    void DrawLineRgb(
        std::vector<uint8_t>& rgb,
        UINT width,
        UINT height,
        int x0,
        int y0,
        int x1,
        int y1,
        uint8_t r,
        uint8_t g,
        uint8_t b
    )
    {
        int dx = std::abs(x1 - x0);
        int sx = x0 < x1 ? 1 : -1;

        int dy = -std::abs(y1 - y0);
        int sy = y0 < y1 ? 1 : -1;

        int err = dx + dy;

        while (true) {
            SetPixelRgb(rgb, width, height, x0, y0, r, g, b);

            if (x0 == x1 && y0 == y1) {
                break;
            }

            const int e2 = 2 * err;

            if (e2 >= dy) {
                err += dy;
                x0 += sx;
            }

            if (e2 <= dx) {
                err += dx;
                y0 += sy;
            }
        }
    }

} // namespace

void SaveNchwFloatTensorD3D12BufferAsBmp(
    D3D12Core& d3d12_core,
    ID3D12Resource* tensor_buffer,
    D3D12_RESOURCE_STATES& tensor_buffer_state,
    UINT width,
    UINT height,
    const wchar_t* output_path
)
{
    if (!d3d12_core.device()) {
        throw std::runtime_error("SaveNchwFloatTensorD3D12BufferAsBmp: D3D12Core device is null");
    }
    if (!tensor_buffer) {
        throw std::runtime_error("SaveNchwFloatTensorD3D12BufferAsBmp: tensor_buffer is null");
    }
    if (width == 0 || height == 0) {
        throw std::runtime_error("SaveNchwFloatTensorD3D12BufferAsBmp: invalid image size");
    }
    if (!output_path) {
        throw std::runtime_error("SaveNchwFloatTensorD3D12BufferAsBmp: output_path is null");
    }

    const UINT64 hw =
        static_cast<UINT64>(width) *
        static_cast<UINT64>(height);

    const UINT64 tensor_bytes =
        hw * 3ull * sizeof(float);

    D3D12_RESOURCE_DESC src_desc =
        tensor_buffer->GetDesc();

    if (src_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
        throw std::runtime_error("SaveNchwFloatTensorD3D12BufferAsBmp: tensor_buffer is not buffer");
    }

    if (src_desc.Width < tensor_bytes) {
        throw std::runtime_error("SaveNchwFloatTensorD3D12BufferAsBmp: tensor_buffer is too small");
    }

    // ------------------------------------------------------------
    // Readback buffer 作成
    // ------------------------------------------------------------

    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_READBACK;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC readback_desc{};
    readback_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    readback_desc.Alignment = 0;
    readback_desc.Width = tensor_bytes;
    readback_desc.Height = 1;
    readback_desc.DepthOrArraySize = 1;
    readback_desc.MipLevels = 1;
    readback_desc.Format = DXGI_FORMAT_UNKNOWN;
    readback_desc.SampleDesc.Count = 1;
    readback_desc.SampleDesc.Quality = 0;
    readback_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    readback_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    Microsoft::WRL::ComPtr<ID3D12Resource> readback_buffer =
        d3d12_core.create_committed_resource(
            heap_props,
            D3D12_HEAP_FLAG_NONE,
            readback_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr
        );

    // ------------------------------------------------------------
    // GPU tensor buffer -> readback buffer
    // ------------------------------------------------------------

    d3d12_core.reset_command_list();

    ID3D12GraphicsCommandList* command_list =
        d3d12_core.command_list();

    TransitionResource(
        command_list,
        tensor_buffer,
        tensor_buffer_state,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );

    tensor_buffer_state = D3D12_RESOURCE_STATE_COPY_SOURCE;

    command_list->CopyBufferRegion(
        readback_buffer.Get(),
        0,
        tensor_buffer,
        0,
        tensor_bytes
    );

    // 保存後も前処理で使いやすいよう、UNORDERED_ACCESS に戻す。
    TransitionResource(
        command_list,
        tensor_buffer,
        tensor_buffer_state,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    tensor_buffer_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    d3d12_core.close_execute_and_wait();

    // ------------------------------------------------------------
    // readback buffer map
    // ------------------------------------------------------------

    D3D12_RANGE read_range{};
    read_range.Begin = 0;
    read_range.End = static_cast<SIZE_T>(tensor_bytes);

    void* mapped = nullptr;

    HRESULT hr = readback_buffer->Map(
        0,
        &read_range,
        &mapped
    );

    win_util::ThrowIfFailed(
        hr,
        "Map readback buffer failed"
    );

    const float* tensor =
        static_cast<const float*>(mapped);

    // ------------------------------------------------------------
    // BMP buffer 作成
    // ------------------------------------------------------------

    const UINT bmp_row_stride = width * 3;
    const UINT bmp_row_stride_padded =
        (bmp_row_stride + 3u) & ~3u;

    const UINT image_size =
        bmp_row_stride_padded * height;

    constexpr UINT file_header_size = 14;
    constexpr UINT info_header_size = 40;
    constexpr UINT pixel_offset =
        file_header_size + info_header_size;

    const UINT file_size =
        pixel_offset + image_size;

    std::vector<uint8_t> bmp(
        file_size,
        0
    );

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

    auto write_i32 = [&] (size_t offset, int32_t v) {
        write32(offset, static_cast<uint32_t>(v));
        };

    // BITMAPFILEHEADER
    bmp[0] = 'B';
    bmp[1] = 'M';
    write32(2, file_size);
    write16(6, 0);
    write16(8, 0);
    write32(10, pixel_offset);

    // BITMAPINFOHEADER
    write32(14, info_header_size);
    write_i32(18, static_cast<int32_t>(width));

    // 負のheightで top-down BMP
    // tensor の y=0 が画像上端になる
    write_i32(22, -static_cast<int32_t>(height));

    write16(26, 1);    // planes
    write16(28, 24);   // bits per pixel
    write32(30, 0);    // BI_RGB
    write32(34, image_size);
    write_i32(38, 0);
    write_i32(42, 0);
    write32(46, 0);
    write32(50, 0);

    uint8_t* dst_base =
        bmp.data() + pixel_offset;

    const float* r_plane =
        tensor + 0 * hw;

    const float* g_plane =
        tensor + 1 * hw;

    const float* b_plane =
        tensor + 2 * hw;

    for (UINT y = 0; y < height; ++y) {
        uint8_t* dst_row =
            dst_base +
            static_cast<size_t>(y) * bmp_row_stride_padded;

        for (UINT x = 0; x < width; ++x) {
            const UINT64 idx =
                static_cast<UINT64>(y) *
                static_cast<UINT64>(width) +
                static_cast<UINT64>(x);

            const uint8_t r =
                Float01ToU8(r_plane[idx]);

            const uint8_t g =
                Float01ToU8(g_plane[idx]);

            const uint8_t b =
                Float01ToU8(b_plane[idx]);

            // BMP は BGR
            dst_row[x * 3 + 0] = b;
            dst_row[x * 3 + 1] = g;
            dst_row[x * 3 + 2] = r;
        }
    }

    D3D12_RANGE written_range{};
    written_range.Begin = 0;
    written_range.End = 0;

    readback_buffer->Unmap(
        0,
        &written_range
    );

    // ------------------------------------------------------------
    // write file
    // ------------------------------------------------------------

    FILE* fp = nullptr;

    errno_t e = _wfopen_s(
        &fp,
        output_path,
        L"wb"
    );

    if (e != 0 || !fp) {
        throw std::runtime_error("failed to open output BMP file");
    }

    fwrite(
        bmp.data(),
        1,
        bmp.size(),
        fp
    );

    fclose(fp);
}

void SaveNchwFloatTensorWithDetectionsAsBmp(
    const std::vector<float>& nchw_rgb_tensor,
    UINT width,
    UINT height,
    const std::vector<YoloSegPostProcessorD3D12::Detection>& detections,
    const wchar_t* output_path,
    float score_threshold
)
{
    if (width == 0 || height == 0) {
        throw std::runtime_error("invalid image size");
    }

    const size_t hw =
        static_cast<size_t>(width) * static_cast<size_t>(height);

    const size_t expected_size = hw * 3;

    if (nchw_rgb_tensor.size() < expected_size) {
        throw std::runtime_error("nchw_rgb_tensor is too small");
    }

    std::vector<uint8_t> rgb(hw * 3);

    const float* r_plane = nchw_rgb_tensor.data() + 0 * hw;
    const float* g_plane = nchw_rgb_tensor.data() + 1 * hw;
    const float* b_plane = nchw_rgb_tensor.data() + 2 * hw;

    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            const size_t idx =
                static_cast<size_t>(y) * static_cast<size_t>(width) +
                static_cast<size_t>(x);

            rgb[idx * 3 + 0] = Float01ToU8(r_plane[idx]);
            rgb[idx * 3 + 1] = Float01ToU8(g_plane[idx]);
            rgb[idx * 3 + 2] = Float01ToU8(b_plane[idx]);
        }
    }

    for (const auto& det : detections) {
        if (det.score < score_threshold) {
            continue;
        }

        const int x1 = static_cast<int>(std::round(det.x1));
        const int y1 = static_cast<int>(std::round(det.y1));
        const int x2 = static_cast<int>(std::round(det.x2));
        const int y2 = static_cast<int>(std::round(det.y2));

        // まずは全class同じ赤枠。
        // classごとに色を変えたい場合は det.class_id から色を決める。
        DrawRectRgb(
            rgb,
            width,
            height,
            x1,
            y1,
            x2,
            y2,
            255,
            0,
            0,
            2
        );

        // box中心も小さく打つ
        const int cx = (x1 + x2) / 2;
        const int cy = (y1 + y2) / 2;

        DrawSmallMarkerRgb(
            rgb,
            width,
            height,
            cx,
            cy,
            255,
            255,
            0
        );
    }

    WriteRgbAsTopDownBmp(
        rgb,
        width,
        height,
        output_path
    );
}

void SaveFloatMaskAsBmp(
    const std::vector<float>& mask,
    UINT width,
    UINT height,
    const wchar_t* output_path,
    bool normalize_min_max
)
{
    if (width == 0 || height == 0) {
        throw std::runtime_error("SaveFloatMaskAsBmp: invalid image size");
    }

    if (!output_path) {
        throw std::runtime_error("SaveFloatMaskAsBmp: output_path is null");
    }

    const size_t pixel_count =
        static_cast<size_t>(width) * static_cast<size_t>(height);

    if (mask.size() < pixel_count) {
        throw std::runtime_error("SaveFloatMaskAsBmp: mask is too small");
    }

    float min_value = 0.0f;
    float max_value = 1.0f;

    if (normalize_min_max) {
        auto minmax = std::minmax_element(
            mask.begin(),
            mask.begin() + pixel_count
        );

        min_value = *minmax.first;
        max_value = *minmax.second;

        if (std::abs(max_value - min_value) < 1e-8f) {
            max_value = min_value + 1.0f;
        }
    }

    std::vector<uint8_t> rgb(pixel_count * 3);

    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            const size_t idx =
                static_cast<size_t>(y) * static_cast<size_t>(width) +
                static_cast<size_t>(x);

            float v = mask[idx];

            if (normalize_min_max) {
                v = (v - min_value) / (max_value - min_value);
            }

            const uint8_t u8 = Float01ToU8(v);

            rgb[idx * 3 + 0] = u8;
            rgb[idx * 3 + 1] = u8;
            rgb[idx * 3 + 2] = u8;
        }
    }

    WriteRgbAsTopDownBmp(
        rgb,
        width,
        height,
        output_path
    );
}

void SaveDetectionRawMaskAsBmp(
    const YoloSegPostProcessorD3D12::DetectionWithMask& result,
    const wchar_t* output_path,
    bool normalize_min_max
)
{
    if (result.mask_width == 0 || result.mask_height == 0) {
        throw std::runtime_error("SaveDetectionRawMaskAsBmp: invalid mask size");
    }

    SaveFloatMaskAsBmp(
        result.mask,
        result.mask_width,
        result.mask_height,
        output_path,
        normalize_min_max
    );
}

void SaveDetectionRawMasksAsBmp(
    const std::vector<YoloSegPostProcessorD3D12::DetectionWithMask>& results,
    const wchar_t* output_path_prefix,
    bool normalize_min_max,
    float score_threshold
)
{
    if (!output_path_prefix) {
        throw std::runtime_error("SaveDetectionRawMasksAsBmp: output_path_prefix is null");
    }

    size_t saved_index = 0;

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        const auto& det = result.detection;

        if (det.score < score_threshold) {
            continue;
        }

        wchar_t path[512]{};

        // 例:
        // prefix_000_cls2_score0.873.bmp
        swprintf_s(
            path,
            L"%s_%03zu_cls%u_score%.3f.bmp",
            output_path_prefix,
            saved_index,
            det.class_id,
            det.score
        );

        SaveDetectionRawMaskAsBmp(
            result,
            path,
            normalize_min_max
        );

        ++saved_index;
    }
}

void SaveNchwFloatTensorWithMasksAsBmp(
    const std::vector<float>& nchw_rgb_tensor,
    UINT width,
    UINT height,
    const std::vector<YoloSegPostProcessorD3D12::DetectionWithMask>& results,
    const wchar_t* output_path,
    float score_threshold,
    float mask_threshold,
    float alpha,
    bool draw_bbox
)
{
    if (width == 0 || height == 0) {
        throw std::runtime_error("SaveNchwFloatTensorWithMasksAsBmp: invalid image size");
    }

    if (!output_path) {
        throw std::runtime_error("SaveNchwFloatTensorWithMasksAsBmp: output_path is null");
    }

    const size_t hw =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    const size_t expected_size = hw * 3;

    if (nchw_rgb_tensor.size() < expected_size) {
        throw std::runtime_error("SaveNchwFloatTensorWithMasksAsBmp: nchw_rgb_tensor is too small");
    }

    std::vector<uint8_t> rgb(hw * 3);

    const float* r_plane =
        nchw_rgb_tensor.data() + 0 * hw;

    const float* g_plane =
        nchw_rgb_tensor.data() + 1 * hw;

    const float* b_plane =
        nchw_rgb_tensor.data() + 2 * hw;

    // ------------------------------------------------------------
    // NCHW float RGB -> RGB uint8 image
    // ------------------------------------------------------------

    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            const size_t idx =
                static_cast<size_t>(y) *
                static_cast<size_t>(width) +
                static_cast<size_t>(x);

            rgb[idx * 3 + 0] = Float01ToU8(r_plane[idx]);
            rgb[idx * 3 + 1] = Float01ToU8(g_plane[idx]);
            rgb[idx * 3 + 2] = Float01ToU8(b_plane[idx]);
        }
    }

    // ------------------------------------------------------------
    // Overlay masks
    //
    // 前提:
    // - image: 640x640 など YOLO input 空間
    // - mask : 160x160 prototype 空間
    //
    // mask は bbox crop 済みの binary mask を想定。
    // image pixel (x, y) を mask pixel (mx, my) に対応させる。
    // ------------------------------------------------------------

    for (const auto& result : results) {
        const auto& det = result.detection;

        if (det.score < score_threshold) {
            continue;
        }

        if (result.mask_width == 0 || result.mask_height == 0) {
            continue;
        }

        const size_t mask_pixel_count =
            static_cast<size_t>(result.mask_width) *
            static_cast<size_t>(result.mask_height);

        if (result.mask.size() < mask_pixel_count) {
            continue;
        }

        uint8_t cr = 255;
        uint8_t cg = 0;
        uint8_t cb = 0;

        GetClassColorRgb(
            det.class_id,
            cr,
            cg,
            cb
        );

        // bbox範囲だけ見れば十分なので、処理範囲をbboxに限定する。
        int x1 = static_cast<int>(std::floor(det.x1));
        int y1 = static_cast<int>(std::floor(det.y1));
        int x2 = static_cast<int>(std::ceil(det.x2));
        int y2 = static_cast<int>(std::ceil(det.y2));

        x1 = std::clamp(x1, 0, static_cast<int>(width) - 1);
        y1 = std::clamp(y1, 0, static_cast<int>(height) - 1);
        x2 = std::clamp(x2, 0, static_cast<int>(width) - 1);
        y2 = std::clamp(y2, 0, static_cast<int>(height) - 1);

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        for (int y = y1; y <= y2; ++y) {
            for (int x = x1; x <= x2; ++x) {
                const float mx_f =
                    (static_cast<float>(x) + 0.5f) *
                    static_cast<float>(result.mask_width) /
                    static_cast<float>(width);

                const float my_f =
                    (static_cast<float>(y) + 0.5f) *
                    static_cast<float>(result.mask_height) /
                    static_cast<float>(height);

                int mx = static_cast<int>(mx_f);
                int my = static_cast<int>(my_f);

                mx = std::clamp(
                    mx,
                    0,
                    static_cast<int>(result.mask_width) - 1
                );

                my = std::clamp(
                    my,
                    0,
                    static_cast<int>(result.mask_height) - 1
                );

                const size_t mask_idx =
                    static_cast<size_t>(my) *
                    static_cast<size_t>(result.mask_width) +
                    static_cast<size_t>(mx);

                const float mask_value =
                    result.mask[mask_idx];

                if (mask_value < mask_threshold) {
                    continue;
                }

                BlendPixelRgb(
                    rgb,
                    width,
                    height,
                    x,
                    y,
                    cr,
                    cg,
                    cb,
                    alpha
                );
            }
        }

        if (draw_bbox) {
            DrawRectRgb(
                rgb,
                width,
                height,
                x1,
                y1,
                x2,
                y2,
                cr,
                cg,
                cb,
                2
            );

            const int cx =
                (x1 + x2) / 2;

            const int cy =
                (y1 + y2) / 2;

            DrawSmallMarkerRgb(
                rgb,
                width,
                height,
                cx,
                cy,
                255,
                255,
                255
            );
        }
    }

    WriteRgbAsTopDownBmp(
        rgb,
        width,
        height,
        output_path
    );
}

void SaveNchwFloatTensorWithToolTipsAsBmp(
    const std::vector<float>& nchw_rgb_tensor,
    UINT width,
    UINT height,
    const std::vector<ToolTipDetectorD3D12::TipResult>& tip_results,
    const wchar_t* output_path,
    bool draw_candidates,
    bool draw_axis
)
{
    if (width == 0 || height == 0) {
        throw std::runtime_error("SaveNchwFloatTensorWithToolTipsAsBmp: invalid image size");
    }

    if (!output_path) {
        throw std::runtime_error("SaveNchwFloatTensorWithToolTipsAsBmp: output_path is null");
    }

    const size_t hw =
        static_cast<size_t>(width) *
        static_cast<size_t>(height);

    const size_t expected_size = hw * 3;

    if (nchw_rgb_tensor.size() < expected_size) {
        throw std::runtime_error("SaveNchwFloatTensorWithToolTipsAsBmp: nchw_rgb_tensor is too small");
    }

    std::vector<uint8_t> rgb(hw * 3);

    const float* r_plane =
        nchw_rgb_tensor.data() + 0 * hw;

    const float* g_plane =
        nchw_rgb_tensor.data() + 1 * hw;

    const float* b_plane =
        nchw_rgb_tensor.data() + 2 * hw;

    // ------------------------------------------------------------
    // NCHW float RGB -> RGB uint8 image
    // ------------------------------------------------------------

    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            const size_t idx =
                static_cast<size_t>(y) *
                static_cast<size_t>(width) +
                static_cast<size_t>(x);

            rgb[idx * 3 + 0] = Float01ToU8(r_plane[idx]);
            rgb[idx * 3 + 1] = Float01ToU8(g_plane[idx]);
            rgb[idx * 3 + 2] = Float01ToU8(b_plane[idx]);
        }
    }

    // ------------------------------------------------------------
    // Plot tool tip results
    // tip_x_input / tip_y_input は YOLO input 空間の座標を想定
    // つまり width,height が 640x640 ならそのまま描画できる
    // ------------------------------------------------------------

    for (const auto& tip : tip_results) {
        if (tip.valid == 0) {
            continue;
        }

        uint8_t cr = 255;
        uint8_t cg = 0;
        uint8_t cb = 0;

        GetTipClassColorRgb(
            tip.class_id,
            cr,
            cg,
            cb
        );

        const int tip_x =
            static_cast<int>(std::round(tip.tip_x_input));

        const int tip_y =
            static_cast<int>(std::round(tip.tip_y_input));

        const int c1_x =
            static_cast<int>(
                std::round(
                    tip.candidate1_x_mask *
                    static_cast<float>(width) /
                    160.0f
                )
                );

        const int c1_y =
            static_cast<int>(
                std::round(
                    tip.candidate1_y_mask *
                    static_cast<float>(height) /
                    160.0f
                )
                );

        const int c2_x =
            static_cast<int>(
                std::round(
                    tip.candidate2_x_mask *
                    static_cast<float>(width) /
                    160.0f
                )
                );

        const int c2_y =
            static_cast<int>(
                std::round(
                    tip.candidate2_y_mask *
                    static_cast<float>(height) /
                    160.0f
                )
                );

        // candidate同士を薄い線で結ぶ
        if (draw_candidates) {
            DrawLineRgb(
                rgb,
                width,
                height,
                c1_x,
                c1_y,
                c2_x,
                c2_y,
                255,
                255,
                255
            );

            // candidate1: yellow
            DrawCrossRgb(
                rgb,
                width,
                height,
                c1_x,
                c1_y,
                255,
                255,
                0,
                5
            );

            // candidate2: cyan
            DrawCrossRgb(
                rgb,
                width,
                height,
                c2_x,
                c2_y,
                0,
                255,
                255,
                5
            );
        }

        // 主軸方向を表示
        if (draw_axis) {
            const float axis_scale = 40.0f;

            const int ax0 =
                static_cast<int>(
                    std::round(
                        static_cast<float>(tip_x) -
                        tip.axis_x * axis_scale
                    )
                    );

            const int ay0 =
                static_cast<int>(
                    std::round(
                        static_cast<float>(tip_y) -
                        tip.axis_y * axis_scale
                    )
                    );

            const int ax1 =
                static_cast<int>(
                    std::round(
                        static_cast<float>(tip_x) +
                        tip.axis_x * axis_scale
                    )
                    );

            const int ay1 =
                static_cast<int>(
                    std::round(
                        static_cast<float>(tip_y) +
                        tip.axis_y * axis_scale
                    )
                    );

            DrawLineRgb(
                rgb,
                width,
                height,
                ax0,
                ay0,
                ax1,
                ay1,
                cr,
                cg,
                cb
            );
        }

        // 採用された先端中心: class色の大きめクロス
        DrawCrossRgb(
            rgb,
            width,
            height,
            tip_x,
            tip_y,
            cr,
            cg,
            cb,
            8
        );

        // 中心点を白く強調
        DrawSmallMarkerRgb(
            rgb,
            width,
            height,
            tip_x,
            tip_y,
            255,
            255,
            255
        );
    }

    WriteRgbAsTopDownBmp(
        rgb,
        width,
        height,
        output_path
    );
}