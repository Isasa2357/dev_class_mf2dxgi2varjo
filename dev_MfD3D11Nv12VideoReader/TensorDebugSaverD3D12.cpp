#include "TensorDebugSaverD3D12.hpp"

#include <vector>
#include <stdexcept>
#include <cstdio>
#include <algorithm>

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