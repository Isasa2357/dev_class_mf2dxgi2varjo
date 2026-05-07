#include "TensorReadbackD3D12.hpp"

#include <stdexcept>
#include <cstring>

#include "HResultUtil.hpp"

namespace {

    void TransitionResource(
        ID3D12GraphicsCommandList* command_list,
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES before,
        D3D12_RESOURCE_STATES after
    )
    {
        if (!command_list) {
            throw std::runtime_error("TransitionResource: command_list is null");
        }
        if (!resource) {
            throw std::runtime_error("TransitionResource: resource is null");
        }

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

        command_list->ResourceBarrier(
            1,
            &barrier
        );
    }

} // namespace

std::vector<float> ReadbackNchwFloatTensorD3D12Buffer(
    D3D12Core& d3d12_core,
    ID3D12Resource* tensor_buffer,
    D3D12_RESOURCE_STATES& tensor_buffer_state,
    size_t element_count
)
{
    if (!d3d12_core.device()) {
        throw std::runtime_error("ReadbackNchwFloatTensorD3D12Buffer: D3D12Core device is null");
    }
    if (!tensor_buffer) {
        throw std::runtime_error("ReadbackNchwFloatTensorD3D12Buffer: tensor_buffer is null");
    }
    if (element_count == 0) {
        throw std::runtime_error("ReadbackNchwFloatTensorD3D12Buffer: element_count is zero");
    }

    const UINT64 tensor_bytes =
        static_cast<UINT64>(element_count) * sizeof(float);

    D3D12_RESOURCE_DESC src_desc =
        tensor_buffer->GetDesc();

    if (src_desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) {
        throw std::runtime_error("ReadbackNchwFloatTensorD3D12Buffer: tensor_buffer is not a buffer");
    }

    if (src_desc.Width < tensor_bytes) {
        throw std::runtime_error("ReadbackNchwFloatTensorD3D12Buffer: tensor_buffer is smaller than requested tensor size");
    }

    // ------------------------------------------------------------
    // readback buffer 作成
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

    // 次の preprocess でそのまま使いやすいように UAV 状態へ戻す
    TransitionResource(
        command_list,
        tensor_buffer,
        tensor_buffer_state,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );

    tensor_buffer_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    d3d12_core.close_execute_and_wait();

    // ------------------------------------------------------------
    // Map
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
        "ReadbackNchwFloatTensorD3D12Buffer: Map readback buffer failed"
    );

    std::vector<float> tensor(element_count);

    std::memcpy(
        tensor.data(),
        mapped,
        static_cast<size_t>(tensor_bytes)
    );

    D3D12_RANGE written_range{};
    written_range.Begin = 0;
    written_range.End = 0;

    readback_buffer->Unmap(
        0,
        &written_range
    );

    return tensor;
}

std::vector<float> ReadbackNchwFloatTensorD3D12Buffer(
    D3D12Core& d3d12_core,
    ID3D12Resource* tensor_buffer,
    D3D12_RESOURCE_STATES& tensor_buffer_state,
    UINT width,
    UINT height,
    UINT channels
)
{
    if (width == 0 || height == 0 || channels == 0) {
        throw std::runtime_error("ReadbackNchwFloatTensorD3D12Buffer: invalid tensor shape");
    }

    const size_t element_count =
        static_cast<size_t>(channels) *
        static_cast<size_t>(height) *
        static_cast<size_t>(width);

    return ReadbackNchwFloatTensorD3D12Buffer(
        d3d12_core,
        tensor_buffer,
        tensor_buffer_state,
        element_count
    );
}