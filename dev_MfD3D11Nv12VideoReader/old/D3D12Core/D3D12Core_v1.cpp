#include "D3D12Core.hpp"

#include <stdexcept>
#include <iostream>

#include "DXGI_util.hpp"
#include "D3D12_mutil.hpp"
#include "HResultUtil.hpp"

D3D12Core::~D3D12Core()
{
    if (this->fence_event_) {
        CloseHandle(this->fence_event_);
        this->fence_event_ = nullptr;
    }
}

void D3D12Core::initialize(
    bool enable_debug_layer
) {
    create_d3d12device(
        this->device_,
        this->adapter_,
        enable_debug_layer
    );

    this->cache_adapter_info();

    this->create_command_objects();
    this->create_fence_object();
}

void D3D12Core::initialize_with_adapter_LUID(
    const LUID adapter_LUID,
    bool enable_debug_layer
) {
    auto found_adapter = find_adapter_by_LUID(adapter_LUID);

    create_d3d12device_on_adapter(
        this->device_,
        found_adapter.Get(),
        enable_debug_layer
    );

    this->adapter_ = found_adapter;

    this->cache_adapter_info();

    if (!is_same_LUID(this->adapter_desc_.AdapterLuid, adapter_LUID)) {
        throw std::runtime_error(
            "D3D12Core::initialize_with_adapter_LUID: created adapter LUID mismatch"
        );
    }

    this->create_command_objects();
    this->create_fence_object();
}

void D3D12Core::print_adapter_info() const
{
    std::wcout
        << L"D3D12 Adapter: " << this->adapter_desc_.Description << L"\n"
        << L"D3D12 Adapter LUID: "
        << this->adapter_desc_.AdapterLuid.HighPart
        << L":"
        << this->adapter_desc_.AdapterLuid.LowPart
        << L"\n";
}

void D3D12Core::reset_command_list()
{
    if (!this->command_allocator_ || !this->command_list_) {
        throw std::runtime_error("D3D12Core::reset_command_list: command objects are null");
    }

    // このクラスでは close_execute_and_wait() によりGPU完了後に次のresetを行う前提
    HRESULT hr = this->command_allocator_->Reset();
    win_util::ThrowIfFailed(hr, "ID3D12CommandAllocator::Reset failed");

    hr = this->command_list_->Reset(
        this->command_allocator_.Get(),
        nullptr
    );
    win_util::ThrowIfFailed(hr, "ID3D12GraphicsCommandList::Reset failed");
}

void D3D12Core::close_command_list()
{
    if (!this->command_list_) {
        throw std::runtime_error("D3D12Core::close_command_list: command_list is null");
    }

    HRESULT hr = this->command_list_->Close();
    win_util::ThrowIfFailed(hr, "ID3D12GraphicsCommandList::Close failed");
}

void D3D12Core::execute_command_list()
{
    if (!this->command_queue_ || !this->command_list_) {
        throw std::runtime_error("D3D12Core::execute_command_list: command objects are null");
    }

    ID3D12CommandList* command_lists[] = {
        this->command_list_.Get()
    };

    this->command_queue_->ExecuteCommandLists(
        1,
        command_lists
    );
}

void D3D12Core::close_execute_and_wait()
{
    this->close_command_list();
    this->execute_command_list();
    this->wait_for_gpu();
}

void D3D12Core::wait_for_gpu()
{
    if (!this->command_queue_ || !this->fence_) {
        throw std::runtime_error("D3D12Core::wait_for_gpu: queue/fence is null");
    }
    if (!this->fence_event_) {
        throw std::runtime_error("D3D12Core::wait_for_gpu: fence_event is null");
    }

    const UINT64 signal_value = ++this->fence_value_;

    HRESULT hr = this->command_queue_->Signal(
        this->fence_.Get(),
        signal_value
    );
    win_util::ThrowIfFailed(hr, "ID3D12CommandQueue::Signal failed");

    if (this->fence_->GetCompletedValue() < signal_value) {
        hr = this->fence_->SetEventOnCompletion(
            signal_value,
            this->fence_event_
        );
        win_util::ThrowIfFailed(hr, "ID3D12Fence::SetEventOnCompletion failed");

        WaitForSingleObject(this->fence_event_, INFINITE);
    }
}

Microsoft::WRL::ComPtr<ID3D12Resource> D3D12Core::create_committed_resource(
    const D3D12_HEAP_PROPERTIES& heap_properties,
    D3D12_HEAP_FLAGS heap_flags,
    const D3D12_RESOURCE_DESC& resource_desc,
    D3D12_RESOURCE_STATES initial_state,
    const D3D12_CLEAR_VALUE* clear_value
) const {
    Microsoft::WRL::ComPtr<ID3D12Resource> resource;

    HRESULT hr = this->device_->CreateCommittedResource(
        &heap_properties,
        heap_flags,
        &resource_desc,
        initial_state,
        clear_value,
        IID_PPV_ARGS(resource.GetAddressOf())
    );

    win_util::ThrowIfFailed(hr, "ID3D12Device::CreateCommittedResource failed");

    return resource;
}

void D3D12Core::cache_adapter_info()
{
    if (!this->adapter_) {
        throw std::runtime_error("D3D12Core::cache_adapter_info: adapter is null");
    }

    auto hr = this->adapter_->GetDesc1(&this->adapter_desc_);
    win_util::ThrowIfFailed(hr, "IDXGIAdapter1::GetDesc1 failed");

    this->adapter_LUID_ = this->adapter_desc_.AdapterLuid;
}

void D3D12Core::create_command_objects()
{
    create_d3d12_command_queue(
        this->device_.Get(),
        this->command_queue_,
        D3D12_COMMAND_LIST_TYPE_DIRECT
    );

    create_d3d12_command_allocator(
        this->device_.Get(),
        this->command_allocator_,
        D3D12_COMMAND_LIST_TYPE_DIRECT
    );

    create_d3d12_graphics_command_list(
        this->device_.Get(),
        this->command_allocator_.Get(),
        this->command_list_,
        D3D12_COMMAND_LIST_TYPE_DIRECT
    );
}

void D3D12Core::create_fence_object()
{
    create_d3d12_fence(
        this->device_.Get(),
        this->fence_,
        0
    );

    this->fence_value_ = 0;

    this->fence_event_ = CreateEvent(
        nullptr,
        FALSE,
        FALSE,
        nullptr
    );

    if (!this->fence_event_) {
        throw std::runtime_error("CreateEvent for D3D12 fence failed");
    }
}