#pragma once

#define NOMINMAX

#include <windows.h>
#include <wrl/client.h>

#include <d3d12.h>
#include <dxgi1_6.h>

class D3D12Core {
public:
    D3D12Core() = default;

    D3D12Core(const D3D12Core&) = delete;
    D3D12Core& operator=(const D3D12Core&) = delete;

    ~D3D12Core();

    void initialize(
        bool enable_debug_layer = false
    );

    void initialize_with_adapter_LUID(
        const LUID adapter_LUID,
        bool enable_debug_layer = false
    );

    ID3D12Device* device() const
    {
        return this->device_.Get();
    }

    ID3D12CommandQueue* command_queue() const
    {
        return this->command_queue_.Get();
    }

    ID3D12CommandAllocator* command_allocator() const
    {
        return this->command_allocator_.Get();
    }

    ID3D12GraphicsCommandList* command_list() const
    {
        return this->command_list_.Get();
    }

    IDXGIAdapter1* adapter() const
    {
        return this->adapter_.Get();
    }

    LUID adapter_LUID() const
    {
        return this->adapter_LUID_;
    }

    DXGI_ADAPTER_DESC1 adapter_desc() const
    {
        return this->adapter_desc_;
    }

    void print_adapter_info() const;

    void reset_command_list();

    void close_command_list();

    void execute_command_list();

    void close_execute_and_wait();

    void wait_for_gpu();

    Microsoft::WRL::ComPtr<ID3D12Resource> create_committed_resource(
        const D3D12_HEAP_PROPERTIES& heap_properties,
        D3D12_HEAP_FLAGS heap_flags,
        const D3D12_RESOURCE_DESC& resource_desc,
        D3D12_RESOURCE_STATES initial_state,
        const D3D12_CLEAR_VALUE* clear_value = nullptr
    ) const;

private:
    void cache_adapter_info();

    void create_command_objects();

    void create_fence_object();

private:
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter_;

    Microsoft::WRL::ComPtr<ID3D12CommandQueue> command_queue_;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> command_allocator_;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> command_list_;

    Microsoft::WRL::ComPtr<ID3D12Fence> fence_;
    UINT64 fence_value_ = 0;
    HANDLE fence_event_ = nullptr;

    DXGI_ADAPTER_DESC1 adapter_desc_{};
    LUID adapter_LUID_{};
};