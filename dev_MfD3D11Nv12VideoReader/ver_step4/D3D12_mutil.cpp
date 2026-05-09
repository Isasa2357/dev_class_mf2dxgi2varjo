#include "D3D12_mutil.hpp"

#include <stdexcept>
#include <iostream>

#include <d3d12sdklayers.h>

#include "HResultUtil.hpp"

void enable_d3d12_debug_layer()
{
#if defined(_DEBUG)
    Microsoft::WRL::ComPtr<ID3D12Debug> debug_controller;

    HRESULT hr = D3D12GetDebugInterface(
        IID_PPV_ARGS(debug_controller.GetAddressOf())
    );

    if (SUCCEEDED(hr) && debug_controller) {
        debug_controller->EnableDebugLayer();
        std::wcout << L"D3D12 Debug Layer enabled\n";
    } else {
        std::wcout << L"D3D12 Debug Layer is not available\n";
    }
#else
    // Releaseビルドでは何もしない
#endif
}

void create_d3d12device(
    Microsoft::WRL::ComPtr<ID3D12Device>& device,
    Microsoft::WRL::ComPtr<IDXGIAdapter1>& adapter,
    const bool enable_debug_layer
) {
#if defined(_DEBUG)
    if (enable_debug_layer) {
        enable_d3d12_debug_layer();
    }
#else
    (void)enable_debug_layer;
#endif

    Microsoft::WRL::ComPtr<IDXGIFactory6> factory;

    HRESULT hr = CreateDXGIFactory1(
        IID_PPV_ARGS(factory.GetAddressOf())
    );
    win_util::ThrowIfFailed(hr, "CreateDXGIFactory1 failed");

    Microsoft::WRL::ComPtr<IDXGIAdapter1> selected_adapter;

    for (UINT i = 0;; ++i) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> current_adapter;

        hr = factory->EnumAdapters1(
            i,
            current_adapter.GetAddressOf()
        );

        if (hr == DXGI_ERROR_NOT_FOUND) {
            break;
        }

        win_util::ThrowIfFailed(hr, "EnumAdapters1 failed");

        DXGI_ADAPTER_DESC1 desc{};
        hr = current_adapter->GetDesc1(&desc);
        win_util::ThrowIfFailed(hr, "IDXGIAdapter1::GetDesc1 failed");

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }

        // D3D12CreateDevice が通る adapter を選ぶ
        if (SUCCEEDED(D3D12CreateDevice(
            current_adapter.Get(),
            D3D_FEATURE_LEVEL_11_0,
            __uuidof(ID3D12Device),
            nullptr
        ))) {
            selected_adapter = current_adapter;
            break;
        }
    }

    if (!selected_adapter) {
        throw std::runtime_error("No suitable D3D12 adapter found");
    }

    hr = D3D12CreateDevice(
        selected_adapter.Get(),
        D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(device.ReleaseAndGetAddressOf())
    );
    win_util::ThrowIfFailed(hr, "D3D12CreateDevice failed");

    adapter = selected_adapter;

#if defined(_DEBUG)
    if (enable_debug_layer) {
        enable_d3d12_info_queue_break_on_error(device.Get());
    }
#endif
}

void create_d3d12device_on_adapter(
    Microsoft::WRL::ComPtr<ID3D12Device>& device,
    IDXGIAdapter1* adapter,
    const bool enable_debug_layer
) {
    if (!adapter) {
        throw std::runtime_error("adapter is null");
    }

#if defined(_DEBUG)
    if (enable_debug_layer) {
        enable_d3d12_debug_layer();
    }
#else
    (void)enable_debug_layer;
#endif

    HRESULT hr = D3D12CreateDevice(
        adapter,
        D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(device.ReleaseAndGetAddressOf())
    );
    win_util::ThrowIfFailed(hr, "D3D12CreateDevice on adapter failed");

#if defined(_DEBUG)
    if (enable_debug_layer) {
        enable_d3d12_info_queue_break_on_error(device.Get());
    }
#endif
}

void create_d3d12_command_queue(
    ID3D12Device* device,
    Microsoft::WRL::ComPtr<ID3D12CommandQueue>& command_queue,
    D3D12_COMMAND_LIST_TYPE type
) {
    if (!device) {
        throw std::runtime_error("device is null");
    }

    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = type;
    desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    desc.NodeMask = 0;

    HRESULT hr = device->CreateCommandQueue(
        &desc,
        IID_PPV_ARGS(command_queue.ReleaseAndGetAddressOf())
    );
    win_util::ThrowIfFailed(hr, "CreateCommandQueue failed");
}

void create_d3d12_command_allocator(
    ID3D12Device* device,
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator>& command_allocator,
    D3D12_COMMAND_LIST_TYPE type
) {
    if (!device) {
        throw std::runtime_error("device is null");
    }

    HRESULT hr = device->CreateCommandAllocator(
        type,
        IID_PPV_ARGS(command_allocator.ReleaseAndGetAddressOf())
    );
    win_util::ThrowIfFailed(hr, "CreateCommandAllocator failed");
}

void create_d3d12_graphics_command_list(
    ID3D12Device* device,
    ID3D12CommandAllocator* command_allocator,
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>& command_list,
    D3D12_COMMAND_LIST_TYPE type
) {
    if (!device) {
        throw std::runtime_error("device is null");
    }
    if (!command_allocator) {
        throw std::runtime_error("command_allocator is null");
    }

    HRESULT hr = device->CreateCommandList(
        0,
        type,
        command_allocator,
        nullptr,
        IID_PPV_ARGS(command_list.ReleaseAndGetAddressOf())
    );
    win_util::ThrowIfFailed(hr, "CreateCommandList failed");

    // 作成直後の command list は open 状態なので、Coreでは「未使用時はclosed」を基本にする。
    hr = command_list->Close();
    win_util::ThrowIfFailed(hr, "Initial command list Close failed");
}

void create_d3d12_fence(
    ID3D12Device* device,
    Microsoft::WRL::ComPtr<ID3D12Fence>& fence,
    const UINT64 initial_value
) {
    if (!device) {
        throw std::runtime_error("device is null");
    }

    HRESULT hr = device->CreateFence(
        initial_value,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(fence.ReleaseAndGetAddressOf())
    );
    win_util::ThrowIfFailed(hr, "CreateFence failed");
}

void enable_d3d12_info_queue_break_on_error(ID3D12Device* device)
{
#if defined(_DEBUG)
    if (!device) {
        throw std::runtime_error("device is null");
    }

    Microsoft::WRL::ComPtr<ID3D12InfoQueue> info_queue;

    HRESULT hr = device->QueryInterface(
        IID_PPV_ARGS(info_queue.GetAddressOf())
    );

    if (SUCCEEDED(hr) && info_queue) {
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
        info_queue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, FALSE);
    }
#else
    (void)device;
#endif
}