#pragma once

#define NOMINMAX

#include <windows.h>
#include <wrl/client.h>

#include <d3d12.h>
#include <dxgi1_6.h>

/*
 * D3D12のデバッグレイヤの有効化
 */
void enable_d3d12_debug_layer();

/*
 * @brief D3D12Deviceの作成
 * @param device: デバイスの出力引数
 * @param adapter: アダプタの出力引数
 * @param enable_debug_layer: デバッグレイヤの有効化
 */
void create_d3d12device(
    Microsoft::WRL::ComPtr<ID3D12Device>& device,
    Microsoft::WRL::ComPtr<IDXGIAdapter1>& adapter,
    const bool enable_debug_layer
);

/*
 * @brief D3D12Deviceの作成
 * @param device: デバイスの出力引数
 * @param adapter: デバイスを作成するアダプタ
 * @param enable_debug_layer: デバッグレイヤの有効化
 */
void create_d3d12device_on_adapter(
    Microsoft::WRL::ComPtr<ID3D12Device>& device,
    IDXGIAdapter1* adapter,
    const bool enable_debug_layer
);

/*
 * @brief D3D12CommandQueueの作成
 */
void create_d3d12_command_queue(
    ID3D12Device* device,
    Microsoft::WRL::ComPtr<ID3D12CommandQueue>& command_queue,
    D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT
);

/*
 * @brief D3D12CommandAllocatorの作成
 */
void create_d3d12_command_allocator(
    ID3D12Device* device,
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator>& command_allocator,
    D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT
);

/*
 * @brief D3D12CommandListの作成
 */
void create_d3d12_graphics_command_list(
    ID3D12Device* device,
    ID3D12CommandAllocator* command_allocator,
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>& command_list,
    D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT
);

/*
 * @brief D3D12Fenceの作成
 */
void create_d3d12_fence(
    ID3D12Device* device,
    Microsoft::WRL::ComPtr<ID3D12Fence>& fence,
    const UINT64 initial_value = 0
);

void enable_d3d12_info_queue_break_on_error(ID3D12Device* device);