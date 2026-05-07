#pragma once

#define NOMINMAX

#include <windows.h>
#include <wrl/client.h>

#include <d3d10_1.h>
#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_6.h>

#include <vector>
#include <stdexcept>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

void create_d3d11device(
	Microsoft::WRL::ComPtr<ID3D11Device>& device,
	Microsoft::WRL::ComPtr<ID3D11DeviceContext>& context,
	D3D_FEATURE_LEVEL& feature_level,
	const bool enable_debug_layer = false
);

void create_d3d11device_on_adapter(
	Microsoft::WRL::ComPtr<ID3D11Device>& device,
	Microsoft::WRL::ComPtr<ID3D11DeviceContext>& context,
	D3D_FEATURE_LEVEL& feature_level,
	IDXGIAdapter1* adapter,
	const bool enable_debug_layer = false
);

Microsoft::WRL::ComPtr<IDXGIAdapter> get_adapter_from_d3d11device(ID3D11Device* device);

void enable_d3d11_multithread_protection(ID3D11Device* device);