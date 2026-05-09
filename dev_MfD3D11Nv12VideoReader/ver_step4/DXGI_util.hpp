#pragma once

#define NOMINMAX

#include <windows.h>
#include <wrl/client.h>

#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_6.h>

#include <vector>
#include <stdexcept>

#include "HResultUtil.hpp"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

inline bool is_same_LUID(const LUID& luid1, const LUID& luid2) {
	return luid1.HighPart == luid2.HighPart && luid1.LowPart == luid2.LowPart;
}

inline Microsoft::WRL::ComPtr<IDXGIAdapter1> find_adapter_by_LUID(LUID target_luid) {
	Microsoft::WRL::ComPtr<IDXGIFactory6> factory;

	auto hr = CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf()));
	win_util::ThrowIfFailed(hr, "CreateDXGIFactory1 failed");

	for (auto i = 0;; ++i) {
		Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter_cond;

		hr = factory->EnumAdapters1(i, adapter_cond.GetAddressOf());

		if (hr == DXGI_ERROR_NOT_FOUND) break;

		win_util::ThrowIfFailed(hr, "EnumAdapters1 failed");

		DXGI_ADAPTER_DESC desc{};
		hr = adapter_cond->GetDesc(&desc);
		win_util::ThrowIfFailed(hr, "IDXGIAdapter1::GetDesc1 failed");

		if (is_same_LUID(desc.AdapterLuid, target_luid)) {
			return adapter_cond;
		}
	}

	throw std::runtime_error("Target adapter is not found");
}