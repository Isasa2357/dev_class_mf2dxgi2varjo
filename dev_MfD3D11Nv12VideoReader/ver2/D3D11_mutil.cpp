
#include "D3D11_mutil.hpp"

#include "HResultUtil.hpp"

void create_d3d11device(
	Microsoft::WRL::ComPtr<ID3D11Device>& device, 
	Microsoft::WRL::ComPtr<ID3D11DeviceContext>& context, 
	D3D_FEATURE_LEVEL& feature_level, 
	const bool enable_debug_layer
) {

	UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

#if defined(_DEBUG)
	if (enable_debug_layer) {
		flags |= D3D11_CREATE_DEVICE_DEBUG;
	}
#else
	// 未使用引数の警告を消すため
	(void)enable_debug_layer;
#endif

	D3D_FEATURE_LEVEL levels[] = {
		D3D_FEATURE_LEVEL_11_1,
		D3D_FEATURE_LEVEL_11_0
	};

	HRESULT hr = D3D11CreateDevice(
		nullptr,
		D3D_DRIVER_TYPE_HARDWARE,
		nullptr,
		flags,
		levels,
		ARRAYSIZE(levels),
		D3D11_SDK_VERSION,
		device.ReleaseAndGetAddressOf(),
		&feature_level,
		context.ReleaseAndGetAddressOf()
	);

#if defined(_DEBUG)
	if (FAILED(hr) && enable_debug_layer) {
		flags &= ~D3D11_CREATE_DEVICE_DEBUG;

		hr = D3D11CreateDevice(
			nullptr,
			D3D_DRIVER_TYPE_HARDWARE,
			nullptr,
			flags,
			levels,
			ARRAYSIZE(levels),
			D3D11_SDK_VERSION,
			device.ReleaseAndGetAddressOf(),
			&feature_level,
			context.ReleaseAndGetAddressOf()
		);
	}
#endif

	win_util::ThrowIfFailed(hr, "D3D11CreateDevice failed");
}

void create_d3d11device_on_adapter(
	Microsoft::WRL::ComPtr<ID3D11Device>& device, 
	Microsoft::WRL::ComPtr<ID3D11DeviceContext>& context, 
	D3D_FEATURE_LEVEL& feature_level, 
	IDXGIAdapter1* adapter, 
	const bool enable_debug_layer
) {
	if (!adapter) {
		throw std::runtime_error("adapter is null");
	}

	UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

#if defined(_DEBUG)
	if (enable_debug_layer) {
		flags |= D3D11_CREATE_DEVICE_DEBUG;
	}
#else
	// 未使用引数の警告を消すため
	(void)enable_debug_layer;
#endif

	D3D_FEATURE_LEVEL levels[] = {
		D3D_FEATURE_LEVEL_11_1,
		D3D_FEATURE_LEVEL_11_0
	};

	HRESULT hr = D3D11CreateDevice(
		adapter,
		D3D_DRIVER_TYPE_UNKNOWN,
		nullptr,
		flags,
		levels,
		ARRAYSIZE(levels),
		D3D11_SDK_VERSION,
		device.ReleaseAndGetAddressOf(),
		&feature_level,
		context.ReleaseAndGetAddressOf()
	);

#if defined(_DEBUG)
	if (FAILED(hr) && enable_debug_layer) {
		flags &= ~D3D11_CREATE_DEVICE_DEBUG;

		hr = D3D11CreateDevice(
			nullptr,
			D3D_DRIVER_TYPE_HARDWARE,
			nullptr,
			flags,
			levels,
			ARRAYSIZE(levels),
			D3D11_SDK_VERSION,
			device.ReleaseAndGetAddressOf(),
			&feature_level,
			context.ReleaseAndGetAddressOf()
		);
	}
#endif

	win_util::ThrowIfFailed(hr, "D3D11CreateDevice failed");
}

Microsoft::WRL::ComPtr<IDXGIAdapter> get_adapter_from_d3d11device(ID3D11Device* device)
{
	if (!device) {
		throw std::runtime_error("Device is null");
	}

	// ID3D11DeviceをIDXGIDeviceに変換する
	Microsoft::WRL::ComPtr<IDXGIDevice> dxgi_device;

	HRESULT hr = device->QueryInterface(
		IID_PPV_ARGS(dxgi_device.ReleaseAndGetAddressOf())
	);
	win_util::ThrowIfFailed(hr, "ID3D11Device QueryInterface IDXGIDevice failed");

	// デバイスからアダプタを生成

	Microsoft::WRL::ComPtr<IDXGIAdapter> adapter;

	hr = dxgi_device->GetAdapter(adapter.ReleaseAndGetAddressOf());
	win_util::ThrowIfFailed(hr, "IDXGIDevice::GetAdapter failed");

	return adapter;
}

void enable_d3d11_multithread_protection(ID3D11Device* device)
{
	if (!device) {
		throw std::runtime_error("Device is null");
	}

	// ID3D11DeviceをID3D10Multithreadに変換
	Microsoft::WRL::ComPtr<ID3D10Multithread> multithread;

	HRESULT hr = device->QueryInterface(
		IID_PPV_ARGS(multithread.ReleaseAndGetAddressOf())
	);
	win_util::ThrowIfFailed(hr, "ID3D11Device QueryInterface ID3D10Multithread failed");

	// マルチスレッド保護を有効化
	multithread->SetMultithreadProtected(TRUE);
}

