#pragma once

#define NOMINMAX

#include <windows.h>
#include <wrl/client.h>

#include <d3d11.h>

#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

class D3D11Core {

private:

	using ComPtrD3D11Device = Microsoft::WRL::ComPtr<ID3D11Device>;
	using ComPtrD3D11Context = Microsoft::WRL::ComPtr<ID3D11DeviceContext>;
	using ComPtrDXGIAdapter = Microsoft::WRL::ComPtr<IDXGIAdapter>;

public:

	/*
	 * @brief コンストラクタ
	 */
	D3D11Core() = default;

	D3D11Core(const D3D11Core&) = delete;
	D3D11Core& operator=(const D3D11Core&) = delete;

	/*
	 * @brief 初期化
	 */
	void initialize(
		bool enable_debug_layer = false,
		bool enable_multithread_protection = false
	);

	/*
	 * @brief 使用するアダプタを指定した初期化
	 */
	void initialize_with_adapter_LUID(
		const LUID adapter_LUID, 
		bool enable_debug_layer = false,
		bool enable_multithread_protection = false
	);

	ID3D11Device* device() const { return this->device_.Get(); }					// デバイス
	ID3D11DeviceContext* context() const { return this->context_.Get(); }			// コンテキスト
	IDXGIAdapter* adapter() const { return this->adapter_.Get(); }					// アダプタ
	LUID adapter_LUID() const { return this->adapter_LUID_; }						// アダプタのLUID
	const DXGI_ADAPTER_DESC& adapter_desc() const { return this->adapter_desc_; }	// アダプタのディスクリプタ

	void print_adapter_info() const;

	Microsoft::WRL::ComPtr<ID3D11Texture2D> create_texture2D(
		const D3D11_TEXTURE2D_DESC& desc,
		const D3D11_SUBRESOURCE_DATA* init_data = nullptr
	) const;

	Microsoft::WRL::ComPtr<ID3D11Buffer> create_buffer(
		const D3D11_BUFFER_DESC& desc,
		const D3D11_SUBRESOURCE_DATA* init_data = nullptr
	) const;

private:

	ComPtrD3D11Device device_;
	ComPtrD3D11Context context_;
	ComPtrDXGIAdapter adapter_;

	DXGI_ADAPTER_DESC adapter_desc_{};
	D3D_FEATURE_LEVEL feature_level_ = D3D_FEATURE_LEVEL_11_0;

	inline static const std::vector<D3D_FEATURE_LEVEL> d3d11_feature_level_list = {
		D3D_FEATURE_LEVEL_11_1,
		D3D_FEATURE_LEVEL_11_0
	};

	LUID adapter_LUID_{};
};

