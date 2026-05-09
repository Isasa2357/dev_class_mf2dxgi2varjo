#include "D3D11Core.hpp"

#include <stdexcept>
#include <iostream>

#include "DXGI_util.hpp"
#include "D3D11_mutil.hpp"
#include "HResultUtil.hpp"

void D3D11Core::initialize(
	bool enable_debug_layer, 
	bool enable_multithread_protection
) {

	// デバイスの作成
	create_d3d11device(this->device_, this->context_, this->feature_level_, enable_debug_layer);

	// アダプタの作成
	this->adapter_ = get_adapter_from_d3d11device(this->device_.Get());

	// アダプタのディスクリプタの作成
	auto hr = this->adapter_->GetDesc(&this->adapter_desc_);
	win_util::ThrowIfFailed(hr, "IDXGIAdapter::GetDesc failed");

	this->adapter_LUID_ = this->adapter_desc_.AdapterLuid;

	if (enable_multithread_protection) {
		enable_d3d11_multithread_protection(this->device_.Get());
	}
}

void D3D11Core::initialize_with_adapter_LUID(
	const LUID adapter_LUID, 
	bool enable_debug_layer, 
	bool enable_multithread_protection
) {
	// アダプタの作成
	auto found_adapter = find_adapter_by_LUID(adapter_LUID);

	// デバイスの作成
	create_d3d11device_on_adapter(
		this->device_,
		this->context_,
		this->feature_level_,
		found_adapter.Get(),
		enable_debug_layer
	);

	this->adapter_ = get_adapter_from_d3d11device(this->device_.Get());

	auto hr = this->adapter_->GetDesc(&this->adapter_desc_);
	win_util::ThrowIfFailed(hr, "IDXGIAdapter::GetDesc failed");

	// デバイスのアダプタのLUIDと指定されたLUIDが等しいかどうかを確認
	if (!is_same_LUID(this->adapter_desc_.AdapterLuid, adapter_LUID)) {
		throw std::runtime_error("D3D11Core::InitializeWithAdapterLuid: created adapter LUID mismatch");
	}

	this->adapter_LUID_ = this->adapter_desc_.AdapterLuid;

	// コンテキストのマルチスレッド保護の有効化
	if (enable_multithread_protection) {
		enable_d3d11_multithread_protection(this->device_.Get());
	}
}

void D3D11Core::print_adapter_info() const
{
	std::wcout
		<< L"D3D11 Adapter: " << this->adapter_desc_.Description << L"\n"
		<< L"D3D11 Adapter LUID: "
		<< this->adapter_desc_.AdapterLuid.HighPart
		<< L":"
		<< this->adapter_desc_.AdapterLuid.LowPart
		<< L"\n"
		<< L"D3D11 Feature Level: 0x"
		<< std::hex
		<< static_cast<unsigned int>(this->feature_level_)
		<< std::dec
		<< L"\n";
}

Microsoft::WRL::ComPtr<ID3D11Texture2D> D3D11Core::create_texture2D(
	const D3D11_TEXTURE2D_DESC& desc, 
	const D3D11_SUBRESOURCE_DATA* init_data) const
{
	Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;

	auto hr = this->device_->CreateTexture2D(
		&desc,
		init_data,
		texture.GetAddressOf()
	);

	win_util::ThrowIfFailed(hr, "CreateTexture2D is failed");

	return texture;
}

Microsoft::WRL::ComPtr<ID3D11Buffer> D3D11Core::create_buffer(
	const D3D11_BUFFER_DESC& desc, 
	const D3D11_SUBRESOURCE_DATA* init_data
) const {
	Microsoft::WRL::ComPtr<ID3D11Buffer> buffer;

	auto hr = this->device_->CreateBuffer(
		&desc,
		init_data,
		buffer.GetAddressOf()
	);

	win_util::ThrowIfFailed(hr, "CreateBuffer is failed");

	return buffer;
}
