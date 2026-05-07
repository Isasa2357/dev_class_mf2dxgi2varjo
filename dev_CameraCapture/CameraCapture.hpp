#pragma once

#define NOMINMAX
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <mfobjects.h>
#include <d3d10_1.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <d3d11_3.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>

#include <vector>
#include <string>
#include <chrono>

struct MFFrame {
	bool is_valid = false;
	bool gpu_backed = false;
	int width = 0;
	int height = 0;
	int64_t frame_id = 0;
	int64_t timestamp;
	ID3D11Texture2D* gpu_texture = nullptr;
};

class BaseCameraCapture {
public:

	~BaseCameraCapture() {
		this->close();
	}

	/**
	 * @brief D3D11デバイスでMFを初期化
	 * @param d3d_device D3D11デバイス
	 */
	bool initialize(ID3D11Device* d3d_device);

	bool open(const std::wstring& symbolic_link);

	void close();

	bool read_frame(MFFrame& frame);

	virtual DXGI_FORMAT desired_format() const = 0;

protected:

	void close_source_reader();

	void close_dxgi_device_mgr();

	virtual IMFMediaType* chooseBestType() = 0;

protected:

	IMFSourceReader* source_reader_ = nullptr;
	IMFDXGIDeviceManager* dxgi_device_mgr_ = nullptr;
	UINT reset_token_ = 0;
	int width_ = 0;
	int height_ = 0;
	int64_t next_frame_id_ = 1;
};

class ELP_USBFHD08S_LC1100_CameraCapture : public BaseCameraCapture {

public:

	DXGI_FORMAT desired_format() const override;

private:

	IMFMediaType* chooseBestType() override;

};