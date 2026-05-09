#pragma once

#define NOMINMAX

#include <windows.h>
#include <wrl/client.h>

#include <d3d11.h>
#include <dxgi.h>

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mfobjects.h>

#include <cstdint>

#include "D3D11Core.hpp"

struct D3D11VideoFrame {
    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    UINT subresourceIndex = 0;

    UINT width = 0;
    UINT height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;

    LONGLONG timestamp100ns = 0;
    uint64_t frameIndex = 0;
};

class MfD3D11Nv12VideoReader {
public:
    MfD3D11Nv12VideoReader(
        const wchar_t* file_path,
        D3D11Core& d3d11_core
    );

    MfD3D11Nv12VideoReader(const MfD3D11Nv12VideoReader&) = delete;
    MfD3D11Nv12VideoReader& operator=(const MfD3D11Nv12VideoReader&) = delete;

    bool read_next_frame(D3D11VideoFrame& out_frame);

    UINT width() const;
    UINT height() const;

private:
    void initialize_dxgi_device_manager();
    void create_source_reader(const wchar_t* file_path);
    void configure_video_stream_to_nv12();
    void update_current_video_type();

private:
    D3D11Core& d3d11_core_;

    Microsoft::WRL::ComPtr<IMFDXGIDeviceManager> dxgi_device_manager_;
    Microsoft::WRL::ComPtr<IMFSourceReader> reader_;

    UINT width_ = 0;
    UINT height_ = 0;

    uint64_t frame_index_ = 0;
};