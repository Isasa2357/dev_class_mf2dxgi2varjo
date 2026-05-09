#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <wrl/client.h>

#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_2.h>

#include <cstdint>

#include "D3D11Core.hpp"
#include "D3D12Core.hpp"

class SharedNv12FrameBridge11To12 {
public:
    SharedNv12FrameBridge11To12(
        D3D11Core& d3d11_core,
        D3D12Core& d3d12_core,
        UINT width,
        UINT height
    );

    SharedNv12FrameBridge11To12(const SharedNv12FrameBridge11To12&) = delete;
    SharedNv12FrameBridge11To12& operator=(const SharedNv12FrameBridge11To12&) = delete;

    void copy_from_d3d11_frame_and_wait(
        ID3D11Texture2D* src_texture,
        UINT src_subresource_index
    );

    ID3D11Texture2D* d3d11_shared_nv12_texture() const;
    ID3D12Resource* d3d12_nv12_texture() const;

    void acquire_for_d3d11_read();
    void release_from_d3d11_read();

    void acquire_for_d3d12_read_guard();
    void release_from_d3d12_read_guard();

    UINT width() const;
    UINT height() const;

private:
    void create_shared_nv12_texture();
    void open_shared_texture_on_d3d12();
    void create_d3d11_copy_done_query();

private:
    D3D11Core& d3d11_core_;
    D3D12Core& d3d12_core_;

    UINT width_ = 0;
    UINT height_ = 0;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> shared_texture_11_;
    Microsoft::WRL::ComPtr<ID3D12Resource> shared_texture_12_;

    Microsoft::WRL::ComPtr<ID3D11Query> copy_done_query_;
    Microsoft::WRL::ComPtr<IDXGIKeyedMutex> keyed_mutex_11_;
};