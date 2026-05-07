#pragma once

#define NOMINMAX

#include <windows.h>
#include <wrl/client.h>
#include <comdef.h>

#include <d3d11.h>
#include <d3d10.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <thread>

#include "util.hpp"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "d3d12.lib")

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

class SharedNv12FrameBridge11To12 {
public:
    SharedNv12FrameBridge11To12(
        ID3D11Device* device11,
        ID3D11DeviceContext* context11,
        ID3D12Device* device12,
        UINT width,
        UINT height
    )
        : m_device11(device11)
        , m_context11(context11)
        , m_device12(device12)
        , m_width(width)
        , m_height(height)
    {
        if (!m_device11 || !m_context11 || !m_device12) {
            throw std::runtime_error("SharedNv12FrameBridge11To12: null device/context");
        }

        CreateSharedNv12Texture();
        OpenSharedTextureOnD3D12();
        CreateD3D11CopyDoneQuery();
    }

    void CopyFromD3D11FrameAndWait(
        ID3D11Texture2D* srcTexture,
        UINT srcSubresourceIndex
    )
    {
        if (!srcTexture) {
            throw std::runtime_error("CopyFromD3D11FrameAndWait: srcTexture is null");
        }

        D3D11_TEXTURE2D_DESC srcDesc{};
        srcTexture->GetDesc(&srcDesc);

        if (srcDesc.Format != DXGI_FORMAT_NV12) {
            throw std::runtime_error("CopyFromD3D11FrameAndWait: source is not NV12");
        }

        if (srcDesc.Width != m_width || srcDesc.Height != m_height) {
            throw std::runtime_error("CopyFromD3D11FrameAndWait: size mismatch");
        }

        HRESULT hr = m_keyedMutex11->AcquireSync(0, INFINITE);
        ThrowIfFailed(hr, "D3D11 keyed mutex AcquireSync(0) failed");

        m_context11->CopySubresourceRegion(
            m_sharedTexture11.Get(),
            0,
            0, 0, 0,
            srcTexture,
            srcSubresourceIndex,
            nullptr
        );

        m_context11->End(m_copyDoneQuery.Get());
        m_context11->Flush();

        BOOL done = FALSE;
        while (true) {
            hr = m_context11->GetData(
                m_copyDoneQuery.Get(),
                &done,
                sizeof(done),
                0
            );

            if (hr == S_OK && done) {
                break;
            }

            if (hr != S_FALSE) {
                m_keyedMutex11->ReleaseSync(0);
                ThrowIfFailed(hr, "D3D11 GetData(copyDoneQuery) failed");
            }

            std::this_thread::yield();
        }

        // 次の利用者に key=1 で渡す
        hr = m_keyedMutex11->ReleaseSync(1);
        ThrowIfFailed(hr, "D3D11 keyed mutex ReleaseSync(1) failed");
    }

    ID3D12Resource* GetD3D12Nv12Texture() const
    {
        return m_sharedTexture12.Get();
    }

    ID3D11Texture2D* GetD3D11SharedNv12Texture() const
    {
        return m_sharedTexture11.Get();
    }

    void AcquireForD3D11Read()
    {
        HRESULT hr = m_keyedMutex11->AcquireSync(1, INFINITE);
        ThrowIfFailed(hr, "AcquireForD3D11Read: AcquireSync(1) failed");
    }

    void ReleaseFromD3D11Read()
    {
        // 読み終わったので key=0 に戻す
        HRESULT hr = m_keyedMutex11->ReleaseSync(0);
        ThrowIfFailed(hr, "ReleaseFromD3D11Read: ReleaseSync(0) failed");
    }

    void AcquireForD3D12ReadGuard()
    {
        HRESULT hr = m_keyedMutex11->AcquireSync(1, INFINITE);
        ThrowIfFailed(hr, "AcquireForD3D12ReadGuard: AcquireSync(1) failed");
    }

    void ReleaseFromD3D12ReadGuard()
    {
        HRESULT hr = m_keyedMutex11->ReleaseSync(0);
        ThrowIfFailed(hr, "ReleaseFromD3D12ReadGuard: ReleaseSync(0) failed");
    }

    UINT Width() const { return m_width; }
    UINT Height() const { return m_height; }

private:
    void CreateSharedNv12Texture()
    {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = m_width;
        desc.Height = m_height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_NV12;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;

        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

        desc.CPUAccessFlags = 0;

        desc.MiscFlags =
            D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
            D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

        HRESULT hr = m_device11->CreateTexture2D(
            &desc,
            nullptr,
            m_sharedTexture11.GetAddressOf()
        );

        if (FAILED(hr)) {
            std::stringstream ss;
            ss << "Create shared NV12 texture on D3D11 failed. HRESULT=0x"
                << std::hex << static_cast<unsigned long>(hr)
                << " width=0x" << desc.Width
                << " height=0x" << desc.Height
                << " format=0x" << desc.Format
                << " bind=0x" << desc.BindFlags
                << " misc=0x" << desc.MiscFlags;

            throw std::runtime_error(ss.str());
        }

        hr = m_sharedTexture11.As(&m_keyedMutex11);
        ThrowIfFailed(hr, "Query IDXGIKeyedMutex from shared NV12 texture failed");

        std::wcout
            << L"Created D3D11 shared NV12 texture: "
            << m_width << L"x" << m_height
            << L" MiscFlags=0x" << std::hex << desc.MiscFlags << std::dec
            << L"\n";
    }

    void OpenSharedTextureOnD3D12()
    {
        ComPtr<IDXGIResource1> dxgiResource;
        HRESULT hr = m_sharedTexture11.As(&dxgiResource);
        ThrowIfFailed(hr, "Query IDXGIResource1 from D3D11 shared texture failed");

        HANDLE sharedHandle = nullptr;

        hr = dxgiResource->CreateSharedHandle(
            nullptr,
            GENERIC_ALL,
            nullptr,
            &sharedHandle
        );
        ThrowIfFailed(hr, "CreateSharedHandle failed");

        hr = m_device12->OpenSharedHandle(
            sharedHandle,
            IID_PPV_ARGS(m_sharedTexture12.GetAddressOf())
        );

        CloseHandle(sharedHandle);

        ThrowIfFailed(hr, "D3D12 OpenSharedHandle for NV12 texture failed");

        D3D12_RESOURCE_DESC desc12 = m_sharedTexture12->GetDesc();

        std::wcout
            << L"Opened shared NV12 texture on D3D12: "
            << desc12.Width << L"x" << desc12.Height
            << L" Format=" << static_cast<int>(desc12.Format)
            << L"\n";
    }

    void CreateD3D11CopyDoneQuery()
    {
        D3D11_QUERY_DESC desc{};
        desc.Query = D3D11_QUERY_EVENT;
        desc.MiscFlags = 0;

        HRESULT hr = m_device11->CreateQuery(
            &desc,
            m_copyDoneQuery.GetAddressOf()
        );
        ThrowIfFailed(hr, "Create D3D11 copy done query failed");
    }

private:
    ComPtr<ID3D11Device> m_device11;
    ComPtr<ID3D11DeviceContext> m_context11;
    ComPtr<ID3D12Device> m_device12;

    UINT m_width = 0;
    UINT m_height = 0;

    ComPtr<ID3D11Texture2D> m_sharedTexture11;
    ComPtr<ID3D12Resource> m_sharedTexture12;

    ComPtr<ID3D11Query> m_copyDoneQuery;

    ComPtr<IDXGIKeyedMutex> m_keyedMutex11;
};