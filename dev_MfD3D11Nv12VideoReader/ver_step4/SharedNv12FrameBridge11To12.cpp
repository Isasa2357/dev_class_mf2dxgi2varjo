#include "SharedNv12FrameBridge11To12.hpp"

#include <stdexcept>
#include <iostream>
#include <sstream>
#include <thread>

#include "HResultUtil.hpp"

SharedNv12FrameBridge11To12::SharedNv12FrameBridge11To12(
    D3D11Core& d3d11_core,
    D3D12Core& d3d12_core,
    UINT width,
    UINT height
)
    : d3d11_core_(d3d11_core)
    , d3d12_core_(d3d12_core)
    , width_(width)
    , height_(height)
{
    if (!this->d3d11_core_.device()) {
        throw std::runtime_error("SharedNv12FrameBridge11To12: D3D11Core device is null");
    }
    if (!this->d3d11_core_.context()) {
        throw std::runtime_error("SharedNv12FrameBridge11To12: D3D11Core context is null");
    }
    if (!this->d3d12_core_.device()) {
        throw std::runtime_error("SharedNv12FrameBridge11To12: D3D12Core device is null");
    }
    if (this->width_ == 0 || this->height_ == 0) {
        throw std::runtime_error("SharedNv12FrameBridge11To12: invalid texture size");
    }

    this->create_shared_nv12_texture();
    this->open_shared_texture_on_d3d12();
    this->create_d3d11_copy_done_query();
}


SharedNv12FrameBridge11To12::ScopedD3D11ReadAccess::ScopedD3D11ReadAccess(
    SharedNv12FrameBridge11To12& bridge
)
    : bridge_(&bridge)
{
    this->bridge_->acquire_for_d3d11_read();
}

SharedNv12FrameBridge11To12::ScopedD3D11ReadAccess::~ScopedD3D11ReadAccess()
{
    if (!this->bridge_) {
        return;
    }

    try {
        this->bridge_->release_from_d3d11_read();
    } catch (...) {
        // Destructors must not throw. In debug-save paths, a release failure
        // will normally be accompanied by a later synchronization/device error.
    }
}

SharedNv12FrameBridge11To12::ScopedD3D11ReadAccess::ScopedD3D11ReadAccess(
    ScopedD3D11ReadAccess&& other
) noexcept
    : bridge_(other.bridge_)
{
    other.bridge_ = nullptr;
}

SharedNv12FrameBridge11To12::ScopedD3D11ReadAccess
SharedNv12FrameBridge11To12::acquire_for_d3d11_read_guard()
{
    return ScopedD3D11ReadAccess(*this);
}

void SharedNv12FrameBridge11To12::copy_from_d3d11_frame_and_wait(
    ID3D11Texture2D* src_texture,
    UINT src_subresource_index
)
{
    if (!src_texture) {
        throw std::runtime_error(
            "SharedNv12FrameBridge11To12::copy_from_d3d11_frame_and_wait: src_texture is null"
        );
    }

    D3D11_TEXTURE2D_DESC src_desc{};
    src_texture->GetDesc(&src_desc);

    if (src_desc.Format != DXGI_FORMAT_NV12) {
        throw std::runtime_error(
            "SharedNv12FrameBridge11To12::copy_from_d3d11_frame_and_wait: source is not NV12"
        );
    }

    if (src_desc.Width != this->width_ || src_desc.Height != this->height_) {
        throw std::runtime_error(
            "SharedNv12FrameBridge11To12::copy_from_d3d11_frame_and_wait: size mismatch"
        );
    }

    HRESULT hr = this->keyed_mutex_11_->AcquireSync(
        0,
        INFINITE
    );
    win_util::ThrowIfFailed(
        hr,
        "D3D11 keyed mutex AcquireSync(0) failed"
    );

    this->d3d11_core_.context()->CopySubresourceRegion(
        this->shared_texture_11_.Get(),
        0,
        0,
        0,
        0,
        src_texture,
        src_subresource_index,
        nullptr
    );

    this->d3d11_core_.context()->End(
        this->copy_done_query_.Get()
    );

    this->d3d11_core_.context()->Flush();

    BOOL done = FALSE;

    while (true) {
        hr = this->d3d11_core_.context()->GetData(
            this->copy_done_query_.Get(),
            &done,
            sizeof(done),
            0
        );

        if (hr == S_OK && done) {
            break;
        }

        if (hr != S_FALSE) {
            this->keyed_mutex_11_->ReleaseSync(0);
            win_util::ThrowIfFailed(
                hr,
                "D3D11 GetData(copy_done_query) failed"
            );
        }

        std::this_thread::yield();
    }

    hr = this->keyed_mutex_11_->ReleaseSync(
        1
    );
    win_util::ThrowIfFailed(
        hr,
        "D3D11 keyed mutex ReleaseSync(1) failed"
    );
}

ID3D11Texture2D* SharedNv12FrameBridge11To12::d3d11_shared_nv12_texture() const
{
    return this->shared_texture_11_.Get();
}

ID3D12Resource* SharedNv12FrameBridge11To12::d3d12_nv12_texture() const
{
    return this->shared_texture_12_.Get();
}

void SharedNv12FrameBridge11To12::acquire_for_d3d11_read()
{
    // After D3D12 read finishes, release_from_d3d12_read_guard() releases key 0.
    // A D3D11-side debug read must therefore acquire key 0, not key 1.
    HRESULT hr = this->keyed_mutex_11_->AcquireSync(
        0,
        INFINITE
    );

    win_util::ThrowIfFailed(
        hr,
        "SharedNv12FrameBridge11To12::acquire_for_d3d11_read: AcquireSync(0) failed"
    );
}

void SharedNv12FrameBridge11To12::release_from_d3d11_read()
{
    // This is read-only access from the D3D11 side, so return ownership to key 0.
    // The next D3D11 copy_from_d3d11_frame_and_wait() also acquires key 0.
    HRESULT hr = this->keyed_mutex_11_->ReleaseSync(
        0
    );

    win_util::ThrowIfFailed(
        hr,
        "SharedNv12FrameBridge11To12::release_from_d3d11_read: ReleaseSync(0) failed"
    );
}

void SharedNv12FrameBridge11To12::acquire_for_d3d12_read_guard()
{
    HRESULT hr = this->keyed_mutex_11_->AcquireSync(
        1,
        INFINITE
    );

    win_util::ThrowIfFailed(
        hr,
        "SharedNv12FrameBridge11To12::acquire_for_d3d12_read_guard: AcquireSync(1) failed"
    );
}

void SharedNv12FrameBridge11To12::release_from_d3d12_read_guard()
{
    HRESULT hr = this->keyed_mutex_11_->ReleaseSync(
        0
    );

    win_util::ThrowIfFailed(
        hr,
        "SharedNv12FrameBridge11To12::release_from_d3d12_read_guard: ReleaseSync(0) failed"
    );
}

UINT SharedNv12FrameBridge11To12::width() const
{
    return this->width_;
}

UINT SharedNv12FrameBridge11To12::height() const
{
    return this->height_;
}

void SharedNv12FrameBridge11To12::create_shared_nv12_texture()
{
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = this->width_;
    desc.Height = this->height_;
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

    HRESULT hr = this->d3d11_core_.device()->CreateTexture2D(
        &desc,
        nullptr,
        this->shared_texture_11_.GetAddressOf()
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

    hr = this->shared_texture_11_.As(
        &this->keyed_mutex_11_
    );

    win_util::ThrowIfFailed(
        hr,
        "Query IDXGIKeyedMutex from shared NV12 texture failed"
    );

    std::wcout
        << L"Created D3D11 shared NV12 texture: "
        << this->width_
        << L"x"
        << this->height_
        << L" MiscFlags=0x"
        << std::hex
        << desc.MiscFlags
        << std::dec
        << L"\n";
}

void SharedNv12FrameBridge11To12::open_shared_texture_on_d3d12()
{
    Microsoft::WRL::ComPtr<IDXGIResource1> dxgi_resource;

    HRESULT hr = this->shared_texture_11_.As(
        &dxgi_resource
    );

    win_util::ThrowIfFailed(
        hr,
        "Query IDXGIResource1 from D3D11 shared texture failed"
    );

    HANDLE shared_handle = nullptr;

    hr = dxgi_resource->CreateSharedHandle(
        nullptr,
        GENERIC_ALL,
        nullptr,
        &shared_handle
    );

    win_util::ThrowIfFailed(
        hr,
        "CreateSharedHandle failed"
    );

    hr = this->d3d12_core_.device()->OpenSharedHandle(
        shared_handle,
        IID_PPV_ARGS(this->shared_texture_12_.GetAddressOf())
    );

    CloseHandle(shared_handle);

    win_util::ThrowIfFailed(
        hr,
        "D3D12 OpenSharedHandle for NV12 texture failed"
    );

    D3D12_RESOURCE_DESC desc12 =
        this->shared_texture_12_->GetDesc();

    std::wcout
        << L"Opened shared NV12 texture on D3D12: "
        << desc12.Width
        << L"x"
        << desc12.Height
        << L" Format="
        << static_cast<int>(desc12.Format)
        << L"\n";
}

void SharedNv12FrameBridge11To12::create_d3d11_copy_done_query()
{
    D3D11_QUERY_DESC desc{};
    desc.Query = D3D11_QUERY_EVENT;
    desc.MiscFlags = 0;

    HRESULT hr = this->d3d11_core_.device()->CreateQuery(
        &desc,
        this->copy_done_query_.GetAddressOf()
    );

    win_util::ThrowIfFailed(
        hr,
        "Create D3D11 copy done query failed"
    );
}