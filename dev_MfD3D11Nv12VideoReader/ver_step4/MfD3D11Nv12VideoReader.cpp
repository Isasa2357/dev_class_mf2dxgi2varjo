#include "MfD3D11Nv12VideoReader.hpp"

#include <stdexcept>
#include <iostream>

#include "HResultUtil.hpp"

MfD3D11Nv12VideoReader::MfD3D11Nv12VideoReader(
    const wchar_t* file_path,
    D3D11Core& d3d11_core
)
    : d3d11_core_(d3d11_core)
{
    if (!file_path) {
        throw std::runtime_error("file_path is null");
    }

    if (!this->d3d11_core_.device()) {
        throw std::runtime_error("D3D11Core device is null");
    }

    this->initialize_dxgi_device_manager();
    this->create_source_reader(file_path);
    this->configure_video_stream_to_nv12();
}

bool MfD3D11Nv12VideoReader::read_next_frame(D3D11VideoFrame& out_frame)
{
    out_frame = {};

    while (true) {
        DWORD stream_index = 0;
        DWORD flags = 0;
        LONGLONG timestamp = 0;

        Microsoft::WRL::ComPtr<IMFSample> sample;

        HRESULT hr = this->reader_->ReadSample(
            MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            0,
            &stream_index,
            &flags,
            &timestamp,
            sample.GetAddressOf()
        );

        win_util::ThrowIfFailed(
            hr,
            "IMFSourceReader::ReadSample failed"
        );

        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
            return false;
        }

        if (flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) {
            this->update_current_video_type();

            if (!sample) {
                continue;
            }
        }

        if (flags & MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED) {
            std::wcout << L"Native media type changed\n";

            if (!sample) {
                continue;
            }
        }

        if (flags & MF_SOURCE_READERF_STREAMTICK) {
            if (!sample) {
                continue;
            }
        }

        if (!sample) {
            continue;
        }

        Microsoft::WRL::ComPtr<IMFMediaBuffer> buffer;

        hr = sample->GetBufferByIndex(
            0,
            buffer.GetAddressOf()
        );

        win_util::ThrowIfFailed(
            hr,
            "IMFSample::GetBufferByIndex failed"
        );

        Microsoft::WRL::ComPtr<IMFDXGIBuffer> dxgi_buffer;

        hr = buffer.As(&dxgi_buffer);

        if (FAILED(hr) || !dxgi_buffer) {
            throw std::runtime_error(
                "The media buffer is not IMFDXGIBuffer. "
                "D3D11 hardware texture output may not be enabled."
            );
        }

        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;

        hr = dxgi_buffer->GetResource(
            __uuidof(ID3D11Texture2D),
            reinterpret_cast<void**>(texture.GetAddressOf())
        );

        win_util::ThrowIfFailed(
            hr,
            "IMFDXGIBuffer::GetResource(ID3D11Texture2D) failed"
        );

        UINT subresource_index = 0;

        hr = dxgi_buffer->GetSubresourceIndex(
            &subresource_index
        );

        win_util::ThrowIfFailed(
            hr,
            "IMFDXGIBuffer::GetSubresourceIndex failed"
        );

        D3D11_TEXTURE2D_DESC desc{};
        texture->GetDesc(&desc);

        /*
        std::wcout
            << L"Decoded texture: "
            << L"Format=" << desc.Format
            << L" Width=" << desc.Width
            << L" Height=" << desc.Height
            << L" ArraySize=" << desc.ArraySize
            << L" MipLevels=" << desc.MipLevels
            << L" BindFlags=0x" << std::hex << desc.BindFlags << std::dec
            << L" MiscFlags=0x" << std::hex << desc.MiscFlags << std::dec
            << L"\n";
        */

        if (desc.Format != DXGI_FORMAT_NV12) {
            throw std::runtime_error(
                "Decoded texture is not DXGI_FORMAT_NV12"
            );
        }

        out_frame.texture = texture;
        out_frame.subresourceIndex = subresource_index;
        out_frame.width = desc.Width;
        out_frame.height = desc.Height;
        out_frame.format = desc.Format;
        out_frame.timestamp100ns = timestamp;
        out_frame.frameIndex = this->frame_index_++;

        return true;
    }
}

UINT MfD3D11Nv12VideoReader::width() const
{
    return this->width_;
}

UINT MfD3D11Nv12VideoReader::height() const
{
    return this->height_;
}

void MfD3D11Nv12VideoReader::initialize_dxgi_device_manager()
{
    UINT reset_token = 0;

    HRESULT hr = MFCreateDXGIDeviceManager(
        &reset_token,
        this->dxgi_device_manager_.GetAddressOf()
    );

    win_util::ThrowIfFailed(
        hr,
        "MFCreateDXGIDeviceManager failed"
    );

    hr = this->dxgi_device_manager_->ResetDevice(
        this->d3d11_core_.device(),
        reset_token
    );

    win_util::ThrowIfFailed(
        hr,
        "IMFDXGIDeviceManager::ResetDevice failed"
    );
}

void MfD3D11Nv12VideoReader::create_source_reader(
    const wchar_t* file_path
)
{
    Microsoft::WRL::ComPtr<IMFAttributes> attrs;

    HRESULT hr = MFCreateAttributes(
        attrs.GetAddressOf(),
        4
    );

    win_util::ThrowIfFailed(
        hr,
        "MFCreateAttributes failed"
    );

    hr = attrs->SetUnknown(
        MF_SOURCE_READER_D3D_MANAGER,
        this->dxgi_device_manager_.Get()
    );

    win_util::ThrowIfFailed(
        hr,
        "Set MF_SOURCE_READER_D3D_MANAGER failed"
    );

    hr = attrs->SetUINT32(
        MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS,
        TRUE
    );

    win_util::ThrowIfFailed(
        hr,
        "Set MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS failed"
    );

    hr = attrs->SetUINT32(
        MF_SOURCE_READER_DISABLE_DXVA,
        FALSE
    );

    win_util::ThrowIfFailed(
        hr,
        "Set MF_SOURCE_READER_DISABLE_DXVA failed"
    );

    // MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING は有効化しない。
    // 有効化すると CPU 側の色変換などが入りやすい。

    hr = MFCreateSourceReaderFromURL(
        file_path,
        attrs.Get(),
        this->reader_.GetAddressOf()
    );

    win_util::ThrowIfFailed(
        hr,
        "MFCreateSourceReaderFromURL failed"
    );

    hr = this->reader_->SetStreamSelection(
        MF_SOURCE_READER_ALL_STREAMS,
        FALSE
    );

    win_util::ThrowIfFailed(
        hr,
        "SetStreamSelection all false failed"
    );

    hr = this->reader_->SetStreamSelection(
        MF_SOURCE_READER_FIRST_VIDEO_STREAM,
        TRUE
    );

    win_util::ThrowIfFailed(
        hr,
        "SetStreamSelection video true failed"
    );
}

void MfD3D11Nv12VideoReader::configure_video_stream_to_nv12()
{
    Microsoft::WRL::ComPtr<IMFMediaType> type;

    HRESULT hr = MFCreateMediaType(
        type.GetAddressOf()
    );

    win_util::ThrowIfFailed(
        hr,
        "MFCreateMediaType failed"
    );

    hr = type->SetGUID(
        MF_MT_MAJOR_TYPE,
        MFMediaType_Video
    );

    win_util::ThrowIfFailed(
        hr,
        "Set MF_MT_MAJOR_TYPE failed"
    );

    hr = type->SetGUID(
        MF_MT_SUBTYPE,
        MFVideoFormat_NV12
    );

    win_util::ThrowIfFailed(
        hr,
        "Set MF_MT_SUBTYPE NV12 failed"
    );

    hr = this->reader_->SetCurrentMediaType(
        MF_SOURCE_READER_FIRST_VIDEO_STREAM,
        nullptr,
        type.Get()
    );

    win_util::ThrowIfFailed(
        hr,
        "SetCurrentMediaType NV12 failed"
    );

    Microsoft::WRL::ComPtr<IMFMediaType> current_type;

    hr = this->reader_->GetCurrentMediaType(
        MF_SOURCE_READER_FIRST_VIDEO_STREAM,
        current_type.GetAddressOf()
    );

    win_util::ThrowIfFailed(
        hr,
        "GetCurrentMediaType failed"
    );

    GUID subtype{};

    hr = current_type->GetGUID(
        MF_MT_SUBTYPE,
        &subtype
    );

    win_util::ThrowIfFailed(
        hr,
        "Get current subtype failed"
    );

    if (subtype != MFVideoFormat_NV12) {
        throw std::runtime_error(
            "Current media type is not MFVideoFormat_NV12"
        );
    }

    UINT32 width = 0;
    UINT32 height = 0;

    hr = MFGetAttributeSize(
        current_type.Get(),
        MF_MT_FRAME_SIZE,
        &width,
        &height
    );

    win_util::ThrowIfFailed(
        hr,
        "MFGetAttributeSize(MF_MT_FRAME_SIZE) failed"
    );

    this->width_ = width;
    this->height_ = height;
}

void MfD3D11Nv12VideoReader::update_current_video_type()
{
    Microsoft::WRL::ComPtr<IMFMediaType> current_type;

    HRESULT hr = this->reader_->GetCurrentMediaType(
        MF_SOURCE_READER_FIRST_VIDEO_STREAM,
        current_type.GetAddressOf()
    );

    win_util::ThrowIfFailed(
        hr,
        "GetCurrentMediaType failed"
    );

    GUID major_type{};

    hr = current_type->GetGUID(
        MF_MT_MAJOR_TYPE,
        &major_type
    );

    win_util::ThrowIfFailed(
        hr,
        "Get MF_MT_MAJOR_TYPE failed"
    );

    if (major_type != MFMediaType_Video) {
        throw std::runtime_error(
            "Current media type is not video"
        );
    }

    GUID subtype{};

    hr = current_type->GetGUID(
        MF_MT_SUBTYPE,
        &subtype
    );

    win_util::ThrowIfFailed(
        hr,
        "Get MF_MT_SUBTYPE failed"
    );

    if (subtype != MFVideoFormat_NV12) {
        throw std::runtime_error(
            "Current media subtype is not NV12"
        );
    }

    UINT32 width = 0;
    UINT32 height = 0;

    hr = MFGetAttributeSize(
        current_type.Get(),
        MF_MT_FRAME_SIZE,
        &width,
        &height
    );

    win_util::ThrowIfFailed(
        hr,
        "MFGetAttributeSize(MF_MT_FRAME_SIZE) failed"
    );

    this->width_ = width;
    this->height_ = height;

    std::wcout
        << L"Current media type changed: "
        << this->width_
        << L" x "
        << this->height_
        << L", subtype=NV12\n";
}