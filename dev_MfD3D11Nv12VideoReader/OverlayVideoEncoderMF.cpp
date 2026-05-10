#include "OverlayVideoEncoderMF.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <sstream>

#include "HResultUtil.hpp"

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

namespace {

    uint8_t ClampToU8(int v)
    {
        return static_cast<uint8_t>(std::clamp(v, 0, 255));
    }

    uint8_t Float01ToU8(float v)
    {
        v = std::clamp(v, 0.0f, 1.0f);
        return static_cast<uint8_t>(v * 255.0f + 0.5f);
    }

    void GetClassColorRgb(UINT class_id, uint8_t& r, uint8_t& g, uint8_t& b)
    {
        static constexpr uint8_t colors[][3] = {
            {255,   0,   0},
            {  0, 255,   0},
            {  0, 128, 255},
            {255, 255,   0},
            {255,   0, 255},
            {  0, 255, 255},
        };

        const auto& c = colors[class_id % (sizeof(colors) / sizeof(colors[0]))];
        r = c[0];
        g = c[1];
        b = c[2];
    }

    void GetTipClassColorRgb(UINT class_id, uint8_t& r, uint8_t& g, uint8_t& b)
    {
        GetClassColorRgb(class_id, r, g, b);
    }

    void SetPixelBgra(
        std::vector<uint8_t>& bgra,
        UINT width,
        UINT height,
        int x,
        int y,
        uint8_t r,
        uint8_t g,
        uint8_t b,
        uint8_t a = 255
    )
    {
        if (x < 0 || y < 0) {
            return;
        }
        if (x >= static_cast<int>(width) || y >= static_cast<int>(height)) {
            return;
        }

        const size_t idx =
            (static_cast<size_t>(y) * static_cast<size_t>(width) +
             static_cast<size_t>(x)) * 4;

        bgra[idx + 0] = b;
        bgra[idx + 1] = g;
        bgra[idx + 2] = r;
        bgra[idx + 3] = a;
    }

    void BlendPixelBgra(
        std::vector<uint8_t>& bgra,
        UINT width,
        UINT height,
        int x,
        int y,
        uint8_t r,
        uint8_t g,
        uint8_t b,
        float alpha
    )
    {
        if (x < 0 || y < 0) {
            return;
        }
        if (x >= static_cast<int>(width) || y >= static_cast<int>(height)) {
            return;
        }

        alpha = std::clamp(alpha, 0.0f, 1.0f);
        const float inv_alpha = 1.0f - alpha;

        const size_t idx =
            (static_cast<size_t>(y) * static_cast<size_t>(width) +
             static_cast<size_t>(x)) * 4;

        const float old_b = static_cast<float>(bgra[idx + 0]);
        const float old_g = static_cast<float>(bgra[idx + 1]);
        const float old_r = static_cast<float>(bgra[idx + 2]);

        bgra[idx + 0] = Float01ToU8((old_b * inv_alpha + static_cast<float>(b) * alpha) / 255.0f);
        bgra[idx + 1] = Float01ToU8((old_g * inv_alpha + static_cast<float>(g) * alpha) / 255.0f);
        bgra[idx + 2] = Float01ToU8((old_r * inv_alpha + static_cast<float>(r) * alpha) / 255.0f);
        bgra[idx + 3] = 255;
    }

    void DrawLineBgra(
        std::vector<uint8_t>& bgra,
        UINT width,
        UINT height,
        int x0,
        int y0,
        int x1,
        int y1,
        uint8_t r,
        uint8_t g,
        uint8_t b
    )
    {
        int dx = std::abs(x1 - x0);
        int sx = x0 < x1 ? 1 : -1;
        int dy = -std::abs(y1 - y0);
        int sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;

        while (true) {
            SetPixelBgra(bgra, width, height, x0, y0, r, g, b);

            if (x0 == x1 && y0 == y1) {
                break;
            }

            const int e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx) {
                err += dx;
                y0 += sy;
            }
        }
    }

    void DrawCrossBgra(
        std::vector<uint8_t>& bgra,
        UINT width,
        UINT height,
        int cx,
        int cy,
        uint8_t r,
        uint8_t g,
        uint8_t b,
        int radius
    )
    {
        for (int i = -radius; i <= radius; ++i) {
            SetPixelBgra(bgra, width, height, cx + i, cy + i, r, g, b);
            SetPixelBgra(bgra, width, height, cx + i, cy - i, r, g, b);
        }
    }

    void DrawFilledCircleBgra(
        std::vector<uint8_t>& bgra,
        UINT width,
        UINT height,
        int cx,
        int cy,
        int radius,
        uint8_t r,
        uint8_t g,
        uint8_t b
    )
    {
        if (radius <= 0) {
            return;
        }

        const int r2 = radius * radius;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx * dx + dy * dy <= r2) {
                    SetPixelBgra(bgra, width, height, cx + dx, cy + dy, r, g, b);
                }
            }
        }
    }

    void DrawCircleOutlineBgra(
        std::vector<uint8_t>& bgra,
        UINT width,
        UINT height,
        int cx,
        int cy,
        int radius,
        uint8_t r,
        uint8_t g,
        uint8_t b,
        int thickness
    )
    {
        if (radius <= 0 || thickness <= 0) {
            return;
        }

        const int outer = radius;
        const int inner = std::max(0, radius - thickness);
        const int outer2 = outer * outer;
        const int inner2 = inner * inner;

        for (int dy = -outer; dy <= outer; ++dy) {
            for (int dx = -outer; dx <= outer; ++dx) {
                const int d2 = dx * dx + dy * dy;
                if (d2 <= outer2 && d2 >= inner2) {
                    SetPixelBgra(bgra, width, height, cx + dx, cy + dy, r, g, b);
                }
            }
        }
    }

    void DrawTipMarkerBgra(
        std::vector<uint8_t>& bgra,
        UINT width,
        UINT height,
        int x,
        int y,
        uint8_t r,
        uint8_t g,
        uint8_t b
    )
    {
        DrawCircleOutlineBgra(bgra, width, height, x, y, 24, 255, 255, 255, 5);
        DrawCircleOutlineBgra(bgra, width, height, x, y, 17, r, g, b, 5);
        DrawFilledCircleBgra(bgra, width, height, x, y, 4, 255, 255, 255);
    }

    float InputToOriginalCoord(float input_coord, float pad, float scale, float max_value)
    {
        if (scale <= 0.0f) {
            return 0.0f;
        }

        const float v = (input_coord - pad) / scale;
        return std::clamp(v, 0.0f, max_value);
    }

} // namespace

OverlayVideoEncoderMF::OverlayVideoEncoderMF(const Config& config)
{
    this->open(config);
}

OverlayVideoEncoderMF::~OverlayVideoEncoderMF()
{
    this->close();
}

void OverlayVideoEncoderMF::open(const Config& config)
{
    if (this->sink_writer_) {
        this->close();
    }

    if (config.output_path.empty()) {
        throw std::runtime_error("OverlayVideoEncoderMF::open: output_path is empty");
    }
    if (config.width == 0 || config.height == 0) {
        throw std::runtime_error("OverlayVideoEncoderMF::open: invalid frame size");
    }
    if (config.fps_num == 0 || config.fps_den == 0) {
        throw std::runtime_error("OverlayVideoEncoderMF::open: invalid frame rate");
    }

    this->config_ = config;
    this->frame_index_ = 0;

    HRESULT hr = MFCreateSinkWriterFromURL(
        this->config_.output_path.c_str(),
        nullptr,
        nullptr,
        this->sink_writer_.ReleaseAndGetAddressOf()
    );
    win_util::ThrowIfFailed(hr, "MFCreateSinkWriterFromURL failed");

    Microsoft::WRL::ComPtr<IMFMediaType> output_type;
    hr = MFCreateMediaType(output_type.GetAddressOf());
    win_util::ThrowIfFailed(hr, "MFCreateMediaType output failed");

    hr = output_type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    win_util::ThrowIfFailed(hr, "Set output MF_MT_MAJOR_TYPE failed");

    hr = output_type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
    win_util::ThrowIfFailed(hr, "Set output MF_MT_SUBTYPE H264 failed");

    hr = output_type->SetUINT32(MF_MT_AVG_BITRATE, this->config_.bitrate);
    win_util::ThrowIfFailed(hr, "Set output MF_MT_AVG_BITRATE failed");

    hr = output_type->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    win_util::ThrowIfFailed(hr, "Set output MF_MT_INTERLACE_MODE failed");

    hr = MFSetAttributeSize(output_type.Get(), MF_MT_FRAME_SIZE, this->config_.width, this->config_.height);
    win_util::ThrowIfFailed(hr, "Set output MF_MT_FRAME_SIZE failed");

    hr = MFSetAttributeRatio(output_type.Get(), MF_MT_FRAME_RATE, this->config_.fps_num, this->config_.fps_den);
    win_util::ThrowIfFailed(hr, "Set output MF_MT_FRAME_RATE failed");

    hr = MFSetAttributeRatio(output_type.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
    win_util::ThrowIfFailed(hr, "Set output MF_MT_PIXEL_ASPECT_RATIO failed");

    hr = this->sink_writer_->AddStream(output_type.Get(), &this->stream_index_);
    win_util::ThrowIfFailed(hr, "IMFSinkWriter::AddStream failed");

    Microsoft::WRL::ComPtr<IMFMediaType> input_type;
    hr = MFCreateMediaType(input_type.GetAddressOf());
    win_util::ThrowIfFailed(hr, "MFCreateMediaType input failed");

    hr = input_type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    win_util::ThrowIfFailed(hr, "Set input MF_MT_MAJOR_TYPE failed");

    hr = input_type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
    win_util::ThrowIfFailed(hr, "Set input MF_MT_SUBTYPE RGB32 failed");

    hr = input_type->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    win_util::ThrowIfFailed(hr, "Set input MF_MT_INTERLACE_MODE failed");

    hr = MFSetAttributeSize(input_type.Get(), MF_MT_FRAME_SIZE, this->config_.width, this->config_.height);
    win_util::ThrowIfFailed(hr, "Set input MF_MT_FRAME_SIZE failed");

    hr = MFSetAttributeRatio(input_type.Get(), MF_MT_FRAME_RATE, this->config_.fps_num, this->config_.fps_den);
    win_util::ThrowIfFailed(hr, "Set input MF_MT_FRAME_RATE failed");

    hr = MFSetAttributeRatio(input_type.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
    win_util::ThrowIfFailed(hr, "Set input MF_MT_PIXEL_ASPECT_RATIO failed");

    // Positive stride for top-down BGRA/RGB32 memory in our buffer.
    hr = input_type->SetUINT32(MF_MT_DEFAULT_STRIDE, this->config_.width * 4);
    win_util::ThrowIfFailed(hr, "Set input MF_MT_DEFAULT_STRIDE failed");

    hr = this->sink_writer_->SetInputMediaType(this->stream_index_, input_type.Get(), nullptr);
    win_util::ThrowIfFailed(hr, "IMFSinkWriter::SetInputMediaType failed");

    hr = this->sink_writer_->BeginWriting();
    win_util::ThrowIfFailed(hr, "IMFSinkWriter::BeginWriting failed");
}

void OverlayVideoEncoderMF::close()
{
    if (this->sink_writer_) {
        this->sink_writer_->Finalize();
        this->sink_writer_.Reset();
    }
}

LONGLONG OverlayVideoEncoderMF::frame_duration_100ns() const
{
    return static_cast<LONGLONG>(
        (10'000'000ULL * static_cast<UINT64>(this->config_.fps_den)) /
        static_cast<UINT64>(this->config_.fps_num)
    );
}

LONGLONG OverlayVideoEncoderMF::frame_time_100ns(uint64_t index) const
{
    return static_cast<LONGLONG>(index) * this->frame_duration_100ns();
}

void OverlayVideoEncoderMF::write_bgra_frame(
    const uint8_t* bgra,
    size_t byte_size
)
{
    if (!this->sink_writer_) {
        throw std::runtime_error("OverlayVideoEncoderMF::write_bgra_frame: encoder is not open");
    }
    if (!bgra) {
        throw std::runtime_error("OverlayVideoEncoderMF::write_bgra_frame: bgra is null");
    }

    const size_t expected_size =
        static_cast<size_t>(this->config_.width) *
        static_cast<size_t>(this->config_.height) * 4u;

    if (byte_size < expected_size) {
        throw std::runtime_error("OverlayVideoEncoderMF::write_bgra_frame: input buffer is too small");
    }

    Microsoft::WRL::ComPtr<IMFMediaBuffer> buffer;
    HRESULT hr = MFCreateMemoryBuffer(
        static_cast<DWORD>(expected_size),
        buffer.GetAddressOf()
    );
    win_util::ThrowIfFailed(hr, "MFCreateMemoryBuffer failed");

    BYTE* dst = nullptr;
    DWORD max_len = 0;
    DWORD current_len = 0;
    hr = buffer->Lock(&dst, &max_len, &current_len);
    win_util::ThrowIfFailed(hr, "IMFMediaBuffer::Lock failed");

    std::memcpy(dst, bgra, expected_size);

    hr = buffer->Unlock();
    win_util::ThrowIfFailed(hr, "IMFMediaBuffer::Unlock failed");

    hr = buffer->SetCurrentLength(static_cast<DWORD>(expected_size));
    win_util::ThrowIfFailed(hr, "IMFMediaBuffer::SetCurrentLength failed");

    Microsoft::WRL::ComPtr<IMFSample> sample;
    hr = MFCreateSample(sample.GetAddressOf());
    win_util::ThrowIfFailed(hr, "MFCreateSample failed");

    hr = sample->AddBuffer(buffer.Get());
    win_util::ThrowIfFailed(hr, "IMFSample::AddBuffer failed");

    hr = sample->SetSampleTime(this->frame_time_100ns(this->frame_index_));
    win_util::ThrowIfFailed(hr, "IMFSample::SetSampleTime failed");

    hr = sample->SetSampleDuration(this->frame_duration_100ns());
    win_util::ThrowIfFailed(hr, "IMFSample::SetSampleDuration failed");

    hr = this->sink_writer_->WriteSample(this->stream_index_, sample.Get());
    win_util::ThrowIfFailed(hr, "IMFSinkWriter::WriteSample failed");

    ++this->frame_index_;
}

std::vector<uint8_t> OverlayVideoEncoderMF::readback_d3d11_nv12_texture_as_bgra(
    D3D11Core& d3d11_core,
    ID3D11Texture2D* nv12_texture,
    UINT source_subresource_index,
    UINT width,
    UINT height
)
{
    if (!d3d11_core.device() || !d3d11_core.context()) {
        throw std::runtime_error("readback_d3d11_nv12_texture_as_bgra: D3D11Core is not initialized");
    }
    if (!nv12_texture) {
        throw std::runtime_error("readback_d3d11_nv12_texture_as_bgra: nv12_texture is null");
    }

    D3D11_TEXTURE2D_DESC src_desc{};
    nv12_texture->GetDesc(&src_desc);

    if (src_desc.Format != DXGI_FORMAT_NV12) {
        throw std::runtime_error("readback_d3d11_nv12_texture_as_bgra: source is not NV12");
    }
    if (width == 0 || height == 0 || width > src_desc.Width || height > src_desc.Height) {
        throw std::runtime_error("readback_d3d11_nv12_texture_as_bgra: invalid size");
    }

    D3D11_TEXTURE2D_DESC staging_desc{};
    staging_desc.Width = width;
    staging_desc.Height = height;
    staging_desc.MipLevels = 1;
    staging_desc.ArraySize = 1;
    staging_desc.Format = DXGI_FORMAT_NV12;
    staging_desc.SampleDesc.Count = 1;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
    HRESULT hr = d3d11_core.device()->CreateTexture2D(
        &staging_desc,
        nullptr,
        staging.GetAddressOf()
    );
    win_util::ThrowIfFailed(hr, "CreateTexture2D staging NV12 failed");

    D3D11_BOX box{};
    box.left = 0;
    box.top = 0;
    box.front = 0;
    box.right = width;
    box.bottom = height;
    box.back = 1;

    d3d11_core.context()->CopySubresourceRegion(
        staging.Get(),
        0,
        0,
        0,
        0,
        nv12_texture,
        source_subresource_index,
        &box
    );

    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = d3d11_core.context()->Map(
        staging.Get(),
        0,
        D3D11_MAP_READ,
        0,
        &mapped
    );
    win_util::ThrowIfFailed(hr, "Map staging NV12 failed");

    std::vector<uint8_t> bgra(
        static_cast<size_t>(width) * static_cast<size_t>(height) * 4u
    );

    const auto* base = static_cast<const uint8_t*>(mapped.pData);
    const UINT pitch = mapped.RowPitch;

    const uint8_t* y_plane = base;
    const uint8_t* uv_plane = base + static_cast<size_t>(pitch) * static_cast<size_t>(height);

    for (UINT y = 0; y < height; ++y) {
        const uint8_t* y_row = y_plane + static_cast<size_t>(pitch) * y;
        const uint8_t* uv_row = uv_plane + static_cast<size_t>(pitch) * (y / 2u);

        for (UINT x = 0; x < width; ++x) {
            const int yy = static_cast<int>(y_row[x]);
            const int u = static_cast<int>(uv_row[(x / 2u) * 2u + 0]);
            const int v = static_cast<int>(uv_row[(x / 2u) * 2u + 1]);

            const int c = yy - 16;
            const int d = u - 128;
            const int e = v - 128;

            const int r = (298 * c + 409 * e + 128) >> 8;
            const int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            const int b = (298 * c + 516 * d + 128) >> 8;

            const size_t idx =
                (static_cast<size_t>(y) * static_cast<size_t>(width) +
                 static_cast<size_t>(x)) * 4u;

            bgra[idx + 0] = ClampToU8(b);
            bgra[idx + 1] = ClampToU8(g);
            bgra[idx + 2] = ClampToU8(r);
            bgra[idx + 3] = 255;
        }
    }

    d3d11_core.context()->Unmap(staging.Get(), 0);

    return bgra;
}

void OverlayVideoEncoderMF::overlay_masks_and_tips_bgra(
    std::vector<uint8_t>& bgra,
    const std::vector<YoloSegPostProcessorD3D12::DetectionWithMask>& mask_results,
    const std::vector<ToolTipDetectorD3D12::TipResult>& tip_results,
    float letterbox_scale,
    float letterbox_pad_x,
    float letterbox_pad_y
) const
{
    if (this->config_.input_width <= 0.0f ||
        this->config_.input_height <= 0.0f ||
        this->config_.mask_width <= 0.0f ||
        this->config_.mask_height <= 0.0f)
    {
        throw std::runtime_error("overlay_masks_and_tips_bgra: invalid coordinate size");
    }
    if (letterbox_scale <= 0.0f) {
        throw std::runtime_error("overlay_masks_and_tips_bgra: invalid letterbox scale");
    }

    const UINT width = this->config_.width;
    const UINT height = this->config_.height;
    const float max_x = static_cast<float>(width - 1);
    const float max_y = static_cast<float>(height - 1);

    // Mask overlay. Iterate original-space bbox only.
    for (const auto& result : mask_results) {
        const auto& det = result.detection;

        if (det.score < this->config_.score_threshold) {
            continue;
        }
        if (result.mask_width == 0 || result.mask_height == 0) {
            continue;
        }

        const size_t mask_pixels =
            static_cast<size_t>(result.mask_width) * static_cast<size_t>(result.mask_height);
        if (result.mask.size() < mask_pixels) {
            continue;
        }

        uint8_t cr = 255;
        uint8_t cg = 0;
        uint8_t cb = 0;
        GetClassColorRgb(det.class_id, cr, cg, cb);

        const float box_orig_x1 = InputToOriginalCoord(det.x1, letterbox_pad_x, letterbox_scale, max_x);
        const float box_orig_y1 = InputToOriginalCoord(det.y1, letterbox_pad_y, letterbox_scale, max_y);
        const float box_orig_x2 = InputToOriginalCoord(det.x2, letterbox_pad_x, letterbox_scale, max_x);
        const float box_orig_y2 = InputToOriginalCoord(det.y2, letterbox_pad_y, letterbox_scale, max_y);

        int x1 = static_cast<int>(std::floor(std::min(box_orig_x1, box_orig_x2)));
        int y1 = static_cast<int>(std::floor(std::min(box_orig_y1, box_orig_y2)));
        int x2 = static_cast<int>(std::ceil(std::max(box_orig_x1, box_orig_x2)));
        int y2 = static_cast<int>(std::ceil(std::max(box_orig_y1, box_orig_y2)));

        x1 = std::clamp(x1, 0, static_cast<int>(width) - 1);
        y1 = std::clamp(y1, 0, static_cast<int>(height) - 1);
        x2 = std::clamp(x2, 0, static_cast<int>(width) - 1);
        y2 = std::clamp(y2, 0, static_cast<int>(height) - 1);

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        for (int y = y1; y <= y2; ++y) {
            const float input_y = static_cast<float>(y) * letterbox_scale + letterbox_pad_y;
            if (input_y < 0.0f || input_y >= this->config_.input_height) {
                continue;
            }

            int my = static_cast<int>(
                (input_y + 0.5f) * static_cast<float>(result.mask_height) / this->config_.input_height
            );
            my = std::clamp(my, 0, static_cast<int>(result.mask_height) - 1);

            for (int x = x1; x <= x2; ++x) {
                const float input_x = static_cast<float>(x) * letterbox_scale + letterbox_pad_x;
                if (input_x < 0.0f || input_x >= this->config_.input_width) {
                    continue;
                }

                int mx = static_cast<int>(
                    (input_x + 0.5f) * static_cast<float>(result.mask_width) / this->config_.input_width
                );
                mx = std::clamp(mx, 0, static_cast<int>(result.mask_width) - 1);

                const size_t mask_idx =
                    static_cast<size_t>(my) * static_cast<size_t>(result.mask_width) +
                    static_cast<size_t>(mx);

                if (result.mask[mask_idx] < this->config_.mask_threshold) {
                    continue;
                }

                BlendPixelBgra(
                    bgra,
                    width,
                    height,
                    x,
                    y,
                    cr,
                    cg,
                    cb,
                    this->config_.mask_alpha
                );
            }
        }
    }

    // Tip / candidate / axis overlay.
    for (const auto& tip : tip_results) {
        if (tip.valid == 0) {
            continue;
        }

        uint8_t cr = 255;
        uint8_t cg = 0;
        uint8_t cb = 0;
        GetTipClassColorRgb(tip.class_id, cr, cg, cb);

        const int tip_x = static_cast<int>(std::round(std::clamp(tip.tip_x_original, 0.0f, max_x)));
        const int tip_y = static_cast<int>(std::round(std::clamp(tip.tip_y_original, 0.0f, max_y)));

        if (this->config_.draw_candidates) {
            const float c1_input_x = tip.candidate1_x_mask * this->config_.input_width / this->config_.mask_width;
            const float c1_input_y = tip.candidate1_y_mask * this->config_.input_height / this->config_.mask_height;
            const float c2_input_x = tip.candidate2_x_mask * this->config_.input_width / this->config_.mask_width;
            const float c2_input_y = tip.candidate2_y_mask * this->config_.input_height / this->config_.mask_height;

            const float c1_orig_x = InputToOriginalCoord(c1_input_x, letterbox_pad_x, letterbox_scale, max_x);
            const float c1_orig_y = InputToOriginalCoord(c1_input_y, letterbox_pad_y, letterbox_scale, max_y);
            const float c2_orig_x = InputToOriginalCoord(c2_input_x, letterbox_pad_x, letterbox_scale, max_x);
            const float c2_orig_y = InputToOriginalCoord(c2_input_y, letterbox_pad_y, letterbox_scale, max_y);

            const int c1_x = static_cast<int>(std::round(c1_orig_x));
            const int c1_y = static_cast<int>(std::round(c1_orig_y));
            const int c2_x = static_cast<int>(std::round(c2_orig_x));
            const int c2_y = static_cast<int>(std::round(c2_orig_y));

            DrawLineBgra(bgra, width, height, c1_x, c1_y, c2_x, c2_y, 255, 255, 255);
            DrawCrossBgra(bgra, width, height, c1_x, c1_y, 255, 255, 0, 10);
            DrawCrossBgra(bgra, width, height, c2_x, c2_y, 0, 255, 255, 10);
        }

        if (this->config_.draw_axis) {
            const float axis_scale_original = 120.0f;
            const int ax0 = static_cast<int>(std::round(static_cast<float>(tip_x) - tip.axis_x * axis_scale_original));
            const int ay0 = static_cast<int>(std::round(static_cast<float>(tip_y) - tip.axis_y * axis_scale_original));
            const int ax1 = static_cast<int>(std::round(static_cast<float>(tip_x) + tip.axis_x * axis_scale_original));
            const int ay1 = static_cast<int>(std::round(static_cast<float>(tip_y) + tip.axis_y * axis_scale_original));

            DrawLineBgra(bgra, width, height, ax0, ay0, ax1, ay1, cr, cg, cb);
        }

        DrawTipMarkerBgra(bgra, width, height, tip_x, tip_y, cr, cg, cb);
    }
}

void OverlayVideoEncoderMF::write_d3d11_nv12_overlay_frame(
    D3D11Core& d3d11_core,
    ID3D11Texture2D* nv12_texture,
    UINT source_subresource_index,
    const std::vector<YoloSegPostProcessorD3D12::DetectionWithMask>& mask_results,
    const std::vector<ToolTipDetectorD3D12::TipResult>& tip_results,
    float letterbox_scale,
    float letterbox_pad_x,
    float letterbox_pad_y
)
{
    std::vector<uint8_t> bgra = readback_d3d11_nv12_texture_as_bgra(
        d3d11_core,
        nv12_texture,
        source_subresource_index,
        this->config_.width,
        this->config_.height
    );

    this->overlay_masks_and_tips_bgra(
        bgra,
        mask_results,
        tip_results,
        letterbox_scale,
        letterbox_pad_x,
        letterbox_pad_y
    );

    this->write_bgra_frame(
        bgra.data(),
        bgra.size()
    );
}
