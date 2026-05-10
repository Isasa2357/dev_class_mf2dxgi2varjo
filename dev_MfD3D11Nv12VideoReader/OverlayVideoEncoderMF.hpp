#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <wrl/client.h>

#include <d3d11.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <cstdint>
#include <string>
#include <vector>

#include "D3D11Core.hpp"
#include "YoloSegPostProcessorD3D12.hpp"
#include "ToolTipDetectorD3D12.hpp"

class OverlayVideoEncoderMF {
public:
    struct Config {
        std::wstring output_path = L"output_overlay.mp4";

        UINT width = 0;
        UINT height = 0;

        UINT fps_num = 30;
        UINT fps_den = 1;

        // H.264 bitrate. 3840x2160なら 20-80 Mbps 程度から調整。
        UINT32 bitrate = 40'000'000;

        // Overlay coordinate settings.
        float input_width = 640.0f;
        float input_height = 640.0f;
        float mask_width = 160.0f;
        float mask_height = 160.0f;

        float score_threshold = 0.25f;
        float mask_threshold = 0.5f;
        float mask_alpha = 0.45f;

        bool draw_candidates = true;
        bool draw_axis = true;
    };

public:
    OverlayVideoEncoderMF() = default;

    explicit OverlayVideoEncoderMF(const Config& config);

    OverlayVideoEncoderMF(const OverlayVideoEncoderMF&) = delete;
    OverlayVideoEncoderMF& operator=(const OverlayVideoEncoderMF&) = delete;

    ~OverlayVideoEncoderMF();

    void open(const Config& config);

    void close();

    bool is_open() const { return this->sink_writer_ != nullptr; }

    // CPU BGRA8 frame input. bgra size must be width * height * 4.
    void write_bgra_frame(
        const uint8_t* bgra,
        size_t byte_size
    );

    // Readback D3D11 NV12 texture to CPU, overlay masks/tips, then encode.
    // This is for confirmation/debug video. It is intentionally CPU based.
    void write_d3d11_nv12_overlay_frame(
        D3D11Core& d3d11_core,
        ID3D11Texture2D* nv12_texture,
        UINT source_subresource_index,
        const std::vector<YoloSegPostProcessorD3D12::DetectionWithMask>& mask_results,
        const std::vector<ToolTipDetectorD3D12::TipResult>& tip_results,
        float letterbox_scale,
        float letterbox_pad_x,
        float letterbox_pad_y
    );

    uint64_t frame_index() const { return this->frame_index_; }

private:
    static std::vector<uint8_t> readback_d3d11_nv12_texture_as_bgra(
        D3D11Core& d3d11_core,
        ID3D11Texture2D* nv12_texture,
        UINT source_subresource_index,
        UINT width,
        UINT height
    );

    void overlay_masks_and_tips_bgra(
        std::vector<uint8_t>& bgra,
        const std::vector<YoloSegPostProcessorD3D12::DetectionWithMask>& mask_results,
        const std::vector<ToolTipDetectorD3D12::TipResult>& tip_results,
        float letterbox_scale,
        float letterbox_pad_x,
        float letterbox_pad_y
    ) const;

    LONGLONG frame_duration_100ns() const;
    LONGLONG frame_time_100ns(uint64_t index) const;

private:
    Config config_{};

    Microsoft::WRL::ComPtr<IMFSinkWriter> sink_writer_;
    DWORD stream_index_ = 0;
    uint64_t frame_index_ = 0;
};
