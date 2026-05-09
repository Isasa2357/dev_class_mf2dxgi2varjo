#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <wrl/client.h>

#include <d3d12.h>
#include <d3d11.h>

#include <vector>
#include <cstdint>

#include "D3D12Core.hpp"
#include "D3D11Core.hpp"
#include "YoloSegPostProcessorD3D12.hpp"
#include "ToolTipDetectorD3D12.hpp"


void SaveNchwFloatTensorD3D12BufferAsBmp(
    D3D12Core& d3d12_core,
    ID3D12Resource* tensor_buffer,
    D3D12_RESOURCE_STATES& tensor_buffer_state,
    UINT width,
    UINT height,
    const wchar_t* output_path
);

void SaveNchwFloatTensorWithDetectionsAsBmp(
    const std::vector<float>& nchw_rgb_tensor,
    UINT width,
    UINT height,
    const std::vector<YoloSegPostProcessorD3D12::Detection>& detections,
    const wchar_t* output_path,
    float score_threshold = 0.0f
);

void SaveFloatMaskAsBmp(
    const std::vector<float>& mask,
    UINT width,
    UINT height,
    const wchar_t* output_path,
    bool normalize_min_max = false
);

void SaveDetectionRawMaskAsBmp(
    const YoloSegPostProcessorD3D12::DetectionWithMask& result,
    const wchar_t* output_path,
    bool normalize_min_max = false
);

void SaveDetectionRawMasksAsBmp(
    const std::vector<YoloSegPostProcessorD3D12::DetectionWithMask>& results,
    const wchar_t* output_path_prefix,
    bool normalize_min_max = false,
    float score_threshold = 0.0f
);

void SaveNchwFloatTensorWithMasksAsBmp(
    const std::vector<float>& nchw_rgb_tensor,
    UINT width,
    UINT height,
    const std::vector<YoloSegPostProcessorD3D12::DetectionWithMask>& results,
    const wchar_t* output_path,
    float score_threshold = 0.0f,
    float mask_threshold = 0.5f,
    float alpha = 0.45f,
    bool draw_bbox = true
);

void SaveNchwFloatTensorWithToolTipsAsBmp(
    const std::vector<float>& nchw_rgb_tensor,
    UINT width,
    UINT height,
    const std::vector<ToolTipDetectorD3D12::TipResult>& tip_results,
    const wchar_t* output_path,
    bool draw_candidates = true,
    bool draw_axis = true
);

// Debug-only helper.
// Saves the original NV12 D3D11 frame as BMP and plots tip_x_original / tip_y_original.
// If draw_candidates is true, candidate1/candidate2 are also mapped from 160x160 mask space
// to original frame space using the supplied letterbox parameters.
void SaveD3D11Nv12TextureWithOriginalToolTipsAsBmp(
    D3D11Core& d3d11_core,
    ID3D11Texture2D* nv12_texture,
    UINT source_subresource_index,
    UINT width,
    UINT height,
    const std::vector<ToolTipDetectorD3D12::TipResult>& tip_results,
    const wchar_t* output_path,
    float input_width = 640.0f,
    float input_height = 640.0f,
    float mask_width = 160.0f,
    float mask_height = 160.0f,
    float letterbox_scale = 1.0f,
    float letterbox_pad_x = 0.0f,
    float letterbox_pad_y = 0.0f,
    bool draw_candidates = true,
    bool draw_axis = true
);
