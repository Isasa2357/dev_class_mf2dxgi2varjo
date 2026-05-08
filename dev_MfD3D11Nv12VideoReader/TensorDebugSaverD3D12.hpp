#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <wrl/client.h>

#include <d3d12.h>

#include <vector>
#include <cstdint>

#include "D3D12Core.hpp"
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