#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "D3D11Core.hpp"
#include "D3D12Core.hpp"
#include "MfD3D11Nv12VideoReader.hpp"
#include "SharedNv12FrameBridge11To12.hpp"
#include "Nv12YoloPreprocessorD3D12.hpp"
#include "OrtDmlYoloSegRunner.hpp"
#include "YoloSegPostProcessorD3D12.hpp"
#include "ToolTipDetectorD3D12.hpp"

class ToolTipPipelineD3D12 {
public:
    struct Config {
        std::wstring model_path = L"model\\best_n_300.onnx";

        bool print_model_info = true;

        UINT input_width = 640;
        UINT input_height = 640;
        bool limited_range_yuv = true;
        float pad_value = 114.0f / 255.0f;

        UINT num_attrs = 41;
        UINT num_candidates = 8400;
        UINT num_classes = 5;
        UINT num_mask_coeffs = 32;
        UINT mask_width = 160;
        UINT mask_height = 160;
        UINT max_candidates = 4096;
        UINT max_detections = 512;
        float conf_threshold = 0.25f;
        float iou_threshold = 0.45f;

        UINT target_class_id = 1;
        float mask_threshold = 0.5f;
        UINT min_area_pixels = 10;
        float end_region_ratio = 0.10f;
        float top_edge_ratio = 0.05f;
        float edge_reject_ratio = 0.02f;

        bool enable_original_tip_debug_image_save = false;
        uint64_t original_tip_debug_save_interval = 250;
        std::string debug_output_directory = "img";
        float debug_score_threshold = 0.25f;
        float debug_mask_threshold = 0.5f;
        float debug_mask_alpha = 0.45f;
        bool debug_draw_candidates = true;
        bool debug_draw_axis = true;
    };

    struct StageTimingMs {
        long long bridge_copy = 0;
        long long preprocess = 0;
        long long ort_inference = 0;
        long long postprocess = 0;
        long long tooltip_detect_and_readback = 0;
        long long debug_save = 0;
        long long total = 0;
    };

    struct PipelineTipResult {
        bool valid = false;

        UINT detection_index = 0;
        UINT class_id = 0;
        UINT selected_candidate = 0;

        float tip_x_mask = 0.0f;
        float tip_y_mask = 0.0f;

        float tip_x_input = 0.0f;
        float tip_y_input = 0.0f;

        float tip_x_original = 0.0f;
        float tip_y_original = 0.0f;

        float confidence = 0.0f;
        float width_ratio = 0.0f;
        float area_ratio = 0.0f;
        float axis_length = 0.0f;
        UINT failure_reason = ToolTipDetectorD3D12::FAILURE_NONE;
    };

    struct PipelineFrameResult {
        uint64_t frame_index = 0;
        LONGLONG timestamp100ns = 0;
        UINT width = 0;
        UINT height = 0;

        std::vector<PipelineTipResult> tips;
        StageTimingMs timing;
    };

public:
    ToolTipPipelineD3D12(
        D3D11Core& d3d11_core,
        D3D12Core& d3d12_core,
        const Config& config
    );

    ToolTipPipelineD3D12(const ToolTipPipelineD3D12&) = delete;
    ToolTipPipelineD3D12& operator=(const ToolTipPipelineD3D12&) = delete;

    PipelineFrameResult process_frame(
        const D3D11VideoFrame& frame
    );

    const Config& config() const;

private:
    void ensure_bridge(UINT width, UINT height);

    static PipelineTipResult convert_tip_result(
        const ToolTipDetectorD3D12::TipResult& tip
    );

    static std::wstring string_to_wstring_utf8(
        const std::string& str
    );

    std::wstring make_debug_output_path(
        uint64_t frame_index
    ) const;

private:
    D3D11Core& d3d11_core_;
    D3D12Core& d3d12_core_;
    Config config_;

    std::unique_ptr<SharedNv12FrameBridge11To12> bridge_;

    std::unique_ptr<Nv12YoloPreprocessorD3D12> preprocessor_;
    OrtDmlYoloSegRunner yolo_runner_;
    std::unique_ptr<YoloSegPostProcessorD3D12> postprocessor_;
    std::unique_ptr<ToolTipDetectorD3D12> tip_detector_;
};
