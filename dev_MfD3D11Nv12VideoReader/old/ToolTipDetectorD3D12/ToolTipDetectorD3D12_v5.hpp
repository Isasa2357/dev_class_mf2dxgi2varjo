#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <wrl/client.h>

#include <d3d12.h>

#include <cstdint>
#include <vector>

#include "D3D12Core.hpp"

class ToolTipDetectorD3D12 {
public:
    enum FailureReason : UINT {
        FAILURE_NONE = 0,
        FAILURE_OUT_OF_RANGE = 1,
        FAILURE_CLASS_MISMATCH = 2,
        FAILURE_AREA_TOO_SMALL = 3,
        FAILURE_INVALID_AXIS = 4,
        FAILURE_INVALID_SPAN = 5,
        FAILURE_INVALID_END_REGION = 6,
        FAILURE_CANDIDATES_NEAR_EDGE = 7,
        FAILURE_UNKNOWN = 255
    };

    struct Config {
        UINT max_detections = 512;

        UINT mask_width = 160;
        UINT mask_height = 160;

        // YOLO input / letterbox image size, usually 640x640.
        float input_width = 640.0f;
        float input_height = 640.0f;

        // Original source frame size. Used to compute tip_x_original / tip_y_original.
        // If not explicitly set, these default to input_width/input_height.
        float original_width = 640.0f;
        float original_height = 640.0f;

        // Letterbox parameters used by Nv12YoloPreprocessorD3D12.
        // input = original * letterbox_scale + letterbox_pad.
        float letterbox_scale = 1.0f;
        float letterbox_pad_x = 0.0f;
        float letterbox_pad_y = 0.0f;

        // 先端検出対象 class
        UINT target_class_id = 1;

        // binary mask threshold
        float mask_threshold = 0.5f;

        // 少なすぎるmaskは無効
        UINT min_area_pixels = 10;

        // 主軸端領域の割合。
        // 0.10なら、主軸方向の端10%を幅推定に使う。
        float end_region_ratio = 0.10f;

        // 旧設定値。互換性のため残しますが、現在の端判定には edge_reject_ratio を使います。
        float top_edge_ratio = 0.05f;

        // 候補点が上下左右の端からこの割合以内なら、その候補を棄却する。
        // 0.03なら、元画像の上下左右端から3%以内を端扱いにする。
        float edge_reject_ratio = 0.2f;
    };

    struct TipResult {
        UINT valid = 0;
        UINT detection_index = 0;
        UINT class_id = 0;
        UINT selected_candidate = 0; // 1 or 2

        // selected tip center in mask space: 160x160
        float tip_x_mask = 0.0f;
        float tip_y_mask = 0.0f;

        // selected tip center in YOLO input space: 640x640
        float tip_x_input = 0.0f;
        float tip_y_input = 0.0f;

        // selected tip center in original source frame space.
        float tip_x_original = 0.0f;
        float tip_y_original = 0.0f;

        // candidate 1: thinner side before top-edge override
        float candidate1_x_mask = 0.0f;
        float candidate1_y_mask = 0.0f;

        // candidate 2: opposite side
        float candidate2_x_mask = 0.0f;
        float candidate2_y_mask = 0.0f;

        float candidate1_width = 0.0f;
        float candidate2_width = 0.0f;

        float area = 0.0f;

        float axis_x = 0.0f;
        float axis_y = 0.0f;

        // Diagnostic values.
        // width_ratio = thinner_width / thicker_width. Smaller means tip-side decision is clearer.
        float confidence = 0.0f;
        float width_ratio = 0.0f;
        float area_ratio = 0.0f;
        float axis_length = 0.0f;
        UINT failure_reason = FAILURE_NONE;
    };

public:
    ToolTipDetectorD3D12(
        D3D12Core& d3d12_core,
        const Config& config
    );

    ToolTipDetectorD3D12(const ToolTipDetectorD3D12&) = delete;
    ToolTipDetectorD3D12& operator=(const ToolTipDetectorD3D12&) = delete;

    // Update original-frame / letterbox mapping.
    // Call this after Nv12YoloPreprocessorD3D12::preprocess_and_wait() when the source size may change.
    void set_original_mapping(
        UINT original_width,
        UINT original_height,
        float letterbox_scale,
        float letterbox_pad_x,
        float letterbox_pad_y
    );

    void detect_and_wait(
        ID3D12Resource* selected_detection_buffer,
        D3D12_RESOURCE_STATES& selected_detection_state,
        ID3D12Resource* selected_counter_buffer,
        D3D12_RESOURCE_STATES& selected_counter_state,
        ID3D12Resource* selected_mask_buffer,
        D3D12_RESOURCE_STATES& selected_mask_state
    );

    // Record tooltip detection commands into an already-open D3D12 command list.
    // This does not reset, close, execute, or wait. It is used by the pipeline
    // to batch postprocess + tooltip detection into a single GPU submission.
    void record_detect_commands(
        ID3D12GraphicsCommandList* command_list,
        ID3D12Resource* selected_detection_buffer,
        D3D12_RESOURCE_STATES& selected_detection_state,
        ID3D12Resource* selected_counter_buffer,
        D3D12_RESOURCE_STATES& selected_counter_state,
        ID3D12Resource* selected_mask_buffer,
        D3D12_RESOURCE_STATES& selected_mask_state
    );

    // Only valid results.
    std::vector<TipResult> readback_results();

    // All result slots. Useful to inspect failure_reason for invalid detections.
    std::vector<TipResult> readback_all_results();

    ID3D12Resource* result_buffer() const;
    D3D12_RESOURCE_STATES& result_buffer_state_ref();

    const Config& config() const;

private:
    struct ShaderParams {
        UINT max_detections;
        UINT mask_width;
        UINT mask_height;
        UINT mask_pixels;

        float input_width;
        float input_height;
        UINT target_class_id;
        UINT min_area_pixels;

        float mask_threshold;
        float end_region_ratio;
        float top_edge_ratio;
        float edge_reject_ratio;

        float original_width;
        float original_height;
        float letterbox_scale;
        float letterbox_pad_x;

        float letterbox_pad_y;
        float reserved1;
        float reserved2;
        float reserved3;
    };

private:
    void create_root_signature();
    void create_pipeline_state();
    void create_descriptor_heap();
    void create_result_buffer();
    void create_constant_buffer();
    void update_params();

    void create_input_descriptors(
        ID3D12Resource* selected_detection_buffer,
        ID3D12Resource* selected_counter_buffer,
        ID3D12Resource* selected_mask_buffer
    );

    void record_detect(
        ID3D12GraphicsCommandList* command_list,
        ID3D12Resource* selected_detection_buffer,
        D3D12_RESOURCE_STATES& selected_detection_state,
        ID3D12Resource* selected_counter_buffer,
        D3D12_RESOURCE_STATES& selected_counter_state,
        ID3D12Resource* selected_mask_buffer,
        D3D12_RESOURCE_STATES& selected_mask_state
    );

    void transition_resource(
        ID3D12GraphicsCommandList* command_list,
        ID3D12Resource* resource,
        D3D12_RESOURCE_STATES& current_state,
        D3D12_RESOURCE_STATES new_state
    );

    static UINT align256(UINT value);

private:
    D3D12Core& d3d12_core_;
    Config config_;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> root_signature_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline_state_;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptor_heap_;
    UINT descriptor_size_ = 0;

    Microsoft::WRL::ComPtr<ID3D12Resource> result_buffer_;
    D3D12_RESOURCE_STATES result_buffer_state_ = D3D12_RESOURCE_STATE_COMMON;

    Microsoft::WRL::ComPtr<ID3D12Resource> constant_buffer_;
    uint8_t* mapped_constants_ = nullptr;
};
