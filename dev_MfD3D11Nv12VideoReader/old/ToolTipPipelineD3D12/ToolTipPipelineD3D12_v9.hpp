#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
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

        // Stage-1 ring buffering: D3D11/D3D12 shared NV12 bridge slots.
        UINT bridge_slot_count = 3;

        // Stage-2 resource slotting: per-slot preprocessor, postprocessor,
        // and tooltip detector resources. The public API remains sequential.
        // This prepares the pipeline for later asynchronous/ring-buffered
        // execution without changing the current frame-by-frame behavior.
        UINT processing_slot_count = 3;
    };

    struct StageTimingMs {
        // The frame this timing belongs to.
        // Keeping this inside timing makes debug logs robust even when results
        // are stored or processed out of order in the future.
        uint64_t frame_index = 0;

        long long bridge_copy = 0;
        long long preprocess = 0;
        long long ort_inference = 0;
        // In the wait-reduced pipeline, this includes one batched GPU
        // submission containing both YOLO postprocess and tooltip detection.
        long long postprocess = 0;

        // Final small CPU readback of tooltip results only.
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
        bool success = false;

        uint64_t frame_index = 0;
        LONGLONG timestamp100ns = 0;
        UINT width = 0;
        UINT height = 0;

        // Internal resource slots used for this frame. Useful for debugging
        // frame/slot correspondence when ring buffering is expanded.
        UINT bridge_slot_index = 0;
        UINT processing_slot_index = 0;

        std::vector<PipelineTipResult> tips;
        StageTimingMs timing;
    };



    // ---------------------------------------------------------------------
    // Resource / synchronization contract for the current SEQUENTIAL pipeline
    // ---------------------------------------------------------------------
    // This class intentionally keeps the frame pipeline sequential:
    //   copy -> preprocess -> ORT/DML -> postprocess -> tooltip -> readback.
    //
    // Important rule:
    //   D3D12_RESOURCE_STATES are owned by each component's *_state_ref().
    //   Callers must pass these references, not cached copies. Callees are
    //   allowed to transition resources and update the referenced state.
    //
    // Current per-stage contract:
    //
    // 1. SharedNv12FrameBridge11To12::copy_from_d3d11_frame_and_wait()
    //    - Input : decoder D3D11 NV12 texture + decoder subresource index.
    //    - Output: bridge shared texture contains the copied frame.
    //    - Keyed mutex state: released with key 1 for D3D12 read.
    //
    // 2. ScopedD3D12ReadAccess / bridge acquire_for_d3d12_read_guard()
    //    - Acquires key 1 before D3D12 reads shared NV12.
    //    - Releases key 0 when D3D12 read/preprocess is finished.
    //    - Any debug D3D11 read of the bridge texture must acquire key 0.
    //
    // 3. Nv12YoloPreprocessorD3D12::preprocess_and_wait()
    //    - Input : bridge->d3d12_nv12_texture() while D3D12 read guard is held.
    //    - Output: preprocessor_->output_tensor_buffer().
    //    - State : tracked by preprocessor_->output_tensor_state_ref().
    //              The next stage must pass this state reference.
    //
    // 4. OrtDmlYoloSegRunner::run_d3d12_input_and_upload_outputs()
    //    - Input : preprocessor output tensor buffer and its state reference.
    //    - Output: yolo_runner output0/output1 D3D12 buffers.
    //    - State : output0_state_ref()/output1_state_ref() track final states.
    //              The postprocessor must consume these references directly.
    //
    // 5. Wait-reduced postprocess + tooltip stage
    //    - The pipeline records:
    //        postprocessor_->record_external_outputs(...)
    //        tip_detector_->record_detect_commands(...)
    //      into the same D3D12 command list.
    //    - This replaces two separate execute/wait calls with one
    //      execute/wait for both GPU compute stages.
    //    - Output: tip result buffer.
    //    - CPU readback is performed only by readback_results()/readback_all_results().
    //
    // 7. Debug image save path
    //    - Heavy debug-only path. Reads masks from postprocessor and bridge NV12.
    //    - D3D11 bridge texture read must be protected by
    //      bridge_->acquire_for_d3d11_read_guard().
    //
    // Ring-buffering or wait reduction should preserve this contract first,
    // then split ownership per slot/frame.

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

    // -----------------------------------------------------------------
    // Async API step 2
    // -----------------------------------------------------------------
    // submit_frame() now records/submits the final postprocess+tooltip GPU
    // command list and returns without immediately reading back tooltip results.
    // poll_result() checks the associated fence; when the GPU work is complete,
    // it performs the small tooltip readback and returns the completed result.
    //
    // This stage still limits the pipeline to one pending GPU frame because
    // D3D12Core currently owns a single command allocator/list. If a previous
    // frame is still pending when submit_frame() is called, the pipeline retires
    // it first before resetting the command list for the next frame.
    bool submit_frame(
        const D3D11VideoFrame& frame
    );

    std::optional<PipelineFrameResult> poll_result();

    bool has_pending_result() const;

    size_t pending_result_count() const;

    // Waits for any pending GPU frame, completes its readback, and pushes the
    // result to the completed-result queue.
    void flush();

    const Config& config() const;

private:
    struct ProcessingSlot {
        std::unique_ptr<Nv12YoloPreprocessorD3D12> preprocessor;
        std::unique_ptr<YoloSegPostProcessorD3D12> postprocessor;
        std::unique_ptr<ToolTipDetectorD3D12> tip_detector;
    };

    using Clock = std::chrono::high_resolution_clock;

    struct PendingGpuFrame {
        PipelineFrameResult result;

        UINT bridge_slot_index = 0;
        UINT processing_slot_index = 0;
        UINT64 fence_value = 0;

        Nv12YoloPreprocessorD3D12::LetterboxParams letterbox{};
        std::wstring debug_output_path;
        bool debug_save = false;

        Clock::time_point start{};
        Clock::time_point end_bridge_copy{};
        Clock::time_point end_preprocess{};
        Clock::time_point end_ort{};
        Clock::time_point end_gpu_submit{};
    };

    bool retire_pending_frame(bool wait_for_completion);

    SharedNv12FrameBridge11To12& acquire_bridge_slot(UINT width, UINT height);
    void recreate_bridge_slots(UINT width, UINT height);

    ProcessingSlot& acquire_processing_slot();
    void recreate_processing_slots();
    std::unique_ptr<Nv12YoloPreprocessorD3D12> create_preprocessor() const;
    std::unique_ptr<YoloSegPostProcessorD3D12> create_postprocessor() const;
    std::unique_ptr<ToolTipDetectorD3D12> create_tip_detector() const;

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

    std::vector<std::unique_ptr<SharedNv12FrameBridge11To12>> bridge_slots_;
    UINT next_bridge_slot_index_ = 0;
    UINT current_bridge_slot_index_ = 0;
    UINT bridge_width_ = 0;
    UINT bridge_height_ = 0;

    std::vector<ProcessingSlot> processing_slots_;
    UINT next_processing_slot_index_ = 0;
    UINT current_processing_slot_index_ = 0;

    std::optional<PendingGpuFrame> pending_gpu_frame_;
    std::deque<PipelineFrameResult> completed_results_;

    // ORT session is shared, but Step 3 gives it per-processing-slot D3D12
    // output buffers. This prevents output0/output1 overwrite collisions when
    // later stages allow multiple in-flight frames.
    OrtDmlYoloSegRunner yolo_runner_;
};
