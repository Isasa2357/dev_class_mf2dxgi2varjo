#include "ToolTipPipelineD3D12.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>

#include "TensorDebugSaverD3D12.hpp"

namespace {

class ScopedD3D12ReadAccess {
public:
    explicit ScopedD3D12ReadAccess(SharedNv12FrameBridge11To12& bridge)
        : bridge_(&bridge)
    {
        bridge_->acquire_for_d3d12_read_guard();
    }

    ~ScopedD3D12ReadAccess()
    {
        if (bridge_) {
            bridge_->release_from_d3d12_read_guard();
        }
    }

    ScopedD3D12ReadAccess(const ScopedD3D12ReadAccess&) = delete;
    ScopedD3D12ReadAccess& operator=(const ScopedD3D12ReadAccess&) = delete;

private:
    SharedNv12FrameBridge11To12* bridge_ = nullptr;
};

long long elapsed_ms(
    const std::chrono::high_resolution_clock::time_point& a,
    const std::chrono::high_resolution_clock::time_point& b
)
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

} // namespace

ToolTipPipelineD3D12::ToolTipPipelineD3D12(
    D3D11Core& d3d11_core,
    D3D12Core& d3d12_core,
    const Config& config
)
    : d3d11_core_(d3d11_core)
    , d3d12_core_(d3d12_core)
    , config_(config)
{
    Nv12YoloPreprocessorD3D12::Config preprocessor_config{};
    preprocessor_config.input_width = config_.input_width;
    preprocessor_config.input_height = config_.input_height;
    preprocessor_config.limited_range_yuv = config_.limited_range_yuv;
    preprocessor_config.pad_value = config_.pad_value;

    preprocessor_ = std::make_unique<Nv12YoloPreprocessorD3D12>(
        d3d12_core_,
        preprocessor_config
    );

    yolo_runner_.initialize_with_d3d12(
        d3d12_core_,
        config_.model_path.c_str()
    );

    if (config_.print_model_info) {
        yolo_runner_.print_model_info();
    }

    YoloSegPostProcessorD3D12::Config post_config{};
    post_config.num_attrs = config_.num_attrs;
    post_config.num_candidates = config_.num_candidates;
    post_config.num_classes = config_.num_classes;
    post_config.num_mask_coeffs = config_.num_mask_coeffs;
    post_config.mask_width = config_.mask_width;
    post_config.mask_height = config_.mask_height;
    post_config.max_candidates = config_.max_candidates;
    post_config.max_detections = config_.max_detections;
    post_config.conf_threshold = config_.conf_threshold;
    post_config.iou_threshold = config_.iou_threshold;
    post_config.input_width = static_cast<float>(config_.input_width);
    post_config.input_height = static_cast<float>(config_.input_height);

    postprocessor_ = std::make_unique<YoloSegPostProcessorD3D12>(
        d3d12_core_,
        post_config
    );

    ToolTipDetectorD3D12::Config tip_config{};
    tip_config.max_detections = config_.max_detections;
    tip_config.mask_width = config_.mask_width;
    tip_config.mask_height = config_.mask_height;
    tip_config.input_width = static_cast<float>(config_.input_width);
    tip_config.input_height = static_cast<float>(config_.input_height);
    tip_config.target_class_id = config_.target_class_id;
    tip_config.mask_threshold = config_.mask_threshold;
    tip_config.min_area_pixels = config_.min_area_pixels;
    tip_config.end_region_ratio = config_.end_region_ratio;
    tip_config.top_edge_ratio = config_.top_edge_ratio;
    tip_config.edge_reject_ratio = config_.edge_reject_ratio;

    tip_detector_ = std::make_unique<ToolTipDetectorD3D12>(
        d3d12_core_,
        tip_config
    );
}

ToolTipPipelineD3D12::PipelineFrameResult
ToolTipPipelineD3D12::process_frame(
    const D3D11VideoFrame& frame
)
{
    if (!frame.texture) {
        throw std::runtime_error("ToolTipPipelineD3D12::process_frame: frame.texture is null");
    }

    this->ensure_bridge(frame.width, frame.height);

    PipelineFrameResult result{};
    result.frame_index = frame.frameIndex;
    result.timestamp100ns = frame.timestamp100ns;
    result.width = frame.width;
    result.height = frame.height;
    result.timing.frame_index = frame.frameIndex;

    const auto start = std::chrono::high_resolution_clock::now();

    // Stage 1: decoder D3D11 NV12 -> bridge shared NV12.
    // After this call, the bridge keyed mutex is released with key 1,
    // so D3D12 may acquire it for reading.
    bridge_->copy_from_d3d11_frame_and_wait(
        frame.texture.Get(),
        frame.subresourceIndex
    );

    const auto end_bridge_copy = std::chrono::high_resolution_clock::now();

    {
        // Stage 2: D3D12 preprocess reads the shared NV12 texture while
        // D3D12 ownership of the keyed mutex is held. The preprocessor
        // updates output_tensor_state_ref() internally; later stages must
        // use that reference instead of assuming a fixed state.
        ScopedD3D12ReadAccess d3d12_read_guard(*bridge_);

        preprocessor_->preprocess_and_wait(
            bridge_->d3d12_nv12_texture(),
            frame.width,
            frame.height
        );
    }

    const auto end_preprocess = std::chrono::high_resolution_clock::now();

    // Stage 3: ORT/DML consumes the D3D12 input tensor.
    // The runner may transition preprocessor_->output_tensor_state_ref()
    // and produces output0/output1 buffers with states tracked by
    // yolo_runner_.output0_state_ref()/output1_state_ref().
    yolo_runner_.run_d3d12_input_and_upload_outputs(
        preprocessor_->output_tensor_buffer(),
        preprocessor_->output_tensor_state_ref(),
        1,
        3,
        preprocessor_->input_height(),
        preprocessor_->input_width()
    );

    const auto end_ort = std::chrono::high_resolution_clock::now();

    const auto& lb = preprocessor_->letterbox_params();

    tip_detector_->set_original_mapping(
        lb.src_width,
        lb.src_height,
        lb.scale,
        lb.pad_x,
        lb.pad_y
    );

    // Stage 4 + 5: record YOLO postprocess and tooltip detection into one
    // D3D12 command list, then execute/wait once.
    //
    // Previous sequential path did:
    //   postprocessor.process_external_outputs_and_wait();
    //   tip_detector.detect_and_wait();
    // which caused two command-list submissions and two GPU waits.
    //
    // This path keeps the same resource-state contract, but batches the two
    // compute stages into a single GPU submission. The tooltip stage consumes
    // the selected_* buffers produced earlier in the same command list.
    d3d12_core_.reset_command_list();

    ID3D12GraphicsCommandList* command_list =
        d3d12_core_.command_list();

    postprocessor_->record_external_outputs(
        command_list,
        yolo_runner_.output0_buffer(),
        yolo_runner_.output0_state_ref(),
        yolo_runner_.output1_buffer(),
        yolo_runner_.output1_state_ref()
    );

    tip_detector_->record_detect_commands(
        command_list,
        postprocessor_->selected_detection_buffer(),
        postprocessor_->selected_detection_state_ref(),
        postprocessor_->selected_counter_buffer(),
        postprocessor_->selected_counter_state_ref(),
        postprocessor_->selected_mask_buffer(),
        postprocessor_->selected_mask_state_ref()
    );

    d3d12_core_.close_execute_and_wait();

    const auto end_postprocess = std::chrono::high_resolution_clock::now();

    // Stage 6: final normal-runtime CPU readback.
    // This should remain the only readback in the non-debug path.
    const std::vector<ToolTipDetectorD3D12::TipResult> tip_results =
        tip_detector_->readback_results();

    result.tips.reserve(tip_results.size());
    for (const auto& tip : tip_results) {
        result.tips.push_back(convert_tip_result(tip));
    }

    const auto end_tooltip = std::chrono::high_resolution_clock::now();

    if (config_.enable_original_tip_debug_image_save &&
        config_.original_tip_debug_save_interval != 0 &&
        (frame.frameIndex % config_.original_tip_debug_save_interval) == 0)
    {
        const auto start_debug = std::chrono::high_resolution_clock::now();

        // Debug-only readback of masks. Do not move this outside the debug
        // branch, or normal runtime will no longer be tip-result-only readback.
        const std::vector<YoloSegPostProcessorD3D12::DetectionWithMask> mask_results =
            postprocessor_->readback_results();

        const std::wstring wpath = this->make_debug_output_path(frame.frameIndex);

        std::wcout
            << L"save original tooltip + mask overlay plot as "
            << wpath
            << L"\n";

        std::cout
            << "debug save frameIndex=" << frame.frameIndex
            << " timestamp100ns=" << frame.timestamp100ns
            << " tip_count=" << tip_results.size()
            << " mask_count=" << mask_results.size()
            << "\n";

        {
            // Debug-only D3D11 read of the bridge shared texture.
            // The bridge texture is keyed-mutex protected; read it only while
            // this guard is alive.
            auto d3d11_debug_read_guard =
                bridge_->acquire_for_d3d11_read_guard();

            SaveD3D11Nv12TextureWithOriginalMasksAndToolTipsAsBmp(
                d3d11_core_,
                bridge_->d3d11_shared_nv12_texture(),
                0,
                frame.width,
                frame.height,
                mask_results,
                tip_results,
                wpath.c_str(),
                static_cast<float>(preprocessor_->input_width()),
                static_cast<float>(preprocessor_->input_height()),
                static_cast<float>(config_.mask_width),
                static_cast<float>(config_.mask_height),
                lb.scale,
                lb.pad_x,
                lb.pad_y,
                config_.debug_score_threshold,
                config_.debug_mask_threshold,
                config_.debug_mask_alpha,
                config_.debug_draw_candidates,
                config_.debug_draw_axis
            );
        }

        const auto end_debug = std::chrono::high_resolution_clock::now();
        result.timing.debug_save = elapsed_ms(start_debug, end_debug);
    }

    const auto end_total = std::chrono::high_resolution_clock::now();

    result.timing.bridge_copy = elapsed_ms(start, end_bridge_copy);
    result.timing.preprocess = elapsed_ms(end_bridge_copy, end_preprocess);
    result.timing.ort_inference = elapsed_ms(end_preprocess, end_ort);
    // postprocess now includes the batched GPU submission for both
    // YOLO postprocess and tooltip detection. The following tooltip timing
    // is the final small CPU readback only.
    result.timing.postprocess = elapsed_ms(end_ort, end_postprocess);
    result.timing.tooltip_detect_and_readback = elapsed_ms(end_postprocess, end_tooltip);
    result.timing.total = elapsed_ms(start, end_total);
    result.success = true;

    return result;
}

const ToolTipPipelineD3D12::Config&
ToolTipPipelineD3D12::config() const
{
    return this->config_;
}

void ToolTipPipelineD3D12::ensure_bridge(UINT width, UINT height)
{
    if (!bridge_ || bridge_->width() != width || bridge_->height() != height) {
        bridge_ = std::make_unique<SharedNv12FrameBridge11To12>(
            d3d11_core_,
            d3d12_core_,
            width,
            height
        );
    }
}

ToolTipPipelineD3D12::PipelineTipResult
ToolTipPipelineD3D12::convert_tip_result(
    const ToolTipDetectorD3D12::TipResult& tip
)
{
    PipelineTipResult out{};
    out.valid = tip.valid != 0;
    out.detection_index = tip.detection_index;
    out.class_id = tip.class_id;
    out.selected_candidate = tip.selected_candidate;

    out.tip_x_mask = tip.tip_x_mask;
    out.tip_y_mask = tip.tip_y_mask;
    out.tip_x_input = tip.tip_x_input;
    out.tip_y_input = tip.tip_y_input;
    out.tip_x_original = tip.tip_x_original;
    out.tip_y_original = tip.tip_y_original;

    out.confidence = tip.confidence;
    out.width_ratio = tip.width_ratio;
    out.area_ratio = tip.area_ratio;
    out.axis_length = tip.axis_length;
    out.failure_reason = tip.failure_reason;

    return out;
}

std::wstring ToolTipPipelineD3D12::string_to_wstring_utf8(
    const std::string& str
)
{
    if (str.empty()) {
        return std::wstring{};
    }

    const int size_needed = MultiByteToWideChar(
        CP_UTF8,
        0,
        str.c_str(),
        -1,
        nullptr,
        0
    );

    if (size_needed <= 0) {
        throw std::runtime_error("MultiByteToWideChar failed");
    }

    std::wstring wstr(static_cast<size_t>(size_needed), L'\0');

    MultiByteToWideChar(
        CP_UTF8,
        0,
        str.c_str(),
        -1,
        wstr.data(),
        size_needed
    );

    if (!wstr.empty() && wstr.back() == L'\0') {
        wstr.pop_back();
    }

    return wstr;
}

std::wstring ToolTipPipelineD3D12::make_debug_output_path(
    uint64_t frame_index
) const
{
    std::string path =
        config_.debug_output_directory +
        "\\debug_original_tooltips_" +
        std::to_string(frame_index) +
        ".bmp";

    return string_to_wstring_utf8(path);
}
