#include "ToolTipPipelineD3D12.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

#include "TensorDebugSaverD3D12.hpp"
#include "HResultUtil.hpp"

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
    yolo_runner_.initialize_with_d3d12(
        d3d12_core_,
        config_.model_path.c_str()
    );

    if (config_.print_model_info) {
        yolo_runner_.print_model_info();
    }

    yolo_runner_.set_output_slot_count(
        std::max<UINT>(1u, config_.processing_slot_count)
    );

    this->create_gpu_timestamp_resources();

    this->recreate_processing_slots();
}

ToolTipPipelineD3D12::PipelineFrameResult
ToolTipPipelineD3D12::process_frame(
    const D3D11VideoFrame& frame
)
{
    // Synchronous compatibility wrapper around the Step-2 async API.
    // It submits the frame, waits for the pending GPU work, then returns the
    // completed result. Existing callers of process_frame() can keep working,
    // while submit_frame()/poll_result() can be used for delayed readback.
    this->submit_frame(frame);
    this->flush();

    std::optional<PipelineFrameResult> result = this->poll_result();

    if (!result) {
        throw std::runtime_error(
            "ToolTipPipelineD3D12::process_frame: no result after flush"
        );
    }

    return std::move(*result);
}

bool ToolTipPipelineD3D12::submit_frame(
    const D3D11VideoFrame& frame
)
{
    if (!frame.texture) {
        throw std::runtime_error("ToolTipPipelineD3D12::submit_frame: frame.texture is null");
    }

    // Step 2 still uses D3D12Core's single command allocator/list. Therefore,
    // before we reset the command list for a new frame, any previous pending
    // GPU submission must be retired. Try non-blocking first; if it is still
    // running, wait here to preserve correctness. Later stages can remove this
    // restriction by introducing per-slot command allocators/lists.
    this->retire_pending_frame(false);

    if (this->pending_gpu_frame_) {
        this->retire_pending_frame(true);
    }

    SharedNv12FrameBridge11To12& bridge =
        this->acquire_bridge_slot(frame.width, frame.height);

    ProcessingSlot& slot = this->acquire_processing_slot();

    PendingGpuFrame pending{};
    pending.result.frame_index = frame.frameIndex;
    pending.result.timestamp100ns = frame.timestamp100ns;
    pending.result.width = frame.width;
    pending.result.height = frame.height;
    pending.result.bridge_slot_index = this->current_bridge_slot_index_;
    pending.result.processing_slot_index = this->current_processing_slot_index_;
    pending.result.timing.frame_index = frame.frameIndex;

    pending.bridge_slot_index = this->current_bridge_slot_index_;
    pending.processing_slot_index = this->current_processing_slot_index_;

    pending.start = Clock::now();

    // Stage 1: decoder D3D11 NV12 -> bridge shared NV12.
    bridge.copy_from_d3d11_frame_and_wait(
        frame.texture.Get(),
        frame.subresourceIndex
    );

    pending.end_bridge_copy = Clock::now();

    {
        // Stage 2: D3D12 preprocess reads shared NV12 while the keyed mutex is
        // held for D3D12. preprocess_and_wait() still waits internally so the
        // bridge can safely release the keyed mutex after this scope.
        ScopedD3D12ReadAccess d3d12_read_guard(bridge);

        slot.preprocessor->preprocess_and_wait(
            bridge.d3d12_nv12_texture(),
            frame.width,
            frame.height
        );
    }

    pending.end_preprocess = Clock::now();

    // Stage 3: ORT/DML inference. This currently remains synchronous inside
    // the runner, but it consumes a D3D12 input buffer and produces D3D12
    // output buffers.
    yolo_runner_.run_d3d12_input_and_upload_outputs_to_slot(
        pending.processing_slot_index,
        slot.preprocessor->output_tensor_buffer(),
        slot.preprocessor->output_tensor_state_ref(),
        1,
        3,
        slot.preprocessor->input_height(),
        slot.preprocessor->input_width()
    );

    pending.end_ort = Clock::now();

    const auto& lb = slot.preprocessor->letterbox_params();
    pending.letterbox = lb;

    slot.tip_detector->set_original_mapping(
        lb.src_width,
        lb.src_height,
        lb.scale,
        lb.pad_x,
        lb.pad_y
    );

    // Stage 4 + 5: record YOLO postprocess and tooltip detection into one
    // command list. Unlike process_external_outputs_and_wait()/detect_and_wait(),
    // Step 2 submits this command list and returns without reading back the
    // tooltip results immediately.
    d3d12_core_.reset_command_list();

    ID3D12GraphicsCommandList* command_list =
        d3d12_core_.command_list();

    // GPU timestamp query layout for this frame:
    //   0: before YOLO postprocess
    //   1: after YOLO postprocess / before tooltip detector
    //   2: after tooltip detector
    pending.gpu_timestamps_enabled =
        this->gpu_timestamp_query_heap_ &&
        this->gpu_timestamp_readback_buffer_ &&
        this->gpu_timestamp_frequency_ != 0;
    pending.gpu_timestamp_query_base = 0;

    if (pending.gpu_timestamps_enabled) {
        this->write_gpu_timestamp(command_list, pending.gpu_timestamp_query_base + 0);
    }

    slot.postprocessor->record_external_outputs(
        command_list,
        yolo_runner_.output0_buffer(pending.processing_slot_index),
        yolo_runner_.output0_state_ref(pending.processing_slot_index),
        yolo_runner_.output1_buffer(pending.processing_slot_index),
        yolo_runner_.output1_state_ref(pending.processing_slot_index)
    );

    if (pending.gpu_timestamps_enabled) {
        this->write_gpu_timestamp(command_list, pending.gpu_timestamp_query_base + 1);
    }

    slot.tip_detector->record_detect_commands(
        command_list,
        slot.postprocessor->selected_detection_buffer(),
        slot.postprocessor->selected_detection_state_ref(),
        slot.postprocessor->selected_counter_buffer(),
        slot.postprocessor->selected_counter_state_ref(),
        slot.postprocessor->selected_mask_buffer(),
        slot.postprocessor->selected_mask_state_ref()
    );

    if (pending.gpu_timestamps_enabled) {
        this->write_gpu_timestamp(command_list, pending.gpu_timestamp_query_base + 2);
        this->resolve_gpu_timestamps(
            command_list,
            pending.gpu_timestamp_query_base,
            this->gpu_timestamp_query_count_
        );
    }

    // Signal a fence but do not wait here. poll_result()/flush() will retire
    // this pending frame when the fence is complete.
    pending.fence_value = d3d12_core_.close_execute_and_signal();
    pending.end_gpu_submit = Clock::now();

    pending.result.timing.bridge_copy = elapsed_ms(pending.start, pending.end_bridge_copy);
    pending.result.timing.preprocess = elapsed_ms(pending.end_bridge_copy, pending.end_preprocess);
    pending.result.timing.ort_inference = elapsed_ms(pending.end_preprocess, pending.end_ort);

    // In Step 2 this is CPU-side command recording/submission time, not a GPU
    // timestamp. A later GPU timestamp-query pass can measure real GPU time.
    pending.result.timing.postprocess = elapsed_ms(pending.end_ort, pending.end_gpu_submit);

    pending.debug_save =
        config_.enable_original_tip_debug_image_save &&
        config_.original_tip_debug_save_interval != 0 &&
        (frame.frameIndex % config_.original_tip_debug_save_interval) == 0;

    if (pending.debug_save) {
        pending.debug_output_path = this->make_debug_output_path(frame.frameIndex);
    }

    this->pending_gpu_frame_ = std::move(pending);

    return true;
}

std::optional<ToolTipPipelineD3D12::PipelineFrameResult>
ToolTipPipelineD3D12::poll_result()
{
    if (!this->completed_results_.empty()) {
        PipelineFrameResult result = std::move(this->completed_results_.front());
        this->completed_results_.pop_front();
        return result;
    }

    this->retire_pending_frame(false);

    if (this->completed_results_.empty()) {
        return std::nullopt;
    }

    PipelineFrameResult result = std::move(this->completed_results_.front());
    this->completed_results_.pop_front();
    return result;
}

bool ToolTipPipelineD3D12::has_pending_result() const
{
    return !this->completed_results_.empty() || this->pending_gpu_frame_.has_value();
}

size_t ToolTipPipelineD3D12::pending_result_count() const
{
    return this->completed_results_.size() + (this->pending_gpu_frame_ ? 1u : 0u);
}

void ToolTipPipelineD3D12::flush()
{
    this->retire_pending_frame(true);
}

bool ToolTipPipelineD3D12::retire_pending_frame(bool wait_for_completion)
{
    if (!this->pending_gpu_frame_) {
        return false;
    }

    if (wait_for_completion) {
        d3d12_core_.wait_for_fence(this->pending_gpu_frame_->fence_value);
    }
    else if (!d3d12_core_.is_fence_complete(this->pending_gpu_frame_->fence_value)) {
        return false;
    }

    PendingGpuFrame pending = std::move(*this->pending_gpu_frame_);
    this->pending_gpu_frame_.reset();

    ProcessingSlot& slot =
        this->processing_slots_.at(pending.processing_slot_index);

    SharedNv12FrameBridge11To12& bridge =
        *this->bridge_slots_.at(pending.bridge_slot_index);

    PipelineFrameResult result = std::move(pending.result);

    this->fill_gpu_timestamp_timing(pending, result.timing);

    const auto start_readback = Clock::now();

    const std::vector<ToolTipDetectorD3D12::TipResult> tip_results =
        slot.tip_detector->readback_results();

    result.tips.reserve(tip_results.size());
    for (const auto& tip : tip_results) {
        result.tips.push_back(convert_tip_result(tip));
    }

    const auto end_readback = Clock::now();

    result.timing.tooltip_detect_and_readback =
        elapsed_ms(start_readback, end_readback);

    if (pending.debug_save) {
        const auto start_debug = Clock::now();

        const std::vector<YoloSegPostProcessorD3D12::DetectionWithMask> mask_results =
            slot.postprocessor->readback_results();

        std::wcout
            << L"save original tooltip + mask overlay plot as "
            << pending.debug_output_path
            << L"\n";

        std::cout
            << "debug save frameIndex=" << result.frame_index
            << " timestamp100ns=" << result.timestamp100ns
            << " tip_count=" << tip_results.size()
            << " mask_count=" << mask_results.size()
            << " bridge_slot=" << result.bridge_slot_index
            << " processing_slot=" << result.processing_slot_index
            << "\n";

        {
            auto d3d11_debug_read_guard =
                bridge.acquire_for_d3d11_read_guard();

            SaveD3D11Nv12TextureWithOriginalMasksAndToolTipsAsBmp(
                d3d11_core_,
                bridge.d3d11_shared_nv12_texture(),
                0,
                result.width,
                result.height,
                mask_results,
                tip_results,
                pending.debug_output_path.c_str(),
                static_cast<float>(config_.input_width),
                static_cast<float>(config_.input_height),
                static_cast<float>(config_.mask_width),
                static_cast<float>(config_.mask_height),
                pending.letterbox.scale,
                pending.letterbox.pad_x,
                pending.letterbox.pad_y,
                config_.debug_score_threshold,
                config_.debug_mask_threshold,
                config_.debug_mask_alpha,
                config_.debug_draw_candidates,
                config_.debug_draw_axis
            );
        }

        const auto end_debug = Clock::now();
        result.timing.debug_save = elapsed_ms(start_debug, end_debug);
    }

    const auto end_total = Clock::now();
    result.timing.total = elapsed_ms(pending.start, end_total);
    result.success = true;

    this->completed_results_.push_back(std::move(result));
    return true;
}

void ToolTipPipelineD3D12::create_gpu_timestamp_resources()
{
    this->gpu_timestamp_query_heap_.Reset();
    this->gpu_timestamp_readback_buffer_.Reset();
    this->gpu_timestamp_frequency_ = 0;

    if (!d3d12_core_.device() || !d3d12_core_.command_queue()) {
        return;
    }

    HRESULT hr = d3d12_core_.command_queue()->GetTimestampFrequency(
        &this->gpu_timestamp_frequency_
    );

    if (FAILED(hr) || this->gpu_timestamp_frequency_ == 0) {
        this->gpu_timestamp_frequency_ = 0;
        return;
    }

    D3D12_QUERY_HEAP_DESC query_desc{};
    query_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    query_desc.Count = this->gpu_timestamp_query_count_;
    query_desc.NodeMask = 0;

    hr = d3d12_core_.device()->CreateQueryHeap(
        &query_desc,
        IID_PPV_ARGS(this->gpu_timestamp_query_heap_.GetAddressOf())
    );

    win_util::ThrowIfFailed(
        hr,
        "ToolTipPipelineD3D12::create_gpu_timestamp_resources: CreateQueryHeap failed"
    );

    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_READBACK;
    heap_props.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heap_props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heap_props.CreationNodeMask = 1;
    heap_props.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC buffer_desc{};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Alignment = 0;
    buffer_desc.Width = static_cast<UINT64>(this->gpu_timestamp_query_count_) * sizeof(UINT64);
    buffer_desc.Height = 1;
    buffer_desc.DepthOrArraySize = 1;
    buffer_desc.MipLevels = 1;
    buffer_desc.Format = DXGI_FORMAT_UNKNOWN;
    buffer_desc.SampleDesc.Count = 1;
    buffer_desc.SampleDesc.Quality = 0;
    buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buffer_desc.Flags = D3D12_RESOURCE_FLAG_NONE;

    this->gpu_timestamp_readback_buffer_ =
        d3d12_core_.create_committed_resource(
            heap_props,
            D3D12_HEAP_FLAG_NONE,
            buffer_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr
        );
}

void ToolTipPipelineD3D12::write_gpu_timestamp(
    ID3D12GraphicsCommandList* command_list,
    UINT query_index
)
{
    if (!command_list || !this->gpu_timestamp_query_heap_) {
        return;
    }

    command_list->EndQuery(
        this->gpu_timestamp_query_heap_.Get(),
        D3D12_QUERY_TYPE_TIMESTAMP,
        query_index
    );
}

void ToolTipPipelineD3D12::resolve_gpu_timestamps(
    ID3D12GraphicsCommandList* command_list,
    UINT query_base,
    UINT query_count
)
{
    if (!command_list ||
        !this->gpu_timestamp_query_heap_ ||
        !this->gpu_timestamp_readback_buffer_ ||
        query_count == 0)
    {
        return;
    }

    command_list->ResolveQueryData(
        this->gpu_timestamp_query_heap_.Get(),
        D3D12_QUERY_TYPE_TIMESTAMP,
        query_base,
        query_count,
        this->gpu_timestamp_readback_buffer_.Get(),
        0
    );
}

void ToolTipPipelineD3D12::fill_gpu_timestamp_timing(
    PendingGpuFrame& pending,
    StageTimingMs& timing
)
{
    timing.gpu_postprocess = 0.0;
    timing.gpu_tooltip = 0.0;
    timing.gpu_postprocess_tooltip_total = 0.0;

    if (!pending.gpu_timestamps_enabled ||
        !this->gpu_timestamp_readback_buffer_ ||
        this->gpu_timestamp_frequency_ == 0)
    {
        return;
    }

    const UINT64 byte_size =
        static_cast<UINT64>(this->gpu_timestamp_query_count_) * sizeof(UINT64);

    D3D12_RANGE read_range{};
    read_range.Begin = 0;
    read_range.End = static_cast<SIZE_T>(byte_size);

    UINT64* mapped = nullptr;
    HRESULT hr = this->gpu_timestamp_readback_buffer_->Map(
        0,
        &read_range,
        reinterpret_cast<void**>(&mapped)
    );

    win_util::ThrowIfFailed(
        hr,
        "ToolTipPipelineD3D12::fill_gpu_timestamp_timing: timestamp readback Map failed"
    );

    const UINT64 t0 = mapped[0];
    const UINT64 t1 = mapped[1];
    const UINT64 t2 = mapped[2];

    D3D12_RANGE written_range{};
    written_range.Begin = 0;
    written_range.End = 0;

    this->gpu_timestamp_readback_buffer_->Unmap(
        0,
        &written_range
    );

    const double tick_to_ms =
        1000.0 / static_cast<double>(this->gpu_timestamp_frequency_);

    if (t1 >= t0) {
        timing.gpu_postprocess = static_cast<double>(t1 - t0) * tick_to_ms;
    }

    if (t2 >= t1) {
        timing.gpu_tooltip = static_cast<double>(t2 - t1) * tick_to_ms;
    }

    if (t2 >= t0) {
        timing.gpu_postprocess_tooltip_total =
            static_cast<double>(t2 - t0) * tick_to_ms;
    }
}

const ToolTipPipelineD3D12::Config&
ToolTipPipelineD3D12::config() const
{
    return this->config_;
}

SharedNv12FrameBridge11To12& ToolTipPipelineD3D12::acquire_bridge_slot(
    UINT width,
    UINT height
)
{
    const UINT requested_slot_count =
        std::max<UINT>(1u, config_.bridge_slot_count);

    const bool need_recreate =
        bridge_slots_.empty() ||
        bridge_slots_.size() != static_cast<size_t>(requested_slot_count) ||
        bridge_width_ != width ||
        bridge_height_ != height;

    if (need_recreate) {
        this->recreate_bridge_slots(width, height);
    }

    current_bridge_slot_index_ = next_bridge_slot_index_;

    SharedNv12FrameBridge11To12* slot =
        bridge_slots_.at(current_bridge_slot_index_).get();

    if (!slot) {
        throw std::runtime_error(
            "ToolTipPipelineD3D12::acquire_bridge_slot: bridge slot is null"
        );
    }

    next_bridge_slot_index_ =
        (next_bridge_slot_index_ + 1u) % static_cast<UINT>(bridge_slots_.size());

    return *slot;
}

void ToolTipPipelineD3D12::recreate_bridge_slots(UINT width, UINT height)
{
    const UINT slot_count = std::max<UINT>(1u, config_.bridge_slot_count);

    bridge_slots_.clear();
    bridge_slots_.reserve(slot_count);

    for (UINT i = 0; i < slot_count; ++i) {
        bridge_slots_.push_back(
            std::make_unique<SharedNv12FrameBridge11To12>(
                d3d11_core_,
                d3d12_core_,
                width,
                height
            )
        );
    }

    bridge_width_ = width;
    bridge_height_ = height;
    next_bridge_slot_index_ = 0;
    current_bridge_slot_index_ = 0;
}


ToolTipPipelineD3D12::ProcessingSlot& ToolTipPipelineD3D12::acquire_processing_slot()
{
    const UINT requested_slot_count =
        std::max<UINT>(1u, config_.processing_slot_count);

    if (processing_slots_.empty() ||
        processing_slots_.size() != static_cast<size_t>(requested_slot_count))
    {
        this->recreate_processing_slots();
    }

    current_processing_slot_index_ = next_processing_slot_index_;

    ProcessingSlot& slot =
        processing_slots_.at(current_processing_slot_index_);

    if (!slot.preprocessor || !slot.postprocessor || !slot.tip_detector) {
        throw std::runtime_error(
            "ToolTipPipelineD3D12::acquire_processing_slot: incomplete processing slot"
        );
    }

    next_processing_slot_index_ =
        (next_processing_slot_index_ + 1u) % static_cast<UINT>(processing_slots_.size());

    return slot;
}

void ToolTipPipelineD3D12::recreate_processing_slots()
{
    const UINT slot_count = std::max<UINT>(1u, config_.processing_slot_count);

    processing_slots_.clear();
    processing_slots_.reserve(slot_count);

    for (UINT i = 0; i < slot_count; ++i) {
        ProcessingSlot slot{};
        slot.preprocessor = this->create_preprocessor();
        slot.postprocessor = this->create_postprocessor();
        slot.tip_detector = this->create_tip_detector();
        processing_slots_.push_back(std::move(slot));
    }

    next_processing_slot_index_ = 0;
    current_processing_slot_index_ = 0;
}

std::unique_ptr<Nv12YoloPreprocessorD3D12>
ToolTipPipelineD3D12::create_preprocessor() const
{
    Nv12YoloPreprocessorD3D12::Config preprocessor_config{};
    preprocessor_config.input_width = config_.input_width;
    preprocessor_config.input_height = config_.input_height;
    preprocessor_config.limited_range_yuv = config_.limited_range_yuv;
    preprocessor_config.pad_value = config_.pad_value;

    return std::make_unique<Nv12YoloPreprocessorD3D12>(
        d3d12_core_,
        preprocessor_config
    );
}

std::unique_ptr<YoloSegPostProcessorD3D12>
ToolTipPipelineD3D12::create_postprocessor() const
{
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

    return std::make_unique<YoloSegPostProcessorD3D12>(
        d3d12_core_,
        post_config
    );
}

std::unique_ptr<ToolTipDetectorD3D12>
ToolTipPipelineD3D12::create_tip_detector() const
{
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

    return std::make_unique<ToolTipDetectorD3D12>(
        d3d12_core_,
        tip_config
    );
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
