#define NOMINMAX

#include <windows.h>

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <chrono>
#include <iostream>
#include <stdexcept>

#include "D3D11Core.hpp"
#include "D3D12Core.hpp"
#include "MfD3D11Nv12VideoReader.hpp"
#include "ToolTipPipelineD3D12.hpp"

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

namespace {

    long long elapsed_ms(
        std::chrono::high_resolution_clock::time_point begin,
        std::chrono::high_resolution_clock::time_point end
    )
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            end - begin
        ).count();
    }

    void print_duration_and_fps(
        const char* label,
        long long duration_ms,
        long long frame_count
    )
    {
        std::cout << label << ":\n";
        std::cout << "duration: " << duration_ms << " ms\n";

        if (duration_ms > 0 && frame_count > 0) {
            std::cout
                << "fps     : "
                << frame_count / (static_cast<float>(duration_ms) / 1000.0f)
                << "\n";
        }
        else {
            std::cout << "fps     : inf / not measurable\n";
        }
    }

    void print_sum_and_average(
        const char* label,
        long long duration_ms_sum,
        long long frame_count
    )
    {
        std::cout << label << ":\n";
        std::cout << "sum     : " << duration_ms_sum << " ms\n";

        if (frame_count > 0) {
            std::cout
                << "avg     : "
                << static_cast<double>(duration_ms_sum) /
                static_cast<double>(frame_count)
                << " ms/frame\n";
        }
        else {
            std::cout << "avg     : n/a\n";
        }
    }

} // namespace

int wmain(int argc, wchar_t** argv)
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    if (argc < 2) {
        std::wcerr
            << L"Usage: dev_MfD3D11Nv12VideoReader.exe input.mp4 [model.onnx]\n";
        return 1;
    }

    HRESULT hr = CoInitializeEx(
        nullptr,
        COINIT_MULTITHREADED
    );

    if (FAILED(hr)) {
        std::cerr << "CoInitializeEx failed\n";
        return 1;
    }

    hr = MFStartup(MF_VERSION);

    if (FAILED(hr)) {
        std::cerr << "MFStartup failed\n";
        CoUninitialize();
        return 1;
    }

    int result_code = 0;

    long long bridge_copy_duration_sum = 0;
    long long preprocess_duration_sum = 0;
    long long ort_duration_sum = 0;
    long long postprocess_duration_sum = 0;
    long long tooltip_duration_sum = 0;
    long long debug_save_duration_sum = 0;

    // 非同期版では total は「submitからpollまでのlatency寄り」になりやすい。
    // throughput fps の計算には使わず、latency観察用として扱う。
    long long latency_total_duration_sum = 0;

    long long submitted_frame_count = 0;
    long long completed_frame_count = 0;

    try {
        // ------------------------------------------------------------
        // D3D11 / D3D12
        // ------------------------------------------------------------

        D3D11Core d3d11;
        d3d11.initialize(
            false,  // enable_debug_layer
            true    // enable_multithread_protection
        );
        d3d11.print_adapter_info();

        D3D12Core d3d12;
        d3d12.initialize_with_adapter_LUID(
            d3d11.adapter_LUID(),
            true    // enable_debug_layer
        );
        d3d12.print_adapter_info();

        // ------------------------------------------------------------
        // Video reader
        // ------------------------------------------------------------

        MfD3D11Nv12VideoReader video_reader(
            argv[1],
            d3d11
        );

        // ------------------------------------------------------------
        // Pipeline config
        // ------------------------------------------------------------

        ToolTipPipelineD3D12::Config pipeline_config{};
        pipeline_config.model_path =
            (argc >= 3) ? argv[2] : L"model\\best_n_300.onnx";

        pipeline_config.input_width = 640;
        pipeline_config.input_height = 640;
        pipeline_config.limited_range_yuv = true;
        pipeline_config.pad_value = 114.0f / 255.0f;

        pipeline_config.num_attrs = 41;
        pipeline_config.num_candidates = 8400;
        pipeline_config.num_classes = 5;
        pipeline_config.num_mask_coeffs = 32;
        pipeline_config.mask_width = 160;
        pipeline_config.mask_height = 160;
        pipeline_config.max_candidates = 4096;
        pipeline_config.max_detections = 512;
        pipeline_config.conf_threshold = 0.25f;
        pipeline_config.iou_threshold = 0.45f;

        pipeline_config.target_class_id = 1;
        pipeline_config.mask_threshold = 0.5f;
        pipeline_config.min_area_pixels = 10;
        pipeline_config.end_region_ratio = 0.10f;
        pipeline_config.top_edge_ratio = 0.05f;
        pipeline_config.edge_reject_ratio = 0.02f;

        // slot settings
        pipeline_config.bridge_slot_count = 3;
        pipeline_config.processing_slot_count = 3;

        // 性能比較時は必ず false。
        // trueにすると元画像/mask/tip debug保存のreadbackとBMP書き込みが入る。
        pipeline_config.enable_original_tip_debug_image_save = false;
        pipeline_config.original_tip_debug_save_interval = 250;
        pipeline_config.debug_output_directory = "img";
        pipeline_config.debug_score_threshold = 0.25f;
        pipeline_config.debug_mask_threshold = 0.5f;
        pipeline_config.debug_mask_alpha = 0.45f;
        pipeline_config.debug_draw_candidates = true;
        pipeline_config.debug_draw_axis = true;

        ToolTipPipelineD3D12 pipeline(
            d3d11,
            d3d12,
            pipeline_config
        );

        D3D11VideoFrame frame{};

        auto consume_result = [&](
            const ToolTipPipelineD3D12::PipelineFrameResult& frame_result
            ) {
                bridge_copy_duration_sum += frame_result.timing.bridge_copy;
                preprocess_duration_sum += frame_result.timing.preprocess;
                ort_duration_sum += frame_result.timing.ort_inference;
                postprocess_duration_sum += frame_result.timing.postprocess;
                tooltip_duration_sum += frame_result.timing.tooltip_detect_and_readback;
                debug_save_duration_sum += frame_result.timing.debug_save;
                latency_total_duration_sum += frame_result.timing.total;

                ++completed_frame_count;

                // 必要なら一部フレームだけログ表示
                /*
                if (frame_result.frame_index % 250 == 0) {
                    std::cout
                        << "frame=" << frame_result.frame_index
                        << " bridge_slot=" << frame_result.bridge_slot_index
                        << " processing_slot=" << frame_result.processing_slot_index
                        << " tips=" << frame_result.tips.size()
                        << " latency_ms=" << frame_result.timing.total
                        << "\n";

                    for (const auto& tip : frame_result.tips) {
                        std::cout
                            << "  tip det=" << tip.detection_index
                            << " class=" << tip.class_id
                            << " selected=" << tip.selected_candidate
                            << " conf=" << tip.confidence
                            << " orig=(" << tip.tip_x_original
                            << ", " << tip.tip_y_original << ")"
                            << " failure=" << tip.failure_reason
                            << "\n";
                    }
                }
                */
            };

        const auto wall_start =
            std::chrono::high_resolution_clock::now();

        while (video_reader.read_next_frame(frame)) {
            if (pipeline.submit_frame(frame)) {
                ++submitted_frame_count;
            }

            while (auto frame_result = pipeline.poll_result()) {
                consume_result(*frame_result);
            }
        }

        pipeline.flush();

        while (auto frame_result = pipeline.poll_result()) {
            consume_result(*frame_result);
        }

        const auto wall_end =
            std::chrono::high_resolution_clock::now();

        const long long wall_duration_ms =
            elapsed_ms(wall_start, wall_end);

        std::wcout << L"Finished\n";

        // ------------------------------------------------------------
        // Timing summary
        // ------------------------------------------------------------

        std::cout << "\n===== Throughput summary =====\n";
        std::cout << "submitted frames : " << submitted_frame_count << "\n";
        std::cout << "completed frames : " << completed_frame_count << "\n";

        print_duration_and_fps(
            "Wall clock throughput",
            wall_duration_ms,
            completed_frame_count
        );

        std::cout << "\n===== Per-stage timing sums =====\n";
        std::cout
            << "Note: stage timing sums are useful for observation, but their fps values\n"
            << "      are not the true async throughput. Use wall clock throughput above.\n";

        print_sum_and_average(
            "Bridge copy",
            bridge_copy_duration_sum,
            completed_frame_count
        );

        print_sum_and_average(
            "Preprocess",
            preprocess_duration_sum,
            completed_frame_count
        );

        print_sum_and_average(
            "ORT/DML inference",
            ort_duration_sum,
            completed_frame_count
        );

        print_sum_and_average(
            "Postprocess + tooltip GPU",
            postprocess_duration_sum,
            completed_frame_count
        );

        print_sum_and_average(
            "Tooltip final readback",
            tooltip_duration_sum,
            completed_frame_count
        );

        print_sum_and_average(
            "Debug save",
            debug_save_duration_sum,
            completed_frame_count
        );

        std::cout << "\n===== Latency-derived total =====\n";
        std::cout
            << "This is submit-to-result latency sum, not throughput time.\n";

        print_sum_and_average(
            "Latency-derived total",
            latency_total_duration_sum,
            completed_frame_count
        );
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        result_code = 1;
    }

    MFShutdown();
    CoUninitialize();

    return result_code;
}