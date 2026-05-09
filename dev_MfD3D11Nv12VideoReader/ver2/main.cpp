#define NOMINMAX

#include <windows.h>

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <string>
#include <vector>

#include "D3D11Core.hpp"
#include "D3D12Core.hpp"

#include "MfD3D11Nv12VideoReader.hpp"
#include "SharedNv12FrameBridge11To12.hpp"
#include "Nv12YoloPreprocessorD3D12.hpp"
#include "TensorDebugSaverD3D12.hpp"
#include "TensorReadbackD3D12.hpp"
#include "OrtDmlYoloSegRunner.hpp"
#include "YoloSegPostProcessorD3D12.hpp"
#include "ToolTipDetectorD3D12.hpp"

#include "HResultUtil.hpp"

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

namespace {

    std::wstring string_to_wstring_utf8(const std::string& str)
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

        // remove terminating null from std::wstring length
        if (!wstr.empty() && wstr.back() == L'\0') {
            wstr.pop_back();
        }

        return wstr;
    }

    void print_timing(
        const char* label,
        long long duration_ms,
        long long frame_count
    )
    {
        std::cout << label << ":\n";
        std::cout << "duration: " << duration_ms << " ms\n";

        if (duration_ms > 0) {
            std::cout
                << "fps     : "
                << frame_count / (static_cast<float>(duration_ms) / 1000.0f)
                << "\n";
        }
        else {
            std::cout << "fps     : inf / not measurable\n";
        }
    }

} // namespace

int wmain(int argc, wchar_t** argv)
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    if (argc < 2) {
        std::wcerr << L"Usage: dev_MfD3D11Nv12VideoReader.exe input.mp4\n";
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

    int result = 0;

    long long bridge_copy_duration = 0;
    long long preprocess_duration = 0;
    long long input_readback_duration = 0;
    long long predict_and_upload_duration = 0;
    long long postprocess_duration = 0;
    long long tooltip_duration = 0;
    long long total_duration = 0;
    long long frame_count = 0;

    try {
        // ------------------------------------------------------------
        // 1. D3D11 / D3D12 Core
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
        // 2. Video reader
        // ------------------------------------------------------------

        MfD3D11Nv12VideoReader video_reader(
            argv[1],
            d3d11
        );

        // ------------------------------------------------------------
        // 3. Preprocessor
        // ------------------------------------------------------------

        Nv12YoloPreprocessorD3D12::Config preprocessor_config{};
        preprocessor_config.input_width = 640;
        preprocessor_config.input_height = 640;
        preprocessor_config.limited_range_yuv = true;
        preprocessor_config.pad_value = 114.0f / 255.0f;

        Nv12YoloPreprocessorD3D12 preprocessor(
            d3d12,
            preprocessor_config
        );

        // ------------------------------------------------------------
        // 4. YOLO runner
        //
        // 現段階では input は CPU tensor のまま。
        // output0/output1 は runner 内部で D3D12 buffer に upload し、
        // postprocessor は external GPU buffer として読む。
        // ------------------------------------------------------------

        OrtDmlYoloSegRunner yolo_runner;
        //yolo_runner.initialize(
        //    L"model\\best_n_300.onnx",
        //    true,   // use_directml
        //    0       // dml_device_id
        //);
        yolo_runner.initialize_with_d3d12(d3d12, L"model\\best_n_300.onnx");
        yolo_runner.print_model_info();

        // ------------------------------------------------------------
        // 5. Postprocessor
        // ------------------------------------------------------------

        YoloSegPostProcessorD3D12::Config post_config{};
        post_config.num_attrs = 41;
        post_config.num_candidates = 8400;
        post_config.num_classes = 5;
        post_config.num_mask_coeffs = 32;
        post_config.mask_width = 160;
        post_config.mask_height = 160;
        post_config.max_candidates = 4096;
        post_config.max_detections = 512;
        post_config.conf_threshold = 0.25f;
        post_config.iou_threshold = 0.45f;
        post_config.input_width = 640.0f;
        post_config.input_height = 640.0f;

        YoloSegPostProcessorD3D12 postprocessor(
            d3d12,
            post_config
        );

        // ------------------------------------------------------------
        // 6. Tool tip detector
        // ------------------------------------------------------------

        ToolTipDetectorD3D12::Config tip_config{};
        tip_config.max_detections = post_config.max_detections;
        tip_config.mask_width = post_config.mask_width;
        tip_config.mask_height = post_config.mask_height;
        tip_config.input_width = post_config.input_width;
        tip_config.input_height = post_config.input_height;
        tip_config.target_class_id = 1;
        tip_config.mask_threshold = 0.5f;
        tip_config.min_area_pixels = 10;
        tip_config.end_region_ratio = 0.10f;
        tip_config.top_edge_ratio = 0.05f;

        ToolTipDetectorD3D12 tip_detector(
            d3d12,
            tip_config
        );

        // ------------------------------------------------------------
        // 7. Main loop
        // ------------------------------------------------------------

        D3D11VideoFrame frame{};
        std::unique_ptr<SharedNv12FrameBridge11To12> bridge;

        while (video_reader.read_next_frame(frame)) {
            if (!bridge ||
                bridge->width() != frame.width ||
                bridge->height() != frame.height)
            {
                bridge = std::make_unique<SharedNv12FrameBridge11To12>(
                    d3d11,
                    d3d12,
                    frame.width,
                    frame.height
                );
            }

            const auto start = std::chrono::high_resolution_clock::now();

            // D3D11 decoder texture -> shared D3D11 NV12 texture
            bridge->copy_from_d3d11_frame_and_wait(
                frame.texture.Get(),
                frame.subresourceIndex
            );

            const auto end_bridge_copy = std::chrono::high_resolution_clock::now();

            // D3D12 preprocess from shared NV12 texture
            bridge->acquire_for_d3d12_read_guard();

            preprocessor.preprocess_and_wait(
                bridge->d3d12_nv12_texture(),
                frame.width,
                frame.height
            );

            bridge->release_from_d3d12_read_guard();

            const auto end_preprocess = std::chrono::high_resolution_clock::now();

            // --------------------------------------------------------
            // 現段階では YOLO input は CPU tensor 経由。
            // 次の段階でこの readback を ORT GPU input binding に置き換える。
            // --------------------------------------------------------

            /*
            std::vector<float> input_tensor =
                ReadbackNchwFloatTensorD3D12Buffer(
                    d3d12,
                    preprocessor.output_tensor_buffer(),
                    preprocessor.output_tensor_state_ref(),
                    preprocessor.input_width(),
                    preprocessor.input_height(),
                    3
                );
            */

            const auto end_input_readback = std::chrono::high_resolution_clock::now();

            // CPU input -> ORT run -> runner internal D3D12 output upload
            /*
            yolo_runner.run_cpu_input_and_upload_outputs(
                input_tensor,
                1,
                3,
                preprocessor.input_height(),
                preprocessor.input_width()
            );
            */
            yolo_runner.run_d3d12_input_and_upload_outputs(
                preprocessor.output_tensor_buffer(),
                preprocessor.output_tensor_state_ref(),
                1,
                3,
                preprocessor.input_height(),
                preprocessor.input_width()
            );

            const auto end_predict_upload = std::chrono::high_resolution_clock::now();

            // --------------------------------------------------------
            // ここが今回の主変更点。
            // postprocessor.upload_outputs_from_cpu() は使わず、
            // runner の D3D12 output buffer を直接読む。
            // --------------------------------------------------------

            postprocessor.process_external_outputs_and_wait(
                yolo_runner.output0_buffer(),
                yolo_runner.output0_state_ref(),
                yolo_runner.output1_buffer(),
                yolo_runner.output1_state_ref()
            );

            const auto end_postprocess = std::chrono::high_resolution_clock::now();

            // GPU postprocess output -> GPU tooltip detection
            const auto& lb = preprocessor.letterbox_params();

            tip_detector.set_original_mapping(
                lb.src_width,
                lb.src_height,
                lb.scale,
                lb.pad_x,
                lb.pad_y
            );
            tip_detector.detect_and_wait(
                postprocessor.selected_detection_buffer(),
                postprocessor.selected_detection_state_ref(),
                postprocessor.selected_counter_buffer(),
                postprocessor.selected_counter_state_ref(),
                postprocessor.selected_mask_buffer(),
                postprocessor.selected_mask_state_ref()
            );

            // final small readback only
            const std::vector<ToolTipDetectorD3D12::TipResult> tip_results =
                tip_detector.readback_results();

            const auto end_tooltip = std::chrono::high_resolution_clock::now();

            std::cout
                << "frame idx: " << frame.frameIndex
                << ", valid tips: " << tip_results.size()
                << "\n";

            bridge_copy_duration +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_bridge_copy - start
                ).count();

            preprocess_duration +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_preprocess - end_bridge_copy
                ).count();

            input_readback_duration +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_input_readback - end_preprocess
                ).count();

            predict_and_upload_duration +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_predict_upload - end_input_readback
                ).count();

            postprocess_duration +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_postprocess - end_predict_upload
                ).count();

            tooltip_duration +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_tooltip - end_postprocess
                ).count();

            total_duration +=
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_tooltip - start
                ).count();

            ++frame_count;

            // --------------------------------------------------------
            // Debug image save
            // input_tensor は CPU readback 済みなので、この段階では
            // debug plot にそのまま使える。
            // --------------------------------------------------------

            if (frame.frameIndex % 500 == 0) {
                //const std::string path =
                //    "img\\debug_tooltips_" +
                //    std::to_string(frame.frameIndex) +
                //    ".bmp";

                //const std::wstring wpath =
                //    string_to_wstring_utf8(path);

                //std::cout << "save as " << path << "\n";

                //SaveNchwFloatTensorWithToolTipsAsBmp(
                //    input_tensor,
                //    preprocessor.input_width(),
                //    preprocessor.input_height(),
                //    tip_results,
                //    wpath.c_str(),
                //    true,   // draw_candidates
                //    true    // draw_axis
                //);
            }
        }

        std::wcout << L"Finished\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        result = 1;
    }

    std::cout << "\n===== Timing summary =====\n";
    std::cout << "frames: " << frame_count << "\n";
    print_timing("Bridge copy", bridge_copy_duration, frame_count);
    print_timing("Preprocess", preprocess_duration, frame_count);
    print_timing("Input tensor readback", input_readback_duration, frame_count);
    print_timing("Predict + output upload", predict_and_upload_duration, frame_count);
    print_timing("Postprocess", postprocess_duration, frame_count);
    print_timing("Tooltip detect + final readback", tooltip_duration, frame_count);
    print_timing("Total", total_duration, frame_count);

    MFShutdown();
    CoUninitialize();

    return result;
}
