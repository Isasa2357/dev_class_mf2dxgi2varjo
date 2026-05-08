#define NOMINMAX

#include <windows.h>

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>

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

// 必要なら
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

std::wstring stringToWstring(const std::string& str) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], size_needed);
    return wstr;
}

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
    long long mk_shared_texture_duration = 0;
    long long preprocess_duration = 0;
    long long readback_input_tensor_duration = 0;
    long long predict_duration = 0;
    long long postprocess_duration = 0;
    long long frame_count = 0;

    try {
        // ------------------------------------------------------------
        // 1. D3D11 / D3D12 Core 初期化
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
        // 2. Video reader 初期化
        // ------------------------------------------------------------

        MfD3D11Nv12VideoReader video_reader(
            argv[1],
            d3d11
        );

        //std::wcout
        //    << L"Video size: "
        //    << video_reader.width()
        //    << L" x "
        //    << video_reader.height()
        //    << L"\n";

        // ------------------------------------------------------------
        // 3. D3D12 preprocessor 初期化
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

        OrtDmlYoloSegRunner yolo_runner;
        yolo_runner.initialize(
            L"model\\best_n_300.onnx",
            true,   // use_directml
            0       // dml_device_id
        );
        yolo_runner.print_model_info();

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

        ToolTipDetectorD3D12::Config tip_config{};
        tip_config.max_detections = 512;
        tip_config.mask_width = 160;
        tip_config.mask_height = 160;
        tip_config.input_width = 640.0f;
        tip_config.input_height = 640.0f;
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
        // 4. frame 読み出し -> bridge -> D3D12 preprocess
        // ------------------------------------------------------------

        D3D11VideoFrame frame{};

        std::unique_ptr<SharedNv12FrameBridge11To12> bridge;

        while (video_reader.read_next_frame(frame)) {
            //std::wcout
            //    << L"Frame "
            //    << frame.frameIndex
            //    << L" timestamp100ns="
            //    << frame.timestamp100ns
            //    << L" size="
            //    << frame.width
            //    << L"x"
            //    << frame.height
            //    << L" subresource="
            //    << frame.subresourceIndex
            //    << L"\n";

            // 動画サイズが分かったタイミングで bridge を作る
            if (!bridge ||
                bridge->width() != frame.width ||
                bridge->height() != frame.height) {
                bridge = std::make_unique<SharedNv12FrameBridge11To12>(
                    d3d11,
                    d3d12,
                    frame.width,
                    frame.height
                );
            }

            // --------------------------------------------------------
            // D3D11 decoder texture -> D3D11 shared NV12 texture
            // --------------------------------------------------------

            auto start = std::chrono::high_resolution_clock::now();

            bridge->copy_from_d3d11_frame_and_wait(
                frame.texture.Get(),
                frame.subresourceIndex
            );

            auto end_mk_shared_texture = std::chrono::high_resolution_clock::now();

            // --------------------------------------------------------
            // D3D12 側で shared NV12 texture を読む
            // keyed mutex で、D3D12処理中は D3D11側が上書きしないようにする
            // --------------------------------------------------------

            bridge->acquire_for_d3d12_read_guard();

            preprocessor.preprocess_and_wait(
                bridge->d3d12_nv12_texture(),
                frame.width,
                frame.height
            );

            bridge->release_from_d3d12_read_guard();

            auto end_preprocess = std::chrono::high_resolution_clock::now();

            ID3D12Resource* yolo_input_tensor =
                preprocessor.output_tensor_buffer();

            std::vector<float> input_tensor = ReadbackNchwFloatTensorD3D12Buffer(
                d3d12,
                preprocessor.output_tensor_buffer(),
                preprocessor.output_tensor_state_ref(),
                preprocessor.input_width(),
                preprocessor.input_height(),
                3
            );

            auto end_mk_input_tensor = std::chrono::high_resolution_clock::now();

            std::vector<OrtDmlYoloSegRunner::OutputTensor> outputs = yolo_runner.run_cpu_input(
                input_tensor,
                1,
                3,
                preprocessor.input_height(),
                preprocessor.input_width()
            );

            auto end_predict = std::chrono::high_resolution_clock::now();

            postprocessor.upload_outputs_from_cpu(
                outputs[0].data,
                outputs[1].data
            );
            postprocessor.process_uploaded_outputs_and_wait();
            std::vector<YoloSegPostProcessorD3D12::DetectionWithMask> results = postprocessor.readback_results();

            auto end_postprocess = std::chrono::high_resolution_clock::now();

            tip_detector.detect_and_wait(
                postprocessor.selected_detection_buffer(),
                postprocessor.selected_detection_state_ref(),
                postprocessor.selected_counter_buffer(),
                postprocessor.selected_counter_state_ref(),
                postprocessor.selected_mask_buffer(),
                postprocessor.selected_mask_state_ref()
            );

            auto tip_results = tip_detector.readback_results();

            std::cout << "frame idx: " << frame.frameIndex << ", detections: " << results.size() << "\n";

            for (const auto& r : tip_results) {
                std::cout
                    << "det=" << r.detection_index
                    << " class=" << r.class_id
                    << " selectedCandidate=" << r.selected_candidate
                    << " tipMask=(" << r.tip_x_mask << ", " << r.tip_y_mask << ")"
                    << " tipInput=(" << r.tip_x_input << ", " << r.tip_y_input << ")"
                    << " c1=(" << r.candidate1_x_mask << ", " << r.candidate1_y_mask
                    << ") width1=" << r.candidate1_width
                    << " c2=(" << r.candidate2_x_mask << ", " << r.candidate2_y_mask
                    << ") width2=" << r.candidate2_width
                    << " area=" << r.area
                    << "\n";
            }

            //for (const auto& det : detections) {
            //    std::cout
            //        << "class=" << det.class_id
            //        << " score=" << det.score
            //        << " box=("
            //        << det.x1 << ", "
            //        << det.y1 << ", "
            //        << det.x2 << ", "
            //        << det.y2 << ")"
            //        << " candidate_index=" << det.candidate_index
            //        << "\n";
            //}

            mk_shared_texture_duration += std::chrono::duration_cast<std::chrono::milliseconds>(end_mk_shared_texture - start).count();
            preprocess_duration += std::chrono::duration_cast<std::chrono::milliseconds>(end_preprocess - end_mk_shared_texture).count();
            readback_input_tensor_duration += std::chrono::duration_cast<std::chrono::milliseconds>(end_mk_input_tensor - end_preprocess).count();
            predict_duration += std::chrono::duration_cast<std::chrono::milliseconds>(end_predict - end_mk_input_tensor).count();
            postprocess_duration += std::chrono::duration_cast<std::chrono::milliseconds>(end_postprocess - end_predict).count();
            frame_count += 1;
            std::cout << "frame count: " << frame_count << std::endl;

            //std::wcout
            //    << frame.frameIndex << " "
            //    << L"Readback tensor elemnts: "
            //    << input_tensor.size()
            //    << std::endl;
            //break;

            /*
            std::wcout
                << L"D3D12 YOLO preprocess completed. Tensor elements = "
                << preprocessor.tensor_element_count()
                << L"\n";
            */

            if (frame.frameIndex % 500 == 0) {            
                std::string path = "img\\debug_yolo_input_d3d12_" + std::to_string(frame.frameIndex) + ".bmp";
                std::wstring wpath = stringToWstring(path);
                std::string mask_path = "img\\debug_raw_mask";
                std::wstring wmask_path = stringToWstring(mask_path);

                std::vector<YoloSegPostProcessorD3D12::Detection> detections;
                for (auto& result : results) {
                    detections.push_back(result.detection);
                }

                std::cout << "save as " << path.data() << std::endl;

                //SaveNchwFloatTensorWithDetectionsAsBmp(
                //    input_tensor,
                //    preprocessor.input_width(),
                //    preprocessor.input_height(),
                //    detections,
                //    wpath.data(),
                //    0.5f
                //);

                //SaveDetectionRawMasksAsBmp(
                //    results,
                //    wmask_path.data(),
                //    true,  // normalize_min_max
                //    0.5f   // score_threshold
                //);

                //SaveNchwFloatTensorWithMasksAsBmp(
                //    input_tensor,
                //    preprocessor.input_width(),
                //    preprocessor.input_height(),
                //    results,
                //    wpath.data(),
                //    0.25f,  // score_threshold
                //    0.5f,   // mask_threshold
                //    0.45f,  // alpha
                //    true    // draw_bbox
                //);

                SaveNchwFloatTensorWithToolTipsAsBmp(
                    input_tensor,
                    preprocessor.input_width(),
                    preprocessor.input_height(),
                    tip_results,
                    wpath.data(),
                    true,   // draw_candidates
                    true    // draw_axis
                );
            }
        }

        std::wcout << L"Finished\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        result = 1;
    }

    std::cout << "共有テクスチャ作成: " << std::endl;
    std::cout << "duration: " << mk_shared_texture_duration << " ms\n";
    std::cout << "fps     : " << frame_count / (float(mk_shared_texture_duration) / 1000) << std::endl;
    std::cout << "前処理            : " << std::endl;
    std::cout << "duration: " << preprocess_duration << " ms\n";
    std::cout << "fps     ; " << frame_count / (float(preprocess_duration) / 1000) << std::endl;
    std::cout << "入力テンソル作成  : " << std::endl;
    std::cout << "duration: " << readback_input_tensor_duration << " ms\n";
    std::cout << "fps     : " << frame_count / (float(readback_input_tensor_duration) / 1000) << std::endl;
    std::cout << "Predict: " << std::endl;
    std::cout << "duration: " << predict_duration << " ms" << std::endl;
    std::cout << "fps     : " << frame_count / (float(predict_duration) / 1000) << std::endl;
    std::cout << "Postprocess: " << std::endl;
    std::cout << "duration: " << postprocess_duration << std::endl;
    std::cout << "fps     : " << frame_count / (float(postprocess_duration) / 1000) << std::endl;

    MFShutdown();
    CoUninitialize();

    return result;
}

//#include <iostream>
//#include <onnxruntime_c_api.h>
//#include <windows.h>
//#include <iostream>
//
//void PrintLoadedOnnxRuntimeDllPath()
//{
//    HMODULE h = GetModuleHandleW(L"onnxruntime.dll");
//
//    if (!h) {
//        std::wcout << L"onnxruntime.dll is not loaded yet\n";
//        return;
//    }
//
//    wchar_t path[MAX_PATH]{};
//    DWORD len = GetModuleFileNameW(h, path, MAX_PATH);
//
//    if (len > 0) {
//        std::wcout << L"Loaded onnxruntime.dll: " << path << L"\n";
//    } else {
//        std::wcout << L"GetModuleFileNameW failed\n";
//    }
//}
//
//int main()
//{
//    const OrtApiBase* base = OrtGetApiBase();
//
//    if (!base) {
//        std::cerr << "OrtGetApiBase returned nullptr\n";
//        return 1;
//    }
//
//    std::cout << "ORT version: " << base->GetVersionString() << "\n";
//
//    const OrtApi* api = base->GetApi(ORT_API_VERSION);
//
//    if (!api) {
//        std::cerr
//            << "base->GetApi(ORT_API_VERSION) returned nullptr\n"
//            << "ORT_API_VERSION=" << ORT_API_VERSION << "\n";
//        return 1;
//    }
//
//    std::cout << "OrtApi acquired\n";
//
//    PrintLoadedOnnxRuntimeDllPath();
//
//    return 0;
//}