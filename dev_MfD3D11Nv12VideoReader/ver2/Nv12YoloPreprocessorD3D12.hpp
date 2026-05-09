#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <wrl/client.h>

#include <d3d12.h>
#include <dxgi.h>

#include <cstdint>
#include <cstddef>

#include "D3D12Core.hpp"

class Nv12YoloPreprocessorD3D12 {
public:
    struct Config {
        UINT input_width = 640;
        UINT input_height = 640;

        // true: limited range YUV
        // false: full range YUV
        bool limited_range_yuv = true;

        // Ultralytics系のletterbox背景値
        float pad_value = 114.0f / 255.0f;
    };

    struct LetterboxParams {
        UINT src_width = 0;
        UINT src_height = 0;
        UINT dst_width = 0;
        UINT dst_height = 0;

        float scale = 1.0f;
        float pad_x = 0.0f;
        float pad_y = 0.0f;

        float input_to_original_x(float x) const
        {
            float v = (x - pad_x) / scale;
            if (v < 0.0f) {
                return 0.0f;
            }
            if (v > static_cast<float>(src_width - 1)) {
                return static_cast<float>(src_width - 1);
            }
            return v;
        }

        float input_to_original_y(float y) const
        {
            float v = (y - pad_y) / scale;
            if (v < 0.0f) {
                return 0.0f;
            }
            if (v > static_cast<float>(src_height - 1)) {
                return static_cast<float>(src_height - 1);
            }
            return v;
        }

        float original_to_input_x(float x) const
        {
            return x * scale + pad_x;
        }

        float original_to_input_y(float y) const
        {
            return y * scale + pad_y;
        }
    };

public:
    Nv12YoloPreprocessorD3D12(
        D3D12Core& d3d12_core,
        const Config& config = Config()
    );

    Nv12YoloPreprocessorD3D12(const Nv12YoloPreprocessorD3D12&) = delete;
    Nv12YoloPreprocessorD3D12& operator=(const Nv12YoloPreprocessorD3D12&) = delete;

    ~Nv12YoloPreprocessorD3D12();

    // 既に open 状態の command list に前処理コマンドを記録する。
    // 呼び出し側が d3d12_core.reset_command_list() / close_execute_and_wait() を管理する想定。
    void record_preprocess(
        ID3D12Resource* nv12_texture,
        UINT src_width,
        UINT src_height
    );

    // 簡易版。
    // 内部で reset -> record -> close/execute/wait まで行う。
    void preprocess_and_wait(
        ID3D12Resource* nv12_texture,
        UINT src_width,
        UINT src_height
    );

    ID3D12Resource* output_tensor_buffer() const;

    UINT input_width() const;
    UINT input_height() const;

    size_t tensor_element_count() const;
    size_t tensor_byte_size() const;

    D3D12_RESOURCE_STATES output_tensor_state() const;

    // デバッグ保存やORT接続前に外部で状態遷移したい場合用。
    void transition_output_tensor(
        ID3D12GraphicsCommandList* command_list,
        D3D12_RESOURCE_STATES new_state
    );

    D3D12_RESOURCE_STATES& output_tensor_state_ref();

    const LetterboxParams& letterbox_params() const { return this->letterbox_params_; }

private:
    struct ShaderParams {
        UINT src_width;
        UINT src_height;
        UINT dst_width;
        UINT dst_height;

        float scale;
        float pad_x;
        float pad_y;
        float pad_value;

        UINT limited_range_yuv;
        UINT reserved0;
        UINT reserved1;
        UINT reserved2;
    };

private:
    static UINT align256(UINT value);

    void create_root_signature();
    void create_pipeline_state();
    void create_descriptor_heap();
    void create_output_tensor_buffer();
    void create_constant_buffer();
    void create_output_tensor_uav();

    void create_nv12_srvs(ID3D12Resource* nv12_texture);
    void update_params(UINT src_width, UINT src_height);


private:
    D3D12Core& d3d12_core_;
    Config config_;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> root_signature_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline_state_;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> srv_uav_heap_;
    UINT descriptor_size_ = 0;

    Microsoft::WRL::ComPtr<ID3D12Resource> output_tensor_buffer_;
    D3D12_RESOURCE_STATES output_tensor_state_ = D3D12_RESOURCE_STATE_COMMON;

    Microsoft::WRL::ComPtr<ID3D12Resource> constant_buffer_;
    uint8_t* mapped_constants_ = nullptr;

    LetterboxParams letterbox_params_;
};