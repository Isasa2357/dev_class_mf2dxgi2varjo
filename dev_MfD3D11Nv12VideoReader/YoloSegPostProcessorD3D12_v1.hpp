#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <wrl/client.h>

#include <d3d12.h>

#include <cstdint>
#include <vector>
#include <cstddef>

#include "D3D12Core.hpp"

class YoloSegPostProcessorD3D12 {
public:
    struct Config {
        UINT num_attrs = 116;
        UINT num_candidates = 8400;
        UINT num_classes = 80;
        UINT num_mask_coeffs = 32;

        UINT max_candidates = 4096;
        UINT max_detections = 512;

        float conf_threshold = 0.25f;
        float iou_threshold = 0.45f;

        float input_width = 640.0f;
        float input_height = 640.0f;
    };

    struct Detection {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;

        uint32_t class_id;
        uint32_t candidate_index;
        uint32_t reserved;
    };

public:
    YoloSegPostProcessorD3D12(
        D3D12Core& d3d12_core,
        const Config& config = Config()
    );

    YoloSegPostProcessorD3D12(const YoloSegPostProcessorD3D12&) = delete;
    YoloSegPostProcessorD3D12& operator=(const YoloSegPostProcessorD3D12&) = delete;

    // 現在の CPU output0 経路用。
    // output0 は shape [1, num_attrs, num_candidates] の float 配列を想定。
    void upload_output0_from_cpu(
        const std::vector<float>& output0
    );

    // upload済み output0 に対して GPU decode + NMS + compact を実行する。
    void process_uploaded_output0_and_wait();

    // GPU後処理済み detection を CPU に戻す。
    std::vector<Detection> readback_detections();

    ID3D12Resource* output0_buffer() const;
    ID3D12Resource* selected_detection_buffer() const;
    ID3D12Resource* selected_counter_buffer() const;

    D3D12_RESOURCE_STATES& output0_state_ref();
    D3D12_RESOURCE_STATES& selected_detection_state_ref();
    D3D12_RESOURCE_STATES& selected_counter_state_ref();

    const Config& config() const;

private:
    struct ShaderParams {
        UINT num_attrs;
        UINT num_candidates;
        UINT num_classes;
        UINT num_mask_coeffs;

        UINT max_candidates;
        UINT max_detections;
        float conf_threshold;
        float iou_threshold;

        float input_width;
        float input_height;
        UINT reserved0;
        UINT reserved1;
    };

private:
    void create_root_signature();
    void create_pipeline_states();
    void create_descriptor_heap();
    void create_buffers();
    void create_descriptors();
    void create_constant_buffer();
    void update_params();

    void reset_counters(
        ID3D12GraphicsCommandList* command_list
    );

    void record_process(
        ID3D12GraphicsCommandList* command_list
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

    Microsoft::WRL::ComPtr<ID3D12PipelineState> decode_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> nms_pso_;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> compact_pso_;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptor_heap_;
    UINT descriptor_size_ = 0;

    Microsoft::WRL::ComPtr<ID3D12Resource> output0_buffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> output0_upload_buffer_;
    D3D12_RESOURCE_STATES output0_state_ = D3D12_RESOURCE_STATE_COMMON;

    Microsoft::WRL::ComPtr<ID3D12Resource> candidate_buffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> candidate_counter_buffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> keep_flag_buffer_;

    D3D12_RESOURCE_STATES candidate_state_ = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES candidate_counter_state_ = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES keep_flag_state_ = D3D12_RESOURCE_STATE_COMMON;

    Microsoft::WRL::ComPtr<ID3D12Resource> selected_detection_buffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> selected_counter_buffer_;

    D3D12_RESOURCE_STATES selected_detection_state_ = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES selected_counter_state_ = D3D12_RESOURCE_STATE_COMMON;

    Microsoft::WRL::ComPtr<ID3D12Resource> zero_upload_buffer_;

    Microsoft::WRL::ComPtr<ID3D12Resource> constant_buffer_;
    uint8_t* mapped_constants_ = nullptr;
};