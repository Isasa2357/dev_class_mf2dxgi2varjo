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
        // output0: [1, num_attrs, num_candidates]
        // output1: [1, num_mask_coeffs, mask_width, mask_height]
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

    struct DetectionWithMask {
        Detection detection;
        std::vector<float> mask;
        UINT mask_width = 0;
        UINT mask_height = 0;
    };

public:
    YoloSegPostProcessorD3D12(
        D3D12Core& d3d12_core,
        const Config& config
    );

    YoloSegPostProcessorD3D12(const YoloSegPostProcessorD3D12&) = delete;
    YoloSegPostProcessorD3D12& operator=(const YoloSegPostProcessorD3D12&) = delete;

    ~YoloSegPostProcessorD3D12();

    void upload_outputs_from_cpu(
        const std::vector<float>& output0,
        const std::vector<float>& output1
    );

    void process_uploaded_outputs_and_wait();

    std::vector<DetectionWithMask> readback_results();

    ID3D12Resource* output0_buffer() const;
    ID3D12Resource* output1_buffer() const;

    ID3D12Resource* selected_detection_buffer() const;
    ID3D12Resource* selected_counter_buffer() const;
    ID3D12Resource* selected_mask_buffer() const;

    const Config& config() const;

    D3D12_RESOURCE_STATES& selected_detection_state_ref() { return this->selected_detection_state_; }
    D3D12_RESOURCE_STATES& selected_counter_state_ref() { return this->selected_counter_state_; }
    D3D12_RESOURCE_STATES& selected_mask_state_ref() { return this->selected_mask_state_; }

private:
    struct ShaderParams {
        UINT num_attrs;
        UINT num_candidates;
        UINT num_classes;
        UINT num_mask_coeffs;

        UINT mask_width;
        UINT mask_height;
        UINT mask_pixels;
        UINT max_candidates;

        UINT max_detections;
        float conf_threshold;
        float iou_threshold;
        float input_width;

        float input_height;
        UINT reserved0;
        UINT reserved1;
        UINT reserved2;
    };

private:
    void create_root_signature();
    void create_pipeline_states();
    void create_descriptor_heap();
    void create_buffers();
    void create_descriptors();
    void create_constant_buffer();
    void update_params();

    void upload_buffer_from_cpu(
        ID3D12Resource* dst_buffer,
        D3D12_RESOURCE_STATES& dst_state,
        ID3D12Resource* upload_buffer,
        const void* data,
        UINT64 byte_size
    );

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
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mask_pso_;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptor_heap_;
    UINT descriptor_size_ = 0;

    Microsoft::WRL::ComPtr<ID3D12Resource> output0_buffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> output0_upload_buffer_;
    D3D12_RESOURCE_STATES output0_state_ = D3D12_RESOURCE_STATE_COMMON;

    Microsoft::WRL::ComPtr<ID3D12Resource> output1_buffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> output1_upload_buffer_;
    D3D12_RESOURCE_STATES output1_state_ = D3D12_RESOURCE_STATE_COMMON;

    Microsoft::WRL::ComPtr<ID3D12Resource> candidate_buffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> candidate_counter_buffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> keep_flag_buffer_;

    D3D12_RESOURCE_STATES candidate_state_ = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES candidate_counter_state_ = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES keep_flag_state_ = D3D12_RESOURCE_STATE_COMMON;

    Microsoft::WRL::ComPtr<ID3D12Resource> selected_detection_buffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> selected_counter_buffer_;
    Microsoft::WRL::ComPtr<ID3D12Resource> selected_mask_buffer_;

    D3D12_RESOURCE_STATES selected_detection_state_ = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES selected_counter_state_ = D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES selected_mask_state_ = D3D12_RESOURCE_STATE_COMMON;

    Microsoft::WRL::ComPtr<ID3D12Resource> zero_upload_buffer_;

    Microsoft::WRL::ComPtr<ID3D12Resource> constant_buffer_;
    uint8_t* mapped_constants_ = nullptr;
};