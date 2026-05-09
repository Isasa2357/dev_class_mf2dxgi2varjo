#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <wrl/client.h>

#include <d3d12.h>

#include <vector>
#include <cstdint>

#include "D3D12Core.hpp"

std::vector<float> ReadbackNchwFloatTensorD3D12Buffer(
    D3D12Core& d3d12_core,
    ID3D12Resource* tensor_buffer,
    D3D12_RESOURCE_STATES& tensor_buffer_state,
    size_t element_count
);

std::vector<float> ReadbackNchwFloatTensorD3D12Buffer(
    D3D12Core& d3d12_core,
    ID3D12Resource* tensor_buffer,
    D3D12_RESOURCE_STATES& tensor_buffer_state,
    UINT width,
    UINT height,
    UINT channels = 3
);