#include "CameraCapture.hpp"

#define NOMINMAX
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mfobjects.h>
#include <d3d10_1.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <d3d11_3.h>
#include <d3dcompiler.h>
#include <dxgi1_2.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3d10_1.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

namespace {

    template <class T>
    void release_com(T*& p) {
        if (p) {
            p->Release();
            p = nullptr;
        }
    }

    void ShowHr(const char* where, HRESULT hr) {
        char buf[512];
        sprintf_s(buf, "%s failed\nHRESULT = 0x%08X", where, static_cast<unsigned>(hr));
        MessageBoxA(nullptr, buf, "Error", MB_OK | MB_ICONERROR);
    }

    std::wstring GetAllocatedString(IMFActivate* act, const GUID& key) {
        if (!act) return L"";
        WCHAR* value = nullptr;
        UINT32 cch = 0;
        std::wstring out;
        if (SUCCEEDED(act->GetAllocatedString(key, &value, &cch)) && value) {
            out.assign(value, cch);
            CoTaskMemFree(value);
        }
        return out;
    }

    struct CameraDeviceInfo {
        IMFActivate* activate = nullptr;
        std::wstring friendly_name;
        std::wstring symbolic_link;
    };

    void ReleaseDevices(std::vector<CameraDeviceInfo>& devices) {
        for (auto& d : devices) {
            release_com(d.activate);
        }
        devices.clear();
    }

    bool EnumerateVideoCaptureDevices(std::vector<CameraDeviceInfo>& out_devices) {
        IMFAttributes* attrs = nullptr;
        IMFActivate** activates = nullptr;
        UINT32 count = 0;
        HRESULT hr = S_OK;

        out_devices.clear();

        hr = MFCreateAttributes(&attrs, 1);
        if (FAILED(hr)) goto done;

        hr = attrs->SetGUID(
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
        if (FAILED(hr)) goto done;

        hr = MFEnumDeviceSources(attrs, &activates, &count);
        if (FAILED(hr)) goto done;

        out_devices.reserve(count);
        for (UINT32 i = 0; i < count; ++i) {
            CameraDeviceInfo info;
            info.activate = activates[i];
            activates[i] = nullptr;
            info.friendly_name = GetAllocatedString(info.activate, MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME);
            info.symbolic_link = GetAllocatedString(info.activate, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK);
            out_devices.push_back(std::move(info));
        }

    done:
        if (activates) {
            for (UINT32 i = 0; i < count; ++i) release_com(activates[i]);
            CoTaskMemFree(activates);
        }
        release_com(attrs);
        return SUCCEEDED(hr);
    }

    void PrintDeviceList(const std::vector<CameraDeviceInfo>& devices) {
        std::wcout << L"\n=== Detected video capture devices ===\n";
        if (devices.empty()) {
            std::wcout << L"(none)\n";
            return;
        }

        for (size_t i = 0; i < devices.size(); ++i) {
            std::wcout << L"[" << i << L"] " << devices[i].friendly_name << L"\n";
            std::wcout << L"    symbolic link: " << devices[i].symbolic_link << L"\n";
        }
    }

    bool EnableD3DMultithreadProtection(ID3D11Device* device) {
        if (!device) return false;

        ID3D10Multithread* mt = nullptr;
        const HRESULT hr = device->QueryInterface(__uuidof(ID3D10Multithread), reinterpret_cast<void**>(&mt));
        if (FAILED(hr) || !mt) {
            ShowHr("QueryInterface(ID3D10Multithread)", hr);
            return false;
        }

        mt->SetMultithreadProtected(TRUE);
        release_com(mt);
        return true;
    }

    LRESULT CALLBACK PreviewWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        switch (msg) {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) {
                DestroyWindow(hwnd);
                return 0;
            }
            break;
        }
        return DefWindowProcW(hwnd, msg, wParam, lParam);
    }

    bool EnsureWindowClass() {
        static bool registered = false;
        if (registered) return true;

        WNDCLASSW wc{};
        wc.lpfnWndProc = PreviewWndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"CameraCapturePreviewWindow";
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);

        if (!RegisterClassW(&wc)) {
            const DWORD err = GetLastError();
            if (err != ERROR_CLASS_ALREADY_EXISTS) return false;
        }

        registered = true;
        return true;
    }

    class Nv12PreviewRenderer {
    public:
        ~Nv12PreviewRenderer() = default;

        bool initialize(ID3D11Device3* device, ID3D11DeviceContext1* context) {
            m_device = device;
            m_context = context;
            if (!m_device || !m_context) return false;
            m_device->AddRef();
            m_context->AddRef();

            if (!createShaders()) return false;
            if (!createQuad()) return false;
            if (!createSampler()) return false;
            return true;
        }

        void cleanup() {
            release_com(m_copyUVSRV);
            release_com(m_copyYSRV);
            release_com(m_copyTexture);

            release_com(m_sampler);
            release_com(m_vertexBuffer);
            release_com(m_inputLayout);
            release_com(m_pixelShader);
            release_com(m_vertexShader);

            release_com(m_context);
            release_com(m_device);
        }

        bool prepareFrame(const MFFrame& frame) {
            if (!frame.is_valid || !frame.gpu_backed || !frame.gpu_texture) return false;

            D3D11_TEXTURE2D_DESC srcDesc{};
            frame.gpu_texture->GetDesc(&srcDesc);
            if (srcDesc.Format != DXGI_FORMAT_NV12) return false;

            if (!ensureCopyTexture(static_cast<int>(srcDesc.Width), static_cast<int>(srcDesc.Height))) {
                return false;
            }

            m_context->CopyResource(m_copyTexture, frame.gpu_texture);
            return true;
        }

        void render(ID3D11RenderTargetView* rtv, int dstWidth, int dstHeight) {
            if (!rtv) return;

            const float clearColor[4] = {0.f, 0.f, 0.f, 1.f};
            m_context->OMSetRenderTargets(1, &rtv, nullptr);
            m_context->ClearRenderTargetView(rtv, clearColor);

            if (!m_copyYSRV || !m_copyUVSRV) {
                return;
            }

            D3D11_VIEWPORT vp{};
            vp.TopLeftX = 0.0f;
            vp.TopLeftY = 0.0f;
            vp.Width = static_cast<float>(dstWidth);
            vp.Height = static_cast<float>(dstHeight);
            vp.MinDepth = 0.0f;
            vp.MaxDepth = 1.0f;
            m_context->RSSetViewports(1, &vp);

            UINT stride = sizeof(Vertex);
            UINT offset = 0;
            m_context->IASetInputLayout(m_inputLayout);
            m_context->IASetVertexBuffers(0, 1, &m_vertexBuffer, &stride, &offset);
            m_context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

            m_context->VSSetShader(m_vertexShader, nullptr, 0);
            m_context->PSSetShader(m_pixelShader, nullptr, 0);
            m_context->PSSetSamplers(0, 1, &m_sampler);

            ID3D11ShaderResourceView* srvs[2] = {m_copyYSRV, m_copyUVSRV};
            m_context->PSSetShaderResources(0, 2, srvs);
            m_context->Draw(4, 0);

            ID3D11ShaderResourceView* nullSrvs[2] = {nullptr, nullptr};
            m_context->PSSetShaderResources(0, 2, nullSrvs);
        }

    private:
        struct Vertex {
            float px, py;
            float u, v;
        };

        bool createShaders() {
            static const char* vsSrc = R"(
struct VSIn {
    float2 pos : POSITION;
    float2 uv  : TEXCOORD0;
};
struct VSOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};
VSOut main(VSIn input) {
    VSOut o;
    o.pos = float4(input.pos, 0.0, 1.0);
    o.uv = input.uv;
    return o;
}
)";

            static const char* psSrc = R"(
Texture2D texY  : register(t0);
Texture2D texUV : register(t1);
SamplerState samp0 : register(s0);

struct PSIn {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

float3 YUVToRGB(float y, float u, float v) {
    float Y = y * 255.0;
    float U = u * 255.0;
    float V = v * 255.0;

    float C = Y - 16.0;
    float D = U - 128.0;
    float E = V - 128.0;

    float R = (298.0 * C + 409.0 * E + 128.0) / 256.0;
    float G = (298.0 * C - 100.0 * D - 208.0 * E + 128.0) / 256.0;
    float B = (298.0 * C + 516.0 * D + 128.0) / 256.0;

    return saturate(float3(R, G, B) / 255.0);
}

float4 main(PSIn input) : SV_Target {
    float y = texY.Sample(samp0, input.uv).r;
    float2 uv = texUV.Sample(samp0, input.uv).rg;
    float3 rgb = YUVToRGB(y, uv.x, uv.y);
    return float4(rgb, 1.0);
}
)";

            ID3DBlob* vsBlob = nullptr;
            ID3DBlob* psBlob = nullptr;
            ID3DBlob* errBlob = nullptr;

            HRESULT hr = D3DCompile(vsSrc, strlen(vsSrc), nullptr, nullptr, nullptr, "main", "vs_4_0", 0, 0, &vsBlob, &errBlob);
            if (FAILED(hr)) {
                if (errBlob) {
                    MessageBoxA(nullptr, static_cast<const char*>(errBlob->GetBufferPointer()), "VS compile error", MB_OK | MB_ICONERROR);
                }
                release_com(errBlob);
                release_com(vsBlob);
                return false;
            }
            release_com(errBlob);

            hr = D3DCompile(psSrc, strlen(psSrc), nullptr, nullptr, nullptr, "main", "ps_4_0", 0, 0, &psBlob, &errBlob);
            if (FAILED(hr)) {
                if (errBlob) {
                    MessageBoxA(nullptr, static_cast<const char*>(errBlob->GetBufferPointer()), "PS compile error", MB_OK | MB_ICONERROR);
                }
                release_com(errBlob);
                release_com(vsBlob);
                release_com(psBlob);
                return false;
            }
            release_com(errBlob);

            hr = m_device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_vertexShader);
            if (FAILED(hr)) {
                ShowHr("CreateVertexShader", hr);
                release_com(vsBlob);
                release_com(psBlob);
                return false;
            }

            hr = m_device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &m_pixelShader);
            if (FAILED(hr)) {
                ShowHr("CreatePixelShader", hr);
                release_com(vsBlob);
                release_com(psBlob);
                return false;
            }

            D3D11_INPUT_ELEMENT_DESC layout[] = {
                { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,                     D3D11_INPUT_PER_VERTEX_DATA, 0 },
                { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, sizeof(float) * 2,     D3D11_INPUT_PER_VERTEX_DATA, 0 },
            };

            hr = m_device->CreateInputLayout(
                layout,
                static_cast<UINT>(std::size(layout)),
                vsBlob->GetBufferPointer(),
                vsBlob->GetBufferSize(),
                &m_inputLayout);

            release_com(vsBlob);
            release_com(psBlob);

            if (FAILED(hr)) {
                ShowHr("CreateInputLayout", hr);
                return false;
            }

            return true;
        }

        bool createQuad() {
            const Vertex verts[4] = {
                { -1.0f,  1.0f, 0.0f, 0.0f },
                {  1.0f,  1.0f, 1.0f, 0.0f },
                { -1.0f, -1.0f, 0.0f, 1.0f },
                {  1.0f, -1.0f, 1.0f, 1.0f },
            };

            D3D11_BUFFER_DESC bd{};
            bd.ByteWidth = sizeof(verts);
            bd.Usage = D3D11_USAGE_IMMUTABLE;
            bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;

            D3D11_SUBRESOURCE_DATA init{};
            init.pSysMem = verts;

            const HRESULT hr = m_device->CreateBuffer(&bd, &init, &m_vertexBuffer);
            if (FAILED(hr)) {
                ShowHr("CreateBuffer(vertex)", hr);
                return false;
            }
            return true;
        }

        bool createSampler() {
            D3D11_SAMPLER_DESC sd{};
            sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
            sd.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
            sd.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
            sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
            sd.MaxLOD = D3D11_FLOAT32_MAX;

            const HRESULT hr = m_device->CreateSamplerState(&sd, &m_sampler);
            if (FAILED(hr)) {
                ShowHr("CreateSamplerState", hr);
                return false;
            }
            return true;
        }

        bool createNV12Views(ID3D11Texture2D* tex, ID3D11ShaderResourceView1** outYSRV, ID3D11ShaderResourceView1** outUVSRV) {
            if (!tex || !outYSRV || !outUVSRV) return false;
            *outYSRV = nullptr;
            *outUVSRV = nullptr;

            D3D11_TEXTURE2D_DESC td{};
            tex->GetDesc(&td);
            if (td.Format != DXGI_FORMAT_NV12) return false;

            D3D11_SHADER_RESOURCE_VIEW_DESC1 yDesc{};
            yDesc.Format = DXGI_FORMAT_R8_UNORM;
            yDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            yDesc.Texture2D.MostDetailedMip = 0;
            yDesc.Texture2D.MipLevels = 1;
            yDesc.Texture2D.PlaneSlice = 0;

            HRESULT hr = m_device->CreateShaderResourceView1(tex, &yDesc, outYSRV);
            if (FAILED(hr)) {
                ShowHr("CreateShaderResourceView1(Y plane)", hr);
                return false;
            }

            D3D11_SHADER_RESOURCE_VIEW_DESC1 uvDesc{};
            uvDesc.Format = DXGI_FORMAT_R8G8_UNORM;
            uvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            uvDesc.Texture2D.MostDetailedMip = 0;
            uvDesc.Texture2D.MipLevels = 1;
            uvDesc.Texture2D.PlaneSlice = 1;

            hr = m_device->CreateShaderResourceView1(tex, &uvDesc, outUVSRV);
            if (FAILED(hr)) {
                ShowHr("CreateShaderResourceView1(UV plane)", hr);
                release_com(*outYSRV);
                return false;
            }

            return true;
        }

        bool ensureCopyTexture(int width, int height) {
            if (m_copyTexture && m_width == width && m_height == height) {
                return true;
            }

            release_com(m_copyUVSRV);
            release_com(m_copyYSRV);
            release_com(m_copyTexture);

            D3D11_TEXTURE2D_DESC td{};
            td.Width = static_cast<UINT>(width);
            td.Height = static_cast<UINT>(height);
            td.MipLevels = 1;
            td.ArraySize = 1;
            td.Format = DXGI_FORMAT_NV12;
            td.SampleDesc.Count = 1;
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = D3D11_BIND_SHADER_RESOURCE;

            const HRESULT hr = m_device->CreateTexture2D(&td, nullptr, &m_copyTexture);
            if (FAILED(hr)) {
                ShowHr("CreateTexture2D(NV12 copy)", hr);
                return false;
            }

            if (!createNV12Views(m_copyTexture, &m_copyYSRV, &m_copyUVSRV)) {
                return false;
            }

            m_width = width;
            m_height = height;
            return true;
        }

    private:
        ID3D11Device3* m_device = nullptr;
        ID3D11DeviceContext1* m_context = nullptr;

        ID3D11VertexShader* m_vertexShader = nullptr;
        ID3D11PixelShader* m_pixelShader = nullptr;
        ID3D11InputLayout* m_inputLayout = nullptr;
        ID3D11Buffer* m_vertexBuffer = nullptr;
        ID3D11SamplerState* m_sampler = nullptr;

        ID3D11Texture2D* m_copyTexture = nullptr;
        ID3D11ShaderResourceView1* m_copyYSRV = nullptr;
        ID3D11ShaderResourceView1* m_copyUVSRV = nullptr;
        int m_width = 0;
        int m_height = 0;
    };

} // namespace

int wmain() {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        ShowHr("CoInitializeEx", hr);
        return 1;
    }

    hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) {
        ShowHr("MFStartup", hr);
        CoUninitialize();
        return 1;
    }

    std::vector<CameraDeviceInfo> devices;
    if (!EnumerateVideoCaptureDevices(devices)) {
        std::wcerr << L"Failed to enumerate cameras.\n";
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    PrintDeviceList(devices);
    if (devices.empty()) {
        ReleaseDevices(devices);
        MFShutdown();
        CoUninitialize();
        return 0;
    }

    std::wcout << L"\nSelect camera index: ";
    int index = -1;
    if (!(std::wcin >> index) || index < 0 || index >= static_cast<int>(devices.size())) {
        std::wcerr << L"Invalid camera index.\n";
        ReleaseDevices(devices);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    const std::wstring symbolicLink = devices[static_cast<size_t>(index)].symbolic_link;
    const std::wstring friendlyName = devices[static_cast<size_t>(index)].friendly_name;
    ReleaseDevices(devices);

    if (!EnsureWindowClass()) {
        std::wcerr << L"Failed to register window class.\n";
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    HWND hwnd = CreateWindowExW(
        0,
        L"CameraCapturePreviewWindow",
        (L"CameraCapture Preview - " + friendlyName).c_str(),
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        1280,
        800,
        nullptr,
        nullptr,
        GetModuleHandleW(nullptr),
        nullptr);

    if (!hwnd) {
        std::wcerr << L"CreateWindowExW failed.\n";
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL requested[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0
    };
    D3D_FEATURE_LEVEL obtained = D3D_FEATURE_LEVEL_11_0;

    IDXGIFactory1* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    IDXGIDevice* dxgiDevice = nullptr;
    IDXGIAdapter* dxgiAdapter = nullptr;
    IDXGIFactory2* factory2 = nullptr;
    IDXGISwapChain1* swapChain1 = nullptr;
    IDXGISwapChain* swapChain = nullptr;
    ID3D11Device* baseDevice = nullptr;
    ID3D11DeviceContext* baseContext = nullptr;
    ID3D11Device3* device = nullptr;
    ID3D11DeviceContext1* context = nullptr;
    ID3D11Texture2D* backBuffer = nullptr;
    ID3D11RenderTargetView* backBufferRTV = nullptr;

    hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        flags,
        requested,
        static_cast<UINT>(std::size(requested)),
        D3D11_SDK_VERSION,
        &baseDevice,
        &obtained,
        &baseContext);
    if (FAILED(hr)) {
        ShowHr("D3D11CreateDevice", hr);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    if (!EnableD3DMultithreadProtection(baseDevice)) {
        release_com(baseContext);
        release_com(baseDevice);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    hr = baseDevice->QueryInterface(IID_PPV_ARGS(&device));
    if (FAILED(hr)) {
        ShowHr("QueryInterface(ID3D11Device3)", hr);
        release_com(baseContext);
        release_com(baseDevice);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    hr = baseContext->QueryInterface(IID_PPV_ARGS(&context));
    if (FAILED(hr)) {
        ShowHr("QueryInterface(ID3D11DeviceContext1)", hr);
        release_com(device);
        release_com(baseContext);
        release_com(baseDevice);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    hr = baseDevice->QueryInterface(IID_PPV_ARGS(&dxgiDevice));
    if (FAILED(hr)) {
        ShowHr("QueryInterface(IDXGIDevice)", hr);
        release_com(context);
        release_com(device);
        release_com(baseContext);
        release_com(baseDevice);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    hr = dxgiDevice->GetAdapter(&dxgiAdapter);
    if (FAILED(hr)) {
        ShowHr("IDXGIDevice::GetAdapter", hr);
        release_com(dxgiDevice);
        release_com(context);
        release_com(device);
        release_com(baseContext);
        release_com(baseDevice);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    hr = dxgiAdapter->GetParent(IID_PPV_ARGS(&factory2));
    if (FAILED(hr)) {
        ShowHr("IDXGIAdapter::GetParent", hr);
        release_com(dxgiAdapter);
        release_com(dxgiDevice);
        release_com(context);
        release_com(device);
        release_com(baseContext);
        release_com(baseDevice);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.Width = 0;
    scd.Height = 0;
    scd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    scd.Stereo = FALSE;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount = 2;
    scd.Scaling = DXGI_SCALING_STRETCH;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;

    hr = factory2->CreateSwapChainForHwnd(baseDevice, hwnd, &scd, nullptr, nullptr, &swapChain1);
    if (FAILED(hr)) {
        ShowHr("CreateSwapChainForHwnd", hr);
        release_com(factory2);
        release_com(dxgiAdapter);
        release_com(dxgiDevice);
        release_com(context);
        release_com(device);
        release_com(baseContext);
        release_com(baseDevice);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    hr = swapChain1->QueryInterface(IID_PPV_ARGS(&swapChain));
    if (FAILED(hr)) {
        ShowHr("QueryInterface(IDXGISwapChain)", hr);
        release_com(swapChain1);
        release_com(factory2);
        release_com(dxgiAdapter);
        release_com(dxgiDevice);
        release_com(context);
        release_com(device);
        release_com(baseContext);
        release_com(baseDevice);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    hr = swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
    if (FAILED(hr)) {
        ShowHr("IDXGISwapChain::GetBuffer", hr);
        release_com(swapChain);
        release_com(swapChain1);
        release_com(factory2);
        release_com(dxgiAdapter);
        release_com(dxgiDevice);
        release_com(context);
        release_com(device);
        release_com(baseContext);
        release_com(baseDevice);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    hr = baseDevice->CreateRenderTargetView(backBuffer, nullptr, &backBufferRTV);
    if (FAILED(hr)) {
        ShowHr("CreateRenderTargetView", hr);
        release_com(backBuffer);
        release_com(swapChain);
        release_com(swapChain1);
        release_com(factory2);
        release_com(dxgiAdapter);
        release_com(dxgiDevice);
        release_com(context);
        release_com(device);
        release_com(baseContext);
        release_com(baseDevice);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    release_com(backBuffer);
    release_com(factory2);
    release_com(dxgiAdapter);
    release_com(dxgiDevice);

    Nv12PreviewRenderer renderer;
    if (!renderer.initialize(device, context)) {
        release_com(backBufferRTV);
        release_com(swapChain);
        release_com(swapChain1);
        release_com(context);
        release_com(device);
        release_com(baseContext);
        release_com(baseDevice);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    ELP_USBFHD08S_LC1100_CameraCapture camera;
    if (!camera.initialize(baseDevice)) {
        std::wcerr << L"camera.initialize failed.\n";
        renderer.cleanup();
        release_com(backBufferRTV);
        release_com(swapChain);
        release_com(swapChain1);
        release_com(context);
        release_com(device);
        release_com(baseContext);
        release_com(baseDevice);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    if (!camera.open(symbolicLink)) {
        std::wcerr << L"camera.open failed.\n";
        camera.close();
        renderer.cleanup();
        release_com(backBufferRTV);
        release_com(swapChain);
        release_com(swapChain1);
        release_com(context);
        release_com(device);
        release_com(baseContext);
        release_com(baseDevice);
        DestroyWindow(hwnd);
        MFShutdown();
        CoUninitialize();
        return 1;
    }

    std::wcout << L"\nPreview started. Press ESC or close the window to quit.\n";

    bool running = true;
    while (running) {
        MSG msg{};
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                running = false;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!running) break;

        MFFrame frame{};
        if (camera.read_frame(frame)) {
            renderer.prepareFrame(frame);
        }

        RECT rc{};
        GetClientRect(hwnd, &rc);
        const int width = std::max(1L, rc.right - rc.left);
        const int height = std::max(1L, rc.bottom - rc.top);

        renderer.render(backBufferRTV, width, height);
        swapChain->Present(1, 0);

        release_com(frame.gpu_texture);

        if ((GetAsyncKeyState(VK_ESCAPE) & 0x8000) != 0) {
            running = false;
        }
    }

    camera.close();
    renderer.cleanup();

    release_com(backBufferRTV);
    release_com(swapChain);
    release_com(swapChain1);
    release_com(context);
    release_com(device);
    release_com(baseContext);
    release_com(baseDevice);

    DestroyWindow(hwnd);
    MFShutdown();
    CoUninitialize();
    return 0;
}
