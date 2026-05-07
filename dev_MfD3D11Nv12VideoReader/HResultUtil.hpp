#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <comdef.h>

#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>

namespace win_util {

    inline std::string hresult_to_string(HRESULT hr) {
        switch (hr) {
        case S_OK:
            return "S_OK";
        case E_ABORT:
            return "E_ABORT";
        case E_ACCESSDENIED:
            return "E_ACCESSDENIED";
        case E_FAIL:
            return "E_FAIL";
        case E_HANDLE:
            return "E_HANDLE";
        case E_INVALIDARG:
            return "E_INVALIDARG";
        case E_NOINTERFACE:
            return "E_NOINTERFACE";
        case E_NOTIMPL:
            return "E_NOTIMPL";
        case E_OUTOFMEMORY:
            return "E_OUTOFMEMORY";
        default:
            return "Not convertable error";
        }
    }

    inline void ThrowIfFailed(HRESULT hr, const char* message)
    {
        if (FAILED(hr)) {
            _com_error err(hr);
            std::wstring wmsg = err.ErrorMessage();

            std::stringstream ss;
            ss << message
                << " HRESULT=0x"
                << std::hex
                << static_cast<unsigned long>(hr)
                << " "
                << "str = "
                << hresult_to_string(hr);

            std::string narrow(wmsg.begin(), wmsg.end());
            ss << narrow;

            throw std::runtime_error(ss.str());
        }
    }

}