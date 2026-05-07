#include "CameraCapture.hpp"

#include <winerror.h>

namespace {

	// Release a COM object safely and null the pointer.
	// This helper is used throughout the file for IMF* / ID3D11* / DXGI objects.
	template <class T>
	void safe_release(T*& p) {
		if (p) {
			p->Release();
			p = nullptr;
		}
	}

	std::wstring get_allocated_string(IMFActivate* act, const GUID& key) 
	{
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

	void clean_mfframe(MFFrame& frame) {
		safe_release(frame.gpu_texture);

		frame.is_valid = false;
		frame.gpu_backed = false;
		frame.width = 0;
		frame.height = 0;
		frame.frame_id = 0;
		frame.timestamp = 0;
	}
}

bool BaseCameraCapture::initialize(ID3D11Device* d3d_device)
{
	// 一度全て閉じる
	this->close();

	// デバイスが有効でなければ初期化失敗
	if (!d3d_device) return false;

	// D3D11デバイスを使用してMFのDXGIデバイスマネージャーを作成
	HRESULT hr = MFCreateDXGIDeviceManager(&this->reset_token_, &this->dxgi_device_mgr_);
	if (FAILED(hr)) return false;

	// デバイスマネージャーにD3D11デバイスを関連付ける
	hr = this->dxgi_device_mgr_->ResetDevice(d3d_device, this->reset_token_);
	if (FAILED(hr)) {
		safe_release(this->dxgi_device_mgr_);
		return false;
	}

	return true;
}

bool BaseCameraCapture::open(const std::wstring& symbolic_link)
{
	// source_readerやその他変数の初期化
	this->close_source_reader();

	IMFAttributes* attributes = nullptr;
	IMFActivate** devices = nullptr;
	UINT32 count = 0;
	IMFMediaSource* source = nullptr;
	IMFAttributes* readerAttributes = nullptr;
	IMFMediaType* chosenType = nullptr;
	HRESULT hr = S_OK;
	IMFActivate* chosenActivate = nullptr;

	// attributeの作成
	hr = MFCreateAttributes(&attributes, 1);
	if (FAILED(hr)) goto done;

	hr = attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
	if (FAILED(hr)) goto done;

	// デバイスの列挙
	hr = MFEnumDeviceSources(attributes, &devices, &count);
	if (FAILED(hr)) goto done;
	if (count == 0) {
		hr = E_FAIL;
		goto done;
	}

	// 列挙されたデバイスから、指定されたシンボリックリンクと一致するものを探す
	for (UINT32 i = 0; i < count; ++i) {
		const std::wstring device_symbolic_link = get_allocated_string(devices[i], MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK);
		if (symbolic_link == device_symbolic_link) {
			chosenActivate = devices[i];
			break;
		}
	}

	// 一致するデバイスが見つからない
	if (!chosenActivate) {
		hr = HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
		goto done;
	}

	// デバイスをアクティブ化してメディアソースを取得
	hr = chosenActivate->ActivateObject(IID_PPV_ARGS(&source));
	if (FAILED(hr)) goto done;

	// ソースリーダのAttributeを作成 + 属性を追加
	hr = MFCreateAttributes(&readerAttributes, 8);
	if (FAILED(hr)) goto done;

	readerAttributes->SetUINT32(MF_LOW_LATENCY, TRUE);
	readerAttributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, FALSE);
#ifdef MF_READWRITE_DISABLE_CONVERTERS
	readerAttributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, TRUE);
#endif
#ifdef MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS
	readerAttributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
#endif
#ifdef MF_SOURCE_READER_DISABLE_DXVA
	readerAttributes->SetUINT32(MF_SOURCE_READER_DISABLE_DXVA, FALSE);
#endif
	if (this->dxgi_device_mgr_) {
		readerAttributes->SetUnknown(MF_SOURCE_READER_D3D_MANAGER, this->dxgi_device_mgr_);
	}

	// ソースリーダの作成
	hr = MFCreateSourceReaderFromMediaSource(source, readerAttributes, &this->source_reader_);
	if (FAILED(hr)) goto done;

	hr = this->source_reader_->SetStreamSelection(static_cast<DWORD>(MF_SOURCE_READER_ALL_STREAMS), FALSE);
	if (FAILED(hr)) goto done;

	hr = this->source_reader_->SetStreamSelection(static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM), TRUE);
	if (FAILED(hr)) goto done;

	// 最適なメディアタイプを選択して設定
	chosenType = this->chooseBestType();
	if (!chosenType) {
		hr = E_FAIL;
		goto done;
	}

	hr = this->source_reader_->SetCurrentMediaType(
		static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM),
		nullptr,
		chosenType);
	if (FAILED(hr)) goto done;

	// 選択されたメディアタイプからフレームサイズを取得
	{
		UINT32 w = 0, h = 0;
		hr = MFGetAttributeSize(chosenType, MF_MT_FRAME_SIZE, &w, &h);
		if (FAILED(hr)) goto done;

		this->width_ = static_cast<int>(w);
		this->height_ = static_cast<int>(h);
	}

done:

	// クリーンアップ
	safe_release(chosenType);
	safe_release(readerAttributes);
	safe_release(source);
	if (devices) {
		for (UINT32 i = 0; i < count; ++i) safe_release(devices[i]);
		CoTaskMemFree(devices);
	}
	safe_release(attributes);
	if (FAILED(hr)) {
		this->close_source_reader();
	}

	return SUCCEEDED(hr) && this->source_reader_ != nullptr;
}

void BaseCameraCapture::close()
{
	this->close_source_reader();
	this->close_dxgi_device_mgr();
}

void BaseCameraCapture::close_source_reader()
{
	safe_release(this->source_reader_);
	this->width_ = 0;
	this->height_ = 0;
	this->next_frame_id_ = 0;
}

void BaseCameraCapture::close_dxgi_device_mgr()
{
	safe_release(this->dxgi_device_mgr_);
	this->reset_token_ = 0;
}

bool BaseCameraCapture::read_frame(MFFrame& out_frame)
{
	// ソースリーダが有効でなければ失敗
	if (!this->source_reader_) return false;
	
	// フレーム構造体をクリーンアップ
	clean_mfframe(out_frame);

	DWORD stream_index = 0;
	DWORD flags = 0;
	LONGLONG timestamp = 0;
	IMFSample* sample = nullptr;
	IMFMediaBuffer* buffer = nullptr;
	IMFDXGIBuffer* dxgi_buffer = nullptr;
	HRESULT hr = S_OK;

	hr = this->source_reader_->ReadSample(
		static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM),
		0,
		&stream_index,
		&flags,
		&timestamp,
		&sample);

	if (FAILED(hr)) goto done;								// フレームの読み取りに失敗
	if (flags & MF_SOURCE_READERF_ENDOFSTREAM) goto done;	// ストリームの終わりに達した
	if (flags & MF_SOURCE_READERF_STREAMTICK) goto done;	// ストリームティック（フレームがスキップされた） - これも有効なフレームではない
	if (!sample) goto done;									// サンプルが空 - これも有効なフレームではない

	// サンプルからバッファを取得
	hr = sample->ConvertToContiguousBuffer(&buffer);
	if (FAILED(hr) || !buffer) goto done;

	hr = buffer->QueryInterface(IID_PPV_ARGS(&dxgi_buffer));
	if (FAILED(hr) || !dxgi_buffer) goto done;

	{
		ID3D11Texture2D* tex = nullptr;
		hr = dxgi_buffer->GetResource(IID_PPV_ARGS(&tex));
		if (FAILED(hr) || !tex) {
			safe_release(tex);
			goto done;
		}

		D3D11_TEXTURE2D_DESC desc{};
		tex->GetDesc(&desc);

		if (desc.Format != this->desired_format()) {
			safe_release(tex);
			goto done;
		}

		out_frame.is_valid = true;
		out_frame.gpu_backed = true;
		out_frame.width = width_;
		out_frame.height = height_;
		out_frame.timestamp = timestamp;
		out_frame.frame_id = this->next_frame_id_++;
		out_frame.gpu_texture = tex;

		safe_release(dxgi_buffer);
		safe_release(buffer);
		safe_release(sample);
		return true;
	}

done:

	safe_release(dxgi_buffer);
	safe_release(buffer);
	safe_release(sample);

	return false;
}

DXGI_FORMAT ELP_USBFHD08S_LC1100_CameraCapture::desired_format() const
{
	return DXGI_FORMAT_NV12;
}

IMFMediaType* ELP_USBFHD08S_LC1100_CameraCapture::chooseBestType()
{
	IMFMediaType* best = nullptr;
	uint64_t bestScore = 0;

	for (DWORD i = 0;; ++i) {
		IMFMediaType* mt = nullptr;
		const HRESULT hr = this->source_reader_->GetNativeMediaType(static_cast<DWORD>(MF_SOURCE_READER_FIRST_VIDEO_STREAM), i, &mt);
		if (hr == MF_E_NO_MORE_TYPES) break;
		if (FAILED(hr) || !mt) continue;

		GUID subtype{};
		UINT32 w = 0, h = 0, frNum = 0, frDen = 0;
		if (FAILED(mt->GetGUID(MF_MT_SUBTYPE, &subtype)) ||
			FAILED(MFGetAttributeSize(mt, MF_MT_FRAME_SIZE, &w, &h))) {
			safe_release(mt);
			continue;
		}

		MFGetAttributeRatio(mt, MF_MT_FRAME_RATE, &frNum, &frDen);
		const uint64_t fpsX1000 = (frDen != 0) ? (static_cast<uint64_t>(frNum) * 1000ULL / frDen) : 0ULL;

		if (subtype == MFVideoFormat_NV12) {
			const uint64_t score = static_cast<uint64_t>(w) * h * 100000ULL + fpsX1000;
			if (!best || score > bestScore) {
				safe_release(best);
				best = mt;
				bestScore = score;
			} else {
				safe_release(mt);
			}
		} else {
			safe_release(mt);
		}
	}

	return best;
}
