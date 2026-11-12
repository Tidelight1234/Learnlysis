# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""Windows DPAPI 封装：用「当前登录用户」的凭据加解密一小段秘密（AI 助手的 api_key）。

定位——这是「保护用户本地数据」，不是「防逆向」：
- CryptProtectData 用登录用户的主密钥加密，别的 Windows 账户、被拔走的磁盘、
  拷走的 data/ai_settings.json 都读不出明文；
- 但同一个用户自己仍能解开（那本来就是他的 key），这是设计使然，不是漏洞。

非 Windows 平台没有 DPAPI，protect()/unprotect() 直接返回 None，由上层降级成
明文存储（并在 UI 上照旧警示）。只依赖 ctypes + crypt32.dll，不引任何第三方库。

密文用 `dpapi:` 前缀 + base64 落进 JSON，既 JSON-safe，又能和历史遗留的明文
api_key 区分开：读到不带前缀的旧值时原样返回，老用户升级后不必重填 key。
"""

import sys
import base64
import ctypes

#: 密文标记前缀：读到带它的值才走解密，否则当明文（向后兼容）
DPAPI_PREFIX = "dpapi:"

# 无 UI 确认框：桌面应用后台加解密，不能弹系统的「是否允许」对话框
_CRYPTPROTECT_UI_FORBIDDEN = 0x1

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    from ctypes import wintypes

    class _DATA_BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD),
                    ("pbData", ctypes.POINTER(ctypes.c_char))]

    _crypt32 = ctypes.windll.crypt32
    _kernel32 = ctypes.windll.kernel32

    # 64 位下必须显式声明 argtypes，否则指针会被按 32 位截断，调用直接崩
    _blob_p = ctypes.POINTER(_DATA_BLOB)
    _crypt32.CryptProtectData.restype = wintypes.BOOL
    _crypt32.CryptProtectData.argtypes = [
        _blob_p, wintypes.LPCWSTR, _blob_p, wintypes.LPVOID,
        wintypes.LPVOID, wintypes.DWORD, _blob_p]
    _crypt32.CryptUnprotectData.restype = wintypes.BOOL
    _crypt32.CryptUnprotectData.argtypes = [
        _blob_p, wintypes.LPCWSTR, _blob_p, wintypes.LPVOID,
        wintypes.LPVOID, wintypes.DWORD, _blob_p]
    _kernel32.LocalFree.argtypes = [wintypes.HLOCAL]


def available():
    """DPAPI 是否可用（仅 Windows）。上层据此决定要不要降级成明文。"""
    return _IS_WINDOWS


def _to_blob(data):
    """把 bytes 包成 DATA_BLOB；返回 (blob, buf)，buf 必须在调用期间保持存活。"""
    n = len(data)
    buf = ctypes.create_string_buffer(data, n) if n else ctypes.create_string_buffer(b"\0", 1)
    blob = _DATA_BLOB(n, ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    return blob, buf


def _drain(out_blob):
    """读出 API 填充的输出 blob，随后立刻 LocalFree（文档要求调用方释放）。"""
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        _kernel32.LocalFree(out_blob.pbData)


def protect(plaintext):
    """把明文加密成 `dpapi:<base64>` 串；不可用或失败时返回 None（上层降级明文）。"""
    if not _IS_WINDOWS or plaintext is None:
        return None
    try:
        in_blob, in_buf = _to_blob(plaintext.encode("utf-8"))   # noqa: F841 buf 保活
        out_blob = _DATA_BLOB()
        ok = _crypt32.CryptProtectData(
            ctypes.byref(in_blob), None, None, None, None,
            _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out_blob))
        if not ok:
            return None
        return DPAPI_PREFIX + base64.b64encode(_drain(out_blob)).decode("ascii")
    except Exception:                                            # noqa: BLE001
        return None


def unprotect(token):
    """把 `dpapi:<base64>` 解回明文；不是本格式 / 解不开（换用户换机器）返回 None。"""
    if not _IS_WINDOWS or not isinstance(token, str) or not token.startswith(DPAPI_PREFIX):
        return None
    try:
        data = base64.b64decode(token[len(DPAPI_PREFIX):].encode("ascii"))
        in_blob, in_buf = _to_blob(data)                          # noqa: F841 buf 保活
        out_blob = _DATA_BLOB()
        ok = _crypt32.CryptUnprotectData(
            ctypes.byref(in_blob), None, None, None, None,
            _CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(out_blob))
        if not ok:
            return None
        return _drain(out_blob).decode("utf-8")
    except Exception:                                            # noqa: BLE001
        return None


def is_protected(value):
    """判断一个存下来的 api_key 串是不是 DPAPI 密文。"""
    return isinstance(value, str) and value.startswith(DPAPI_PREFIX)
