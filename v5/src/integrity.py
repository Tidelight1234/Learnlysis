# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""Authenticode 签名自校验：用 WinVerifyTrust 判断「这个 exe 还是不是原样发出去的」。

定位——防护档里的「完整性校验」，配合代码签名才有意义：

- 没买证书时，本模块只会得出 `unsigned`，界面照常在关于页写「未签名构建」，
  **绝不拦启动**（否则开发者自测与本地打包全被堵死）；
- 签过名之后，任何人改过 exe（去水印、改弹窗、塞代码）都会让摘要对不上，
  得出 `tampered`。校验逻辑本身会被 Cython 编成 `.pyd` 再进包，翻一个 `if`
  就绕过的写法在这里不成立。

只做本机离线判定：`WTD_REVOKE_NONE` + `WTD_CACHE_ONLY_URL_RETRIEVAL`，
绝不联网拉 CRL/OCSP——一个纯离线桌面应用不该在关于页里偷偷开一条外连。
非 Windows、文件不存在、API 抛异常一律降级成状态串返回，本模块对外不抛异常。
"""

import sys
import ctypes

#: 状态码 → 界面上的说法。未知状态一律带十六进制错误码回显，不含糊报「出错」。
STATUS_LABELS = {
    "signed": "已签名（发布者可信）",
    "unsigned": "未签名构建",
    "tampered": "签名校验失败：文件可能已被篡改",
    "untrusted": "已签名，但证书链不被本机信任",
    "policy": "已签名，但本机安全策略拒绝校验",
    "skipped": "不适用（非 Windows 或文件不存在）",
    "dev": "源码运行，非打包产物",
    "error": "校验不可用",
}

# --- wintrust 的常量与结构（取值见 Wintrust.h / MS-WVT 文档）-----------------
_WTD_UI_NONE = 0x2
_WTD_REVOKE_NONE = 0x0
_WTD_CHOICE_FILE = 0x1
_WTD_STATEACTION_IGNORE = 0x0
_WTD_SAFER_FLAG = 0x10
_WTD_DISABLE_MD2_MD4 = 0x2000
_WTD_CACHE_ONLY_URL_RETRIEVAL = 0x1000

_ERROR_SUCCESS = 0x00000000
_TRUST_E_NOSIGNATURE = 0x800B0100
_CRYPT_E_NO_SIGNER = 0x8009201C
_CRYPT_E_NOT_IN_CERT_CHAIN = 0x80092027
_TRUST_E_BAD_DIGEST = 0x80096010
_CRYPT_E_HASH_VALUE = 0x8009201B
_TRUST_E_PROVIDER_UNKNOWN = 0x800B0001
_CRYPT_E_SECURITY_SETTINGS = 0x80092023
_TRUST_E_SUBJECT_FORM_UNKNOWN = 0x800B0003
_TRUST_E_SUBJECT_NOT_TRUSTED = 0x800B0004

_CLASSIFY = {
    _TRUST_E_NOSIGNATURE: "unsigned",
    _CRYPT_E_NO_SIGNER: "unsigned",
    _TRUST_E_BAD_DIGEST: "tampered",
    _CRYPT_E_HASH_VALUE: "tampered",
    _CRYPT_E_SECURITY_SETTINGS: "policy",
    # TRUST_E_PROVIDER_UNKNOWN 不归到 untrusted：它意味着「我们这次调用本身没被系统
    # 认账」（结构体版本不对 / 缺 wintrust），拿它报「证书链不可信」是把校验器自己的
    # 故障洗成被检文件的问题。落到 default 的 error 里，连带错误码一起露出来。
    _CRYPT_E_NOT_IN_CERT_CHAIN: "untrusted",
    _TRUST_E_SUBJECT_FORM_UNKNOWN: "untrusted",
    _TRUST_E_SUBJECT_NOT_TRUSTED: "untrusted",
}

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:
    from ctypes import wintypes

    class _GUID(ctypes.Structure):
        _fields_ = [("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", ctypes.c_byte * 8)]

    class _FileInfo(ctypes.Structure):
        _fields_ = [("cbStruct", wintypes.DWORD),
                    ("pcwszFilePath", wintypes.LPCWSTR),
                    ("hFile", wintypes.HANDLE),
                    ("pgKnownSubject", ctypes.POINTER(_GUID))]

    # dwUnionChoice 选 FILE 时只用 pFile，但 union 必须按最大成员对齐，
    # 少了这几个占位成员，64 位下 cbStruct 就会比系统期望的小一圈。
    class _Payload(ctypes.Union):
        _fields_ = [("pFile", ctypes.POINTER(_FileInfo)),
                    ("pCatalog", ctypes.c_void_p),
                    ("pBlob", ctypes.c_void_p),
                    ("pSigner", ctypes.c_void_p)]

    # ⚠⚠ 这个结构体必须停在 v1 布局（x64 = 80 字节）：WinVerifyTrust 拿 cbStruct
    # 当版本号，多一个成员就换一个版本。带上 v2 才有的 pUIContext 会凑成 88 字节，
    # 系统认成「未知策略提供程序」，于是对**任何**文件都返回 0x800B0001
    # （TRUST_E_PROVIDER_UNKNOWN），签名的也报未签名（实测踩过）。
    # dwUIChoice=WTD_UI_NONE 时 v2 字段全用不上，到此为止是对的。
    class _TrustData(ctypes.Structure):
        _fields_ = [("cbStruct", wintypes.DWORD),
                    ("pPolicyCallbackData", ctypes.c_void_p),
                    ("pSIPClientData", ctypes.c_void_p),
                    ("dwUIChoice", wintypes.DWORD),
                    ("fdwRevocationChecks", wintypes.DWORD),
                    ("dwUnionChoice", wintypes.DWORD),
                    ("payload", _Payload),
                    ("dwStateAction", wintypes.DWORD),
                    ("hWVTStateData", wintypes.HANDLE),
                    ("pwszURLReference", wintypes.LPCSTR),
                    ("dwProvFlags", wintypes.DWORD),
                    ("dwUIContext", wintypes.DWORD)]

    # WINTRUST_ACTION_GENERIC_VERIFY_V2 = {00AAC56B-CD44-11D0-8CC2-00C04FC295EE}
    # 尾巴必须是 00C04FC295EE：网上流传的 006C9B001527 是错的，本机注册表
    # （HKLM\...\Cryptography\Providers\Trust\Signature）只登记了前者，用错尾巴
    # 会让**任何**文件都返回 0x800B0001（TRUST_E_PROVIDER_UNKNOWN），
    # 签过名的系统 dll 也报失败——实测 ctypes 与 C# P/Invoke 双双踩坑。
    _ACTION_ID = _GUID(
        0x00AAC56B, 0xCD44, 0x11D0,
        (ctypes.c_byte * 8)(0x8C, 0xC2, 0x00, 0xC0, 0x4F, 0xC2, 0x95, 0xEE))

    _wintrust = ctypes.windll.wintrust
    # 不显式写 argtypes，64 位下三个指针参数都会被按 32 位截断，直接崩。
    # 两个指针参数写 c_void_p 而不是 POINTER(...)：ctypes 不接受把 byref()
    # 返回的 CArgObject 交给 POINTER 型的 argtype（报 TypeError），
    # 而结构体里那个 union 成员也只能靠 cast 过去——两边统一按 void* 走。
    _wintrust.WinVerifyTrust.restype = wintypes.LONG
    _wintrust.WinVerifyTrust.argtypes = [wintypes.HANDLE,
                                         ctypes.c_void_p,
                                         ctypes.c_void_p]


def available():
    """本机能不能真做校验（只有 Windows 能）。"""
    return _IS_WINDOWS


def _classify(code):
    if code == _ERROR_SUCCESS:
        return "signed", ""
    status = _CLASSIFY.get(code & 0xFFFFFFFF, "error")
    return status, "0x%08X" % (code & 0xFFFFFFFF)


def verify_file(path):
    """校验一个 PE 文件的 Authenticode 签名，返回 ``{"status", "code", "path"}``。

    status 取值见 `STATUS_LABELS`；本函数绝不抛异常，任何失败都变成一个可展示的状态。
    """
    result = {"status": "skipped", "code": "", "path": str(path or "")}
    if not _IS_WINDOWS:
        return result
    try:
        import os
        if not path or not os.path.isfile(path):
            return result
        file_info = _FileInfo()
        file_info.cbStruct = ctypes.sizeof(_FileInfo)
        file_info.pcwszFilePath = str(path)
        file_info.hFile = None
        file_info.pgKnownSubject = ctypes.cast(None, ctypes.POINTER(_GUID))

        data = _TrustData()
        data.cbStruct = ctypes.sizeof(_TrustData)
        data.dwUIChoice = _WTD_UI_NONE
        data.fdwRevocationChecks = _WTD_REVOKE_NONE
        data.dwUnionChoice = _WTD_CHOICE_FILE
        data.payload.pFile = ctypes.cast(ctypes.byref(file_info),
                                         ctypes.POINTER(_FileInfo))
        data.dwStateAction = _WTD_STATEACTION_IGNORE
        data.dwProvFlags = (_WTD_SAFER_FLAG | _WTD_DISABLE_MD2_MD4
                            | _WTD_CACHE_ONLY_URL_RETRIEVAL)

        code = _wintrust.WinVerifyTrust(None, ctypes.cast(ctypes.byref(_ACTION_ID),
                                                          ctypes.c_void_p),
                                        ctypes.cast(ctypes.byref(data),
                                                    ctypes.c_void_p))
        status, raw = _classify(code)
        result.update({"status": status, "code": raw})
        return result
    except Exception as exc:                                     # noqa: BLE001
        # 校验器自己出问题（结构不匹配 / 系统缺 wintrust）绝不能连累应用：
        # 报 "error" 并带上异常名，便于在关于页与日志里一眼看出是校验器坏了。
        result.update({"status": "error", "code": type(exc).__name__})
        return result


def self_result():
    """校验当前正在跑的程序：打包后查 exe 自身，源码运行返回 `dev`。"""
    if getattr(sys, "frozen", False):
        return verify_file(sys.executable)
    return {"status": "dev", "code": "", "path": sys.executable}


def label(result):
    """把状态字典变成一句中文（未知状态带错误码回显，不写成「未知错误」）。"""
    status = (result or {}).get("status", "error")
    text = STATUS_LABELS.get(status, STATUS_LABELS["error"])
    code = (result or {}).get("code") or ""
    if status == "error" and code:
        return "%s（%s）" % (text, code)
    return text


def is_verified(result):
    """只有真正验过签才算「完好」；`unsigned` / `policy` / `error` 都不算。"""
    return (result or {}).get("status") == "signed"
