"""签名自校验的行为回归：重点钉住「真能分出已签名 / 未签名」这件事。

这里踩过一个很坑的错：Action GUID 用了网上流传的 006C9B001527 尾巴，
WinVerifyTrust 对**任何**文件都返回 TRUST_E_PROVIDER_UNKNOWN，
签名的系统 dll 也被报成失败，而校验器「不抛异常」的立场把这个问题洗得很干净
——只有拿真实已签名的系统文件做断言才会暴露。所以本文件必须有正向断言。
"""

import sys

import pytest

import integrity


@pytest.mark.skipif(sys.platform != "win32", reason="Authenticode 只在 Windows 上有意义")
def test_系统已签名文件判为_signed():
    result = integrity.verify_file(r"C:\Windows\System32\kernel32.dll")
    assert result["status"] == "signed", result


@pytest.mark.skipif(sys.platform != "win32", reason="Authenticode 只在 Windows 上有意义")
def test_未签名文件判为_unsigned(tmp_path):
    plain = tmp_path / "not_a_pe.exe"
    plain.write_bytes(b"MZ" + b"\x00" * 512)
    result = integrity.verify_file(str(plain))
    # 不是合法 PE 时系统给 FORM_UNKNOWN，也归到"没签名可用"这一类，
    # 关键是不能是 signed；provider 不认识（error）则是 GUID 回归。
    assert result["status"] in ("unsigned", "untrusted", "policy"), result


@pytest.mark.skipif(sys.platform != "win32", reason="Authenticode 只在 Windows 上有意义")
def test_错误码不会静默变_signed():
    # 用一个一定不存在的路径：必须是 skipped，不能是 signed
    assert integrity.verify_file(r"Z:\__no_such_file__.exe")["status"] == "skipped"


def test_源码运行标为_dev():
    result = integrity.self_result()
    if not getattr(sys, "frozen", False):
        # 开发态下绝不去校验解释器：python.exe 在本机可能签名有效，
        # 拿它当「本程序签名正常」会是彻底的误报。
        assert result["status"] == "dev"


def test_label_始终有中文且不抛():
    for status in list(integrity.STATUS_LABELS) + ["__未知__"]:
        text = integrity.label({"status": status, "code": ""})
        assert text and isinstance(text, str)
    # error 带码时要把码露出来，不能只写"校验不可用"
    assert "0x800B0001" in integrity.label({"status": "error", "code": "0x800B0001"})


def test_is_verified_只认真正验过签的():
    assert integrity.is_verified({"status": "signed"}) is True
    for status in ("unsigned", "tampered", "error", "dev", "skipped", "policy"):
        assert integrity.is_verified({"status": status}) is False
    assert integrity.is_verified(None) is False
