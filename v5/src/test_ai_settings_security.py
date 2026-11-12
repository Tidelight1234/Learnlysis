"""AI 设置里 api_key 的存储边界：DPAPI 密文落盘、明文回读、历史文件兼容。

覆盖 P0 防护档第 2 项（保护用户本地数据）。注意 conftest 的 autouse `chart_fonts`
会构造 QApplication，于是 `_disk_writes_blocked()` 默认为真（offscreen 不写盘）——
本文件所有落盘断言都必须显式打开 LEARNLYSIS_AUTOSAVE=1 这个官方逃生口，
并把模块级的 AI_SETTINGS_FILE / data_dir 指到 tmp_path，绝不碰工程 data/。
"""

import json
import os

import pytest

import ai_settings
import secure_store

KEY = "sk-SECRET-abc123-中文也要能过一遍"


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    """把 AI 设置文件重定向到临时目录，并允许 offscreen 下真的写盘。"""
    path = tmp_path / "ai_settings.json"
    monkeypatch.setattr(ai_settings, "AI_SETTINGS_FILE", str(path))
    monkeypatch.setattr(ai_settings, "data_dir", str(tmp_path))
    monkeypatch.setenv("LEARNLYSIS_AUTOSAVE", "1")
    return str(path)


def _disk_provider(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["provider"]


def test_dpapi_available_on_windows():
    """DPAPI 只在 Windows 有；非 Windows 上 protect 返回 None 是设计而非故障。"""
    assert secure_store.available() == (os.name == "nt")
    if not secure_store.available():
        pytest.skip("DPAPI is Windows-only")
    token = secure_store.protect(KEY)
    assert token.startswith(secure_store.DPAPI_PREFIX)
    assert secure_store.unprotect(token) == KEY
    # 不是密文格式的串绝不硬解，交由上层按明文处理
    assert secure_store.unprotect("sk-plain") is None
    assert secure_store.is_protected(token) and not secure_store.is_protected("sk-plain")


def test_key_is_written_as_ciphertext(settings_file):
    """盘上找不到明文，内存里却是明文——「保存后立刻发请求」不必回读文件。"""
    store = ai_settings.AiSettings()
    store.set_provider(api_key=KEY, persist_key=True)

    with open(settings_file, encoding="utf-8") as handle:
        raw = handle.read()
    assert KEY not in raw, "明文 key 泄露到磁盘"
    assert raw.count(secure_store.DPAPI_PREFIX) >= 1
    assert store.api_key() == KEY


def test_key_survives_reload(settings_file):
    """重新装载一份镜像（模拟重启程序）能解回同一个明文。"""
    ai_settings.AiSettings().set_provider(api_key=KEY, persist_key=True)
    assert ai_settings.AiSettings().api_key() == KEY


def test_legacy_plaintext_still_readable(settings_file):
    """升级前存过明文 key 的老配置文件照常能用，不要求用户重填。"""
    with open(settings_file, "w", encoding="utf-8") as handle:
        json.dump({"provider": {"api_key": "sk-legacy-plain"}}, handle)
    assert ai_settings.AiSettings().api_key() == "sk-legacy-plain"


def test_undecryptable_key_degrades_to_empty(settings_file, caplog):
    """密文解不开（配置被搬到别的账户/机器）按空处理，绝不把一串密文当 key 用出去。"""
    with open(settings_file, "w", encoding="utf-8") as handle:
        json.dump({"provider": {"api_key": secure_store.DPAPI_PREFIX + "AAAA不是真密文"}},
                  handle)
    assert ai_settings.AiSettings().api_key() == ""


def test_persist_key_false_erases_on_disk(settings_file):
    """勾了「本次不保存」：盘上抹空，内存里这一次仍然可用。"""
    store = ai_settings.AiSettings()
    store.set_provider(api_key=KEY, persist_key=True)
    store.set_provider(persist_key=False)
    assert _disk_provider(settings_file)["api_key"] == ""
    assert store.api_key() == KEY


def test_empty_key_stays_empty(settings_file):
    """空 key 不该被加密成一串没意义的密文占位。"""
    store = ai_settings.AiSettings()
    store.set_provider(api_key="", persist_key=True)
    assert _disk_provider(settings_file)["api_key"] == ""
