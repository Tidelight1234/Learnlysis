"""构建水印、导出审计与关于页的溯源字段。

钉三件事：

1. **水印真进了产物**——用户手上那份 JSON 里要能查到 build_id，否则外流的副本
   无从反查是哪一次构建发出去的；同时源码库里这份 `build_stamp` 必须永远是 dev
   空值，带了值就说明打包步骤把水印写回源树了。本文件也会被
   `_protect_stage.py --verify` 在**已写水印的副本**里跑一遍，所以断言必须按
   「现在站在哪份副本上」分岔，而不是一律假设 dev。
2. **Cython 化后族名不会静默变空**——旧写法靠 `inspect.getmodule(func)` 推族模块，
   核心模块编成 `.pyd` 后拿不到源文件，它会返回 None，审计里的 family 列悄悄变空
   且不报错。现在族名由注册表盖上（`spec.family`），这里逐方法断言非空。
3. **水印要能被最终用户看见**——关于页是 build_id 唯一的对外出口，也是「签名校验
   失败只标记、不拦启动」这条立场的落点，故两个分支的呈现都钉住（见文件末节）。
"""

import json
import sys
import types

import pandas as pd
import pytest

import about_dialog
import analysis_export
import build_stamp
import statistic_visualization_export as statviz_export
from analysis_methods import MethodSpec, all_methods
from statistic_visualization_data import AnalysisResult


def _result():
    return AnalysisResult(method_id="IND01", method_name="LSA 滞后序列分析",
                          analysis_id="IND01_fixture",
                          tables={"状态转换矩阵": pd.DataFrame({"a": [1]})})


# ------------------------------------------------------------------ 水印本体

def test_水印状态与所在副本一致():
    """同一个断言要能站在两种副本上：源码库（dev 空值）与加固 staging（已写水印）。

    写成单纯的「永远为空」会在 `_protect_stage.py --verify` 那一趟里必红（那份副本
    就是该有值），而那个 skip 又会把「水印真进了 .pyd」这条证据一并丢成没人管。
    """
    if build_stamp.is_release():
        # 发布副本：水印非空且能解析，才算 Cython 编完的 .pyd 真带着它。
        assert build_stamp.BUILD_ID and build_stamp.GIT_COMMIT
        items = build_stamp.hardening_items()
        assert items.get("docstrings") in ("stripped", "kept"), items
        assert "compiled" in items, items
        return
    assert build_stamp.BUILD_ID == ""
    assert build_stamp.is_release() is False
    # dev 也要有完整的键（空串而不是缺键）：下游解析方的 schema 不能因构建而异
    assert build_stamp.audit_dict() == {"build_id": "", "build_time": "",
                                        "git_commit": "", "hardening": ""}


@pytest.mark.parametrize("attr", ["BUILD_ID", "BUILD_TIME", "GIT_COMMIT", "HARDENING"])
def test_四个水印字段各归各(attr):
    monkeypatch_target = {"BUILD_ID": "5.1.0+20260915-a1b2c3d",
                          "BUILD_TIME": "2026-09-15T08:00:00Z",
                          "GIT_COMMIT": "a1b2c3d",
                          "HARDENING": "docstrings=stripped"}[attr]
    import pytest as _p  # noqa  保持局部引用清晰
    with _p.MonkeyPatch.context() as mp:
        mp.setattr(build_stamp, attr, monkeypatch_target)
        assert build_stamp.audit_dict()[attr.lower()] == monkeypatch_target


def test_加固档位可解析成清单():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(build_stamp, "HARDENING",
                   "docstrings=stripped;compiled=analysis_stats_sequence,interaction_model")
        items = build_stamp.hardening_items()
        assert items["docstrings"] == "stripped"
        assert items["compiled"].startswith("analysis_stats_sequence")


# ------------------------------------------------------------------ 水印进产物

def test_水印进了_config_audit_与_model_card():
    fields = {"BUILD_ID": "5.1.0+20260915-a1b2c3d", "BUILD_TIME": "2026-09-15T08:00:00Z",
              "GIT_COMMIT": "a1b2c3d", "HARDENING": "docstrings=stripped;compiled=2"}
    with pytest.MonkeyPatch.context() as mp:
        for key, value in fields.items():
            mp.setattr(build_stamp, key, value)
        result = _result()
        for produced in (statviz_export.build_config(result, None),
                         statviz_export.build_audit(result, None),
                         analysis_export.build_model_card(result, None)):
            build = produced["build"]
            assert build["build_id"] == fields["BUILD_ID"], produced
            assert build["hardening"] == fields["HARDENING"]


def test_水印真落到磁盘上的_json(tmp_path):
    """内存字典对不算数：写盘那条路（jsonable → json.dump）也得带着走一遍。"""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(build_stamp, "BUILD_ID", "5.1.0+D-C0FFEE")
        path = tmp_path / "run_audit.json"
        statviz_export.export_audit(_result(), None, str(path))
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk["build"]["build_id"] == "5.1.0+D-C0FFEE"


def test_extra_与水印各占各的格子():
    """extra 直接 update 到顶层，同名键调用方赢——所以水印**不能**写在全局可读的
    顶层键上，否则谁传一个 extra 带 build 就把溯源信息抹了。这里
    钉住两件事：extra 照常生效，build 子对象不被它染成一个子。
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(build_stamp, "BUILD_ID", "5.1.0+X")
        audit = statviz_export.build_audit(_result(), None,
                                           extra={"analysis_id": "caller-wins"})
        assert audit["analysis_id"] == "caller-wins"
        assert isinstance(audit["build"], dict) and audit["build"]["build_id"] == "5.1.0+X"


# ------------------------------------------------------ 族名解析（Cython 前置条件）

def test_每个注册方法都带族名():
    specs = all_methods()
    assert specs, "方法注册表空了"
    missing = [spec.id for spec in specs if not getattr(spec, "family", "")]
    assert not missing, "这些方法没有族名，导出审计会静默写空：%s" % missing


@pytest.mark.parametrize("suffix,expect",
                         [(".pyd", "compiled"), (".so", "compiled"),
                          (".dll", "compiled"), (".py", "source"), (None, "unknown")])
def test_族来源能区分源码与编译产物(suffix, expect):
    name = "_fake_family_module"
    fake = types.ModuleType(name)
    if suffix is not None:
        fake.__file__ = name + suffix
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, name, fake)
        spec = MethodSpec(id="X01", name="占位", group="g", question="q",
                          family=name, compute=lambda ctx, p: None)
        version = analysis_export.method_version(spec)
        assert version["family"] == name
        assert version["source"] == expect
        # 版本号在没有 METHOD_VERSION 常量时如实回落应用版本，不编造第三套版本
        assert version["app"] == version["family_version"]


def test_没有族名时仍能从函数自身模块回落():
    """手造的 spec（测试、脚本）不该因为没走注册表就拿不到族名。"""
    spec = MethodSpec(id="X02", name="占位", group="g", question="q",
                      compute=statviz_export.slug)
    version = analysis_export.method_version(spec)
    assert version["family"] == "statistic_visualization_export"
    assert version["source"] == "source"


# ------------------------------------------------------------------ 关于页呈现
# 水印写进 .pyd 只是半成品：拿到 exe 的人看不见 build_id，就等于没有溯源入口。
# 这两项把 `_info_rows()` 的两个分支都钉住（含「完整性」行只在发布态出现这条立场），
# 并且刻意用 monkeypatch 强制分支，这样同一份断言在源码树与已写水印的 stage 副本
# 里跑出来的结果一致，不必再按副本分岔。

def _rows_as_dict():
    return dict(about_dialog._info_rows())


def test_关于页发布态显示构建标识与完整性():
    import integrity
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(build_stamp, "is_release", lambda: True)
        mp.setattr(build_stamp, "BUILD_ID", "5.1.0+20260915-a1b2c3d")
        mp.setattr(build_stamp, "BUILD_TIME", "2026-09-15T08:00:00Z")
        mp.setattr(integrity, "self_result",
                   lambda: {"status": "unsigned", "code": "", "path": "Learnlysis.exe"})
        rows = _rows_as_dict()
        assert rows["构建标识"] == "5.1.0+20260915-a1b2c3d", rows
        assert rows["构建时间"] == "2026-09-15T08:00:00Z"
        # 「未签名」要写在脸上而不是拦启动：这一行是 P1-7 立场的唯一对外出口
        assert rows["完整性"] == integrity.STATUS_LABELS["unsigned"], rows

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(build_stamp, "is_release", lambda: True)
        mp.setattr(integrity, "self_result",
                   lambda: {"status": "tampered", "code": "0x80096010",
                            "path": "Learnlysis.exe"})
        assert "篡改" in _rows_as_dict()["完整性"]


def test_关于页源码态不摆完整性空行():
    """dev 下 `self_result()` 恒为 `dev`，那一行只是噪声；改成露提交与文件时间。"""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(build_stamp, "is_release", lambda: False)
        rows = _rows_as_dict()
        assert "完整性" not in rows, rows
        assert rows["提交"] and rows["构建日期"]


def test_关于对话框带着版权声明与法律提示(qapp):
    from PySide6.QtWidgets import QLabel

    from config import APP_COPYRIGHT, APP_LEGAL_NOTE, APP_NAME

    dialog = about_dialog.AboutDialog()
    texts = "\n".join(l.text() for l in dialog.findChildren(QLabel))
    assert APP_COPYRIGHT in texts, texts
    assert APP_LEGAL_NOTE in texts, texts
    assert APP_NAME in texts
