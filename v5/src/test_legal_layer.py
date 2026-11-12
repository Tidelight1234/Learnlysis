# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""署名法律层的口径一致性回归（P0-4）。

这一层不需要跑起来就能验，但它是**最容易各自漂移**的一层：版权行同时出现在四个地方
（仓库根 LICENSE、NOTICE、关于弹窗、exe 属性页），任何一处手改年份或措辞，都不会有
测试红起来——只会留下一份「三份互相矛盾的版权声明」的发行物，法务上比不写更糟。
所以这里钉的是「同一句话只有一个来源」，而不是「这句话写得好不好」。

判据全部来自 ``config``：文案本身可以改，但只能改一处。
"""

import re
from pathlib import Path

import pytest

import config

APP_DIR = Path(__file__).resolve().parent


def _find_legal_root():
    """向上找同时装着 LICENSE 与 NOTICE 的目录。

    不写死 ``parents[2]``：源码树里是 ``version/v5/``，而构建加固会把这个测试
    拷进 ``version/v5/_stage/``，多套一层，硬编码层级会指到不存在的目录。
    """
    here = APP_DIR
    for _ in range(6):
        if (here / "LICENSE").is_file() and (here / "NOTICE").is_file():
            return here
        if here.parent == here:
            break
        here = here.parent
    return None


LEGAL_ROOT = _find_legal_root()


# ---------------------------------------------------------------- 仓库根法律文本

@pytest.mark.skipif(LEGAL_ROOT is None, reason="仓库根没有 LICENSE / NOTICE（不在源码树里跑）")
def test_根目录法律文件齐备():
    """关于弹窗的版权行让用户「保留安装包目录下的 LICENSE 与 NOTICE」，它们就得真在。"""
    for name in ("LICENSE", "NOTICE"):
        text = (LEGAL_ROOT / name).read_text(encoding="utf-8")
        assert len(text) > 200, "%s 几乎是空的" % name


@pytest.mark.skipif(LEGAL_ROOT is None, reason="仓库根没有 LICENSE / NOTICE")
def test_LICENSE_的版权行与_config_逐字一致():
    text = (LEGAL_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert config.APP_COPYRIGHT in text, (
        "LICENSE 里的版权声明与 config.APP_COPYRIGHT 不是同一句话：%r vs %r"
        % (text.splitlines()[1] if text.splitlines() else "", config.APP_COPYRIGHT))
    # MIT 授权文本及其版权/许可声明保留要求必须存在。
    assert "MIT License" in text and "permission notice" in text


@pytest.mark.skipif(LEGAL_ROOT is None, reason="仓库根没有 LICENSE / NOTICE")
def test_NOTICE_如实交代第三方与_PySide6_风险():
    text = (LEGAL_ROOT / "NOTICE").read_text(encoding="utf-8")
    for comp in ("PySide6", "pandas", "NumPy", "MiSans", "PyInstaller", "Cython"):
        assert comp in text, "NOTICE 少了 %s" % comp
    # PySide6 的 LGPL/GPL/商业多重授权与替换库义务必须在发行说明中交代
    assert "LGPL" in text and "商业" in text and "替换" in text, \
        "NOTICE 没有写明 PySide6 的许可风险"


# ---------------------------------------------------------------- 源文件版权头

def test_随包分发的源文件都带版权头():
    """归属声明写进文件本身：拿到一份 .py 副本的人，不用翻仓库也能看见权利人。

    刻意用 ``#`` 注释而不是 docstring——构建加固会剥掉 docstring，
    版权声明不该跟着没（见 ``_protect_stage.py``）。
    开发期脚本（``_*.py`` / ``test_*.py``）不进分发物，故不在检查范围内；
    stage 副本里没有这些脚本，整个用例就地跳过。
    """
    header = pytest.importorskip("_set_source_header")
    paths = list(header.shipped_sources())
    assert paths, "一个随包源文件都没找到，判据本身坏了"
    missing = [p.name for p in paths if not header.has_header(p.read_text(encoding="utf-8"))]
    assert not missing, "缺版权头：%s（跑一次 python _set_source_header.py 补齐）" % missing
    assert header.HEADER_LINES[0] == "# " + config.APP_COPYRIGHT


def test_版权头是注释不是文档串():
    """写进 docstring 的话，加固一剥就没了——这条钉的是「放在第几行」这件事。"""
    header = pytest.importorskip("_set_source_header")
    assert all(line.startswith("#") for line in header.HEADER_LINES)
    first = next(iter(header.shipped_sources())).read_text(encoding="utf-8").splitlines()[0]
    assert first == header.HEADER_LINES[0]


# ---------------------------------------------------------------- 版本与署名单一来源

def test_版本号只在_config_里定义():
    """别的模块只许 ``from config import APP_VERSION``；再开一份常量必漏改一处。"""
    pattern = re.compile(r"^\s*(?:APP_)?VERSION\s*=", re.MULTILINE)
    offenders = []
    for path in APP_DIR.glob("*.py"):
        if path.name.startswith(("_", "test_")) or path.name == "conftest.py":
            continue
        if path.name != "config.py" and pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert not offenders, "第二份版本号定义：%s" % offenders


@pytest.mark.skipif(LEGAL_ROOT is None, reason="不在源码树里")
def test_exe_属性页的版权行同口径(tmp_path):
    """属性页（LegalCopyright）由构建脚本生成，口径同样只能来自 config。"""
    stage = pytest.importorskip("_protect_stage")
    name = stage.write_version_file(tmp_path, config.APP_VERSION, "9.9.9+TEST",
                                    stage.config_literals(stage.CONFIG_KEYS))
    text = (tmp_path / name).read_text(encoding="utf-8")
    assert config.APP_COPYRIGHT in text
    assert config.APP_NAME in text
    # build_id 进 FileVersion/ProductVersion；数字版本号则是 (5, 1, 0, 0) 这种元组，
    # 所以不能拿 "5.1.0" 去子串里找。
    assert "9.9.9+TEST" in text

