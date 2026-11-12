r"""给「会随包分发的」源文件统一加两行版权头（幂等，可重复跑）。

为什么要有这一步：软著登记与侵权取证看的是**文件本身**能不能说清归属。
关于弹窗、exe 属性页、根目录 LICENSE 都是「外圈」证据，而拿到一份 .py 副本的人
如果看不到版权声明，就等于把「这是谁写的」这件事完全交给对方自觉。

三条口径必须一致，所以都从 config.py 取（见 ``config.APP_COPYRIGHT``）：
根目录 LICENSE 的版权行、关于弹窗的版权行、每个源文件的第一行注释。
这条一致性由 ``test_legal_layer.py`` 钉住。

只处理 `_protect_stage.py` 认定「会进 stage」的那些文件：``_*.py``（开发期脚本）、
``test_*.py``、``conftest.py`` 不进分发物，给它们加商业版权头只是噪声。

用法：
    python _set_source_header.py            # 缺的补上
    python _set_source_header.py --check    # 只报告，不改；有缺就退出码 1
"""

from __future__ import annotations

import sys
from pathlib import Path

import _protect_stage as ps
import config

#: 版权头正文：第一行是归属（与 LICENSE / 关于弹窗逐字相同），第二行指向 MIT 条款全文。
#: 刻意用 ``#`` 注释而不是 docstring —— docstring 会被构建加固剥掉，归属声明不该跟着没。
HEADER_LINES = ("# %s" % config.APP_COPYRIGHT,
                "# Licensed under the MIT License; see the repository root LICENSE file.")

APP_DIR = ps.APP_DIR


def shipped_sources(root: Path = APP_DIR):
    """会随包分发的 .py：判据直接复用 stage 的过滤规则，不在此处重画一份清单。"""
    for path in sorted(root.rglob("*.py")):
        if any(ps._excluded_dir(part) for part in path.relative_to(root).parts[:-1]):
            continue
        if ps._keep_file(path.name):
            yield path


def has_header(text: str) -> bool:
    """头部已存在且内容就是当前口径（口径变了也要认出来，好让 --check 报缺）。"""
    lines = text.splitlines()
    if len(lines) < len(HEADER_LINES):
        return False
    return [ln.rstrip() for ln in lines[:len(HEADER_LINES)]] == list(HEADER_LINES)


def apply_one(path: Path, check_only: bool) -> bool:
    """返回该文件**修改前**是否已合规（合规 = 无需改动）。"""
    text = path.read_text(encoding="utf-8")
    if has_header(text):
        return True
    if not check_only:
        path.write_text("\n".join(HEADER_LINES) + "\n" + text, encoding="utf-8")
    return False


def run(check_only: bool = False) -> int:
    paths = list(shipped_sources())
    # 先数一遍再改：apply_one 报的是「改之前的状态」，不然 apply 模式下永远报 0。
    todo = [p for p in paths if not has_header(p.read_text(encoding="utf-8"))]
    for path in todo:
        apply_one(path, check_only)
    tag = "check" if check_only else "apply"
    print("[source-header] %s: %d shipped .py, %d %s"
          % (tag, len(paths), len(todo),
             "missing header" if check_only else "headers added"))
    for path in todo:
        print("  %s %s" % ("MISSING" if check_only else "ADDED  ", path.name))
    return 1 if (check_only and todo) else 0


def main(argv=None) -> int:
    return run("--check" in (argv or sys.argv[1:]))


if __name__ == "__main__":
    sys.exit(main())
