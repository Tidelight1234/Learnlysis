r"""构建加固 staging：把 version\v5 复制成一份「可以发给别人」的副本，再交给 PyInstaller。

为什么要有这一步：PyInstaller 打的包可以被 pyinstxtractor 秒拆，拆出来的 ``.pyc``
在 Python 3.11 上又能被 pycdc / pylingual 还原成接近原样的源码。``#`` 注释在编译期就
被丢掉了，但三引号文档串作为 ``__doc__`` 会**完整保留**——本项目的 docstring
里写满了公式口径、设计理由与方法内部结构，等于在包里放了一份自带说明书的源码。

所以发布构建不在源码树上跑，而是先做三件事：

1. **剥 docstring**（AST 定点手术，不跑 `-OO`，不动源码）；
2. **写构建水印**（build_stamp 的四个常量，dev 那份永远是空串）；
3. **Cython 编译皇冠明珠**：算法族 + 契约 + 安全/校验模块编成 `.pyd`，
   并把同名 `.py` 从 stage 里删掉（留着 `.py`，import 会优先取源码，等于白编）。

产物清单落 `_ly_manifest.json`，供 build_exe.ps1 / CI 判定「这次到底加固了没有」。
本脚本自身与所有 `_*.py` / `test_*.py` / `data/` / `_testdata*/` 都不进 stage。

用法：
    python _protect_stage.py --out _stage                       # 全套（缺 MSVC 时降级）
    python _protect_stage.py --out _stage --require-compiled    # 编不动就当失败
    python _protect_stage.py --out _stage --verify -k "not r_bridge"   # 编完在 stage 里跑回归
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

#: 整目录排除：工作现场、测试夹具、构建中间产物都不属于「可分发副本」
EXCLUDE_DIRS = {
    # 构建环境不是运行期资产：build_macos.command 会把 .mac-build-venv 建在版本树里，
    # 少了这几项，copy_tree 就把整个解释器环境当「随包源码」抄进发布物（实测 3772 个 .py）。
    ".mac-build-venv", ".build-venv", ".venv", "venv",
    "data", "dist", "build", "__pycache__", ".pytest_cache", ".mypy_cache",
    "_testdata", "_testdata_real", "_diag_plots", "_diag_shots", "_tmp_rwork",
    "_stage", ".idea", ".vscode",
}
#: 单文件排除：开发期脚本（下划线开头）与测试；R 参考脚本另论——它是运行期资产，留着
EXCLUDE_FILES = re.compile(r"^(?:_.*|test_.*|conftest)\.py$|\.spec$|\.pyd$|\.pth$")

#: Cython 编译目标。分两类，任何一类缺席都要在 manifest 里写清楚原因：
#  - 算法族 + 数据契约：值钱的口径与公式（P1-5 原本点名的 5 个）
#  - 加固与自身校验：水印/签名校验/DPAPI 接线，留在 .py 里就是「翻一个 if 就绕过」
CORE_TARGETS = ("analysis_stats_sequence", "analysis_stats_group",
                "analysis_stats_network", "analysis_features", "interaction_model")
SECURITY_TARGETS = ("build_stamp", "secure_store", "integrity")
CythonTargets = CORE_TARGETS + SECURITY_TARGETS
#: 水印携带者必须**最后**编：先编完别的模块，才知道该往 HARDENING 里写谁。
WATERMARK_MODULE = "build_stamp"
#: 递给 PyInstaller `--version-file` 的文件名（也落在 stage 里，不进包体）。
VERSION_INFO_NAME = "_ly_version_info.txt"

#: 编译指令：binding=True 让函数行为贴近 CPython（关键字参数 / __name__ / __module__）。
#: **注意** Cython 3.x 没有 `docstrings` 这个 compiler directive——它是
#: ``Cython.Compiler.Options.docstrings`` 这个模块级全局（对应命令行 ``-D``），
#: 塞进 directives 会当场 ``ValueError: got unknown compiler directive``。
#: 所以剥文档串在下面的 setup 模板里单独设，别写回这份字典。
CYTHON_DIRECTIVES = {
    "binding": True,
    "embedsignature": False,
    "emit_code_comments": False,
    "warn.unused": False,
    "language_level": 3,
}

_SETUP_TEMPLATE = r'''# -*- coding: utf-8 -*-
# 由 _protect_stage.py 生成的一次性 build 脚本（跑完即删，不进 stage 分发物）。
from setuptools import setup
from Cython.Build import cythonize
from Cython.Compiler import Options as _cython_options

# 文档串与源码位置一律不进 C 源：这一步之后 .pyd 里翻不出 docstring，
# 也就翻不出「这个公式为什么这么算」的设计理由。
_cython_options.docstrings = False              # == cython -D / --no-docstrings
_cython_options.embed_pos_in_docstring = False
_cython_options.annotate = False                 # 不产 .html 标注（里面全是源码）

DIRECTIVES = {directives!r}
setup(
    name="learnlysis_protected",
    ext_modules=cythonize({targets!r}, compiler_directives=DIRECTIVES, quiet=True),
)
'''


# ------------------------------------------------------------------ 1. 复制副本

def _keep_file(name: str) -> bool:
    if name.lower().endswith((".pyc", ".pyo", ".log", ".tmp")):
        return False
    return not EXCLUDE_FILES.search(name)


def _excluded_dir(name: str) -> bool:
    return name in EXCLUDE_DIRS or name.startswith("_diag") or name.startswith("_testdata")


def copy_tree(src: Path, dst: Path) -> int:
    """源码树 → stage，只带运行期需要的东西。返回复制的文件数。

    dst 必须在遍历时显式挡掉：目录是先 mkdir 再 walk 的，不挡的话输出目录就在
    源码树里，os.walk 懒求值会把它当成新目录一路复制下去（自嵌套，无限套娃）。
    """
    dst_resolved = dst.resolve()
    copied = 0
    for root, dirs, files in os.walk(src):
        here = Path(root)
        if here == dst_resolved or dst_resolved in here.parents:
            dirs[:] = []
            continue
        dirs[:] = sorted(d for d in dirs
                         if not _excluded_dir(d) and (here / d).resolve() != dst_resolved)
        target = dst / here.relative_to(src)
        target.mkdir(parents=True, exist_ok=True)
        for name in sorted(files):
            if not _keep_file(name):
                continue
            shutil.copy2(str(here / name), str(target / name))
            copied += 1
    return copied


# ------------------------------------------------------------------ 2. 剥 docstring

def _docstring_nodes(tree):
    """所有「语句本身就是字符串」的 docstring 节点，连同所在节点的 body 长度。"""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            out.append((first, len(body)))
    return out


def strip_one(path: Path) -> int:
    """剥掉一个文件里的 docstring，返回成功剥掉的条数。

    刻意不用 `-OO`：那是「编译时才剥」，`.py` 源文件照发；这里要的是**副本里就没有**。
    也不是正则删三引号——本项目里有 f-string、嵌套引号、`__doc__` 参与拼接的写法，
    只有 AST 能准确说清「这一串是 docstring」。
    """
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return 0                                     # 读不懂的文件一律原样保留
    nodes = _docstring_nodes(tree)
    if not nodes:
        return 0
    lines = src.splitlines(keepends=True)
    stripped = 0
    # 自下而上改，前面的行号才不会因为删了后面的行而错位
    for node, body_len in sorted(nodes, key=lambda item: -item[0].lineno):
        start, end = node.lineno - 1, node.end_lineno - 1
        col, end_col = node.col_offset, node.end_col_offset
        prefix = lines[start][:col]
        suffix = lines[end][end_col:]
        # 前后都必须只剩空白：`x = ("doc")` 这类同行表达式、或 docstring 后面还跟着
        # 代码的行，一律放过。宁可少剥一条，也不能产出改不回来的源文件。
        if prefix.strip() and not prefix.strip().startswith("#"):
            continue
        if suffix.strip():
            continue
        if body_len == 1:
            # docstring 是该作用域唯一的语句：换成 pass，否则语法直接塌掉
            lines[start:end + 1] = ["%spass\n" % prefix]
        else:
            lines[start:end + 1] = []
        stripped += 1
    if stripped:
        new_text = "".join(lines)
        # 三道保险：剪坏了语法就当场回退。宁可少剥一条，也不能交出一份编译不过的副本
        # （第二道是文件级 compile，第三道是 CI 里的 compileall）。
        try:
            compile(new_text, str(path), "exec")
        except SyntaxError:
            return 0
        path.write_text(new_text, encoding="utf-8")
    return stripped


def _count_docstrings(path: Path) -> int:
    """文件里还剩几条 docstring（抛不出 AST 就返回 -1，让调用方另记一笔）。"""
    try:
        return len(_docstring_nodes(ast.parse(path.read_text(encoding="utf-8"))))
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return -1


def strip_docstrings(root: Path) -> dict:
    """剥完顺手数一遍残留：`docstrings_left` 是「这一档防护真做了」的唯一硬证据，
    也是 build_exe.ps1 / CI 的 gate 条件（不为 0 就是剥漏了）。"""
    files = changed = left = unreadable = 0
    for path in sorted(root.rglob("*.py")):
        files += 1
        if strip_one(path):
            changed += 1
        remain = _count_docstrings(path)
        if remain < 0:
            unreadable += 1
        else:
            left += remain
    return {"py_files": files, "files_stripped": changed,
            "docstrings_left": left, "ast_unreadable": unreadable}


# ------------------------------------------------------------------ 3. 写水印

def stamp(root: Path, values: dict) -> None:
    """重写 stage 里 build_stamp.py 的常量（只认 `NAME = "..."` 这一行形态）。"""
    target = root / "build_stamp.py"
    text = target.read_text(encoding="utf-8")
    for name, value in values.items():
        pattern = re.compile(r'^%s = (?:u|r)?"[^"]*"$' % name, re.MULTILINE)
        literal = '%s = "%s"' % (name, value.replace("\\", "\\\\").replace('"', "'"))
        if not pattern.search(text):
            raise RuntimeError("build_stamp.py 里找不到常量 %s，水印写不进去" % name)
        text = pattern.sub(lambda _m: literal, text, count=1)
    target.write_text(text, encoding="utf-8")


# ------------------------------------------------------------------ 4. Cython 编译

def _find_msvc() -> str:
    """cl.exe 在不在 PATH。返回一句可展示的状态，不抛异常。"""
    if os.environ.get("VCINSTALLDIR"):
        return "vcvars=%s" % os.environ["VCINSTALLDIR"].rstrip("\\")
    found = shutil.which("cl.exe") or shutil.which("cl")
    return "cl=%s" % found if found else "cl-not-found"


#: vcvars64.bat 的常见位置。写硬路径并不丢人：vswhere 装在 C 盘，而安装路径
#: 完全可能在 D 盘（本机就是），两个都要试。
_VCVARS_GLOBS = (
    r"C:\Program Files*\Microsoft Visual Studio\2*\*\VC\Auxiliary\Build\vcvars64.bat",
    r"D:\Program Files*\Microsoft Visual Studio\2*\*\VC\Auxiliary\Build\vcvars64.bat",
    r"C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    r"D:\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
)
_VSWHERE = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
#: 只导入编译器需要的几个变量：把 vcvars 的全部环境回灌进当前进程，
#: 会连带改掉 PYTHONPATH / TMP 一类东西，把后面的 PyInstaller 带阴沟。
_MSVC_VARS = ("PATH", "INCLUDE", "LIB", "LIBPATH",
              "VCINSTALLDIR", "VSINSTALLDIR", "WINDOWSSDKDIR", "Platform")


def find_vcvars() -> str:
    if os.path.isfile(_VSWHERE):
        try:
            probe = subprocess.run(
                [_VSWHERE, "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            base = (probe.stdout or "").strip()
            cand = Path(base) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat" if base else None
            if cand and cand.is_file():
                return str(cand)
        except OSError:
            pass
    for pattern in _VCVARS_GLOBS:
        for hit in sorted(glob.glob(pattern)):
            if os.path.isfile(hit):
                return hit
    return ""


def ensure_msvc_env(root: Path = APP_DIR) -> str:
    """cl.exe 不在 PATH 时，自己调 vcvars64.bat 把编译器环境拉进来。

    返回一个可写进 manifest 的状态串。拉不到不抛异常：编译目标自动降级为「不编」，
    剩下的加固（剥 docstring + 水印）照常交付，并诚实记在 compile_note 里。
    """
    if shutil.which("cl.exe"):
        return "cl-already-on-path"
    bat = find_vcvars()
    if not bat:
        return "vcvars-not-found"
    # 必须经临时 .bat 转发：直接把 `call "D:\Program Files (x86)\..." && set` 交给
    # cmd /c，路径里的括号会把命令行解析弄碎，结果拿到一纸空环境（实测 0 vars）。
    shim = root / "_ly_vcvars_env.bat"
    shim.write_text("@echo off\r\ncall \"%s\" >nul 2>&1\r\nset\r\n" % bat, encoding="ascii")
    try:
        # errors="replace" 不是客套：vcvars 的 `set` 会回灌用户目录，非 ASCII 路径
        # 一多，按 utf-8 硬解就当场 UnicodeDecodeError（下面编译那趟同病同源）。
        probe = subprocess.run(["cmd", "/c", shim.name], cwd=str(root),
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        return "vcvars-spawn-failed:%s" % type(exc).__name__
    finally:
        shim.unlink(missing_ok=True)
    applied = 0
    for line in (probe.stdout or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.upper() in _MSVC_VARS:
            os.environ[key] = value
            applied += 1
    if not applied:
        return "vcvars-empty-env:%s" % bat
    return "cl=%s via %s (%d vars)" % (shutil.which("cl.exe") or "still-missing",
                                       bat, applied)


def _extension_files(root: Path, name: str) -> list:
    """该模块已编出来的扩展模块文件（含 CPython tag）。

    产物叫 ``analysis_features.cp311-win_amd64.pyd`` 而不是 ``analysis_features.pyd``：
    只拼 `name + ".pyd"` 会把一次完全成功的编译误判成「没产出」，既不删 .py
    （等于白编：源码优先），又把失败原因报给 gate（实测踩过）。
    """
    hits = []
    for pattern in (name + ".pyd", name + ".*.pyd",
                    name + ".so", name + ".*.so", name + ".*.abi3.so"):
        hits.extend(sorted(root.glob(pattern)))
    return hits


def compile_modules(root: Path, targets) -> tuple:
    """把 targets 编成 .pyd/.so 并删掉同名 .py。返回 (成功模块, 失败原因)。"""
    present = [name for name in targets if (root / (name + ".py")).is_file()]
    if not present:
        return [], "stage 里没有可编译的目标"
    msvc = ensure_msvc_env(root)
    if "still-missing" in msvc or msvc.startswith(("vcvars-not-found", "vcvars-spawn",
                                                   "vcvars-empty-env")):
        return [], "no-compiler: %s" % msvc
    setup_py = root / "_ly_cython_setup.py"
    setup_py.write_text(_SETUP_TEMPLATE.format(directives=CYTHON_DIRECTIVES,
                                              targets=[name + ".py" for name in present]),
                        encoding="utf-8")
    # MSVC/ninja 的输出是**本机 ANSI 代码页**（中文 Windows = cp936），而本脚本常以
    # `-X utf8` 跑，text=True 的默认解码就变成 utf-8：编译器一写中文警告，
    # communicate() 就在「编译已经成功」之后抛 UnicodeDecodeError，
    # 把刚编出来的 .pyd 和删 .py 的收尾一起打断（实测踩过）。解码策略必须是
    # 「按 utf-8 解，解不动就替换」，绝不因为日志读不懂而放弃这次编译。
    proc = subprocess.run([sys.executable, setup_py.name, "build_ext", "--inplace"],
                          cwd=str(root), env=os.environ.copy(),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace")
    compiled = []
    failed = {}
    for name in present:
        exts = _extension_files(root, name)
        source = root / (name + ".py")
        if exts:
            # 留着 .py 就白编了：同目录下源码的 import 优先级高于扩展模块
            if source.exists():
                source.unlink()
            compiled.append(name)
        else:
            failed[name] = "no-extension-produced"
        c_file = root / (name + ".c")
        if c_file.exists():
            c_file.unlink()
    setup_py.unlink(missing_ok=True)
    build_dir = root / "build"
    if build_dir.is_dir():
        shutil.rmtree(build_dir, ignore_errors=True)
    reason = "" if not failed else json.dumps(failed, ensure_ascii=False)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-12:])
        reason = "rc=%d %s | %s" % (proc.returncode, reason, tail)
    return compiled, ("" if compiled and not reason
                      else ("msvc=%s %s" % (msvc, reason)).strip())


# ------------------------------------------------------------------ 5. exe 属性页

def _numeric(version: str) -> tuple:
    """"5.1.0" → (5, 1, 0, 0)；VERSIONINFO 要四段整数，多一段是惯例而非意义。"""
    parts = []
    for chunk in re.split(r"[.\-+]", version or ""):
        if chunk.isdigit():
            parts.append(int(chunk))
    parts = (parts + [0, 0, 0, 0])[:4]
    return tuple(parts)


def write_version_file(root: Path, version: str, build_id: str, meta=None) -> str:
    """生成 PyInstaller 的 `--version-file`（exe 右键→属性→详细信息）。

    为什么住在本脚本而不是两个打包脚本里各写一份：本地与 CI 必须逐字一致，
    而这两处唯一共享的就是本脚本；而且 build_id 也只有这里算得出来。

    口径唯一：版本号、公司名、版权行全部从 config.py 抓（``config_literals``），
    这里**不另写一份 legal 文案**——属性页写着 A、关于弹窗写着 B、LICENSE 写着 C
    是三份互相矛盾的版权声明，法务上比不写更糟。meta 缺省时回落到通用值，
    方便单测拿假版本号调用。

    输出格式不是拍脑袋写的：已用 PyInstaller 自己的解析器验过——
    ``PyInstaller.utils.win32.versioninfo.load_version_info_from_text_file(path)``
    能把它读成 VSVersionInfo（test_protect_stage.py 里钉着这条，缺 PyInstaller 时跳过）。
    """
    meta = meta or {}
    company = meta.get("APP_NAME") or "Learnlysis"
    copyright = meta.get("APP_COPYRIGHT") or "Copyright \u00a9 2026 %s. Licensed under the MIT License." % company
    numbers = _numeric(version)
    strings = {
        "CompanyName": company,
        "FileDescription": meta.get("APP_TAGLINE") or "协作学习行为编码与分析工具",
        "FileVersion": build_id or version,
        "InternalName": company,
        "LegalCopyright": copyright,
        "OriginalFilename": "%s.exe" % company,
        "ProductName": company,
        "ProductVersion": build_id or version,
    }
    body = "\n".join("    StringStruct(%r, %r)," % (k, v) for k, v in sorted(strings.items()))
    text = ("# UTF-8\n"
            "# 由 _protect_stage.py 生成，跑完即进 stage；人手改它没意义。\n"
            "VSVersionInfo(\n"
            "  ffi=FixedFileInfo(filevers=%r, prodvers=%r,\n"
            "                    mask=0x3f, flags=0x0, OS=0x40004,\n"
            "                    fileType=0x1, subtype=0x0, date=(0, 0)),\n"
            "  kids=[\n"
            "    StringFileInfo([\n"
            "      StringTable('040904b0', [\n%s\n      ]),\n"
            "    ]),\n"
            "    VarFileInfo([VarStruct('Translation', [1033, 1200])]),\n"
            "  ],\n"
            ")\n" % (numbers, numbers, body))
    target = root / VERSION_INFO_NAME
    target.write_text(text, encoding="utf-8")
    return target.name


# ------------------------------------------------------------------ 6. 就地回归

def verify(root: Path, extra_args) -> int:
    """把测试搬进 stage 跑一遍：验证的是**加固后的那份**，不是源码树。

    这一步不能省：Cython 编译后 `locals()` 的语义、模块级 `if` 里的 class、
    以及 `__file__` 指向 `.pyd` 这几处都可能改变行为，只有拿编译产物重跑公式回归
    与 R 逐格对照才算数。
    """
    import shutil as _sh
    for name in ("conftest.py",):
        src = APP_DIR / name
        if src.exists():
            _sh.copy2(str(src), str(root / name))
    for src in sorted(APP_DIR.glob("test_*.py")):
        _sh.copy2(str(src), str(root / src.name))
    for fixture in ("_testdata", "_testdata_real"):
        src = APP_DIR / fixture
        if src.is_dir():
            _sh.copytree(str(src), str(root / fixture), dirs_exist_ok=True)
    cmd = [sys.executable, "-m", "pytest", "-q", "--no-header"] + list(extra_args or [])
    return subprocess.call(cmd, cwd=str(root), env=os.environ.copy())


# ------------------------------------------------------------------ 7. 主流程

def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(APP_DIR),
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, encoding="utf-8", errors="replace")
        return out.stdout.strip() or "nogit"
    except OSError:
        return "nogit"


def config_literals(names) -> dict:
    """按源码顺序求值 config.py 顶层的简单赋值，拿到版本号 / 应用名 / 版权行。

    为什么不 ``import config``：那是运行期模块（会 ``os.makedirs(data)``、按 frozen
    解析路径），构建脚本要在任何解释器环境下都能跑；而这里要的只是三个字符串常量。
    只认「右侧能在已收集的字面量命名空间里求值」的赋值（``"..." % APP_NAME`` 这种
    拼接算得过），其余一律跳过——真改成复杂写法时 test_protect_stage.py 会红，
    而不是让构建悄悄拿到一个空公司名。
    """
    wanted = set(names)
    src = (APP_DIR / "config.py").read_text(encoding="utf-8")
    ns = {}
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        try:
            expr = compile(ast.Expression(node.value), "config.py", "eval")
            value = eval(expr, {}, dict(ns))          # noqa: S307 - 只吃本仓库的配置文本
        except Exception:
            continue
        ns[target.id] = value
    for name in wanted:
        ns.setdefault(name, "")
    return ns


#: 构建侧需要的 config 字面量。加一项就要在下面 write_version_file / main 里用上。
CONFIG_KEYS = ("APP_NAME", "APP_TAGLINE", "APP_VERSION", "APP_COPYRIGHT")


def app_version() -> str:
    """版本号只认 config.APP_VERSION 这一个来源（不 import config，见上）。"""
    return str(config_literals(("APP_VERSION",))["APP_VERSION"]) or "0.0.0"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Learnlysis 构建加固 staging")
    # 默认落到版本树根而不是 src/ 里：src/ 只放码，_stage/dist/build 是产物。
    parser.add_argument("--out", default=os.path.join("..", "_stage"),
                        help="输出目录（默认版本树根的 _stage，不进 src/）")
    parser.add_argument("--no-strip", action="store_true", help="调试用：保留 docstring")
    parser.add_argument("--no-cython", action="store_true", help="只剥 docstring + 写水印")
    parser.add_argument("--require-compiled", action="store_true",
                        help="目标模块必须全部编成 .pyd，缺一个就当失败退出（发布构建该开）")
    parser.add_argument("--targets", default=",".join(CythonTargets),
                        help="逗号分隔的编译目标；默认 5 个算法族/契约 + 3 个加固模块")
    parser.add_argument("--verify", action="store_true", help="编完在 stage 里跑 pytest")
    parser.add_argument("--pytest-args", default="", help="传给 pytest 的附加参数")
    parser.add_argument("--build-id", default="", help="覆盖默认 build_id（CI 用）")
    args = parser.parse_args(argv)

    dst = Path(args.out)
    if not dst.is_absolute():
        dst = APP_DIR / dst
    if dst.exists():
        shutil.rmtree(str(dst))
    dst.mkdir(parents=True, exist_ok=True)

    committed = git_commit()
    meta = config_literals(CONFIG_KEYS)
    version = str(meta["APP_VERSION"]) or "0.0.0"
    build_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    build_id = args.build_id or "%s+%s-%s" % (version, time.strftime("%Y%m%d", time.gmtime()),
                                              committed)

    copied = copy_tree(APP_DIR, dst)
    strip_info = ({"py_files": 0, "files_stripped": 0, "skipped": "no-strip"}
                  if args.no_strip else strip_docstrings(dst))

    targets = [t.strip() for t in (args.targets or "").split(",") if t.strip()]
    # 编译分两趟：build_stamp 自己必须最后编。HARDENING 要写「哪些模块编成了 .pyd」，
    # 而这个事实只有编完才知道；先编其余目标，再写水印，最后才把带着水印的
    # build_stamp 本身编掉——否则水印只能写进一个已经不该存在的 .py。
    first_pass = [t for t in targets if t != WATERMARK_MODULE]
    last_pass = [WATERMARK_MODULE] if WATERMARK_MODULE in targets else []
    skip = args.no_cython or not targets

    def _hardening(compiled):
        # 用「stripped/kept」而不是「on/off」：off 到底是指关了剥离还是把文档串剥没了，
        # 事后看 manifest 的人猜不到。
        return "docstrings=%s;compiled=%s" % (
            "kept" if args.no_strip else "stripped", ",".join(compiled) or "none")

    def _write_stamp(hardening):
        stamp(dst, {"BUILD_ID": build_id, "BUILD_TIME": build_time,
                    "GIT_COMMIT": committed, "HARDENING": hardening})

    compiled_a, note_a = ([], "skipped-by-flag") if skip else compile_modules(dst, first_pass)
    compiled_b, note_b = [], ""
    if last_pass and not skip:
        _write_stamp(_hardening(compiled_a + last_pass))
        compiled_b, note_b = compile_modules(dst, last_pass)
        if not compiled_b:
            # 第二趟没编动：build_stamp.py 还在，那句「我也被编掉了」得当场收回去。
            # 宁可水印少写一项，也不能让产物里留着一句吹出去圆不回来的声明。
            _write_stamp(_hardening(compiled_a))
    else:
        _write_stamp(_hardening(compiled_a))

    compiled = compiled_a + compiled_b
    fail_reason = "; ".join(x for x in (note_a, note_b) if x)
    ext_files = [p.name for name in compiled for p in _extension_files(dst, name)]
    version_info = write_version_file(dst, version, build_id, meta)

    manifest = {"build_id": build_id, "build_time": build_time, "git_commit": committed,
                "version": version, "app_name": meta["APP_NAME"],
                "hardening": _hardening(compiled),
                "compiler": _find_msvc(), "copied_files": copied,
                "python": sys.version.split()[0],
                "strip": strip_info, "compiled": compiled,
                "compiled_files": ext_files, "version_info": version_info,
                "compile_note": fail_reason or ("" if compiled else "nothing-compiled")}
    (dst / "_ly_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[protect-stage] %s" % json.dumps(manifest, ensure_ascii=False), flush=True)
    print("[protect-stage] stage = %s" % dst, flush=True)

    code = 0
    if args.verify:
        code = verify(dst, args.pytest_args.split() if args.pytest_args else [])
        print("[protect-stage] pytest in stage exit=%d" % code, flush=True)
    missing = [t for t in targets if t not in compiled]
    if args.require_compiled and missing:
        print("[protect-stage] FAIL: 要求目标全部编成扩展模块，但 %s 没产出（%s）"
              % (missing, fail_reason), file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
