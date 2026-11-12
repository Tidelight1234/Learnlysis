# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""构建水印：每个发布构建的身份证，源码库里这份永远停在 dev。

打包脚本（build_exe.ps1 / CI 的 staging 步骤）会在**送进 PyInstaller 的那份副本**里
重写本文件的三个常量，于是：

- 关于弹窗显示「构建标识 / 构建时间」，用户截图反馈时天然带着版本；
- 每一次分析导出的 model_card / config 里都落一个 ``build_id``；
- 外流的副本可以按 build_id 定位到是哪一次构建、哪个时间点发出去的。

刻意**不绑硬件、不做激活**（理由见 AGENTS.md 与 documents/ 的防护决策记录）：
这只是溯源用的水印，对普通用户零摩擦。dev 值全为空串时，界面上相应行直接不显示。
"""

#: 构建标识：形如 "5.1.0+20260915-a1b2c3d"，源码库里的 dev 构建是空串
BUILD_ID = ""

#: 构建时间（UTC ISO8601，打包机上生成），dev 为空串
BUILD_TIME = ""

#: 参与构建的 git commit 短号，dev 为空串
GIT_COMMIT = ""

#: 加固档位记录，形如 "docstrings=stripped;compiled=analysis_stats_sequence,..."，dev 为空串
HARDENING = ""


def is_release():
    """是否是一个打过水印的发布构建（dev / 直接从源码跑返回 False）。"""
    return bool(BUILD_ID)


def audit_dict():
    """导出侧（model_card / config / audit）统一取这一份。

    为什么要进导出而不只进界面：外流的那份副本里最难改的就是产物里的元数据，
    拿 build_id 反查「哪一次构建、哪个时间点发出去」必须能在用户手上的文件里查到。
    dev 下返回空串而不是删键，保证下游解析方的 schema 稳定。
    """
    return {"build_id": BUILD_ID, "build_time": BUILD_TIME,
            "git_commit": GIT_COMMIT, "hardening": HARDENING}


def hardening_items():
    """HARDENING 串（``k=v;k2=v2``）→ dict；dev 下为空。供关于页展示。"""
    out = {}
    for chunk in (HARDENING or "").split(";"):
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            out[key.strip()] = value.strip()
    return out
