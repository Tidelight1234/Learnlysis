# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""用户运行时状态：文件对话框的记忆目录、最近打开清单、自动保存开关。

与 theme.py 的 ui_settings.json 分开存：那份每次写整文件（只装 theme 键），
两边混用会互相覆盖。本模块只装「用户操作留下的现场线索」，丢了我不影响任何数据。

分桶（bucket）= 一个取数入口一份记忆：视频、统计及可视化、个体/小组工作台、
编码框架、导出目录各记各的，互不串味。
"""

import os
import json
import time
import logging

from config import data_dir, USER_STATE_FILE

# ---------- 分桶名：调用方只写常量名，避免各处拼错字串各记一份 ----------
BUCKET_VIDEO = "video"
BUCKET_STATVIZ = "statviz"
BUCKET_INDIVIDUAL = "individual"
BUCKET_GROUP = "group"
BUCKET_FRAMEWORK = "framework"
BUCKET_EXPORT = "export"

#: 进「最近打开」菜单的桶；导出类只记目录，不列最近文件
RECENT_BUCKETS = (BUCKET_VIDEO, BUCKET_STATVIZ, BUCKET_INDIVIDUAL,
                  BUCKET_GROUP, BUCKET_FRAMEWORK)
RECENT_LABELS = {
    BUCKET_VIDEO: "视频",
    BUCKET_STATVIZ: "统计数据",
    BUCKET_INDIVIDUAL: "个体分析数据",
    BUCKET_GROUP: "小组分析数据",
    BUCKET_FRAMEWORK: "编码框架",
    BUCKET_EXPORT: "导出",
}

#: 每个桶最多记几条；菜单里按桶分组展示，再多也点不过来
MAX_RECENT_PER_BUCKET = 5


def default_dir():
    """没有记忆时的落点：用户主目录（比上次进程的工作目录稳定）。"""
    return os.path.expanduser("~")


def _is_dir(path):
    return bool(path) and os.path.isdir(path)


class UserState:
    """data/user_state.json 的内存镜像；写盘一律走原子替换。"""

    def __init__(self):
        self._payload = None

    # ---------------------------------------------------------------- 读写
    def payload(self):
        if self._payload is None:
            self._payload = self._read()
        return self._payload

    def _read(self):
        data = {"dirs": {}, "recent": [], "autosave": True, "language": "zh-CN",
                "coding_ai_visible": True, "coding_font_size": "normal"}
        try:
            with open(USER_STATE_FILE, encoding="utf-8") as handle:
                stored = json.load(handle)
        except FileNotFoundError:
            return data
        except Exception as exc:                       # noqa: BLE001 坏文件按没存过处理
            logging.warning("用户状态读取失败，按空白处理：%s", exc)
            return data
        if isinstance(stored, dict):
            data["dirs"].update(stored.get("dirs") or {})
            for record in stored.get("recent") or []:
                if isinstance(record, dict) and record.get("path"):
                    data["recent"].append({"bucket": str(record.get("bucket", "")),
                                           "path": str(record["path"]),
                                           "at": record.get("at", 0)})
            data["autosave"] = bool(stored.get("autosave", True))
            data["coding_ai_visible"] = bool(stored.get("coding_ai_visible", True))
            data["coding_dimension_mode"] = stored.get("coding_dimension_mode", "standard")
            data["coding_layout"] = stored.get("coding_layout")
            size = stored.get("coding_font_size", "normal")
            data["coding_font_size"] = self._valid_coding_size(size)
            language = str(stored.get("language") or "zh-CN")
            data["language"] = language if language in ("zh-CN", "en-US") else "zh-CN"
        return data

    def save(self):
        """原子写：先写 .tmp 再替换，进程被杀也不会留下半份 JSON。"""
        try:
            os.makedirs(data_dir, exist_ok=True)
            tmp = USER_STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self.payload(), handle, ensure_ascii=False, indent=2)
            os.replace(tmp, USER_STATE_FILE)
        except Exception as exc:                       # noqa: BLE001 状态丢了不该打扰用户
            logging.warning("用户状态保存失败：%s", exc)
            return False
        return True

    # ---------------------------------------------------------------- 目录
    def dir_for(self, bucket):
        """该桶上次用过的目录；失效（被删/挪盘）时退回用户主目录。"""
        path = self.payload()["dirs"].get(bucket) or ""
        return path if _is_dir(path) else default_dir()

    def remember_dir(self, bucket, path):
        """记住一次目录：传文件路径取其所在目录，传目录直接记。"""
        if not path:
            return
        folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
        if not _is_dir(folder):
            return
        dirs = self.payload()["dirs"]
        if dirs.get(bucket) == folder:
            return
        dirs[bucket] = folder
        self.save()

    # ---------------------------------------------------------------- 最近
    def recent(self, bucket=None, existing_only=True):
        """最近打开清单，按桶分组（桶序固定、桶内新→旧）；默认剔掉已失效的文件。"""
        records = self.payload()["recent"]
        if existing_only:
            records = [item for item in records if os.path.exists(item["path"])]
        picked = [item for item in records
                  if bucket is None or item["bucket"] == bucket]
        order = {name: index for index, name in enumerate(RECENT_BUCKETS)}
        return sorted(picked, key=lambda item: (order.get(item["bucket"], len(order)),
                                                -float(item.get("at") or 0)))

    def remember_open(self, bucket, paths):
        """登记一次「打开」：同桶同文件只留最新一条，每桶限量。"""
        paths = [paths] if isinstance(paths, str) else list(paths or [])
        paths = [os.path.abspath(path) for path in paths if path]
        if not paths:
            return
        records = self.payload()["recent"]
        stamp = time.time()
        for path in reversed(paths):                   # 反着插：选多个文件时第一个排在最前
            records = [item for item in records
                       if not (item["bucket"] == bucket and item["path"] == path)]
            records.insert(0, {"bucket": bucket, "path": path, "at": stamp})
        self.remember_dirs_from_paths(bucket, paths)   # 顺带把所在目录记进该桶
        self.trim()
        self.save()

    def remember_dirs_from_paths(self, bucket, paths):
        """多选时以最后一个文件的目录为准（资源管理器记住的就是那一片）。"""
        folder = os.path.dirname(os.path.abspath(paths[-1]))
        dirs = self.payload()["dirs"]
        if _is_dir(folder) and dirs.get(bucket) != folder:
            dirs[bucket] = folder

    def trim(self):
        """每个桶只留前 N 条，防止清单无限膨胀。"""
        records = self.payload()["recent"]
        kept, counts = [], {}
        for item in records:                           # records 已按新→旧排列
            bucket = item["bucket"]
            counts[bucket] = counts.get(bucket, 0)
            if counts[bucket] >= MAX_RECENT_PER_BUCKET:
                continue
            counts[bucket] += 1
            kept.append(item)
        self.payload()["recent"] = kept

    def forget(self, path):
        """从最近清单里去掉一条（文件被删掉后点它只会报错，不如不列）。"""
        path = os.path.abspath(path)
        records = self.payload()["recent"]
        kept = [item for item in records if item["path"] != path]
        if len(kept) != len(records):
            self.payload()["recent"] = kept
            self.save()

    def clear_recent(self):
        self.payload()["recent"] = []
        self.save()

    # ---------------------------------------------------------------- 开关
    def autosave_enabled(self):
        return bool(self.payload().get("autosave", True))

    def set_autosave_enabled(self, value):
        self.payload()["autosave"] = bool(value)
        self.save()

    def language(self):
        value = self.payload().get("language", "zh-CN")
        return value if value in ("zh-CN", "en-US") else "zh-CN"

    def set_language(self, value):
        value = value if value in ("zh-CN", "en-US") else "zh-CN"
        self.payload()["language"] = value
        self.save()

    def coding_ai_visible(self):
        return bool(self.payload().get("coding_ai_visible", True))

    def set_coding_ai_visible(self, value):
        self.payload()["coding_ai_visible"] = bool(value)
        self.save()

    def coding_dimension_mode(self):
        mode = self.payload().get("coding_dimension_mode", "standard")
        return mode if mode in ("standard", "compact") else "standard"

    def set_coding_dimension_mode(self, mode):
        self.payload()["coding_dimension_mode"] = (
            "compact" if mode == "compact" else "standard")
        self.save()

    def coding_layout(self):
        stored = self.payload().get('coding_layout') or {}
        order = stored.get('order') if isinstance(stored, dict) else None
        valid = {'dimensions', 'controls', 'ai', 'table'}
        if not isinstance(order, list) or len(order) != 4 or set(order) != valid:
            order = ['dimensions', 'table', 'controls', 'ai']
        span = stored.get('span') if isinstance(stored, dict) else None
        side = stored.get('span_side') if isinstance(stored, dict) else None
        return {'order': list(order), 'span': span if span in valid - {'ai'} else 'table',
                'span_side': side if side in ('start', 'end') else 'end'}

    def set_coding_layout(self, order, span, span_side='end'):
        self.payload()['coding_layout'] = {'order': list(order), 'span': span,
                                           'span_side': span_side}
        self.save()

    def coding_font_size(self):
        value = self.payload().get("coding_font_size", "normal")
        return self._valid_coding_size(value)

    def set_coding_font_size(self, value):
        self.payload()["coding_font_size"] = self._valid_coding_size(value)
        self.save()

    @staticmethod
    def _valid_coding_size(value):
        if value in ("normal", "medium", "small"):
            return value
        if isinstance(value, dict) and type(value.get("custom")) is int and 10 <= value["custom"] <= 20:
            return {"custom": value["custom"]}
        return "normal"


state = UserState()


# ------------------------------------------------------------------ 对话框包装
# 所有 QFileDialog 走这里：起始目录取自记忆，选完把目录（打开还带最近清单）记回去。
# 解释器没装 Qt 的脚本（如 _crosscheck_r.py）也能 import 本模块，故 Qt 延迟到函数内。

def open_file_names(parent, bucket, title, name_filter=""):
    """多选文件对话框；取消返回 []。"""
    from PySide6.QtWidgets import QFileDialog
    paths, _ = QFileDialog.getOpenFileNames(parent, title, state.dir_for(bucket), name_filter)
    paths = [path for path in paths if path]
    if paths:
        state.remember_open(bucket, paths)
    return paths


def open_file_name(parent, bucket, title, name_filter=""):
    """单选文件对话框；取消返回 ""。"""
    from PySide6.QtWidgets import QFileDialog
    path, _ = QFileDialog.getOpenFileName(parent, title, state.dir_for(bucket), name_filter)
    if path:
        state.remember_open(bucket, [path])
    return path


def save_file_name(parent, bucket, title, suggested="", name_filter=""):
    """保存对话框：起点 = 记忆目录 + 建议文件名；只记目录，导出文件不进「最近打开」。"""
    from PySide6.QtWidgets import QFileDialog
    folder = state.dir_for(bucket)
    start = os.path.join(folder, os.path.basename(suggested)) if suggested else folder
    path, _ = QFileDialog.getSaveFileName(parent, title, start, name_filter)
    if path:
        state.remember_dir(bucket, path)
    return path


def pick_directory(parent, bucket, title):
    """目录选择对话框（一键导出用）。"""
    from PySide6.QtWidgets import QFileDialog
    directory = QFileDialog.getExistingDirectory(parent, title, state.dir_for(bucket))
    if directory:
        state.remember_dir(bucket, directory)
    return directory


def reveal_in_explorer(path):
    """在资源管理器里定位该文件；非 Windows 或失败时退回「什么都不做 + 记日志」。"""
    import sys
    import subprocess
    absolute = os.path.abspath(path)
    try:
        if sys.platform == "win32":
            # /select, 需要文件存在，否则只剩一个打开到父目录的资源管理器窗口
            target = absolute if os.path.exists(absolute) else os.path.dirname(absolute)
            subprocess.Popen(["explorer", "/select,", os.path.normpath(target)])
        else:
            folder = absolute if os.path.isdir(absolute) else os.path.dirname(absolute)
            subprocess.Popen(["xdg-open", folder])
        return True
    except Exception as exc:                            # noqa: BLE001 打不开只提示，不抛
        logging.warning("打开所在目录失败：%s", exc)
        return False
