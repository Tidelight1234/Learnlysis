# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""框架设计项目的独立本地存储；不含 API key，也不进入 archive.json。"""

import json
import logging
import os

from config import data_dir, AI_FRAMEWORK_PROJECTS_FILE
from framework_schema import clean_project


def _disk_writes_blocked():
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None and app.platformName() == "offscreen":
            return os.environ.get("LEARNLYSIS_AUTOSAVE") != "1"
    except Exception:
        pass
    return False


class FrameworkProjectStore:
    def __init__(self, path=None):
        self.path = path or AI_FRAMEWORK_PROJECTS_FILE
        self._projects = None

    def projects(self):
        if self._projects is None:
            self._projects = self._read()
        return [clean_project(value) for value in self._projects]

    def _read(self):
        try:
            with open(self.path, encoding="utf-8") as handle:
                raw = json.load(handle)
        except FileNotFoundError:
            return []
        except Exception as exc:
            logging.warning("AI 框架项目读取失败，按空处理：%s", exc)
            return []
        values = raw.get("projects") if isinstance(raw, dict) else raw
        return [clean_project(value) for value in (values or []) if isinstance(value, dict)]

    def get(self, project_id):
        for value in self.projects():
            if value["project_id"] == str(project_id):
                return value
        return None

    def upsert(self, project):
        value = clean_project(project)
        current = self.projects()
        replaced = False
        for index, item in enumerate(current):
            if item["project_id"] == value["project_id"]:
                current[index] = value
                replaced = True
                break
        if not replaced:
            current.append(value)
        self._projects = current
        return self.save()

    def remove(self, project_id):
        current = self.projects()
        kept = [value for value in current if value["project_id"] != str(project_id)]
        if len(kept) == len(current):
            return False
        self._projects = kept
        return self.save()

    def save(self):
        if _disk_writes_blocked():
            return True
        try:
            os.makedirs(data_dir, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump({"schema_version": 1, "projects": self._projects or []}, handle,
                          ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
            return True
        except Exception as exc:
            logging.warning("AI 框架项目保存失败：%s", exc)
            return False


store = FrameworkProjectStore()
