# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""分析结果落盘：统计表（CSV / XLSX）、图形（PNG / SVG / PDF）、配置与质检 JSON。

四类产物共用同一个文件名模板 ``{kind}_{method_id}_{name_slug}_{analysis_id}.{ext}``，
因此一次运行的表、图、config、audit 可以互相追溯。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from statistic_visualization_data import merge_qa, qa_summary
from config import APP_VERSION, APP_NAME
import build_stamp

TABLE_KIND = "tables"
FIGURE_KIND = "figure"
CONFIG_KIND = "run_config"
AUDIT_KIND = "run_audit"

_ILLEGAL = re.compile(r'[\\/:*?"<>|\s]+')
# Excel sheet 名只禁这几个字符（另限 31 长），空格要留着，否则表名与 sheet 名对不上
_SHEET_ILLEGAL = re.compile(r'[\[\]:*?/\\]+')


def slug(text):
    """把方法名压成安全的文件名片段（保留中文，去掉非法字符）。"""
    cleaned = _ILLEGAL.sub("_", str(text).strip())
    return cleaned.strip("_")[:48] or "method"


def file_name(kind, result, ext):
    return "%s_%s_%s_%s.%s" % (kind, result.method_id, slug(result.method_name),
                               result.analysis_id, ext)


def jsonable(value):
    """递归转成 json.dump 能吃的类型（numpy / pandas / datetime / set 都要过一遍）。"""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, set):
        return [jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat(sep=" ")
    if isinstance(value, (pd.DataFrame,)):
        return jsonable(value.to_dict(orient="records"))
    return value


# ------------------------------------------------------------------ 统计表

def _safe_sheet_name(name, used):
    title = _SHEET_ILLEGAL.sub("_", str(name).strip())[:31] or "Sheet"
    candidate = title
    index = 2
    while candidate in used:
        suffix = "_%d" % index
        candidate = title[:31 - len(suffix)] + suffix
        index += 1
    used.add(candidate)
    return candidate


def export_tables_xlsx(result, path):
    """每张结果表一个 sheet；空表也保留，便于复核「为什么这张表没数据」。"""
    used = set()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in result.tables.items():
            sheet = _safe_sheet_name(name, used)
            frame.to_excel(writer, sheet_name=sheet, index=False)
    return path


def export_tables_csv(result, path):
    """单表直接导出；多表时按块写入并用 ``# 表：`` 行分隔。"""
    if len(result.tables) <= 1:
        frame = result.table()
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        blocks = []
        for name, frame in result.tables.items():
            blocks.append("# 表：%s（%d 行）" % (name, len(frame)))
            blocks.append(frame.to_csv(index=False).rstrip("\r\n"))
        handle.write("\n\n".join(blocks) + "\n")
    return path


def export_tables(result, path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return export_tables_csv(result, path)
    if ext in (".xlsx", ".xlsm"):
        return export_tables_xlsx(result, path)
    if ext == ".xls":
        return export_tables_xlsx(result, os.path.splitext(path)[0] + ".xlsx")
    raise ValueError("不支持的统计表格式：%s" % ext)


# ------------------------------------------------------------------ 图形

def export_figure(fig, path, dpi=None, format=None):
    from theme import tokens
    face = tokens()["card"]
    ext = (format or os.path.splitext(path)[1]).lower().lstrip(".")
    if ext == "svg":
        fig.savefig(path, format="svg", facecolor=face, bbox_inches="tight")
    elif ext == "pdf":
        fig.savefig(path, format="pdf", facecolor=face, bbox_inches="tight")
    else:
        fig.savefig(path, format="png", dpi=dpi or 300, facecolor=face,
                    bbox_inches="tight")
    return path


# ------------------------------------------------------------------ 配置与质检

def build_config(result, ds, spec=None, sheet=None, extra=None):
    """run_config：输入、列映射、时间口径、分箱边界、方法参数、过滤条件、缺失规则。"""
    meta = result.meta or {}
    sampling = dict(meta.get("sampling", {}))
    sampling.update({"describe": ds.time_info.describe() if ds else ""})
    config = {
        "app": {"name": APP_NAME, "version": APP_VERSION},
        # 构建水印：让用户手上的每一份产物都自带「哪次构建发出去的」坐标。
        # 单独一个 build 子对象而不是塞进顶层：extra 是调用方合并进顶层的，
        # 同名键会被覆盖，水印被顺手盖掉就是白做。
        "build": jsonable(build_stamp.audit_dict()),
        "generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "analysis_id": result.analysis_id,
        "method": {
            "id": result.method_id,
            "name": result.method_name,
            "spec_id": getattr(spec, "id", result.method_id),
            "group": getattr(spec, "group", ""),
            "question": getattr(spec, "question", ""),
            "descriptive_only": True,
        },
        "input": {
            "file": ds.source if ds else "",
            "sheet": sheet or "",
            "time_column": ds.time_col if ds else "",
            "column_mapping": {col: {"student_id": stu, "dimension": dim}
                               for col, (stu, dim) in (ds.column_map.items() if ds else {})},
        },
        "sampling": sampling,
        "time_bins": {
            "mode": ds.bin_mode if ds else "",
            "value": jsonable(ds.bin_value) if ds else None,
            "bounds": jsonable(meta.get("bin_bounds", [])),
        },
        "params": jsonable(meta.get("params", {})),
        "sort": meta.get("params", {}).get("sort") or meta.get("params", {}).get("sort_by") or "",
        "filter": jsonable(meta.get("filter", {})),
        "missing_rule": meta.get("missing_rule", ""),
        "dataset": jsonable(meta.get("dataset", {})),
    }
    if extra:
        config.update(jsonable(extra))
    return jsonable(config)


def build_audit(result, ds, extra=None):
    """run_audit：全部质检条目、表行列数、valid N / missing、被剔除维度、耗时。

    qa 取“数据集级 + 方法级”合并后的清单，与 UI 质检卡一字不差。
    """
    frames = {}
    for name, frame in result.tables.items():
        frames[name] = {"rows": int(len(frame)), "columns": [str(c) for c in frame.columns]}
    valid_n = missing_n = 0
    dropped = []
    if ds is not None and len(ds.long_df):
        valid_n = int((~ds.long_df["is_missing"]).sum())
        missing_n = int(ds.long_df["is_missing"].sum())
        dropped = [item.message for item in ds.qa if item.code == "dimension_all_empty"]
    items = merge_qa(getattr(ds, "qa", None), result.qa)
    errors, warns, infos = qa_summary(items)
    audit = {
        "app_version": APP_VERSION,
        "build": jsonable(build_stamp.audit_dict()),
        "analysis_id": result.analysis_id,
        "method_id": result.method_id,
        "method_name": result.method_name,
        "generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "blocked": bool(result.blocked),
        "blocked_reason": result.blocked_reason,
        "elapsed_ms": int(result.elapsed_ms),
        "counts": {"error": errors, "warn": warns, "info": infos},
        "qa": jsonable([{"level": item.level, "code": item.code, "message": item.message,
                         "blocking": bool(getattr(item, "blocking", False))}
                        for item in items]),
        "tables": frames,
        "observations": {
            "valid_cells": valid_n,
            "missing_cells": missing_n,
            "missing_percent": round(missing_n / float(valid_n + missing_n) * 100.0, 2)
                               if (valid_n + missing_n) else 0.0,
        },
        "dropped_dimensions": dropped,
        "notes": jsonable(list(result.notes)),
    }
    if extra:
        audit.update(jsonable(extra))
    return jsonable(audit)


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return path


def export_config(result, ds, path, spec=None, sheet=None, extra=None):
    return write_json(build_config(result, ds, spec, sheet, extra), path)


def export_audit(result, ds, path, extra=None):
    return write_json(build_audit(result, ds, extra), path)


# ------------------------------------------------------------------ 一键导出

def export_all(result, ds, directory, spec=None, fig=None, sheet=None, dpi=300,
               formats=("png", "svg", "pdf")):
    """把一次运行的四类产物写进同一目录，返回写出的文件路径列表。

    图形默认一次出 PNG / SVG / PDF 三份：位图用于汇报、矢量用于论文插图，
    四类产物共用同一个 analysis_id 前缀，便于事后追溯。
    """
    if not result.tables and result.blocked:
        raise ValueError("结果被质检阻断，没有可导出的统计表")
    os.makedirs(directory, exist_ok=True)
    written = []
    if result.tables:
        written.append(export_tables_xlsx(
            result, os.path.join(directory, file_name(TABLE_KIND, result, "xlsx"))))
    if fig is not None:
        for fmt in formats:
            written.append(export_figure(
                fig, os.path.join(directory, file_name(FIGURE_KIND, result, fmt)),
                dpi=dpi, format=fmt))
    written.append(export_config(
        result, ds, os.path.join(directory, file_name(CONFIG_KIND, result, "json")),
        spec=spec, sheet=sheet))
    written.append(export_audit(
        result, ds, os.path.join(directory, file_name(AUDIT_KIND, result, "json"))))
    return written
