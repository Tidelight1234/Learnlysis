# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""IND / GRP 分析结果落盘：复用 statviz 的四件套，再补模型卡与 R 校验表。

为什么复用而不是重写：表 / 图 / config / audit 的文件名模板与 analysis_id 追溯关系
已经在「统计及可视化」里定死了，两套工作台各长一套导出会让同一次编码出现两种产物
命名。这里只做三件事：

1. ``ConfigView`` —— 把 ``AnalysisContext``（一批组）投影成 statviz 导出层期待的
   「一个数据集」形状，input / 列映射 / 有效格计数全部按组求并集与求和；
2. ``model_card.json`` —— FR-09/FR-10 要求的可复盘凭据：方法版本、参数 JSON、随机
   种子、词表指纹、切分策略、独立分析单元数；
3. ``r_crosscheck.csv`` —— 阶段 5 的 R 参考实现差异表，Python 与 R 逐格对照。

三个产物里都带一份 ``build``（build_id / 构建时间 / commit / 加固档位，见
``build_stamp.py``），而且**不靠** ``extra`` 传：extra 是调用方可用的字段，
拿它当水印的传送带，下一次谁改参数面板就会顺手把水印盖掉。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
from datetime import datetime

import pandas as pd

import build_stamp
from config import APP_NAME, APP_VERSION
from statistic_visualization_export import (AUDIT_KIND, CONFIG_KIND, FIGURE_KIND,
                                            TABLE_KIND, export_audit, export_config,
                                            export_figure, export_tables,
                                            export_tables_xlsx, file_name,
                                            jsonable, write_json)

MODEL_CARD_KIND = "model_card"
CROSSCHECK_KIND = "r_crosscheck"


# ------------------------------------------------------------------ 上下文投影

class ConfigView:
    """把一批组投影成「一个数据集」：导出层只认属性，不认 AnalysisContext。

    ``long_df`` 在这里只为算「有效格 / 缺失格」的总量而并置，**不参与任何统计**
    —— 统计层的铁律（跨组不得拼接序列）在导出侧不适用，但也不能反过来被误用。
    """

    def __init__(self, ctx):
        self.ctx = ctx
        bundles = list(getattr(ctx, "bundles", []) or [])
        self.bundles = bundles
        datasets = [bundle.dataset for bundle in bundles
                    if getattr(bundle, "dataset", None) is not None]
        self._datasets = datasets
        self.source = "；".join(bundle.source_file or bundle.group_id for bundle in bundles)
        head = datasets[0] if datasets else None
        self.time_info = getattr(head, "time_info", None)
        self.time_col = getattr(head, "time_col", "Time")
        self.bin_mode = "原始时点（个体/小组模式不分箱）"
        self.bin_value = None
        self.column_map = {}
        for dataset in datasets:
            self.column_map.update(getattr(dataset, "column_map", {}) or {})
        self.qa = list(ctx.quality()) if ctx is not None else []
        frames = [dataset.long_df for dataset in datasets
                  if getattr(dataset, "long_df", None) is not None and len(dataset.long_df)]
        self.long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def describe(self):
        return {
            "mode": getattr(self.ctx, "mode", ""),
            "n_groups": len(self.bundles),
            "group_ids": [bundle.group_id for bundle in self.bundles],
            "n_units": getattr(self.ctx, "n_students", 0),
            "sources": [bundle.source_file or "session" for bundle in self.bundles],
        }


# ------------------------------------------------------------------ 模型卡

def codebook_version(codebook):
    """词表指纹：维度名 + 编码表的稳定散列，词表一改就能在导出里看出来。"""
    text = "\n".join("%s\t%s" % (name, ",".join(map(str, codes)))
                     for name, codes in (codebook or []))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] if text else ""


def method_version(spec):
    """方法版本 = 应用版本 + 算法族模块名（族内实现改了，版本号跟着应用版本走）。

    族名优先取 ``spec.family``（``all_methods()`` 汇总时盖上去的），只有手造的
    spec 才回落到 ``inspect.getmodule``：核心模块被 Cython 编成 .pyd 后拿不到
    源文件，getmodule 会返回 None，回落到“静默写空”而不是报错——审计里最坑的
    就是这种看得到列但值全空的字段。

    ``source`` 记该族当前是源码还是编译产物：同一份导出要能区分强化版与源码版。
    """
    family = getattr(spec, "family", "") or ""
    module = sys.modules.get(family) if family else None
    if module is None:
        func = getattr(spec, "compute", None)
        module = inspect.getmodule(func) if func is not None else None
        family = getattr(module, "__name__", "") or ""
    origin = str(getattr(module, "__file__", "") or "")
    if origin.lower().endswith((".pyd", ".so", ".dll")):
        source = "compiled"
    elif origin:
        source = "source"
    else:
        source = "unknown"
    return {"app": APP_VERSION, "family": family, "family_version": APP_VERSION,
            "source": source,
            "engine": getattr(spec, "engine", "python"),
            "r_reference": getattr(spec, "r_reference", "")}


def random_seed(params, result):
    meta = getattr(result, "meta", None) or {}
    for source in (params, meta):
        for key in ("seed", "random_seed"):
            value = source.get(key) if hasattr(source, "get") else None
            if value is not None and value != "":
                return int(value)
    return None


def build_model_card(result, ctx, spec=None, params=None, extra=None):
    """模型卡：任何一次分析（尤其是有监督/深度学习）都要能凭它复盘。"""
    params = dict(params or {})
    meta = getattr(result, "meta", None) or {}
    unit = getattr(spec, "unit", "student")
    card = {
        "app": {"name": APP_NAME, "version": APP_VERSION},
        "build": jsonable(build_stamp.audit_dict()),
        "generated_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "analysis_id": getattr(result, "analysis_id", ""),
        "method_id": getattr(result, "method_id", ""),
        "method_name": getattr(result, "method_name", ""),
        "mode": getattr(ctx, "mode", ""),
        "method_version": jsonable(method_version(spec)),
        "parameter_json": jsonable({key: value for key, value in params.items()
                                    if key != "codebook"}),
        "random_seed": random_seed(params, result),
        "codebook_version": codebook_version(getattr(ctx, "codebook", None)),
        "codebook_dimensions": len(getattr(ctx, "codebook", None) or []),
        "split": jsonable(meta.get("split", {"strategy": "不适用（非预测类方法）"})),
        "unit_of_independence": unit,
        "independent_units": int(ctx.n_students if unit == "student" else ctx.n_groups)
                             if ctx is not None else 0,
        "dataset": ConfigView(ctx).describe() if ctx is not None else {},
        "formula_note": jsonable(getattr(spec, "formula_note", "") or meta.get("formula_note", "")),
        "blocked": bool(getattr(result, "blocked", False)),
    }
    if extra:
        card.update(jsonable(extra))
    return jsonable(card)


# ------------------------------------------------------------------ R 交叉校验

def crosscheck_frame(result):
    """从结果里取 R 校验差异表：优先 tables 里的 ``r_crosscheck``，其次 meta 里的行列表。"""
    tables = getattr(result, "tables", None) or {}
    frame = tables.get("r_crosscheck")
    if isinstance(frame, pd.DataFrame) and len(frame):
        return frame
    rows = (getattr(result, "meta", None) or {}).get("r_crosscheck")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def has_crosscheck(result):
    return bool(len(crosscheck_frame(result)))


def export_crosscheck(result, path):
    frame = crosscheck_frame(result)
    if not len(frame):
        return None
    if os.path.splitext(path)[1].lower() in (".xlsx", ".xlsm"):
        frame.to_excel(path, sheet_name="r_crosscheck", index=False)
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ------------------------------------------------------------------ 四件套适配

def export_model_card(result, ctx, path, spec=None, params=None, extra=None):
    return write_json(build_model_card(result, ctx, spec, params, extra), path)


def export_config_ctx(result, ctx, path, spec=None, sheet=None, extra=None):
    return export_config(result, ConfigView(ctx), path, spec=spec, sheet=sheet, extra=extra)


def export_audit_ctx(result, ctx, path, extra=None):
    return export_audit(result, ConfigView(ctx), path, extra=extra)


def export_all(result, ctx, directory, spec=None, params=None, figures=None,
               sheet=None, dpi=300, formats=("png", "svg", "pdf")):
    """一键导出：表 / 每张图三种格式 / config / audit / 模型卡 /（有的话）R 校验表。

    图可能有多张（``render`` 返回 ``{图名: Figure}``），文件名沿用四件套同一模板
    再加 ``_序号``，因此一次运行的表、图、config、audit、模型卡仍同前缀可追溯。
    """
    if not result.tables and result.blocked:
        raise ValueError("结果被质检阻断，没有可导出的统计表")
    os.makedirs(directory, exist_ok=True)
    written = []
    if result.tables:
        written.append(export_tables_xlsx(
            result, os.path.join(directory, file_name(TABLE_KIND, result, "xlsx"))))
    for index, (title, fig) in enumerate((figures or {}).items(), start=1):
        # 共用同一模板，只多一个图序号：图可能不止一张，不加后缀会互相覆盖
        stem = os.path.splitext(file_name(FIGURE_KIND, result, "png"))[0]
        for fmt in formats:
            written.append(export_figure(
                fig, os.path.join(directory, "%s_%d.%s" % (stem, index, fmt)),
                dpi=dpi, format=fmt))
    written.append(export_config_ctx(
        result, ctx, os.path.join(directory, file_name(CONFIG_KIND, result, "json")),
        spec=spec, sheet=sheet))
    written.append(export_audit_ctx(
        result, ctx, os.path.join(directory, file_name(AUDIT_KIND, result, "json"))))
    written.append(export_model_card(
        result, ctx, os.path.join(directory, file_name(MODEL_CARD_KIND, result, "json")),
        spec=spec, params=params))
    cross = export_crosscheck(
        result, os.path.join(directory, file_name(CROSSCHECK_KIND, result, "csv")))
    if cross:
        written.append(cross)
    return written


__all__ = ["ConfigView", "build_model_card", "export_model_card", "crosscheck_frame",
           "has_crosscheck", "export_crosscheck", "export_all", "export_tables",
           "export_figure", "export_config_ctx", "export_audit_ctx", "file_name",
           "jsonable", "write_json", "TABLE_KIND", "FIGURE_KIND", "CONFIG_KIND",
           "AUDIT_KIND", "MODEL_CARD_KIND", "CROSSCHECK_KIND"]
