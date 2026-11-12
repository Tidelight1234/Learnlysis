# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""R 参考层桥接：Python 算一遍，R 参考实现再算一遍，两边逐格对照。

立场与 `documents/traminer-method-crosscheck.md` 一致 —— **运行时不要求 R**：
R 或参考包缺席时，本模块只返回 `available=False` 的探测结果与一句降级说明，
所有方法照常用 Python 实现出结果、照常导出。R 在场时，工作台第三张卡给出
逐格差异表（`r_crosscheck.csv` 一并落盘），用于佐证 Python 实现没有口径漂移。

三条契约：

    probe() -> {"available": bool, "version": str, "packages": {名: bool}, "found": [名]}
    cross_check(py, r, atol, rtol) -> [dict]              # 纯数值差异表
    reference_check(spec, result, ctx, params) -> (差异表, 说明)   # 载荷 + 跑 R + 对照

对照取的是 **Python 侧全精度中间量**（直接复用 stats 层的同一批函数），不是界面上
取整到两位小数的展示表 —— 否则两侧永远差一个舍入，校验就成了比谁的四舍五入更勤。

所有 R 子进程调用都放在 `QThread`（`ProbeWorker` / `ReferenceWorker`）里跑：探测要读
注册表、复算要等 Rscript，任何一项都不许冻住 UI 线程。

R 侧输入不依赖 jsonlite：载荷一律写成 CSV/TSV（`sequences.csv` / `grid.csv` /
`codebook.csv` / `dimension_labels.csv` / `state_order.csv` / `params.tsv`），同时另存
一份 `payload.json` 供装有 jsonlite 的参考脚本使用。输出统一是 `result.json`
（`r_refs/_common.R` 自带的极简序列化器）。
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime

import pandas as pd

from config import R_SETTINGS_FILE

#: R 参考脚本目录（随应用一起发布，非产品数据）
R_REFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "r_refs")
#: 单次 Rscript 调用的硬超时：TraMineR 大规模 seqdist 也够用，又不至于挂死 UI
TIMEOUT_SECONDS = 120
DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-4
#: 差异表最多列多少行，超了就汇总（一张 5000 行的对照表没人看）
MAX_ROWS = 300
MISSING_TEXT = "—"

#: 参考包：TraMineR/seqHMM/dtwclust/rENA 只在对应方法上用得到，缺包只降级不报错
REFERENCE_PACKAGES = ("jsonlite", "TraMineR", "seqHMM", "dtwclust", "rENA", "readxl",
                      "cluster", "seqinr")

#: Windows 上的常见安装位置；版本号变化时用 glob 兜底
_COMMON_RSCRIPTS = (
    r"D:\Program Files (x86)\R-4.5.0\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.5.0\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.4.0\bin\Rscript.exe",
    r"C:\Program Files\R\R-4.3.0\bin\Rscript.exe",
)
_R_GLOBS = (r"C:\Program Files\R\R-*\bin\Rscript.exe",
            r"D:\Program Files\R\R-*\bin\Rscript.exe",
            r"D:\Program Files (x86)\R-*\bin\Rscript.exe")

_PROBE_CACHE = {}


# ------------------------------------------------------------------ R 位置探测

def _read_settings():
    """`data/r_settings.json`：用户手工指定的 Rscript 路径 / 额外库目录。"""
    try:
        with open(R_SETTINGS_FILE, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:                       # noqa: BLE001  没配 / 配坏都按未配置处理
        return {}


def set_rscript(path):
    """把探测到的绝对路径写回 `r_settings.json`，下次启动免再翻注册表。"""
    settings = _read_settings()
    settings["Rscript"] = str(path or "")
    try:
        os.makedirs(os.path.dirname(R_SETTINGS_FILE), exist_ok=True)
        with open(R_SETTINGS_FILE, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, ensure_ascii=False, indent=2)
    except Exception:                       # noqa: BLE001  写不进去不影响本次运行
        return False
    reset_probe_cache()
    return True


def _which_rscript():
    from shutil import which
    for name in ("Rscript", "Rscript.exe", "Rscript.bat"):
        found = which(name)
        if found and os.path.exists(found):
            return found
    return ""


def _registry_r_home():
    """HKLM / HKCU 的 `SOFTWARE\\R-core\\R` 的 InstallPath（仅 Windows）。"""
    if not sys.platform.startswith("win"):
        return ""
    try:
        import winreg
    except ImportError:
        return ""
    for hive in (getattr(winreg, "HKEY_LOCAL_MACHINE", None),
                 getattr(winreg, "HKEY_CURRENT_USER", None)):
        if hive is None:
            continue
        try:
            with winreg.OpenKey(hive, r"SOFTWARE\R-core\R") as key:
                value, _type = winreg.QueryValueEx(key, "InstallPath")
            if value:
                return str(value)
        except OSError:
            continue                        # 这个 hive 里没装，换下一个
    return ""


def _glob_rscripts():
    import glob
    out = []
    for pattern in _R_GLOBS:
        out.extend(glob.glob(pattern))
    return sorted(out, reverse=True)        # 版本号写在路径里，倒序即优先新版


def find_rscript():
    """PATH → 注册表 → 常见目录 → `r_settings.json` 手工指定；找不到返回空串。"""
    found = _which_rscript()
    if found:
        return found
    home = _registry_r_home()
    if home:
        candidate = os.path.join(home, "bin", "Rscript.exe")
        if os.path.exists(candidate):
            return candidate
    for path in _COMMON_RSCRIPTS + tuple(_glob_rscripts()):
        if os.path.exists(path):
            return path
    manual = str(_read_settings().get("Rscript") or "").strip()
    if manual and os.path.exists(manual):
        return manual
    return ""


def _run(args, timeout=TIMEOUT_SECONDS, cwd=None, env=None):
    """统一的子进程入口：Rscript 的版本与报错都打在 stderr，两侧都得收回来。"""
    try:
        done = subprocess.run(args, capture_output=True, timeout=timeout, cwd=cwd,
                              env=env, text=True, encoding="utf-8", errors="replace")
        return int(done.returncode or 0), done.stdout or "", done.stderr or ""
    except FileNotFoundError as exc:
        return 127, "", "Rscript 不存在：%s" % exc
    except subprocess.TimeoutExpired:
        return 124, "", "Rscript 超时（> %d 秒）" % timeout
    except Exception as exc:                # noqa: BLE001  平台问题不该冒泡到 UI
        traceback.print_exc()
        return 1, "", "Rscript 调用失败：%s" % exc


def rscript_version(rscript=None):
    """`Rscript --version` 的版本串（信息打在 stderr，两边都要看）。"""
    rscript = rscript or find_rscript()
    if not rscript:
        return ""
    _code, out, err = _run([rscript, "--version"], timeout=20)
    lines = (out or err or "").strip().splitlines()
    for line in lines:
        match = re.search(r"version\s+(\d[\w.]*)", line)
        if match:
            return "R %s" % match.group(1)
    return lines[0][:60] if lines else ""


def user_library():
    """R 参考包的用户库：装在可写 data 目录下，不要求管理员权限。"""
    from config import data_dir
    return os.path.join(data_dir, "Rlibrary")


def _r_env():
    """R 子进程环境：把用户库顶到最前，`--vanilla` 才找得到一键安装的参考包。"""
    env = dict(os.environ)
    env["R_LIBS_USER"] = user_library()
    return env


def probe_packages(rscript=None):
    """{包名: 版本串或 None}：一次 Rscript 调用问完，缺包不报错。"""
    rscript = rscript or find_rscript()
    result = {name: None for name in REFERENCE_PACKAGES}
    if not rscript:
        return result
    snippet = ("for (p in c(%s)) { v <- tryCatch(as.character(packageVersion(p)), "
               "error=function(e) NA_character_); "
               "cat(p, if (is.na(v)) '-' else v, sep='\\t'); cat('\\n') }"
               % ", ".join('"%s"' % name for name in REFERENCE_PACKAGES))
    code, out, _err = _run([rscript, "--vanilla", "-e", snippet], timeout=60,
                           env=_r_env())
    if code != 0:
        return result
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[0] in result and parts[1] != "-":
            result[parts[0]] = parts[1].strip()
    return result


def probe(force=False):
    """R 参考层整体状态（进程内缓存）：工作台质检卡与 `ctx.has_r` 都读这里。"""
    if _PROBE_CACHE and not force:
        return dict(_PROBE_CACHE)
    rscript = find_rscript()
    packages = probe_packages(rscript) if rscript else {}
    installed = {name: bool(version) for name, version in packages.items()}
    status = {
        "available": bool(rscript),
        "rscript": rscript,
        "version": rscript_version(rscript) if rscript else "",
        "packages": installed,
        "package_versions": packages,
        "found": sorted(name for name, ok in installed.items() if ok),
        "missing": sorted(name for name, ok in installed.items() if not ok),
        "refs_dir": R_REFS_DIR,
        "user_library": user_library(),
        "checked_at": datetime.now().isoformat(sep=" ", timespec="seconds"),
    }
    _PROBE_CACHE.clear()
    _PROBE_CACHE.update(status)
    return dict(status)


def reset_probe_cache():
    _PROBE_CACHE.clear()
    return True


# ------------------------------------------------------------------ 载荷 → R 输入

def _flatten_params(params):
    """参数 → {key: 标量文本}：列表写成逗号串，R 侧 `ly_list()` 再拆回来。"""
    out = {}
    for key, value in dict(params or {}).items():
        if isinstance(value, bool):
            out[key] = "TRUE" if value else "FALSE"
        elif isinstance(value, (list, tuple, set)):
            out[key] = ",".join(str(item) for item in value)
        elif value is None:
            out[key] = ""
        elif isinstance(value, dict):
            out[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        else:
            out[key] = str(value)
    return out


def sequences_frame(sequences, on="raw"):
    """序列 → R 输入表。

    `codes` 是**去掉空编码后**的 `|` 串（与 `valid_codes` 同口径），`stream` 是
    `token_stream` 的结果（`on="compressed"` 时连续同码已合并）；`label` 与
    `build_distance_matrix` 的行列标签一致（``student_uid / 维度键``），这样两侧
    的距离对可以直接按键对齐。
    """
    import analysis_stats_sequence as seq

    rows = []
    for item in sequences:
        raw = seq.valid_codes(item)
        rows.append({
            "label": "%s / %s" % (item.uid, item.dimension),
            "group_id": item.group_id, "uid": item.uid, "student": item.student,
            "dimension": item.dimension,
            "codes": "|".join(raw),
            "stream": "|".join(seq.token_stream(item, on)),
            "length": len(raw),
        })
    return pd.DataFrame(rows)


_SAFE_NAME = re.compile(r"[^0-9A-Za-z_.\-]+")


def write_inputs(workdir, payload, params):
    """落 R 侧输入：CSV/TSV 是无依赖通路，payload.json 给装了 jsonlite 的脚本。

    payload 里**每一张**非空表都按 `<表名>.csv` 落盘（sequences / grid / codebook /
    dimension_labels / state_order / model_pi …），空表直接跳过 —— R 侧读不到该文件
    就按「本次参数不需要它」处理，脚本自己会回 `no-input`。
    """
    os.makedirs(workdir, exist_ok=True)
    flat = _flatten_params(params)
    for name, frame in (payload or {}).items():
        if not isinstance(frame, pd.DataFrame) or not len(frame):
            continue
        safe = _SAFE_NAME.sub("_", str(name)).strip("_") or "table"
        frame.to_csv(os.path.join(workdir, "%s.csv" % safe), index=False,
                     encoding="utf-8")
    with open(os.path.join(workdir, "params.tsv"), "w", encoding="utf-8", newline="") as handle:
        handle.write("key\tvalue\n")
        for key, value in sorted(flat.items()):
            text = str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")
            handle.write("%s\t%s\n" % (key, text))
    tables = {key: (value.to_dict("records") if isinstance(value, pd.DataFrame) else value)
              for key, value in (payload or {}).items()}
    with open(os.path.join(workdir, "payload.json"), "w", encoding="utf-8") as handle:
        json.dump({"params": flat, "tables": tables}, handle,
                  ensure_ascii=False, default=str)
    return workdir


def run_reference(script, payload, params=None, timeout=TIMEOUT_SECONDS, rscript=None):
    """跑一个 R 参考脚本：临时目录 + `Rscript --vanilla` + JSON 回传 + stderr 捕获。

    返回永远是字典、永远不抛：
        {"available", "ok", "status", "error", "stderr", "tables", "values",
         "engine", "elapsed_ms", ...}
    """
    empty = {"available": False, "ok": False, "status": "no-r", "error": "",
             "stderr": "", "tables": {}, "values": {}, "engine": {}, "elapsed_ms": 0}
    rscript = rscript or find_rscript()
    if not rscript:
        empty["error"] = ("未探测到 Rscript：可在 data/r_settings.json 里手工指定，"
                          "或把 R 的 bin 目录加入 PATH")
        return empty
    script_path = os.path.join(R_REFS_DIR, str(script))
    if not os.path.exists(script_path):
        empty.update({"available": True, "status": "no-script",
                      "error": "没有这个 R 参考脚本：%s" % script})
        return empty
    workdir = tempfile.mkdtemp(prefix="learnlysis_r_")
    write_inputs(workdir, payload or {}, params or {})
    env = _r_env()
    marker = time.perf_counter()
    code, out, err = _run([rscript, "--vanilla", script_path.replace("\\", "/"),
                           workdir.replace("\\", "/"), R_REFS_DIR.replace("\\", "/")],
                          timeout=timeout, cwd=workdir, env=env)
    elapsed = int((time.perf_counter() - marker) * 1000)
    result_path = os.path.join(workdir, "result.json")
    parsed = {}
    if os.path.exists(result_path):
        try:
            with open(result_path, encoding="utf-8") as handle:
                parsed = json.load(handle)
        except Exception as exc:            # noqa: BLE001  回传坏了也要报出来
            parsed = {"status": "bad-json", "error": str(exc)}
    if not isinstance(parsed, dict):
        parsed = {"status": "bad-json", "error": "result.json 不是对象"}
    tables = {}
    for name, records in (parsed.get("tables") or {}).items():
        if isinstance(records, list):
            try:
                tables[str(name)] = pd.DataFrame.from_records(records)
            except Exception:               # noqa: BLE001
                continue
        elif isinstance(records, dict):
            tables[str(name)] = pd.DataFrame(records)
    status = str(parsed.get("status") or ("ok" if code == 0 else "failed"))
    values = dict(parsed.get("values") or {})
    if "package" in parsed and parsed.get("package"):
        values.setdefault("package", parsed.get("package"))
    keep = bool(os.environ.get("LEARNLYSIS_R_KEEP_WORKDIR"))
    if not keep and status == "ok":
        # 一跑一个临时目录：算完就清，不然 %TEMP% 里会一路堆 CSV；
        # 失败时留着目录（连输入一起）才能事后查为什么对不上
        shutil.rmtree(workdir, ignore_errors=True)
        workdir = ""
    return {
        "available": True,
        "ok": status == "ok" and bool(tables or values),
        "status": status,
        "error": str(parsed.get("error") or ("" if code == 0 else (err or "")[-800:])),
        "stderr": (err or "")[-2000:], "stdout": (out or "")[-500:],
        "tables": tables, "values": values,
        "engine": dict(parsed.get("engine") or {}),
        "exit_code": code, "elapsed_ms": elapsed,
        "r_version": str(parsed.get("r_version") or ""),
        "workdir": workdir,
    }


# ------------------------------------------------------------------ 差异表

def _numeric_columns(frame, skip=()):
    out = []
    for column in frame.columns:
        if column in skip:
            continue
        if pd.to_numeric(frame[column], errors="coerce").notna().any():
            out.append(column)
    return out


def _clean(value):
    """两侧的数收敛成 float：`""` / `NaN` / `"-"` 记为 None（不参与判定，另计缺失）。"""
    if value is None:
        return None
    try:
        if isinstance(value, str):
            text = value.strip()
            if not text or text in ("-", "—", "nan", "None"):
                return None
            value = float(text)
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def _text_of(value):
    if value is None:
        return MISSING_TEXT
    return "%.10g" % value


def _row(label, py_value, r_value, atol, rtol):
    absolute = relative = None
    if py_value is None or r_value is None:
        verdict = "缺失"
    else:
        absolute = abs(py_value - r_value)
        relative = absolute / max(abs(py_value), abs(r_value), 1e-12)
        verdict = "一致" if (absolute <= atol or relative <= rtol) else "超差"
    return {"量": label, "Python": _text_of(py_value), "R": _text_of(r_value),
            "绝对差": None if absolute is None else round(absolute, 12),
            "相对差": None if relative is None else round(relative, 12),
            "判定": verdict}


def _as_frames(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, pd.DataFrame):
        return {"": value}
    return {}


def _key_columns(first, second, keys):
    """键列：优先调用方给的，其次两侧共有的「非数值列」（字符串状态名一类）。"""
    if keys:
        return [column for column in keys if column in first.columns
                and column in second.columns]
    numeric = set(_numeric_columns(first)) | set(_numeric_columns(second))
    return [column for column in first.columns
            if column in second.columns and column not in numeric]


def _row_label(record, keys):
    return "·".join(str(record.get(key, "")) for key in keys)


def cross_check(py, r, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL, keys=None, label=""):
    """两侧数值逐格对照 → 差异表（键见工作台 `CROSS_COLUMNS`）。

    ``py`` / ``r`` 可以是 DataFrame，也可以是 ``{表名: DataFrame}``；同名表才对照。
    ``keys`` 给元组（所有表共用）或 ``{表名: 元组}``。判定门槛是
    ``绝对差 ≤ atol`` **或** ``相对差 ≤ rtol``，与规格书 §9 的 1e-6 / 1e-4 双门槛一致。
    """
    left, right = _as_frames(py), _as_frames(r)
    rows = []
    for name, first in left.items():
        prefix = (("%s · " % label) if label else "") + (name or "")
        second = right.get(name)
        if second is None or not isinstance(first, pd.DataFrame) \
                or not isinstance(second, pd.DataFrame):
            rows.append({"量": prefix or "（表）", "Python": "有", "R": MISSING_TEXT,
                         "绝对差": None, "相对差": None, "判定": "缺失"})
            continue
        table_keys = keys.get(name) if isinstance(keys, dict) else keys
        shared = _key_columns(first, second, table_keys)
        common = [column for column in _numeric_columns(first, skip=shared)
                  if column in set(_numeric_columns(second, skip=shared))]
        if not common:
            continue
        left_records = first.to_dict("records")
        if shared:
            index = {}
            for record in left_records:
                index.setdefault(_row_label(record, shared), record)
            right_records = second.to_dict("records")
            for record in right_records:
                twin = index.get(_row_label(record, shared))
                tag = _row_label(record, shared)
                if twin is None:
                    rows.append({"量": "%s · %s" % (prefix, tag), "Python": MISSING_TEXT,
                                 "R": "仅 R 侧有", "绝对差": None, "相对差": None,
                                 "判定": "缺失"})
                    continue
                for column in common:
                    rows.append(_row("%s · %s · %s" % (prefix, tag, column),
                                     _clean(twin.get(column)), _clean(record.get(column)),
                                     atol, rtol))
            # Python 算了、R 没算的格也得报出来：只查单向会把「参照缺格」当成全部一致
            seen = {_row_label(record, shared) for record in right_records}
            for tag in index:
                if tag not in seen:
                    rows.append({"量": "%s · %s" % (prefix, tag), "Python": "仅 Python 侧有",
                                 "R": MISSING_TEXT, "绝对差": None, "相对差": None,
                                 "判定": "缺失"})
        else:                                # 无键可对齐（矩阵类）：按位置对照
            right_records = second.to_dict("records")
            for position in range(min(len(left_records), len(right_records))):
                for column in common:
                    rows.append(_row("%s · 第%d行 · %s" % (prefix, position + 1, column),
                                     _clean(left_records[position].get(column)),
                                     _clean(right_records[position].get(column)),
                                     atol, rtol))
            if len(left_records) != len(right_records):
                # 行数不一样长：多出来的那截无从对照，当成缺失报出来而不是默默截掉
                more = "Python" if len(left_records) > len(right_records) else "R"
                extra = abs(len(left_records) - len(right_records))
                rows.append({"量": "%s · 行数" % (prefix or "（表）"),
                             "Python": "%d 行" % len(left_records),
                             "R": "%d 行" % len(right_records),
                             "绝对差": None, "相对差": None,
                             "判定": "缺失"})
                rows[-1]["量"] += "（仅 %s 侧多 %d 行）" % (more, extra)
    return rows


def summarize_rows(rows):
    """差异表 → (一致, 超差, 缺失, 最大绝对差)：提示文案与断言都读这四个数。"""
    agreed = sum(1 for row in rows if row.get("判定") == "一致")
    failed = sum(1 for row in rows if row.get("判定") == "超差")
    missing = sum(1 for row in rows if row.get("判定") == "缺失")
    worst = max([value for value in (row.get("绝对差") for row in rows)
                 if isinstance(value, (int, float))], default=None)
    return agreed, failed, missing, worst


def trim_rows(rows, limit=MAX_ROWS):
    """超长差异表折叠成「前 N 行 + 一条汇总」：判定结论一行都不许丢。"""
    if len(rows) <= limit:
        return list(rows)
    head, rest = list(rows[:limit]), list(rows[limit:])
    agreed, failed, missing, worst = summarize_rows(rest)
    head.append({"量": "其余 %d 项（已折叠）" % len(rest),
                 "Python": "一致 %d / 超差 %d / 缺失 %d" % (agreed, failed, missing),
                 "R": "最大绝对差 %s" % (MISSING_TEXT if worst is None else worst),
                 "绝对差": worst, "相对差": None,
                 "判定": "超差" if failed else "一致"})
    return head


# ------------------------------------------------------------------ 各方法的对照方案

def _lsa_max_lag(params, aggregate="dimension"):
    """lag 上限与各方法自身一致：IND01 到 5，GRP05 到 3（不然两侧算的 lag 数不同）。"""
    if aggregate == "group+dimension":
        return max(1, min(int(params.get("max_lag") or 1), 3))
    return max(1, min(int(params.get("max_lag") or 3), 5))


def _lsa_cells(ctx, params, aggregate="dimension"):
    """IND01 / GRP05（成员合并口径）的全精度 O / E / Z。"""
    import analysis_stats_sequence as seq
    import analysis_stats_group as grp

    max_lag = _lsa_max_lag(params, aggregate)
    rows = []
    if aggregate == "group+dimension":       # GRP05：逐组 × 维度，成员序列相加
        for group_id in grp.wanted_groups(ctx, params):
            for dimension in grp.wanted_dimensions(ctx, params):
                sequences = ctx.sequences(group_id=group_id, dimension=dimension)
                states = seq.states_of(sequences)
                if len(states) < 2:
                    continue
                for lag in range(1, max_lag + 1):
                    _states, counts = seq.pooled_counts(sequences, states, lag=lag)
                    off, expected, residuals = seq.lsa_residuals(counts)
                    for i, from_state in enumerate(states):
                        for j, to_state in enumerate(states):
                            if i == j:
                                continue
                            rows.append({"group_id": group_id, "dimension": dimension,
                                         "lag": lag, "from": from_state, "to": to_state,
                                         "O": off[i, j], "E": expected[i, j],
                                         "Z": residuals[i, j]})
        return pd.DataFrame(rows)
    chosen = seq.pick_sequences(ctx, params)
    for dimension, group in seq.by_dimension(chosen).items():
        states = seq.states_of(group)
        if len(states) < 2:
            continue
        for lag in range(1, max_lag + 1):
            _states, counts = seq.pooled_counts(group, states, lag=lag)
            off, expected, residuals = seq.lsa_residuals(counts)
            for i, from_state in enumerate(states):
                for j, to_state in enumerate(states):
                    if i == j:
                        continue
                    rows.append({"dimension": dimension, "lag": lag, "from": from_state,
                                 "to": to_state, "O": off[i, j], "E": expected[i, j],
                                 "Z": residuals[i, j]})
    return pd.DataFrame(rows)


def _markov_cells(ctx, params):
    """IND02 的全精度 N / P / 平稳分布 / 转移熵率。"""
    import analysis_stats_sequence as seq
    import numpy as np

    include_self = bool(params.get("self_transition", True))
    counts_rows, probability_rows, stationary_rows, value_rows, values = \
        [], [], [], [], {}
    chosen = seq.pick_sequences(ctx, params)
    for dimension, group in seq.by_dimension(chosen).items():
        states = seq.states_of(group)
        if len(states) < 2:
            continue
        _states, counts = seq.pooled_counts(group, states, lag=1)
        if not include_self:
            counts = counts.copy()
            np.fill_diagonal(counts, 0.0)
        initial = np.array([sum(1 for item in group if seq.valid_codes(item)[:1] == [state])
                            for state in states], dtype=float)
        stats = seq.markov_statistics(counts, initial)
        for i, from_state in enumerate(states):
            for j, to_state in enumerate(states):
                counts_rows.append({"dimension": dimension, "from": from_state,
                                    "to": to_state, "n": float(counts[i, j])})
                probability_rows.append({"dimension": dimension, "from": from_state,
                                         "to": to_state,
                                         "p": float(stats["probabilities"][i, j])})
        for position, state in enumerate(states):
            stationary_rows.append({"dimension": dimension, "state": state,
                                    "pi": float(stats["stationary"][position])})
        values["entropy_rate_%s" % dimension] = float(stats["entropy_rate"])
        value_rows.append({"dimension": dimension,
                           "entropy_rate": float(stats["entropy_rate"]),
                           "initial_entropy": float(stats["initial_entropy"]),
                           "ratio": float(stats["ratio"]),
                           "transitions": float(np.asarray(counts, dtype=float).sum())})
    return {"markov_N": pd.DataFrame(counts_rows), "markov_P": pd.DataFrame(probability_rows),
            "markov_pi": pd.DataFrame(stationary_rows),
            "markov_values": pd.DataFrame(value_rows)}, values


#: R 参照能覆盖的距离方法：traminer_om.R（OM/HAM/DHD）、traminer_multichannel.R（MCO/MCM）、
#: dtwclust_ref.R（DTW/SDTW）。名单外的距离不参与逐格对照，免得拿错数互证。
R_DISTANCES = ("OM", "HAM", "DHD", "MCO", "MCM", "DTW", "SDTW")

#: DTW 是「两两一对 DP」且 R 侧是纯解释型循环，序列一多就要跑十几分钟：
#: 距离参照按标签排序取前 N 条，两侧同一份抽样，仍是逐格对照、只是覆盖面小一点。
REF_DISTANCE_LIMIT = 24


def _distance_sequences(ctx, flat):
    """距离参照的取序列：DP 类全量，DTW/SDTW 抽样前 `REF_DISTANCE_LIMIT` 条。"""
    import analysis_stats_sequence as seq

    chosen = seq.pick_sequences(ctx, flat)
    if str((flat or {}).get("method") or "").upper() in ("DTW", "SDTW"):
        chosen = sorted(chosen, key=lambda item: (item.uid, item.dimension))
        chosen = chosen[:REF_DISTANCE_LIMIT]
    return chosen


def _distance_cells(ctx, params, method=None):
    """IND05 / IND07 / IND06 的全精度成对距离（上三角，标签与 R 侧一致）。"""
    import analysis_stats_sequence as seq

    flat = dict(params or {})
    if method:
        flat["method"] = method
    if str(flat.get("method") or "OM").upper() not in R_DISTANCES:
        # R 参照只覆盖上面点名的距离；不在名单里（如尚未写参照的）宁可不出对照
        return pd.DataFrame()
    chosen = _distance_sequences(ctx, flat)
    labels, matrix, _note = seq.build_distance_matrix(chosen, flat, ctx=ctx)
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rows.append({"a": str(labels[i]), "b": str(labels[j]),
                         "d": float(matrix[i, j])})
    return pd.DataFrame(rows)


def _ngram_cells(ctx, params):
    """IND04 的全精度 n-gram：逐 n 累加，只取两侧能精确对齐的量。"""
    import analysis_stats_sequence as seq

    chosen = seq.pick_sequences(ctx, params)
    max_n = max(2, min(int(params.get("max_n") or 3), 6))
    min_sup = float(params.get("min_sup") or 0.1)
    on = "compressed" if params.get("stream") == "compressed" else "raw"
    frames = [frame for frame in
              (seq.ngram_table(chosen, n, min_sup, on) for n in range(2, max_n + 1))
              if len(frame)]
    if not frames:
        return pd.DataFrame()
    joined = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "n": joined["n"].astype(int),
        "gram": joined["gram"].astype(str),
        "occurrences": joined["occurrences"].astype(float),
        "docs": joined["序列数"].astype(float),
        "support": joined["序列数"].astype(float) / joined["总序列"].astype(float),
    })
    return out.sort_values(["support", "n", "gram"], ascending=[False, True, True]) \
               .reset_index(drop=True)


def _network_cells(ctx, params):
    """GRP09 / IND10 的边权：逐组 × 维度的滞后转移计数（不含自环），与 R 侧同径。"""
    import analysis_stats_group as grp

    lag = max(1, int(params.get("lag") or 1))
    min_edge = int(params.get("min_edge") or 0)
    rows = []
    for group_id in grp.wanted_groups(ctx, params):
        for dimension in grp.wanted_dimensions(ctx, params):
            sequences = ctx.sequences(group_id=group_id, dimension=dimension)
            if not sequences:
                continue
            states, counts = grp.pooled_counts(sequences, lag=lag)
            for i, left in enumerate(states):
                for j, right in enumerate(states):
                    value = float(counts[i, j])
                    if i != j and value > 0 and value >= min_edge:
                        rows.append({"group_id": group_id, "dimension": dimension,
                                     "source": left, "target": right, "weight": value})
    return pd.DataFrame(rows)


def _cooccurrence_cells(ctx, params):
    """GRP09 的 ENA 共现表：直接拿方法层的表，列名拉平成 R 侧的写法。"""
    import analysis_stats_group as grp

    frame = grp.pooled_cooccurrence(ctx, params)
    if not len(frame):
        return frame
    out = frame.rename(columns={"节点A": "node_a", "节点B": "node_b",
                                "共现次数": "weight"})
    return out[["group_id", "node_a", "node_b", "weight"]]


def _cooccurrence_member_cells(ctx, params):
    """IND10 的逐成员 ENA 共现表：成员短名作 `member`，与 R 侧 grid.csv 同口径。"""
    import analysis_stats_network as net

    frame = net.cooccurrence_frame(ctx, params)
    if not len(frame):
        return frame
    return frame[["group_id", "member", "node_a", "node_b", "weight"]]


def _hmm_cells(ctx, params, result):
    """IND08 的推断参照：拿本次拟合的参数逐序列重算 logLik / γ / Viterbi。

    EM 训练两侧不可能同径（初值与迭代路径都不同），所以只比推断，
    与 r_refs/seqhmm_fit.R 的约定一致：逐 (维度 → 一套模型)，发射阵按行（隐状态）
    归一后走缩放前向后向；position 从 0 计（与 R 的 `t - 1L` 同径）。
    多通道 MHMM 的发射是各通道之积，R 参照不覆盖 → 由 CHECK_PLANS 的 skip 挡住。
    """
    import analysis_stats_sequence as seq
    import numpy as np

    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    model = dict((getattr(result, "meta", None) or {}).get("hmm_model") or {}) \
        if result is not None else {}
    frames = {name: model.get(name) for name in ("pi", "A", "B")}
    if not all(isinstance(item, pd.DataFrame) and len(item) for item in frames.values()):
        return empty
    flat = dict(params or {})
    if str(flat.get("model") or "hmm").lower() != "hmm":
        return empty
    on = "compressed" if flat.get("compress", True) else "raw"
    chosen = seq.pick_sequences(ctx, flat)
    if not chosen:
        return empty
    loglik_rows, gamma_rows, viterbi_rows = [], [], []
    for dimension, group in seq.by_dimension(chosen).items():
        pi_table = frames["pi"][frames["pi"]["dimension"] == dimension]
        a_table = frames["A"][frames["A"]["dimension"] == dimension]
        b_frame = frames["B"]
        if "channel" in b_frame.columns:
            b_frame = b_frame[pd.to_numeric(b_frame["channel"], errors="coerce") == 0]
        b_table = b_frame[b_frame["dimension"] == dimension]
        if not len(pi_table) or not len(a_table) or not len(b_table):
            continue
        states = [str(item) for item in pi_table["state"]]
        symbols = list(dict.fromkeys(str(item) for item in b_table["symbol"]))
        k, v = len(states), len(symbols)
        if k < 2 or v < 1:
            continue
        pi = np.zeros(k, dtype=float)
        for position, state in enumerate(states):
            hit = pi_table[pi_table["state"].astype(str) == state]
            if len(hit):
                pi[position] = float(hit["pi"].iloc[0])
        if pi.sum() > 0:
            pi = pi / pi.sum()
        else:
            pi = np.full(k, 1.0 / k)
        A = np.zeros((k, k), dtype=float)
        index = {state: position for position, state in enumerate(states)}
        for _i, record in a_table.iterrows():
            left, right = index.get(str(record["from"])), index.get(str(record["to"]))
            if left is not None and right is not None:
                A[left, right] = float(record["p"])
        for row in range(k):
            total = float(A[row].sum())
            if total > 0:
                A[row] = A[row] / total
        B = np.zeros((k, v), dtype=float)
        column = {symbol: position for position, symbol in enumerate(symbols)}
        for _i, record in b_table.iterrows():
            left, right = index.get(str(record["state"])), column.get(str(record["symbol"]))
            if left is not None and right is not None:
                B[left, right] = float(record["p"])
        for row in range(k):
            total = float(B[row].sum())
            if total > 0:
                B[row] = B[row] / total
        for item in group:
            label = "%s / %s" % (item.uid, item.dimension)
            emit = seq.hmm_emission(seq.token_stream(item, on), symbols, B)
            out = seq.hmm_inference(emit, pi, A)
            if out is None:
                continue
            if out["degenerate"]:
                loglik_rows.append({"label": label, "loglik": float("-inf")})
                continue
            loglik_rows.append({"label": label, "loglik": float(out["loglik"])})
            for step in range(len(out["gamma"])):
                for position, state in enumerate(states):
                    gamma_rows.append({"label": label, "position": step,
                                       "state": state,
                                       "posterior": float(out["gamma"][step, position])})
                viterbi_rows.append({"label": label, "position": step,
                                     "state": states[int(out["path"][step])]})
    return (pd.DataFrame(loglik_rows), pd.DataFrame(gamma_rows),
            pd.DataFrame(viterbi_rows))


def _codebook_frame(ctx, dimensions):
    """codebook.csv：多通道 MD 要拿它算「一个码跨几个通道」。"""
    rows = []
    for dimension in dimensions:
        for code in ctx.codebook_of(dimension):
            rows.append({"dimension": dimension, "code": code})
    return pd.DataFrame(rows)


def _dimension_labels(ctx, dimensions):
    """key → 显示名：R 侧只认数据键，但共现节点名里写的是显示名，需要这张映射表。"""
    return pd.DataFrame([{"key": dimension, "label": ctx.label_of(dimension)}
                         for dimension in dimensions])


def _channel_grid(ctx, chosen, params):
    """多通道距离的对齐码轨（数据整形，不是被校验的算法）→ grid.csv。"""
    import analysis_stats_sequence as seq

    dimensions = sorted({item.dimension for item in chosen})
    students = {item.uid for item in chosen}
    tracks = seq.aligned_channels(ctx, students, dimensions)
    group_of = {item.uid: item.group_id for item in chosen}
    rows = []
    for uid in sorted(tracks):
        if uid not in students:
            continue
        for dimension in dimensions:
            for position, code in enumerate(tracks[uid].get(dimension, [])):
                rows.append({"group_id": group_of.get(uid, ""), "uid": uid,
                             "dimension": dimension, "position": position,
                             "code": code})
    return pd.DataFrame(rows), dimensions


def _member_grid(ctx, params):
    """ENA 共现口径的 member × position × dimension 码表 → grid.csv。

    直接复用 `analysis_stats_network.member_grid`：两侧同一份取数（含按学生筛选），
    才不会 Python 算了 12 个成员、R 只拿到 15 个。
    """
    import analysis_stats_network as net

    rows, dimensions = net.member_grid(ctx, params)
    out = [{"group_id": row["group_id"], "uid": row["member"],
            "dimension": row["dimension"], "position": row["position"],
            "code": row["code"]} for row in rows]
    return pd.DataFrame(out), dimensions


#: 方法号 → R 参考方案。未列入的方法在交叉校验卡上直接说明「未配置 R 参考」。
#: scripts = 按距离方法改跑的脚本（多通道走另一套 DP）。
CHECK_PLANS = {
    "IND01": {"script": "lsa_residual.R", "kind": "lsa", "aggregate": "dimension",
              "tables": {"lsa_cells": ("dimension", "lag", "from", "to")},
              "skip": {"param": "scope", "values": ("cross",),
                       "note": "本次只算跨维共现：R 参照只覆盖同维口径，本轮不做逐格对照。"},
              "note": "R 对照覆盖同维口径；跨维共现只在 Python 侧生成"},
    "GRP05": {"script": "lsa_residual.R", "kind": "lsa", "aggregate": "group+dimension",
              "tables": {"lsa_cells": ("group_id", "dimension", "lag", "from", "to")},
              "skip": {"param": "scope", "values": ("consensus",),
                       "note": "本次只算共识码：R 参照只覆盖「成员合并」口径，本轮不做逐格对照。"},
              "note": "R 对照覆盖「成员合并」口径；共识码序列只在 Python 侧生成"},
    "IND02": {"script": "markov_ref.R", "kind": "markov",
              "tables": {"markov_N": ("dimension", "from", "to"),
                         "markov_P": ("dimension", "from", "to"),
                         "markov_pi": ("dimension", "state"),
                         "markov_values": ("dimension",)}},
    "IND04": {"script": "traminer_seqefsub.R", "kind": "ngram",
              "tables": {"patterns": ("n", "gram")},
              "note": "R 侧 seqefsub（可不连续子序列）只作旁证，不参与逐格对照"},
    "IND05": {"script": "traminer_om.R", "kind": "distance", "tables":
              {"distance_matrix": ("a", "b")},
              "scripts": {"MCO": "traminer_multichannel.R",
                          "MCM": "traminer_multichannel.R"}},
    "IND07": {"script": "traminer_om.R", "kind": "distance",
              "tables": {"distance_matrix": ("a", "b")},
              "scripts": {"MCO": "traminer_multichannel.R",
                          "MCM": "traminer_multichannel.R",
                          "DTW": "dtwclust_ref.R", "SDTW": "dtwclust_ref.R"}},
    "IND06": {"script": "dtwclust_ref.R", "kind": "distance", "method": "DTW",
              "tables": {"distance_matrix": ("a", "b"),
                         "soft_distance_matrix": ("a", "b")},
              "note": "R 参照契约：one-hot 逐时点 + 欧氏局部代价 + 经典 DTW（不做长度归一化）"},
    "IND10": {"script": "rena_network.R", "kind": "network",
              "tables": {"cooccurrence_member": ("group_id", "member", "node_a", "node_b")},
              "note": "R 对照逐成员的 ENA 共现权；坐标/载荷是 SVD 产物，不比逐格"},
    "GRP09": {"script": "rena_network.R", "kind": "network",
              "tables": {"edges": ("group_id", "dimension", "source", "target"),
                         "cooccurrence": ("group_id", "node_a", "node_b")}},
    "IND08": {"script": "seqhmm_fit.R", "kind": "hmm",
              "tables": {"hmm_loglik": ("label",),
                         "hmm_gamma": ("label", "position", "state"),
                         "hmm_viterbi": ("label", "position")},
              "skip": {"param": "model", "values": ("mhmm",),
                       "note": "多通道 MHMM 的发射是各维之积：R 参照只覆盖单通道模型，"
                               "本轮不做逐格对照。"},
              "note": "只比推断（前向后向 / Viterbi / logLik）：EM 训练两侧不可能同径"},
}


def _group_sequences(ctx, flat):
    """逐 (组 × 维度) 取成员序列，**不做长度过滤**：GRP05 / GRP09 的取数口径。"""
    import analysis_stats_group as grp

    chosen = []
    for group_id in grp.wanted_groups(ctx, flat):
        for dimension in grp.wanted_dimensions(ctx, flat):
            chosen.extend(ctx.sequences(group_id=group_id, dimension=dimension))
    return chosen


def _payload_for(plan, ctx, params, result=None):
    """按方案组装 R 输入：序列（含标签与两种口径的码流）、状态顺序、时间栅格、参数。

    写进 `flat` 的那几个键就是 R 脚本的输入契约：`aggregate` / `max_lag` / `method` /
    `channel_order` / `dimension_order` 一律由桥接层**先按 Python 的夹取规则算好**再下发，
    否则两侧可能算不同数量的 lag / 不同顺序的通道。
    """
    import analysis_stats_sequence as seq

    kind = plan["kind"]
    flat = dict(params or {})
    payload = {}
    method = str(flat.get("method") or plan.get("method") or "").upper()
    if plan.get("method"):
        flat["method"] = method
    if kind == "hmm":
        chosen = seq.pick_sequences(ctx, flat)
        payload["sequences"] = sequences_frame(chosen,
                                               on="compressed" if flat.get("compress", True)
                                               else "raw")
        model = dict((getattr(result, "meta", None) or {}).get("hmm_model") or {}) \
            if result is not None else {}
        for name in ("pi", "A", "B"):
            frame = model.get(name)
            if isinstance(frame, pd.DataFrame) and len(frame):
                payload["model_%s" % name] = frame
        flat["kind"] = kind
        return payload, flat
    if kind in ("lsa", "markov", "distance", "ngram", "network"):
        on = "compressed" if (kind == "distance" and flat.get("compress")) else "raw"
        if kind == "ngram":
            on = flat.get("stream") or "raw"
        if kind == "network" or (kind == "lsa" and plan.get("aggregate") == "group+dimension"):
            chosen = _group_sequences(ctx, flat)
        elif kind == "distance":
            chosen = _distance_sequences(ctx, flat)
        else:
            chosen = seq.pick_sequences(ctx, flat)
        payload["sequences"] = sequences_frame(chosen, on=on)
        order = []
        for dimension in sorted({item.dimension for item in chosen}):
            for position, state in enumerate(seq.states_of(
                    [item for item in chosen if item.dimension == dimension])):
                order.append({"dimension": dimension, "state": state,
                              "position": position})
        payload["state_order"] = pd.DataFrame(order)
    if kind == "network":
        grid, dimensions = _member_grid(ctx, flat)
        if len(grid):
            payload["grid"] = grid
        payload["dimension_labels"] = _dimension_labels(ctx, dimensions)
        flat["dimension_order"] = list(dimensions)
    if kind == "distance" and method in ("MCO", "MCM"):
        grid, dimensions = _channel_grid(ctx, seq.pick_sequences(ctx, flat), flat)
        if len(grid):
            payload["grid"] = grid
        payload["codebook"] = _codebook_frame(ctx, dimensions)
        flat["channel_order"] = list(dimensions)
    flat["kind"] = kind
    if plan.get("aggregate"):
        flat["aggregate"] = plan["aggregate"]
        flat["max_lag"] = _lsa_max_lag(flat, plan["aggregate"])
    if kind == "ngram":
        flat["max_n"] = max(2, min(int(flat.get("max_n") or 3), 6))
    return payload, flat


def python_reference(spec, ctx, params, result=None):
    """Python 侧的全精度参照值 → ``{表名: DataFrame}``（只留方案点名的表）。"""
    plan = CHECK_PLANS.get(spec.id)
    if not plan:
        return {}
    kind = plan["kind"]
    flat = dict(params or {})
    if kind == "lsa":
        tables = {"lsa_cells": _lsa_cells(ctx, flat, plan.get("aggregate", "dimension"))}
    elif kind == "markov":
        tables = _markov_cells(ctx, flat)[0]
    elif kind == "distance":
        tables = {"distance_matrix": _distance_cells(ctx, flat, plan.get("method"))}
        if flat.get("soft_dtw"):
            tables["soft_distance_matrix"] = _distance_cells(ctx, flat, "SDTW")
    elif kind == "ngram":
        tables = {"patterns": _ngram_cells(ctx, flat)}
    elif kind == "network":
        tables = {"edges": _network_cells(ctx, flat),
                  "cooccurrence": _cooccurrence_cells(ctx, flat),
                  "cooccurrence_member": _cooccurrence_member_cells(ctx, flat)}
    elif kind == "hmm":
        loglik, gamma, viterbi = _hmm_cells(ctx, flat, result)
        tables = {"hmm_loglik": loglik, "hmm_gamma": gamma, "hmm_viterbi": viterbi}
    else:
        return {}
    wanted = plan.get("tables") or {}
    if wanted:
        tables = {name: frame for name, frame in tables.items() if name in wanted}
    return tables


def _labelize(frame, ctx, column="dimension"):
    """R 只认数据键（D1），Python 表里写的是显示名（D1 认知投入）：对照前拉平两侧。"""
    if not isinstance(frame, pd.DataFrame) or not len(frame) or column not in frame.columns:
        return frame
    out = frame.copy()
    out[column] = [ctx.label_of(value) for value in out[column]]
    return out


def _plan_script(plan, params):
    """选哪个 R 脚本：多通道距离（MCO/MCM）走另一套 DP。"""
    method = str((params or {}).get("method") or "").upper()
    scripts = plan.get("scripts") or {}
    return scripts.get(method) or plan.get("script") or ""


def _pending_note(plan, script):
    """Python 参照拿不出数时的实话，不说「已校验通过」。"""
    if plan["kind"] == "hmm":
        return ("本次没有可对照的 HMM 模型（没拟出来或选了多通道 MHMM），"
                "R 参照（r_refs/%s）本轮跳过。" % script)
    if plan["kind"] == "distance" and str(plan.get("method") or "").upper() == "DTW":
        return ("DTW 参照需要 ≥ 2 条有效序列（r_refs/%s）：本次取数为空，本轮跳过。"
                % script)
    return ("R 参照契约已钉在 r_refs/%s，但本次取数拿不出可对照的全精度量。" % script)


#: R 侧回传的降级状态 → 人话（都是“不影响计算与导出”的提示，不是错误）
STATUS_NOTES = {
    "no-r": "未探测到 Rscript：R 交叉校验降级为提示。",
    "no-script": "R 参考脚本缺席（安装包里没带上 r_refs？）：本轮不对照。",
    "no-input": "R 缺本次需要的输入表：%s",
    "empty": "R 复算没有可对照的格：%s（数据太薄或参数选空）",
    "unsupported": "R 参考实现不覆盖本参数组合：%s",
    "package-missing": "R 缺包 %s：可跑 r_refs/setup_r_reference.R 一键安装",
    "no-packages": "R 已就位但参考包未安装（缺 %s）",
}


def reference_check(spec, result, ctx, params, atol=DEFAULT_ATOL, rtol=DEFAULT_RTOL):
    """跑一次 R 复算并与 Python 全精度值对照 → ``(差异表, 说明)``。

    不抛异常：R 缺席 / 缺包 / 脚本失败都只回一句人话，工作台与导出一律不受影响。
    """
    if spec is None or ctx is None or not ctx.usable:
        return [], "尚未装载数据，无从对照。"
    plan = CHECK_PLANS.get(spec.id)
    declared = getattr(spec, "r_reference", "") or ""
    if not plan:
        return [], ("本方法没有配置 R 参考实现（注册表声明：%s）——以 Python 为准。"
                    % (declared or "无"))
    script = _plan_script(plan, params)
    skip = plan.get("skip") or {}
    if skip and str((params or {}).get(skip.get("param"))) in tuple(skip.get("values") or ()):
        return [], (skip.get("note") or "本参数组合不在 R 参照范围内。")
    py_tables = python_reference(spec, ctx, params, result)
    if not any(len(frame) for frame in py_tables.values()):
        if plan["kind"] == "hmm" or plan.get("method"):
            return [], _pending_note(plan, script)
        return [], "Python 侧没有可对照的全精度量（选中的序列/参数太薄）。"
    status = probe()
    if not status.get("available"):
        return [], ("未探测到 Rscript：R 交叉校验降级为提示，Python 实现照常为准（参照脚本 %s）。"
                    % script)
    try:
        payload, flat = _payload_for(plan, ctx, params, result)
        flat["lm_method"] = spec.id
        out = run_reference(script, payload, flat)
    except Exception as exc:                # noqa: BLE001  桥接层自身出错也不能影响分析
        traceback.print_exc()
        return [], "R 复算异常：%s" % exc
    if isinstance(result, object) and getattr(result, "meta", None) is not None:
        result.meta["r_script"] = script
        result.meta["r_status"] = out.get("status") or ""
        result.meta["r_version"] = status.get("version") or ""
        result.meta["r_engine"] = out.get("engine") or {}
    if not out.get("ok"):
        values = out.get("values") or {}
        missing = "、".join(status.get("missing") or [])
        code_status = out.get("status")
        detail = (out.get("error") or "")[:200]
        if code_status == "package-missing":
            detail = values.get("package") or missing or "未知"
        elif code_status == "no-packages":
            detail = missing or "未知"
        template = STATUS_NOTES.get(code_status)
        if template is None:
            message = "R 复算未成功（status=%s）：%s" % (code_status, detail)
        else:
            message = template % detail if "%s" in template else template
        return [], message
    r_tables = {name: frame for name, frame in (out.get("tables") or {}).items()}
    if not r_tables:
        return [], "R 未回传对照表：%s" % (out.get("error") or "（无 stderr）")
    keys = {name: tuple(columns) for name, columns in (plan.get("tables") or {}).items()}
    rows = cross_check({name: _labelize(frame, ctx) for name, frame in py_tables.items()},
                       {name: _labelize(frame, ctx) for name, frame in r_tables.items()},
                       atol=atol, rtol=rtol, keys=keys, label=spec.id)
    if not rows:
        return [], "两侧没有可比的对齐表（R 返回：%s）" % ("、".join(r_tables) or "空")
    agreed, failed, missing, worst = summarize_rows(rows)
    engine = out.get("engine") or {}
    engine_text = " · ".join("%s：%s" % (name, value) for name, value in engine.items()) \
        or "base R 手写实现"
    coverage = "全量"
    if plan["kind"] == "distance":
        import analysis_stats_sequence as seq

        total = len(seq.pick_sequences(ctx, flat))
        used = len(_distance_sequences(ctx, flat))
        if used < total:
            coverage = "抽样 %d/%d 条序列" % (used, total)
    note = "R %s ｜ %s ｜ %s：对照 %d 格，一致 %d、超差 %d、缺失 %d；最大绝对差 %s。%s" % (
        status.get("version") or "?", script + "（" + coverage + "）", engine_text,
        len(rows), agreed, failed, missing,
        MISSING_TEXT if worst is None else "%g" % worst,
        plan.get("note") or "")
    if isinstance(result, object) and getattr(result, "meta", None) is not None:
        result.meta["r_crosscheck"] = rows
    return trim_rows(rows), note


def install_script():
    return os.path.join(R_REFS_DIR, "setup_r_reference.R").replace("\\", "/")


def install_command(repo=""):
    """一键安装命令（给「复制命令」按钮）：库目录写成参数，装到 `data/Rlibrary`。"""
    rscript = probe().get("rscript") or find_rscript() or "Rscript"
    parts = ['"%s"' % rscript.replace("\\", "/"), "--vanilla", '"%s"' % install_script(),
             '"%s"' % user_library().replace("\\", "/")]
    if repo:
        parts.append('"%s"' % str(repo).strip())
    return " ".join(parts)


def install_references(repo="", timeout=1800, rscript=None):
    """直接执行安装脚本（同步，给 `InstallWorker` 调）：永远返回字典，不抛。

    装不上（离线 / 没有二进制）只把报告文本带回去：R 参考层是可选校验，
    Python 实现始终为准，不能因为装包失败就报一个红错误给用户。
    """
    rscript = rscript or probe().get("rscript") or find_rscript()
    if not rscript:
        return {"ok": False, "status": "no-r",
                "text": "未探测到 Rscript：请先安装 R（≥ 4.1）或在 data/r_settings.json 里指定路径。"}
    if not os.path.exists(install_script()):
        return {"ok": False, "status": "no-script", "text": "缺少 %s" % install_script()}
    args = [rscript, "--vanilla", install_script(), user_library()] + \
           ([str(repo).strip()] if repo else [])
    os.makedirs(user_library(), exist_ok=True)
    code, out, err = _run(args, timeout=timeout, env=_r_env())
    text = ((out or "") + ("\n" if out and err else "") + (err or "")).strip()
    reset_probe_cache()
    return {"ok": code == 0, "status": "ok" if code == 0 else "failed",
            "exit_code": code, "text": text[-6000:], "library": user_library(),
            "packages": probe(force=True).get("packages") or {}}


# ------------------------------------------------------------------ QThread 包装

try:                                        # QtCore 只在有 Qt 的应用里才需要
    from PySide6.QtCore import QThread, Signal

    class ProbeWorker(QThread):
        """后台探测 R 与参考包：首次进入工作台时跑一次，结果写进质检卡。"""

        done = Signal(dict)

        def __init__(self, force=False, parent=None):
            super().__init__(parent)
            self.force = bool(force)

        def run(self):
            try:
                self.done.emit(probe(force=self.force))
            except Exception:               # noqa: BLE001
                self.done.emit({"available": False, "error": "R 探测失败"})

    class ReferenceWorker(QThread):
        """后台跑 R 复算 + 对照：最长 120 秒的 Rscript 不该让 UI 停住。"""

        done = Signal(list, str)

        def __init__(self, spec, result, ctx, params, parent=None):
            super().__init__(parent)
            self.spec, self.result = spec, result
            self.ctx, self.params = ctx, params

        def run(self):
            try:
                rows, note = reference_check(self.spec, self.result, self.ctx, self.params)
            except Exception as exc:        # noqa: BLE001
                traceback.print_exc()
                rows, note = [], "R 复算异常：%s" % exc
            self.done.emit(rows, note)

    class InstallWorker(QThread):
        """后台装 R 参考包：CRAN 编译可能跑十几分钟，必须在子线程里。"""

        done = Signal(dict)

        def __init__(self, repo="", parent=None):
            super().__init__(parent)
            self.repo = str(repo or "")

        def run(self):
            try:
                self.done.emit(install_references(repo=self.repo))
            except Exception as exc:        # noqa: BLE001
                traceback.print_exc()
                self.done.emit({"ok": False, "status": "failed", "text": str(exc)})

except ImportError:                          # pragma: no cover  无 Qt 时仍可 import 本模块
    ProbeWorker = None
    ReferenceWorker = None
    InstallWorker = None


__all__ = ["find_rscript", "probe", "probe_packages", "set_rscript", "reset_probe_cache",
           "rscript_version", "user_library", "run_reference", "write_inputs",
           "cross_check", "reference_check", "python_reference", "summarize_rows",
           "trim_rows", "sequences_frame", "install_command", "install_references",
           "install_script", "CHECK_PLANS", "STATUS_NOTES",
           "ProbeWorker", "ReferenceWorker", "InstallWorker", "R_REFS_DIR",
           "REFERENCE_PACKAGES", "TIMEOUT_SECONDS", "DEFAULT_ATOL", "DEFAULT_RTOL",
           "MISSING_TEXT"]
