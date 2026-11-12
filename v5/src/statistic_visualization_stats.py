# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""M01–M19 的统计计算。

每个方法一个函数，签名统一为 ``f(ds: CodingDataset, p: dict) -> AnalysisResult``，
只产出结果表与元数据，不接触 UI、不做任何推断统计（检验 / 回归 / 模型）。
核心公式与分母口径严格照抄 V1.1 方案 §4 / §8 / §11。
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from statistic_visualization_data import (AnalysisResult, QAItem, LEVEL_ERROR, LEVEL_INFO,
                                          LEVEL_WARN, fmt_round, natural_key, new_analysis_id,
                                          parse_bin_bounds)

MISSING_LABEL = "（缺失）"


# ------------------------------------------------------------------ 通用工具

def _result(method_id, method_name, tables, meta=None, qa=None, notes=None):
    return AnalysisResult(method_id=method_id, method_name=method_name,
                          analysis_id=new_analysis_id(method_id),
                          tables=tables, meta=meta or {}, qa=list(qa or []),
                          notes=list(notes or []))


def _start():
    return time.perf_counter()


def _elapsed(marker):
    return int((time.perf_counter() - marker) * 1000)


def _chosen_students(ds, p):
    return ds._pick(ds.students, p.get("students"))


def _chosen_dims(ds, p, key="dimensions"):
    return ds._pick(ds.dimensions, p.get(key))


def _valid_slice(ds, p, students=None, dimensions=None):
    frame = ds.slice(students if students is not None else p.get("students"),
                     dimensions if dimensions is not None else p.get("dimensions"))
    return frame


def _percent(part, whole):
    if not whole:
        return 0.0
    return fmt_round(part / whole * 100.0)


def _sort_codes(codes, how="count", counts=None):
    """频次降序 / 字母序 / 首次出现的时间序。"""
    if how == "alpha":
        return sorted(codes, key=lambda c: str(c).lower())
    if how == "time":
        return sorted(codes, key=natural_key)
    return sorted(codes, key=lambda c: (-int(counts.get(c, 0)), str(c).lower()))


def run_lengths(values):
    """连续相同编码合并：返回 [(code, start_index, length)]。None 也算一种状态。"""
    runs = []
    for idx, value in enumerate(values):
        if runs and runs[-1][0] == value:
            runs[-1][2] += 1
        else:
            runs.append([value, idx, 1])
    return [(code, start, length) for code, start, length in runs]


def _unit_summary(ds, p):
    """M01/M02/M03 共用的基础聚合：F_c、N_valid、missing、D_c。"""
    frame = _valid_slice(ds, p)
    how = p.get("sort", "count")
    rows = []
    for student in _chosen_students(ds, p):
        for dimension in _chosen_dims(ds, p):
            part = frame[(frame["student_id"] == student)
                         & (frame["dimension"] == dimension)]
            valid_n = int(len(part))
            missing_n = int(ds.unit_missing.get((student, dimension), 0))
            counts = part["code"].value_counts()
            durations = part.groupby("code")["duration"].sum()
            total_duration = float(durations.sum()) if len(durations) else 0.0
            for code in _sort_codes(counts.index.tolist(), how, counts):
                count = int(counts.get(code, 0))
                span = float(durations.get(code, 0.0))
                rows.append({
                    "student_id": student,
                    "dimension": dimension,
                    "code": code,
                    "count": count,
                    "valid_N": valid_n,
                    "missing_N": missing_n,
                    "percentage": _percent(count, valid_n),
                    "duration_s": fmt_round(span),
                    "duration_min": fmt_round(span / 60.0),
                    "time_share_%": _percent(span, total_duration),
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ M01–M03

def frequency(ds, p):
    """M01 Frequency：F_c = Σ_t I(X_t = c)。"""
    marker = _start()
    frame = _unit_summary(ds, p)
    qa = _time_qa(ds)
    if frame.empty:
        # 空表上选列会直接 KeyError：这类数据只能走「阻断 + 质检原因」通道
        return _blocked("M01", "Frequency 频次", qa)
    tables = {"频次统计": frame[["student_id", "dimension", "code", "count",
                                "valid_N", "missing_N", "percentage"]]}
    if p.get("merge_students"):
        merged = (frame.groupby(["dimension", "code"], as_index=False)["count"].sum()
                  .sort_values(["dimension", "count"], ascending=[True, False]))
        merged["percentage"] = [
            _percent(row["count"], int(merged[merged["dimension"] == row["dimension"]]["count"].sum()))
            for _, row in merged.iterrows()]
        tables["跨学生合计"] = merged
        qa.append(QAItem(LEVEL_INFO, "merge_students",
                         "已勾选跨学生合计：该口径把 S1/S2 当作同类观测合并计数，"
                         "不代表任何单个学生的行为比例"))
    return _finish("M01", "Frequency 频次", tables, marker, qa, ds, p,
                   notes=["频次按 student × dimension 独立统计，未跨学生合并分母"])


def percentage(ds, p):
    """M02 Percentage：P_c = F_c / N_valid × 100%，N_valid 按 student × dimension 计。"""
    marker = _start()
    frame = _unit_summary(ds, p)
    qa = _time_qa(ds)
    if frame.empty:
        return _blocked("M02", "Percentage 比例", qa)
    tables = {"比例统计": frame[["student_id", "dimension", "code", "count", "valid_N",
                                "missing_N", "percentage"]]}
    notes = ["分母 N_valid 为该 student × dimension 的有效观测数，缺失值不入分母"]
    check = frame.groupby(["student_id", "dimension"])["percentage"].sum()
    if len(check) and bool((check - 100.0).abs().gt(0.5).any()):
        notes.append("取整导致各单元比例合计与 100% 有 ≤0.5pp 误差")
    return _finish("M02", "Percentage 比例", tables, marker, qa, ds, p,
                   notes=notes)


def duration(ds, p):
    """M03 Duration：D_c = Σ_t duration_t（规则采样等价 F_c × Δt）。"""
    marker = _start()
    frame = _unit_summary(ds, p)
    qa = _time_qa(ds)
    if frame.empty:
        return _blocked("M03", "Duration 持续时间", qa)
    tables = {"持续时间": frame[["student_id", "dimension", "code", "count", "valid_N",
                                 "duration_s", "duration_min", "time_share_%"]]}
    if not ds.time_info.is_regular:
        qa.append(QAItem(LEVEL_WARN, "duration_source",
                         "采样不规则：duration 取相邻时间差而非固定 Δt"))
    return _finish("M03", "Duration 持续时间", tables, marker, qa, ds, p,
                   notes=["持续时间单位同时给秒与分钟；未编码时间点不分配给任何状态",
                          "time_share_% 为该状态时长占本单元总时长的比例"])


# ------------------------------------------------------------------ M04–M05

def _bins_ready(ds, p, qa):
    mode, value = _bin_params(p)
    if not ds.time_info.valid:
        qa.append(QAItem(LEVEL_ERROR, "time_missing",
                         "有效时间点不足 2 个，无法按时间窗口统计", blocking=True))
        return None
    labels, _ = ds.ensure_bins(mode, value)
    return labels


def _bin_params(p):
    mode = p.get("bin_mode", "count")
    value = p.get("bin_value", 6)
    if mode == "custom":
        # UI 交回的是文本框字符串，必须先拆成数值边界再交给 make_bin_edges
        value = parse_bin_bounds(p.get("bin_custom"))
    return mode, value


def _bin_unit_table(ds, p, with_percent):
    frame = _valid_slice(ds, p)
    frame = frame[frame["time_bin"].notna()]
    rows = []
    for student in _chosen_students(ds, p):
        for dimension in _chosen_dims(ds, p):
            part = frame[(frame["student_id"] == student)
                         & (frame["dimension"] == dimension)]
            for label in ds.bin_order:
                window = part[part["time_bin"] == label]
                bin_valid = int(len(window))
                counts = window["code"].value_counts()
                for code in sorted(counts.index.tolist(), key=natural_key):
                    count = int(counts[code])
                    row = {"student_id": student, "dimension": dimension,
                           "time_bin": label, "code": code, "count": count,
                           "bin_valid_N": bin_valid}
                    if with_percent:
                        row["percentage"] = _percent(count, bin_valid)
                    rows.append(row)
    return pd.DataFrame(rows)


def bin_frequency(ds, p):
    """M04 Time-bin Frequency：F_c,b = Σ I(code=c ∧ time∈b)。"""
    marker = _start()
    qa = _time_qa(ds)
    labels = _bins_ready(ds, p, qa)
    if labels is None:
        return _blocked("M04", "Time-bin Frequency 时间窗口频次", qa)
    frame = _bin_unit_table(ds, p, with_percent=False)
    return _finish("M04", "Time-bin Frequency 时间窗口频次",
                   {"时间窗口频次": frame}, marker, qa, ds, p,
                   notes=["窗口边界见导出配置，图上按真实边界标注"])


def bin_percentage(ds, p):
    """M05 Time-bin Percentage：P_c,b = F_c,b / Σ_c F_c,b × 100%（该窗口内有效数为分母）。"""
    marker = _start()
    qa = _time_qa(ds)
    labels = _bins_ready(ds, p, qa)
    if labels is None:
        return _blocked("M05", "Time-bin Percentage 时间窗口构成", qa)
    frame = _bin_unit_table(ds, p, with_percent=True)
    return _finish("M05", "Time-bin Percentage 时间窗口构成",
                   {"时间窗口构成": frame}, marker, qa, ds, p,
                   notes=["分母为该 student × dimension 在该时间窗口内的有效观测数"])


# ------------------------------------------------------------------ M06–M07

def timeline(ds, p):
    """M06 Timeline：单学生单维度的分类时间轴（不做数值折线）。"""
    marker = _start()
    qa = _time_qa(ds)
    if not ds.time_info.valid:
        qa.append(QAItem(LEVEL_ERROR, "time_missing", "无法建立时间轴", blocking=True))
        return _blocked("M06", "Timeline 时间轴", qa)
    student = p.get("student") or ds.students[0]
    dimension = p.get("dimension") or ds.dimensions[0]
    observations = []
    for t in ds.time_positions():
        row = ds.long_df[(ds.long_df["student_id"] == student)
                         & (ds.long_df["dimension"] == dimension)
                         & (ds.long_df["time"] == t)]
        code = None if row.empty or bool(row["is_missing"].iloc[0]) else row["code"].iloc[0]
        observations.append({
            "student_id": student, "dimension": dimension, "time": fmt_round(t),
            "time_bin": row["time_bin"].iloc[0] if not row.empty else "",
            "code": code if code is not None else MISSING_LABEL,
            "duration_s": fmt_round(row["duration"].iloc[0]) if not row.empty else 0.0,
        })
    obs = pd.DataFrame(observations)
    summary = _unit_summary(ds, dict(p, students=[student], dimensions=[dimension]))
    return _finish("M06", "Timeline 时间轴",
                   {"时间轴观测": obs, "编码汇总": summary[["code", "count", "valid_N",
                                                            "percentage", "duration_s"]]},
                   marker, qa, ds, p,
                   unit=(student, dimension),
                   notes=["缺失编码以独立纹理色块显示，可在图例中关闭"])


def multidim_timeline(ds, p):
    """M07 Multi-dimensional Timeline：同一学生 D1/D2/D3 同步时间轴。"""
    marker = _start()
    qa = _time_qa(ds)
    if not ds.time_info.valid:
        qa.append(QAItem(LEVEL_ERROR, "time_missing", "无法建立时间轴", blocking=True))
        return _blocked("M07", "Multi-dimensional Timeline 多维同步时间轴", qa)
    dimensions = _chosen_dims(ds, p)
    tables = {}
    for student in _chosen_students(ds, p):
        columns = {"time": [fmt_round(t) for t in ds.time_positions()]}
        for dimension in dimensions:
            sequence = ds.sequence(student, dimension)
            columns[dimension] = [c if c is not None else MISSING_LABEL for c in sequence]
        tables["学生 %s" % student] = pd.DataFrame(columns)
    return _finish("M07", "Multi-dimensional Timeline 多维同步时间轴", tables, marker, qa, ds, p,
                   notes=["每名学生一组，组内 D1/D2/D3 共享同一 Time 轴；"
                          "S1 与 S2 的比较请改用 State Distribution / Sequence Index"])


# ------------------------------------------------------------------ M08

def heatmap(ds, p):
    """M08 Heatmap：code × time_bin 的 count / percentage 矩阵。"""
    marker = _start()
    qa = _time_qa(ds)
    labels = _bins_ready(ds, p, qa)
    if labels is None:
        return _blocked("M08", "Heatmap 时间窗口热图", qa)
    value = p.get("value", "count")
    frame = _bin_unit_table(ds, p, with_percent=(value == "percentage"))
    tables = {}
    for (student, dimension), part in frame.groupby(["student_id", "dimension"], sort=False):
        pivot = part.pivot_table(index="code", columns="time_bin", values="count",
                                 aggfunc="sum", fill_value=0)
        pivot = pivot.reindex(columns=[l for l in labels if l in pivot.columns],
                              index=sorted(pivot.index.tolist(), key=natural_key))
        if value == "percentage":
            totals = pivot.sum(axis=0).replace(0, np.nan)
            pivot = (pivot / totals * 100.0).round(2)
        pivot.columns.name = None
        pivot.index.name = "code"
        tables["%s×%s" % (student, dimension)] = pivot.reset_index()
    if not tables:
        return _blocked("M08", "Heatmap 时间窗口热图", qa)
    return _finish("M08", "Heatmap 时间窗口热图", tables, marker, qa, ds, p,
                   notes=["默认显示 count，避免把颜色直接误读为概率；"
                          "每列合计等于该时间窗口的有效编码数"])


# ------------------------------------------------------------------ M09–M13

def _single_dimension(ds, p):
    return p.get("dimension") or (ds.dimensions[0] if ds.dimensions else None)


def state_distribution(ds, p):
    """M09 State Distribution：P_c,t = N(code=c at t) / N_valid(t)，跨学生按位置统计。"""
    marker = _start()
    qa = _time_qa(ds)
    dimension = _single_dimension(ds, p)
    if dimension is None or not ds.time_info.valid:
        return _blocked("M09", "State Distribution 状态分布", qa)
    students = _chosen_students(ds, p)
    frame = ds.slice(students, [dimension])
    positions = ds.time_positions()
    cross = pd.crosstab(frame["time"], frame["code"])
    codes = sorted(cross.columns.tolist(), key=natural_key)
    cross = cross.reindex(index=positions, columns=codes, fill_value=0)
    valid_n = cross.sum(axis=1).astype(int)
    wide = pd.DataFrame({"time": [fmt_round(t) for t in positions]})
    wide["valid_N"] = valid_n.to_numpy()
    long_rows = []
    for code in codes:
        counts = cross[code].to_numpy().astype(int)
        wide[code] = counts
        wide["%s %%" % code] = [_percent(c, n) for c, n in zip(counts, wide["valid_N"])]
        for t, count, n in zip(positions, counts, wide["valid_N"]):
            long_rows.append({"student_id": "、".join(students), "dimension": dimension,
                              "time": fmt_round(t), "code": code, "count": int(count),
                              "valid_N": int(n), "percentage": _percent(count, n)})
    notes = []
    if len(students) < 3:
        note = ("当前学生数 N=%d，每个时间位置的比例只能是 0/%s/100%%，"
                "只能作个体比较，不代表稳定的群体分布"
                % (len(students), _num_simple(100.0 / max(len(students), 1))))
        notes.append(note)
        qa.append(QAItem(LEVEL_WARN, "few_students", note))
    return _finish("M09", "State Distribution 状态分布",
                   {"各位置状态比例": wide, "长表": pd.DataFrame(long_rows)},
                   marker, qa, ds, p, dimension=dimension, notes=notes)


def modal_state(ds, p):
    """M10 Modal State：Modal(t) = argmax_c P_c,t，并列必须标记。"""
    marker = _start()
    qa = _time_qa(ds)
    dimension = _single_dimension(ds, p)
    if dimension is None or not ds.time_info.valid:
        return _blocked("M10", "Modal State 主导状态", qa)
    students = _chosen_students(ds, p)
    frame = ds.slice(students, [dimension])
    cross = pd.crosstab(frame["time"], frame["code"])
    positions = ds.time_positions()
    cross = cross.reindex(index=[t for t in positions if t in cross.index])
    cross = cross.reindex(columns=sorted(cross.columns.tolist(), key=natural_key), fill_value=0)
    rows, ties = [], 0
    for t in cross.index:
        counts = cross.loc[t]
        best = int(counts.max()) if len(counts) else 0
        winners = sorted(counts[counts == best].index.tolist(), key=natural_key) if best else []
        if len(winners) > 1:
            ties += 1
        rows.append({"time": fmt_round(t), "modal_code": winners[0] if winners else MISSING_LABEL,
                     "top_count": best, "valid_N": int(counts.sum()),
                     "tie": "是" if len(winners) > 1 else "否",
                     "tie_codes": "、".join(winners) if len(winners) > 1 else ""})
    table = pd.DataFrame(rows)
    notes = ["主导状态只是「最常见状态」的汇总，不表示所有学生都处于该状态"]
    if ties:
        notes.append("有 %d 个时间位置出现并列，图中以斜纹标记" % ties)
    return _finish("M10", "Modal State 主导状态", {"主导状态": table}, marker, qa, ds, p,
                   dimension=dimension, notes=notes)


def sequence_index(ds, p):
    """M11 Sequence Index：固定维度，每个学生一行，连续相同状态合并成矩形。"""
    marker = _start()
    qa = _time_qa(ds)
    if not ds.time_info.valid:
        return _blocked("M11", "Sequence Index 序列索引图", qa)
    dimensions = _chosen_dims(ds, p)
    students = _chosen_students(ds, p)
    tables = {}
    for dimension in dimensions:
        rows = []
        for student in students:
            rows.extend(_student_runs(ds, student, dimension))
        table = pd.DataFrame(rows)
        table = _sort_index_table(table, p.get("sort_by", "student"))
        tables["维度 " + dimension] = table
    if not tables:
        return _blocked("M11", "Sequence Index 序列索引图", qa)
    return _finish("M11", "Sequence Index 序列索引图", tables, marker, qa, ds, p,
                   notes=["排序方式（%s）已写入导出配置" % p.get("sort_by", "student")])


def _student_runs(ds, student, dimension):
    positions = ds.time_positions()
    sequence = ds.sequence(student, dimension)
    durations = [ds.duration_by_time.get(t, 0.0) for t in positions]
    rows = []
    for code, start, length in run_lengths(sequence):
        end = positions[start + length - 1] + (durations[start + length - 1] or 0)
        rows.append({
            "student_id": student, "dimension": dimension,
            "code": code if code is not None else MISSING_LABEL,
            "start_time": fmt_round(positions[start]),
            "end_time": fmt_round(end),
            "run_length": length,
            "duration_s": fmt_round(sum(d for d in durations[start:start + length]
                                        if d == d)),
            "is_missing": "是" if code is None else "否",
        })
    return rows


def _sort_index_table(table, sort_by):
    """按学生 ID / 起始状态 / 主要状态累计时长排序，排序方式由 _finish 写入导出配置。"""
    if table.empty or sort_by == "student":
        return table
    students = list(dict.fromkeys(table["student_id"].tolist()))
    if sort_by == "first_code":
        first = table.drop_duplicates("student_id").set_index("student_id")["code"]
        ranked = sorted(students, key=lambda s: (str(first.get(s, "")).lower(), natural_key(s)))
    else:
        valid = table[table["is_missing"] != "是"]
        if valid.empty:
            return table
        focus = valid.groupby("code")["run_length"].sum().idxmax()
        span = valid[valid["code"] == focus].groupby("student_id")["duration_s"].sum()
        ranked = sorted(students, key=lambda s: (-float(span.get(s, 0.0)), natural_key(s)))
    order = {student: idx for idx, student in enumerate(ranked)}
    return (table.assign(_rank=table["student_id"].map(order))
            .sort_values("_rank", kind="mergesort")
            .drop(columns="_rank").reset_index(drop=True))


def sequence_frequency(ds, p):
    """M12 Sequence Frequency：完整序列（或按窗口压缩后的相位序列）的 Top-N 频次。"""
    marker = _start()
    qa = _time_qa(ds)
    dimension = _single_dimension(ds, p)
    if dimension is None or not ds.time_info.valid:
        return _blocked("M12", "Sequence Frequency 序列频次", qa)
    students = _chosen_students(ds, p)
    compress = p.get("compress", "none")
    keys = {}
    if compress == "bin":
        labels = _bins_ready(ds, dict(p, bin_mode="count", bin_value=int(p.get("compress_bins", 6))), qa)
        frame = ds.slice(students, [dimension])
        for student in students:
            part = frame[frame["student_id"] == student]
            seq = []
            for label in labels or []:
                window = part[part["time_bin"] == label]
                codes = [c for c in window["code"].tolist() if c is not None]
                seq.append(pd.Series(codes).mode().iat[0] if codes else MISSING_LABEL)
            keys[student] = " → ".join(seq)
    else:
        for student in students:
            seq = [c if c is not None else MISSING_LABEL for c in ds.sequence(student, dimension)]
            keys[student] = " → ".join(seq)
    grouped = {}
    for student, key in keys.items():
        grouped.setdefault(key, []).append(student)
    rows = [{"sequence": key, "count": len(members), "students": "、".join(sorted(members)),
             "percentage": _percent(len(members), len(students))}
            for key, members in grouped.items()]
    table = pd.DataFrame(rows).sort_values(["count", "sequence"],
                                           ascending=[False, True]).reset_index(drop=True)
    top_n = int(p.get("top_n", 10))
    notes = []
    if len(table) and len(grouped) / max(len(students), 1) >= 0.8:
        qa.append(QAItem(LEVEL_WARN, "high_sequence_uniqueness",
                         "完整序列几乎全部唯一（%d 名学生产生 %d 条不同序列），"
                         "默认只展示 Top %d，可改用「按时间窗口压缩」"
                         % (len(students), len(grouped), top_n)))
        notes.append("序列唯一性过高：Top-N 之外的序列合计见「全部序列」表")
    shown = table.head(top_n) if top_n > 0 else table
    shown = shown.assign(rank=range(1, len(shown) + 1))[
        ["rank", "sequence", "count", "students", "percentage"]]
    tables = {"Top-%d 序列" % (top_n if top_n > 0 else len(table)): shown}
    if len(table) > len(shown):
        tables["全部序列"] = table
    tables["相位序列"] = pd.DataFrame(
        [{"student_id": s, "sequence": k} for s, k in keys.items()])
    return _finish("M12", "Sequence Frequency 序列频次", tables, marker, qa, ds, p,
                   dimension=dimension, notes=notes or ["压缩方式：%s" % compress])


def mean_time(ds, p):
    """M13 Mean Time：先按 student × code 求总时长，再在学生层聚合 Mean/SD/Median/Min/Max。"""
    marker = _start()
    qa = _time_qa(ds)
    dimensions = _chosen_dims(ds, p)
    students = _chosen_students(ds, p)
    frame = _valid_slice(ds, p, students, dimensions)
    if frame.empty:
        return _blocked("M13", "Mean Time 平均状态占用时间", qa)
    per_student = (frame.groupby(["student_id", "dimension", "code"])
                   ["duration"].sum().reset_index())
    rows = []
    for (dimension, code), part in per_student.groupby(["dimension", "code"], sort=False):
        values = part["duration"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            continue
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
        rows.append({
            "dimension": dimension, "code": code,
            "n_students_with_code": int(values.size),
            "n_students": len(students),
            "mean_s": fmt_round(mean), "sd_s": fmt_round(sd),
            "median_s": fmt_round(float(np.median(values))),
            "min_s": fmt_round(float(values.min())), "max_s": fmt_round(float(values.max())),
            "se_s": fmt_round(sd / np.sqrt(values.size)) if values.size else 0.0,
        })
    table = pd.DataFrame(rows).sort_values(["dimension", "mean_s"],
                                           ascending=[True, False]).reset_index(drop=True)
    notes = ["误差条为描述性 SD/SE，不做 t 检验 / ANOVA / 回归",
             "mean_s 的分母是「出现过该编码的学生」（n_students_with_code）；"
             "若要按全部学生（n_students）计，需把未出现该编码的学生按 0 计入"]
    if len(students) < 3:
        note = "学生数 N=%d 偏少，均值与标准差不具群体稳定性" % len(students)
        notes.append(note)
        qa.append(QAItem(LEVEL_WARN, "few_students", note))
    return _finish("M13", "Mean Time 平均状态占用时间",
                   {"平均占用时间": table, "学生明细": per_student.rename(
                       columns={"duration": "duration_s"})},
                   marker, qa, ds, p, notes=notes)


# ------------------------------------------------------------------ M14

def spells(ds, p):
    """M14 Spell / Run：连续相同编码构成一个 spell，长度 = Σ duration。"""
    marker = _start()
    qa = _time_qa(ds)
    if not ds.time_info.valid:
        return _blocked("M14", "Spell / Run 连续段", qa)
    detail = []
    for student in _chosen_students(ds, p):
        for dimension in _chosen_dims(ds, p):
            for row in _student_runs(ds, student, dimension):
                if row["is_missing"] == "是":
                    continue
                detail.append(dict(row))
    table = pd.DataFrame(detail)
    if table.empty:
        return _blocked("M14", "Spell / Run 连续段", qa)
    index = table.groupby(["student_id", "dimension", "code"]).cumcount() + 1
    table.insert(table.columns.get_loc("code"), "spell_index", index)
    summary_rows = []
    for (student, dimension, code), part in table.groupby(
            ["student_id", "dimension", "code"], sort=False):
        values = part["duration_s"].to_numpy(dtype=float)
        summary_rows.append({
            "student_id": student, "dimension": dimension, "code": code,
            "n_spells": int(len(part)),
            "total_s": fmt_round(float(values.sum())),
            "mean_s": fmt_round(float(values.mean())),
            "sd_s": fmt_round(float(values.std(ddof=1))) if len(values) > 1 else 0.0,
            "median_s": fmt_round(float(np.median(values))),
            "min_s": fmt_round(float(values.min())),
            "max_s": fmt_round(float(values.max())),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["student_id", "dimension", "total_s"], ascending=[True, True, False]
    ).reset_index(drop=True)
    return _finish("M14", "Spell / Run 连续段", {"Spell 明细": table, "Spell 汇总": summary},
                   marker, qa, ds, p,
                   notes=["规则采样下 spell_duration = run_length × Δt；"
                          "不规则采样下为相邻时间差之和"])


# ------------------------------------------------------------------ M15

def cooccurrence(ds, p):
    """M15 共现 / 同步：within-student 与 cross-student synchrony 严格分流。"""
    marker = _start()
    qa = _time_qa(ds)
    scope = p.get("scope")
    if scope not in ("within", "cross"):
        qa.append(QAItem(LEVEL_ERROR, "cooccurrence_scope_unconfirmed",
                         "请先明确选择「同一学生跨维度共现」或「跨学生同维同步比较」",
                         blocking=True))
        return _blocked("M15", "Co-occurrence 共现", qa)
    if scope == "within":
        student = p.get("student") or ds.students[0]
        dims = _chosen_dims(ds, p)
        if len(dims) < 2:
            dims = list(ds.dimensions[:2])
        dim_a, dim_b = p.get("dim_a") or dims[0], p.get("dim_b") or dims[1]
        if dim_a == dim_b:
            dim_b = next((d for d in ds.dimensions if d != dim_a), dim_a)
        left = ds.slice([student], [dim_a]).set_index("time")["code"]
        right = ds.slice([student], [dim_b]).set_index("time")["code"]
        pair = pd.DataFrame({"a": left, "b": right}).dropna()
        title = "同一学生跨维度共现（Within-student Co-occurrence）"
        matrix_name = "%s：%s × %s 共现" % (student, dim_a, dim_b)
        row_label, col_label = "%s 编码" % dim_a, "%s 编码" % dim_b
        notes = ["统计单位是同一学生在同一时间点上的两个维度组合，"
                 "不能把 S1-%s × S2-%s 当作 within-student 共现" % (dim_a, dim_b)]
    else:
        students = _chosen_students(ds, p)
        if len(students) < 2:
            students = list(ds.students[:2])
        student_a, student_b = p.get("student_a") or students[0], \
            p.get("student_b") or (students[1] if len(students) > 1 else students[0])
        dimension = _single_dimension(ds, p)
        left = ds.slice([student_a], [dimension]).set_index("time")["code"]
        right = ds.slice([student_b], [dimension]).set_index("time")["code"]
        pair = pd.DataFrame({"a": left, "b": right}).dropna()
        title = "跨学生同维同步比较（Cross-student Synchrony）"
        matrix_name = "%s × %s（%s）同步" % (student_a, student_b, dimension)
        row_label, col_label = "%s 编码" % student_a, "%s 编码" % student_b
        notes = ["这是跨学生的同时点一致性比较，与 within-student co-occurrence "
                 "不是同一回事，两者不可混用名称与结论"]
    if pair.empty:
        qa.append(QAItem(LEVEL_ERROR, "no_overlap", "两个选择在相同时间点上没有共同有效编码",
                         blocking=True))
        return _blocked("M15", title, qa)
    codes_a = sorted(pair["a"].unique().tolist(), key=natural_key)
    codes_b = sorted(pair["b"].unique().tolist(), key=natural_key)
    matrix = pd.crosstab(pair["a"], pair["b"])
    matrix = matrix.reindex(index=codes_a, columns=codes_b, fill_value=0)
    total = int(matrix.values.sum())
    normalize = p.get("normalize", "none")
    if normalize == "row":
        shown = (matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0) * 100.0).round(2)
    elif normalize == "col":
        shown = (matrix.div(matrix.sum(axis=0).replace(0, np.nan), axis=1) * 100.0).round(2)
    elif normalize == "total":
        shown = (matrix / float(total) * 100.0).round(2) if total else matrix.astype(float)
    else:
        shown = matrix.astype(int)              # 默认 count，保持整数可读
    shown.index.name = row_label
    shown.columns.name = col_label
    matrix_table = shown.reset_index()
    matrix_table.columns.name = None
    numeric = matrix_table.columns[1:].tolist()
    matrix_table["合计"] = matrix_table[numeric].sum(axis=1)
    long_rows = [{"row": a, "column": b, "count": int(matrix.loc[a, b]),
                  "percentage": _percent(int(matrix.loc[a, b]), total)}
                 for a in codes_a for b in codes_b]
    qa.append(QAItem(LEVEL_INFO, "pairwise_valid_N",
                     "共同有效时间点 N=%d（任一维度缺失的时间点已剔除）" % total))
    return _finish("M15", title, {matrix_name: matrix_table, "长表": pd.DataFrame(long_rows)},
                   marker, qa, ds, p, notes=notes + [
                       "矩阵总和 = %d，等于共同参与统计的有效时间点数" % total])


# ------------------------------------------------------------------ M16

def alluvial(ds, p):
    """M16 Alluvial / Sankey：阶段（time-bin）之间的学生路径展示，非转移推断。"""
    marker = _start()
    qa = _time_qa(ds)
    dimension = _single_dimension(ds, p)
    if dimension is None or not ds.time_info.valid:
        return _blocked("M16", "Alluvial / Sankey 阶段路径", qa)
    phases = max(min(int(p.get("phase_count", 4)), 6), 2)
    min_flow = max(int(p.get("min_flow", 1)), 1)
    labels, _bounds = ds.ensure_bins("count", phases)
    students = _chosen_students(ds, p)
    frame = ds.slice(students, [dimension])
    phase_names = ["P%d %s" % (idx + 1, label) for idx, label in enumerate(labels)]

    # 每学生 × 每阶段：取该段众数编码；并列取最早出现者并记 tie
    modal = {}
    state_rows = []
    for student in students:
        for idx, label in enumerate(labels):
            window = frame[(frame["student_id"] == student)
                           & (frame["time_bin"] == label)].sort_values("time")
            codes = [c for c in window["code"].tolist() if c is not None]
            tie = ""
            if codes:
                counts = pd.Series(codes).value_counts()
                best = int(counts.max())
                winners = sorted(counts[counts == best].index.tolist(), key=natural_key)
                code = winners[0] if len(winners) == 1 else codes[0]
                if len(winners) > 1:
                    tie = "、".join(winners)
            else:
                code = None
            modal[(student, idx)] = code
            state_rows.append({"student_id": student, "phase": phase_names[idx],
                               "time_bin": label,
                               "modal_code": code if code is not None else MISSING_LABEL,
                               "valid_n": len(codes),
                               "tie": "是" if tie else "否", "tie_codes": tie})

    node_counts = {}
    for (student, idx), code in modal.items():
        if code is not None:
            node_counts[(idx, code)] = node_counts.get((idx, code), 0) + 1
    # 相邻阶段对（中间阶段缺失时不跨阶段连线）
    flows = {}
    for student in students:
        for idx in range(len(labels) - 1):
            code_a, code_b = modal.get((student, idx)), modal.get((student, idx + 1))
            if code_a is None or code_b is None:
                continue
            flows.setdefault((idx, code_a, code_b), []).append(student)
    flow_rows = []
    for (idx, code_a, code_b), members in sorted(
            flows.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(members) < min_flow:
            continue
        flow_rows.append({
            "from_phase": phase_names[idx], "from_code": code_a,
            "to_phase": phase_names[idx + 1], "to_code": code_b,
            "students": len(members), "student_ids": "、".join(sorted(members)),
            "percentage": _percent(len(members), len(students))})
    node_rows = [{"phase": phase_names[idx], "code": code, "students": int(count),
                  "percentage": _percent(count, len(students))}
                 for (idx, code), count in sorted(node_counts.items())]
    if not flow_rows:
        qa.append(QAItem(LEVEL_WARN, "no_flow",
                         "相邻阶段之间没有达到最小流量的路径，可降低最小流量或减少阶段数"))
        return _blocked("M16", "Alluvial / Sankey 阶段路径", qa)
    return _finish("M16", "Alluvial / Sankey 阶段路径",
                   {"阶段路径流量": pd.DataFrame(flow_rows),
                    "阶段节点": pd.DataFrame(node_rows),
                    "学生阶段众数": pd.DataFrame(state_rows)},
                   marker, qa, ds, p, dimension=dimension,
                   notes=["仅呈现阶段性路径，不构成转移推断；检验转移概率请交给 LSA / Markov 模块",
                          "每学生在每阶段取该段众数编码，并列取最早出现者并在表中标记",
                          "阶段数上限 6，避免流带图可读性崩塌"])


# ------------------------------------------------------------------ M17–M19（熵族）

def _entropies(counts):
    """频次序列 → (H, K)：``H = -Σ p_k ln p_k``，K = **实际出现**的状态数。

    只累加非零项（0·ln0 记 0），所以 K 不能取编码框架里的状态总数：拿框架数当 K
    会把 H_max 抬高、H_norm 压低，同一份数据换个框架写法结果就变了。
    """
    values = [float(c) for c in counts if c]
    total = float(sum(values))
    if not total:
        return 0.0, 0
    entropy = -sum((c / total) * math.log(c / total) for c in values)
    return entropy, len(values)


def _entropy_ceiling(states):
    """H_max = ln K；K≤1 时返回 0.0，由调用方把 H_norm 记 0（不能除）。"""
    return math.log(states) if states > 1 else 0.0


def _ordered_sequence(ds, student, dimension):
    """时间有序、已剔除缺失格的编码序列（缺失格既不是状态也不构成转移）。"""
    return ds.slice([student], [dimension])["code"].tolist()


def state_entropy(ds, p):
    """M17 Entropy 状态熵：H = -Σ p_k ln p_k、H_norm = H / ln K（逐 student × dimension）。"""
    marker = _start()
    qa = []
    students = _chosen_students(ds, p)
    dimensions = _chosen_dims(ds, p)
    frame = _valid_slice(ds, p, students, dimensions)
    if frame.empty:
        return _blocked("M17", "Entropy 状态熵", qa)
    rows, detail = [], []
    for student in students:
        for dimension in dimensions:
            part = frame[(frame["student_id"] == student)
                         & (frame["dimension"] == dimension)]
            if part.empty:
                continue
            counts = part["code"].value_counts()
            total = int(counts.sum())
            entropy, states = _entropies(counts.to_numpy(dtype=float))
            ceiling = _entropy_ceiling(states)
            rows.append({
                "student_id": student, "dimension": dimension,
                "N": total, "n_states": states,
                "entropy": fmt_round(entropy, 4),
                "H_max": fmt_round(ceiling, 4),
                "H_norm": fmt_round(entropy / ceiling, 4) if ceiling else 0.0,
                "missing_N": int(ds.unit_missing.get((student, dimension), 0)),
            })
            for code, count in counts.items():
                share = float(count) / total
                detail.append({
                    "student_id": student, "dimension": dimension,
                    "code": code, "count": int(count), "N": total,
                    "probability": fmt_round(share, 4),
                    "contribution": fmt_round(-share * math.log(share) if share else 0.0, 4),
                })
    if not rows:
        return _blocked("M17", "Entropy 状态熵", qa)
    notes = ["熵一律取自然对数 ln；H_norm = H / ln K，K 取实际出现状态数而非框架规模",
             "单一状态单元（K=1）的 H 恒为 0，H_norm 记 0 而非 0/0",
             "contribution = -p_k ln p_k，逐状态相加等于该单元的 H",
             "N 为剔除缺失格后的有效观测数，missing_N 单独报告"]
    thin = [row for row in rows if row["N"] < 20]
    if thin:
        qa.append(QAItem(LEVEL_WARN, "entropy_small_sample",
                         "存在 N<20 的单元（如 %s×%s N=%d），plug-in 熵系统性偏低，只宜横向比较"
                         % (thin[0]["student_id"], thin[0]["dimension"], thin[0]["N"])))
        notes.append("%d 个单元 N<20：plug-in 熵有小样本负偏，不宜看绝对值" % len(thin))
    return _finish("M17", "Entropy 状态熵",
                   {"状态熵": pd.DataFrame(rows), "各状态概率贡献": pd.DataFrame(detail)},
                   marker, qa, ds, p, notes=notes)


def switch_retention(ds, p):
    """M18 Switch & Retention：switch_rate = Σ I(S_t≠S_t+1)/(T-1)，self_transition = Σ N_ii/Σ N_ij。"""
    marker = _start()
    qa = _time_qa(ds)
    if not ds.time_info.valid:
        return _blocked("M18", "Switch & Retention 切换率/保持率", qa)
    rows = []
    skipped = 0
    for student in _chosen_students(ds, p):
        for dimension in _chosen_dims(ds, p):
            sequence = _ordered_sequence(ds, student, dimension)
            if len(sequence) < 2:
                skipped += 1
                continue
            pairs = list(zip(sequence, sequence[1:]))
            stays = sum(1 for left, right in pairs if left == right)
            switches = len(pairs) - stays
            runs = run_lengths(sequence)
            longest = max(length for _, _, length in runs)
            rows.append({
                "student_id": student, "dimension": dimension,
                "T_valid": len(sequence), "n_transitions": len(pairs),
                "n_switch": switches, "switch_rate": fmt_round(switches / len(pairs), 4),
                "n_stay": stays, "self_transition": fmt_round(stays / len(pairs), 4),
                "n_runs": len(runs),
                "mean_run_length": fmt_round(len(sequence) / len(runs), 4),
                "longest_run": longest,
                "n_states": len(set(sequence)),
                "missing_N": int(ds.unit_missing.get((student, dimension), 0)),
            })
    if not rows:
        return _blocked("M18", "Switch & Retention 切换率/保持率", qa)
    notes = ["分母是转移对数 T-1（T = 有效观测数），不是时间位置数",
             "switch_rate 与 self_transition 共用同一分母、互补为 1，两者只作对照读数",
             "mean_run_length = T_valid / n_runs（时间格数，非秒）"]
    if skipped:
        note = ("%d 个单元有效观测 <2，无法定义任何转移，已跳过（不是切换率为 0）"
                % skipped)
        notes.append(note)
        qa.append(QAItem(LEVEL_WARN, "insufficient_transitions", note))
    if int(sum(row["n_transitions"] for row in rows)) < 10:
        note = "全体转移对不足 10 个，切换率受单条序列支配，不宜跨研究比较"
        notes.append(note)
        qa.append(QAItem(LEVEL_WARN, "few_transitions", note))
    return _finish("M18", "Switch & Retention 切换率/保持率",
                   {"切换与保持": pd.DataFrame(rows)}, marker, qa, ds, p, notes=notes)


def transition_entropy(ds, p):
    """M19 Transition Entropy：H(Y|X) = -Σ_x p(x) Σ_y p(y|x) ln p(y|x)，与 H(Y) 并列。"""
    marker = _start()
    qa = _time_qa(ds)
    if not ds.time_info.valid:
        return _blocked("M19", "Transition Entropy 条件熵", qa)
    rows, detail, breakdown = [], [], []
    skipped = 0
    for student in _chosen_students(ds, p):
        for dimension in _chosen_dims(ds, p):
            sequence = _ordered_sequence(ds, student, dimension)
            codes = sorted(set(sequence), key=natural_key)
            if len(sequence) < 2 or len(codes) < 2:
                skipped += 1
                continue
            slot = {code: idx for idx, code in enumerate(codes)}
            matrix = [[0] * len(codes) for _ in codes]
            for left, right in zip(sequence, sequence[1:]):
                matrix[slot[left]][slot[right]] += 1
            total = sum(sum(row) for row in matrix)
            conditional = 0.0
            row_entropies = []
            for row in matrix:
                row_total = sum(row)
                if not row_total:
                    row_entropies.append(0.0)
                    continue
                entropy, _ = _entropies(row)
                row_entropies.append(entropy)
                conditional += (row_total / total) * entropy
            marginal, _ = _entropies([sum(row[i] for row in matrix) for i in range(len(codes))])
            ceiling = _entropy_ceiling(len(codes))
            # 互信息理论非负，浮点累加可能给出 -1e-16 量级，按 0 处理而不是报负熵差
            reduction = max(0.0, marginal - conditional)
            rows.append({
                "student_id": student, "dimension": dimension,
                "n_transitions": total, "n_states": len(codes),
                "H_Y": fmt_round(marginal, 4),
                "H_Y_given_X": fmt_round(conditional, 4),
                "reduction": fmt_round(reduction, 4),
                "H_max": fmt_round(ceiling, 4),
                "H_norm": fmt_round(conditional / ceiling, 4) if ceiling else 0.0,
                "predictability": fmt_round(reduction / marginal, 4) if marginal else 0.0,
            })
            for from_code, row, row_entropy in zip(codes, matrix, row_entropies):
                row_total = sum(row)
                weight = row_total / total
                # 逐前态一行：contribution = p(x)·H(Y|X=x)，整表相加才是 H(Y|X)
                breakdown.append({
                    "student_id": student, "dimension": dimension,
                    "from_code": from_code, "n_from": int(row_total),
                    "p_from": fmt_round(weight, 4),
                    "H_Y_given_from": fmt_round(row_entropy, 4),
                    "contribution": fmt_round(weight * row_entropy, 4),
                })
                for to_code, count in zip(codes, row):
                    if not count:
                        continue
                    detail.append({
                        "student_id": student, "dimension": dimension,
                        "from_code": from_code, "to_code": to_code,
                        "count": int(count), "row_total": int(row_total),
                        "p_to_given_from": fmt_round(count / row_total, 4) if row_total else 0.0,
                    })
    if not rows:
        return _blocked("M19", "Transition Entropy 条件熵", qa)
    notes = ["X = 前一状态、Y = 后一状态，均取自同一条去缺失序列的相邻对",
             "reduction = H(Y) - H(Y|X)（转移互信息）：越大说明下一状态越可预测",
             "predictability = reduction / H(Y)，H(Y)=0（退化成单一状态）时记 0",
             "H_norm = H(Y|X) / ln K，K 为该单元实际出现状态数",
             "「前态熵分解」表的 contribution = p(x)·H(Y|X=x)，同一单元逐前态相加才等于 H(Y|X)；"
             "「转移条件分布」是逐转移对，每行相加 = 1"]
    if skipped:
        note = ("%d 个单元的有效转移不足（观测 <2 或只有 1 种状态），条件熵无定义，已跳过"
                % skipped)
        notes.append(note)
        qa.append(QAItem(LEVEL_WARN, "insufficient_transitions", note))
    return _finish("M19", "Transition Entropy 条件熵",
                   {"条件熵": pd.DataFrame(rows), "转移条件分布": pd.DataFrame(detail),
                    "前态熵分解": pd.DataFrame(breakdown)},
                   marker, qa, ds, p, notes=notes)


# ------------------------------------------------------------------ 收尾与小工具

def _num_simple(value):
    number = float(value)
    return str(int(round(number))) if abs(number - round(number)) < 1e-9 else "%.2f" % number


def _time_qa(ds):
    return [item for item in ds.qa if item.code in ("time_missing", "irregular_sampling",
                                                    "time_duplicate")]


def _blocked(method_id, method_name, qa):
    result = _result(method_id, method_name, {}, qa=qa)
    result.blocked = True
    result.blocked_reason = next((item.message for item in qa if item.blocking),
                                 "当前数据不满足该方法的输入条件")
    return result


def _finish(method_id, method_name, tables, marker, qa, ds, p, dimension=None,
            unit=None, notes=None):
    result = _result(method_id, method_name, tables, qa=qa, notes=notes)
    result.meta = {
        "method_id": method_id,
        "method_name": method_name,
        "analysis_id": result.analysis_id,
        "params": {k: (list(v) if isinstance(v, (list, tuple, set)) else v)
                   for k, v in p.items()},
        "filter": {
            "students": _chosen_students(ds, p),
            "dimensions": ([dimension] if dimension else _chosen_dims(ds, p)),
            "unit": list(unit) if unit else None,
        },
        "bin_bounds": [[label, fmt_round(low), fmt_round(high)]
                       for label, low, high in ds.bin_bounds],
        "sampling": {"median_dt": fmt_round(ds.time_info.median_dt, 6),
                     "is_regular": bool(ds.time_info.is_regular),
                     "n_positions": int(ds.time_info.n_positions)},
        "missing_rule": "空/NA 不计入频次，也不进入比例分母",
        "descriptive_only": True,
        "dataset": ds.meta({"table_rows": {name: int(len(frame))
                                           for name, frame in tables.items()}}),
    }
    result.elapsed_ms = _elapsed(marker)
    return result
