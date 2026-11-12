# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""IND01–IND04 / IND05 / IND07：个体模式的序列统计内核。

这一层只吃 `AnalysisContext` 给的序列，不碰任何控件；每个方法一个函数，签名统一为
``f(ctx: AnalysisContext, p: dict) -> AnalysisResult``。所有公式都写成独立可测的纯
函数（`lsa_residuals` / `markov_probabilities` / `om_distance` / `silhouette_from_dist`
…），`test_stats_formulas.py` 直接对着手算小样本断言，UI 只是它们的一张脸。

口径约定（计划 §3 统一口径）：
- 熵一律自然对数 ln；
- 比例必报分母；
- 转移只在**同一条序列内部**计数，绝不跨组、也不跨学生的时间断点；
- p 值给 BH-FDR 校正后的 q，同时保留未校正 p，两边都要能看见。
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pandas as pd

import analysis_methods as am
from analysis_methods import (GROUP_DISTANCE, GROUP_MODEL, GROUP_STATE,
                              GROUP_TRANSITION, MODE_INDIVIDUAL, ParamSpec)
from statistic_visualization_data import QAItem, fmt_round

Z_DEFAULT = 1.96          # |Z| > 1.96 ⇔ 双侧 α = 0.05
_SQRT2 = math.sqrt(2.0)


# ------------------------------------------------------------------ 小工具

def _start():
    return time.perf_counter()


def _ms(marker):
    return int((time.perf_counter() - marker) * 1000)


def valid_codes(sequence):
    """序列里的有效编码（按时间升序，去掉空格）。"""
    return [code for code in sequence.codes if code]


def normal_two_sided_p(z):
    """双侧正态 p = erfc(|z|/√2)，不依赖 scipy。"""
    return float(math.erfc(abs(float(z)) / _SQRT2))


def chi2_sf(x, df):
    """卡方上尾概率：scipy 在就用，不在就用 Wilson–Hilferty 近似。"""
    x = float(x)
    df = int(df)
    if df <= 0 or x <= 0:
        return 1.0
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, df))
    except Exception:                       # noqa: BLE001  scipy 缺席仍可出结果
        z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) \
            / math.sqrt(2.0 / (9.0 * df))
        return float(max(min(math.erfc(z / _SQRT2), 1.0), 0.0))


def pick_sequences(ctx, params, min_length=None):
    """按参数卡选序列：太短的（默认 < 10 步）不进转移统计。"""
    floor = int(params.get("min_length") or 0)
    if min_length is not None:
        floor = max(floor, int(min_length))
    floor = max(floor, am.MIN_SEQUENCE_LENGTH)
    return [seq for seq in ctx.sequences_for(params) if len(valid_codes(seq)) >= floor]


def by_dimension(sequences):
    """{维度: [序列]}，保持维度首次出现顺序。"""
    out = {}
    for seq in sequences:
        out.setdefault(seq.dimension, []).append(seq)
    return out


def states_of(sequences):
    """编码全集（按首次出现顺序）：转移矩阵与距离矩阵的行列顺序。"""
    out = []
    for seq in sequences:
        for code in seq.codes:
            if code and code not in out:
                out.append(code)
    return out


def pooled_counts(sequences, states=None, lag=1):
    """把若干条**同一状态空间**的序列的滞后转移加起来。"""
    states = list(states if states is not None else states_of(sequences))
    index = {state: position for position, state in enumerate(states)}
    counts = np.zeros((len(states), len(states)), dtype=float)
    for seq in sequences:
        values = valid_codes(seq)
        for position in range(max(len(values) - lag, 0)):
            left, right = values[position], values[position + lag]
            if left in index and right in index:
                counts[index[left], index[right]] += 1.0
    return states, counts


def matrix_frame(states, matrix, row_name="state"):
    """方阵 → 可读宽表（第一列给状态名，行名列名一致）。"""
    frame = pd.DataFrame(np.round(np.asarray(matrix, dtype=float), 4),
                         index=list(states), columns=list(states))
    frame.index.name = row_name
    return frame.reset_index().rename(columns={row_name: "状态"})


def blocked(method_id, name, note):
    """拿方法号造一条阻断结果：方法声明在文件末尾，函数里不反向引用。"""
    return am.MethodSpec(id=method_id, name=name, group="", question="") \
        .blocked_result(None, note)


def _sparse_warning(counts, states):
    """平均格频 < 5 时给警告：这是转移矩阵解释力的硬底线，不是客套话。"""
    cells = len(states) * (len(states) - 1)
    total = float(counts.sum())
    if cells <= 0 or total / cells >= 5.0:
        return None
    return QAItem("warn", "markov.sparse",
                  "非对角格平均频数 %.2f < 5（共 %.0f 次转移 / %d 格），"
                  "残差与检验只对趋势负责，不建议作结论" % (total / cells, total, cells))


# ------------------------------------------------------------------ IND01 LSA

def lsa_residuals(observed):
    """Sackett 调整残差。

    期望频数按**去掉主对角线**的行列合计算（自转移不参与检验，这是 LSA 的通行做法，
    TraMineR 的 `seqlresid` 与 R 的 `lsayr` 同口径），返回 (去对角观测, 期望, Z)。
    """
    observed = np.asarray(observed, dtype=float)
    off = observed.copy()
    np.fill_diagonal(off, 0.0)
    k = off.shape[0]
    residuals = np.zeros((k, k), dtype=float)
    expected = np.zeros((k, k), dtype=float)
    total = float(off.sum())
    if k < 2 or total <= 0:
        return off, expected, residuals
    rows = off.sum(axis=1)
    cols = off.sum(axis=0)
    expected = np.outer(rows, cols) / total
    factor = (1.0 - rows / total)[:, None] * (1.0 - cols / total)[None, :]
    denom = np.sqrt(np.maximum(expected * factor, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        residuals = np.where(denom > 0, (off - expected) / denom, 0.0)
    np.fill_diagonal(residuals, 0.0)
    return off, expected, np.nan_to_num(residuals)


def yule_matrix(observed):
    """逐对 (i, j) 的 2×2 关联：Q = (ad − bc)/(ad + bc)，取上三角对称填充。"""
    observed = np.asarray(observed, dtype=float)
    k = observed.shape[0]
    out = np.zeros((k, k), dtype=float)
    for i, j in itertools.combinations(range(k), 2):
        block = [[observed[i, i], observed[i, j]], [observed[j, i], observed[j, j]]]
        value = am.yules_q(block)
        out[i, j] = out[j, i] = value
    return out


def aligned_channels(ctx, students, dimensions):
    """``{uid: {维度: 按时间升序的码表}}``：跨维 LSA 要拿同一时间栅格上的两条码轨。"""
    wanted = {str(item) for item in students} or None
    dims = [ctx.dimension_key(item) for item in dimensions]
    out = {}
    for bundle in ctx.bundles:
        frame = bundle.state_frame()
        if not len(frame):
            continue
        part = frame[frame["dimension"].isin(dims)]
        if not len(part):
            continue
        for uid, chunk in part.groupby("uid", sort=False):
            text = str(uid)
            if wanted is not None and text not in wanted:
                continue
            points = sorted(chunk["time"].unique(), key=float)
            position = {moment: idx for idx, moment in enumerate(points)}
            tracks = {dim: [""] * len(points) for dim in dims}
            for row in chunk.itertuples(index=False):
                slot = position.get(row.time)
                key = str(row.dimension)
                if slot is not None and key in tracks:
                    code = str(row.code or "")
                    tracks[key][slot] = "" if code == "nan" else code
            out[text] = tracks
    return out


def cross_dimension_counts(tracks, dimension_a, dimension_b, lag):
    """跨维滞后共现：A_t × B_{t+lag} 的计数（同一学生、同一时间栅格）。"""
    states_a = sorted({code for track in tracks.values() for code in track[dimension_a] if code})
    states_b = sorted({code for track in tracks.values() for code in track[dimension_b] if code})
    index_a = {state: i for i, state in enumerate(states_a)}
    index_b = {state: i for i, state in enumerate(states_b)}
    counts = np.zeros((len(states_a), len(states_b)), dtype=float)
    for track in tracks.values():
        left, right = track[dimension_a], track[dimension_b]
        for position in range(max(len(left) - lag, 0)):
            a, b = left[position], right[position + lag]
            if a in index_a and b in index_b:
                counts[index_a[a], index_b[b]] += 1.0
    return states_a, states_b, counts


def lag_sequence_analysis(ctx, params):
    """IND01 LSA：滞后转移的 O / E / 调整残差 Z / Yule's Q + BH-FDR。"""
    marker = _start()
    max_lag = max(1, min(int(params.get("max_lag") or 3), 5))
    threshold = float(params.get("z_threshold") or Z_DEFAULT)
    scope = params.get("scope") or "same"
    use_fdr = bool(params.get("fdr", True))
    sequences = pick_sequences(ctx, params)
    dims = sorted({seq.dimension for seq in sequences})
    rows, qa, matrices = [], [], []
    warnings = []

    if scope in ("same", "both"):
        for dimension, group in by_dimension(sequences).items():
            states = states_of(group)
            for lag in range(1, max_lag + 1):
                _states, counts = pooled_counts(group, states, lag=lag)
                off, expected, residuals = lsa_residuals(counts)
                q_matrix = yule_matrix(counts)
                cells = [(i, j) for i in range(len(states)) for j in range(len(states)) if i != j]
                p_values = [normal_two_sided_p(residuals[i, j]) for i, j in cells]
                q_values = am.bh_fdr(p_values) if use_fdr else p_values
                for (i, j), p_value, q_value in zip(cells, p_values, q_values):
                    rows.append({
                        "口径": "同维", "lag": lag, "dimension": ctx.label_of(dimension),
                        "from": states[i], "to": states[j],
                        "O": int(off[i, j]), "E": fmt_round(expected[i, j]),
                        "Z": fmt_round(residuals[i, j]), "p": fmt_round(p_value, 4),
                        "q": fmt_round(q_value, 4),
                        "significant": bool(abs(residuals[i, j]) > threshold),
                        "yules_Q": fmt_round(q_matrix[i, j]),
                    })
                for name, source in (("observed", counts), ("expected", expected),
                                     ("residual", residuals)):
                    matrices.append({"口径": "同维", "lag": lag, "dimension": dimension,
                                     "label": ctx.label_of(dimension),
                                     "table": name, "frame": matrix_frame(states, source)})
                warning = _sparse_warning(off, states)
                if warning is not None:
                    warnings.append((warning, dimension, lag))

    if scope in ("cross", "both"):
        students = [seq.uid for seq in sequences]
        tracks = aligned_channels(ctx, students, dims)
        for dimension_a, dimension_b in itertools.permutations(dims, 2):
            for lag in range(1, max_lag + 1):
                states_a, states_b, counts = cross_dimension_counts(
                    tracks, dimension_a, dimension_b, lag)
                if len(states_a) < 2 or len(states_b) < 2 or counts.sum() <= 0:
                    continue
                rows_total = counts.sum(axis=1)
                cols_total = counts.sum(axis=0)
                total = counts.sum()
                expected = np.outer(rows_total, cols_total) / total
                factor = (1.0 - rows_total / total)[:, None] * (1.0 - cols_total / total)[None, :]
                denom = np.sqrt(np.maximum(expected * factor, 0.0))
                with np.errstate(divide="ignore", invalid="ignore"):
                    residuals = np.where(denom > 0, (counts - expected) / denom, 0.0)
                residuals = np.nan_to_num(residuals)
                cells = [(i, j) for i in range(len(states_a)) for j in range(len(states_b))]
                p_values = [normal_two_sided_p(residuals[i, j]) for i, j in cells]
                q_values = am.bh_fdr(p_values) if use_fdr else p_values
                for (i, j), p_value, q_value in zip(cells, p_values, q_values):
                    rows.append({
                        "口径": "跨维", "lag": lag,
                        "dimension": "%s → %s" % (ctx.label_of(dimension_a),
                                                  ctx.label_of(dimension_b)),
                        "from": states_a[i], "to": states_b[j],
                        "O": int(counts[i, j]), "E": fmt_round(expected[i, j]),
                        "Z": fmt_round(residuals[i, j]), "p": fmt_round(p_value, 4),
                        "q": fmt_round(q_value, 4),
                        "significant": bool(abs(residuals[i, j]) > threshold),
                        "yules_Q": "",
                    })
                matrices.append({"口径": "跨维", "lag": lag,
                                 "dimension": "%s → %s" % (dimension_a, dimension_b),
                                 "table": "observed",
                                 "frame": pd.DataFrame(np.round(counts, 4), index=states_a,
                                                       columns=states_b)
                                 .rename_axis("from").reset_index()
                                 .rename(columns={"from": "状态"})})

    if not rows:
        return blocked("IND01", "滞后序列分析 LSA", "没有可检验的转移：序列太短或维度不足 2 个编码")
    for warning, dimension, lag in warnings:
        qa.append(QAItem(warning.level, warning.code,
                         "%s lag=%d：%s" % (dimension, lag, warning.message)))
    frame = pd.DataFrame(rows)
    significant = int(frame["significant"].sum())
    summary = pd.DataFrame([{
        "序列条数": len(sequences), "独立学生数": len({seq.uid for seq in sequences}),
        "检验格数": len(frame), "|Z|>%.2f 的格数" % threshold: significant,
        "最大|Z|": fmt_round(float(np.abs(pd.to_numeric(frame["Z"], errors="coerce")
                                          .fillna(0)).max())),
        "FDR 校正": "BH" if use_fdr else "未校正",
    }])
    tables = {"lsa_cells": frame, "summary": summary}
    for item in matrices:
        tables["matrix_%s_lag%d_%s" % (item["table"], item["lag"], item["dimension"])] = \
            item["frame"]
    notes = ["期望频数按去掉主对角线的行列合计计算（自转移不进 LSA 检验）",
             "q 为 Benjamini–Hochberg 校正后的 FDR，检验族 = 本次导出的全部单元格"]
    return am.result("IND01", "滞后序列分析 LSA", tables,
                     meta={"lags": list(range(1, max_lag + 1)), "scope": scope,
                           "threshold": threshold, "n_sequences": len(sequences),
                           "significant_cells": significant,
                           "unit": "student×dimension"},
                     qa=qa, notes=notes, elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ IND02 一阶 Markov

def markov_probabilities(counts):
    """P_ij = N_ij / Σ_j N_ij；返回 (P, 行合计)。整行为 0 的行给全 0（吸收态）。"""
    counts = np.asarray(counts, dtype=float)
    rows = counts.sum(axis=1)
    probabilities = np.zeros_like(counts)
    for position, total in enumerate(rows):
        if total > 0:
            probabilities[position] = counts[position] / total
    return probabilities, rows


def stationary_distribution(probabilities, iterations=4000, tol=1e-12):
    """平稳分布：幂迭代 + 微小随机扰动保证不可约（缺省分布退化时给极限占比口径）。"""
    matrix = np.asarray(probabilities, dtype=float)
    k = matrix.shape[0]
    if k == 0:
        return np.array([])
    aperiodic = matrix * (1.0 - 1e-9) + 1e-9 / k
    aperiodic = aperiodic / aperiodic.sum(axis=1, keepdims=True)
    vector = np.full(k, 1.0 / k)
    for _ in range(iterations):
        nxt = vector @ aperiodic
        if np.abs(nxt - vector).max() < tol:
            vector = nxt
            break
        vector = nxt
    total = vector.sum()
    return vector / total if total > 0 else np.full(k, 1.0 / k)


def markov_statistics(counts, initial=None):
    """一阶 Markov 的描述量：转移熵率、初始熵、变异性、平稳分布。"""
    states = np.asarray(counts, dtype=float)
    probabilities, rows = markov_probabilities(states)
    pi = stationary_distribution(probabilities)
    total = float(states.sum())
    entropy_rate = 0.0
    for i in range(states.shape[0]):
        if rows[i] <= 0:
            continue
        inner = 0.0
        for j in range(states.shape[1]):
            value = probabilities[i, j]
            if value > 0:
                inner -= value * math.log(value)
        weight = float(rows[i] / total) if total else 0.0
        entropy_rate += weight * inner
    initial = np.asarray(initial, dtype=float) if initial is not None else rows
    labels = [str(position) for position in range(states.shape[0])]
    expanded = [label for label, count in zip(labels, initial) for _ in range(int(count))]
    initial_entropy = am.entropy(expanded) if expanded and total > 0 else 0.0
    return {
        "probabilities": probabilities, "rows": rows, "stationary": pi,
        "entropy_rate": entropy_rate, "initial_entropy": initial_entropy,
        "ratio": (entropy_rate / initial_entropy) if initial_entropy > 0 else 0.0,
    }


def loglik_order(sequences, order):
    """order 阶马尔可夫的对数似然（含 context 未观测时的一级回退）。

    上下文 = 前 ``order`` 个有效状态；计数按序列内部滑窗，绝不跨序列。
    返回 (logLik, 自由参数数, 转移数, {context: {next: n}})。
    """
    contexts = {}
    transitions = 0
    for seq in sequences:
        values = valid_codes(seq)
        for position in range(order, len(values)):
            context = values[position - order:position]
            following = values[position]
            bucket = contexts.setdefault(tuple(context), {})
            bucket[following] = bucket.get(following, 0) + 1
            transitions += 1
    loglik = 0.0
    parameters = 0
    for bucket in contexts.values():
        total = sum(bucket.values())
        parameters += max(len(bucket) - 1, 0)
        for count in bucket.values():
            loglik += count * math.log(count / total)
    return loglik, parameters, transitions, contexts


def markov_first(ctx, params):
    """IND02 一阶 Markov：N_ij、P_ij、平稳分布、转移熵、一阶充分性 LR 检验。"""
    marker = _start()
    sequences = pick_sequences(ctx, params)
    include_self = bool(params.get("self_transition", True))
    qa, tables, notes = [], {}, []
    per_dimension = {}
    for dimension, group in by_dimension(sequences).items():
        states = states_of(group)
        _states, counts = pooled_counts(group, states, lag=1)
        if not include_self:
            counts = counts.copy()
            np.fill_diagonal(counts, 0.0)
        initial = np.array([sum(1 for seq in group if valid_codes(seq)[:1] == [state])
                            for state in states], dtype=float)
        stats = markov_statistics(counts, initial)
        loglik1, params1, transitions1, _c1 = loglik_order(group, 1)
        loglik2, params2, transitions2, _c2 = loglik_order(group, 2)
        degrees = max(params2 - params1, 0)
        statistic = max(2.0 * (loglik2 - loglik1), 0.0)
        p_value = chi2_sf(statistic, degrees) if degrees else 1.0
        per_dimension[dimension] = {
            "states": states, "counts": counts, "stats": stats,
            "loglik1": loglik1, "loglik2": loglik2, "statistic": statistic,
            "df": degrees, "p": p_value, "transitions": transitions1,
        }
        tables["counts_%s" % dimension] = matrix_frame(states, counts, "from")
        tables["probabilities_%s" % dimension] = matrix_frame(states,
                                                             stats["probabilities"], "from")
        tables["stationary_%s" % dimension] = pd.DataFrame({
            "状态": states,
            "平稳分布 π": [fmt_round(value, 4) for value in stats["stationary"]],
            "边际占比": [fmt_round(value / max(counts.sum(), 1.0), 4)
                         for value in counts.sum(axis=0)],
        })
        warning = _sparse_warning(counts, states)
        if warning is not None:
            qa.append(QAItem(warning.level, warning.code, "%s：%s" % (dimension, warning.message)))
    if not per_dimension:
        return blocked("IND02", "一阶马尔可夫链", "没有足够长的序列可估转移矩阵")
    summary = pd.DataFrame([{
        "dimension": ctx.label_of(dimension),
        "状态数": len(info["states"]),
        "转移数": int(info["transitions"]),
        "自转移占比%": fmt_round(
            float(np.trace(info["counts"])) / max(float(info["counts"].sum()), 1.0) * 100.0),
        "转移熵 H(X|prev)": fmt_round(info["stats"]["entropy_rate"], 4),
        "最大熵 lnK": fmt_round(math.log(max(len(info["states"]), 2)), 4),
        "熵率比": fmt_round(info["stats"]["ratio"], 4),
        "logLik(1阶)": fmt_round(info["loglik1"], 4),
        "logLik(2阶)": fmt_round(info["loglik2"], 4),
        "LR(一阶充分性)": fmt_round(info["statistic"], 4),
        "df": info["df"], "p": fmt_round(info["p"], 4),
        "一阶是否充分": ("是" if info["p"] > 0.05 else "否（需高阶）"),
    } for dimension, info in per_dimension.items()])
    tables["summary"] = summary
    notes += ["平稳分布含 1e-9 扰动以保证不可约，退化链上报的是极限占比而非严格 π",
              "LR = 2[logLik(2阶) − logLik(1阶)]，df = 自由参数差；上下文稀疏时该检验偏保守",
              "熵为 ln 口径"]
    return am.result("IND02", "一阶马尔可夫链", tables,
                     meta={"n_sequences": sum(len(group) for group in by_dimension(sequences).values()),
                           "self_transition": include_self,
                           "dimensions": list(per_dimension)},
                     qa=qa, notes=notes, elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ IND03 高阶 / 变阶

def context_lookup(contexts, context, order):
    """变阶回退：从最长上下文往短找，找到第一个见过的上下文（None = 全没见过）。"""
    for length in range(order, 0, -1):
        key = tuple(context[-length:]) if length <= len(context) else tuple(context)
        bucket = contexts.get(key)
        if bucket:
            return length, bucket
    return 0, {}


def higher_order_markov(ctx, params):
    """IND03：2–k 阶转移计数、AIC/BIC 选阶、下一状态命中率。"""
    marker = _start()
    sequences = pick_sequences(ctx, params, min_length=6)
    max_order = max(1, min(int(params.get("max_order") or 3), 5))
    criterion = params.get("criterion") or "BIC"
    mode = params.get("evaluation") or "backoff"
    qa, tables, rows = [], {}, []
    for dimension, group in by_dimension(sequences).items():
        states = states_of(group)
        k = max(len(states), 1)
        candidates = range(1, max_order + 1)
        best = None
        one_order = loglik_order(group, 1)[3]
        for order in candidates:
            loglik, parameters, transitions, contexts = loglik_order(group, order)
            if transitions <= 0:
                continue
            penalty = (2.0 * parameters if criterion == "AIC"
                       else parameters * math.log(max(transitions, 1)))
            information = -2.0 * loglik + penalty
            hit_rate = _hit_rate(group, contexts, order, mode, one_order)
            rows.append({
                "dimension": ctx.label_of(dimension), "order": order, "状态数": k,
                "上下文数": len(contexts), "转移数": int(transitions),
                "自由参数": parameters, "logLik": fmt_round(loglik, 4),
                "AIC": fmt_round(-2.0 * loglik + 2.0 * parameters, 4),
                "BIC": fmt_round(-2.0 * loglik + parameters * math.log(max(transitions, 1)), 4),
                "命中率%": fmt_round(hit_rate * 100.0),
            })
            score = information
            if best is None or score < best[0]:
                best = (score, order, contexts)
        if best is None:
            continue
        tables["tensor_%s" % dimension] = _tensor_frame(group, states, max_order)
        label = ctx.label_of(dimension)
        for row in rows:
            if row["dimension"] == label:
                row["选中阶"] = "★" if row["order"] == best[1] else ""
    if not rows:
        return blocked("IND03", "高阶 / 变阶马尔可夫", "序列太短，估不出 2 阶以上的上下文")
    frame = pd.DataFrame(rows)
    selected = frame[frame["选中阶"] == "★"]
    warning = QAItem("warn", "IND03.sparse",
                     "高阶上下文迅速稀疏：阶数越高「见过该上下文」的比例越低，"
                     "命中率上升不等于可解释性上升", False)
    qa.append(warning)
    tables["summary"] = frame
    return am.result("IND03", "高阶 / 变阶马尔可夫", tables,
                     meta={"criterion": criterion, "max_order": max_order,
                           "selected": {row["dimension"]: int(row["order"])
                                        for _, row in selected.iterrows()}},
                     qa=qa,
                     notes=["AIC/BIC 用同一批转移数算，跨阶可比；选阶只在给定序列集合内有效",
                            "命中率 = 用该阶模型预测下一条状态的比率（含回退到 1 阶）"],
                     elapsed_ms=_ms(marker))


def _hit_rate(group, contexts, order, mode, fallback=None):
    """下一状态命中率：上下文未观测时按 mode 回退（backoff → 降到见过的最长上下文）。"""
    fallback = fallback if fallback is not None else {}
    hit = 0
    tried = 0
    for seq in group:
        values = valid_codes(seq)
        for position in range(order, len(values)):
            context = values[:position]
            actual = values[position]
            if mode == "strict":
                bucket = contexts.get(tuple(context[-order:]), {})
            else:
                _length, bucket = context_lookup(contexts, context, order)
                if not bucket:
                    _length, bucket = context_lookup(fallback, context, 1)
            tried += 1
            if bucket and max(bucket, key=bucket.get) == actual:
                hit += 1
    return hit / tried if tried else 0.0


def _tensor_frame(group, states, max_order):
    """高阶转移张量的可读展开：context | next | n | P(next|context)。"""
    rows = []
    for order in range(1, max_order + 1):
        _loglik, _params, _total, contexts = loglik_order(group, order)
        for context, bucket in sorted(contexts.items()):
            total = sum(bucket.values())
            for following, count in sorted(bucket.items(), key=lambda item: -item[1]):
                rows.append({"dimension_order": order,
                             "context": "→".join(context), "next": following,
                             "n": int(count), "P(next|context)": fmt_round(count / total, 4)})
        if len(rows) > 4000:
            break
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ IND04 n-gram / PrefixSpan

def token_stream(sequence, on="raw"):
    """方法流的原料：raw = 去掉缺码的原始流；compressed = 连续相同码合并。"""
    values = valid_codes(sequence)
    if on == "compressed":
        out = []
        for code in values:
            if not out or out[-1] != code:
                out.append(code)
        return out
    return values


def ngram_table(sequences, n=2, min_sup=0.05, on="raw"):
    """n-gram：support = 含该 gram 的序列数 / 序列总数（**单元 = 学生，不是窗口**）。"""
    streams = [token_stream(seq, on) for seq in sequences]
    streams = [stream for stream in streams if len(stream) >= n]
    if not streams:
        return pd.DataFrame()
    windows, docs = {}, {}
    for position, stream in enumerate(streams):
        # 单字的文档集：算 lift 要的是 P(末字) 这个边际，不能只从窗口起点统计
        for token in set(stream):
            docs.setdefault((token,), set()).add(position)
        for start in range(len(stream) - n + 1):
            gram = tuple(stream[start:start + n])
            windows[gram] = windows.get(gram, 0) + 1
            docs.setdefault(gram, set()).add(position)
            for length in range(2, n):          # 前缀 (n-1)-gram：算 confidence 用
                docs.setdefault(gram[:length], set()).add(position)
    rows = []
    total = len(streams)
    for gram, count in windows.items():
        support = len(docs[gram]) / total
        if support + 1e-12 < float(min_sup):
            continue
        prefix, last = gram[:-1], gram[-1]
        prefix_support = len(docs.get(prefix, set())) / total if prefix else 1.0
        last_support = len(docs.get((last,), set())) / total
        confidence = support / prefix_support if prefix_support else 0.0
        lift = confidence / last_support if last_support else 0.0
        rows.append({
            "n": n, "gram": "→".join(gram), "序列数": len(docs[gram]), "总序列": total,
            "support": fmt_round(support, 4), "occurrences": int(count),
            "confidence": fmt_round(confidence, 4), "lift": fmt_round(lift, 4),
        })
    rows.sort(key=lambda item: (-item["support"], item["gram"]))
    return pd.DataFrame(rows)


PROJECT_CAP = 8      # 每个序列最多取前 8 个出现位置做投影：不封顶时 240 步会爆


def prefix_span(sequences, min_sup=0.1, max_length=4, on="raw"):
    """PrefixSpan 频繁子序列（可不连续）：support 按含该模式的序列数计。"""
    streams = [token_stream(seq, on) for seq in sequences]
    streams = [stream for stream in streams if stream]
    if not streams:
        return pd.DataFrame()
    threshold = max(int(math.ceil(float(min_sup) * len(streams))), 1)
    found = {}

    def walk(prefix, database, depth):
        counts = {}
        for sid, tokens in database:
            seen = set()
            for token in tokens:
                if token in seen:
                    continue
                seen.add(token)
                counts.setdefault(token, set()).add(sid)
        for item, sids in counts.items():
            if len(sids) < threshold:
                continue
            pattern = prefix + (item,)
            found[pattern] = len(sids)
            if depth >= max_length:
                continue
            projected = []
            for sid, tokens in database:
                hits = [pos for pos, token in enumerate(tokens) if token == item]
                for pos in hits[:PROJECT_CAP]:
                    if pos + 1 < len(tokens):
                        projected.append((sid, tokens[pos + 1:]))
            if projected:
                walk(pattern, projected, depth + 1)

    walk((), list(enumerate(streams)), 1)
    rows = []
    total = len(streams)
    for pattern, count in found.items():
        rows.append({"长度": len(pattern), "子序列": "→".join(pattern),
                     "序列数": int(count), "总序列": total,
                     "support": fmt_round(count / total, 4)})
    rows.sort(key=lambda item: (-item["序列数"], item["长度"], item["子序列"]))
    return pd.DataFrame(rows)



def frequent_patterns(ctx, params):
    """IND04：n-gram（连续窗口）+ PrefixSpan（可跳的频繁子序列）。"""
    marker = _start()
    sequences = pick_sequences(ctx, params)
    max_n = max(2, min(int(params.get("max_n") or 3), 6))
    min_sup = float(params.get("min_sup") or 0.1)
    on = "compressed" if params.get("stream") == "compressed" else "raw"
    algorithm = params.get("algorithm") or "both"
    tables, notes, qa = {}, [], []
    if algorithm in ("ngram", "both"):
        frames = [frame for frame in
                  (ngram_table(sequences, n, min_sup, on) for n in range(2, max_n + 1))
                  if len(frame)]
        if frames:
            tables["ngrams"] = pd.concat(frames, ignore_index=True)
    if algorithm in ("span", "both"):
        mined = prefix_span(sequences, min_sup, max_n, on)
        if len(mined):
            tables["subsequences"] = mined
    if not tables:
        return blocked("IND04", "n-gram / 频繁子序列",
                       "当前 min_sup 下没有任何频繁模式，调小阈值或换原始/压缩口径")
    total = len(sequences)
    counts = {name: int(len(frame)) for name, frame in tables.items()}
    tables["summary"] = pd.DataFrame([{
        "序列条数": total, "独立学生数": len({seq.uid for seq in sequences}),
        "形态": "压缩" if on == "compressed" else "原始",
        "n-gram 条数": counts.get("ngrams", 0),
        "子序列条数": counts.get("subsequences", 0),
        "min_sup": fmt_round(min_sup, 3),
    }])
    if total < 10:
        qa.append(QAItem("warn", "IND04.small_n",
                         "只有 %d 条序列，support 的粒度是 1/%d，分档会很粗"
                         % (total, total)))
    notes += ["support 的分母是序列数（同一序列里出现多次只计 1），occurrences 才是次数",
              "PrefixSpan 每条序列最多取 %d 个出现位置做投影（超出会静默截断投影）" % PROJECT_CAP]
    return am.result("IND04", "n-gram / 频繁子序列", tables,
                     meta={"max_n": max_n, "min_sup": min_sup, "stream": on,
                           "algorithm": algorithm, "n_sequences": total},
                     qa=qa, notes=notes, elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ IND05 距离与多通道

def parse_custom_costs(text, states):
    """自定义替换代价：`A,B=1.5; B,C=2`，两个单字码也可写 `AB=1.5`；未给出的非对角格默认 1。"""
    matrix = np.ones((len(states), len(states)), dtype=float)
    np.fill_diagonal(matrix, 0.0)
    known = {state: position for position, state in enumerate(states)}
    if not text:
        return matrix
    for chunk in str(text).replace("；", ";").replace("，", ",").split(";"):
        if "=" not in chunk:
            continue
        pair, _, value = chunk.partition("=")
        try:
            cost = float(value)
        except ValueError:
            continue
        left, comma, right = pair.partition(",")
        if not comma:
            letters = [char for char in left.strip() if not char.isspace()]
            if len(letters) != 2:             # 只认两个单字码并写（AB），其余一律忽略
                continue
            left, right = letters[0], letters[1]
        left, right = left.strip(), right.strip()
        if left in known and right in known:
            matrix[known[left], known[right]] = matrix[known[right], known[left]] = cost
    return matrix


def transition_rate_costs(sequences, states):
    """TRATE：d(i,j) = 2 − p(i→j) − p(j→i)（Gabadinho 等，值域 [0,2]，i=j 时为 0）。"""
    _states, counts = pooled_counts(sequences, states, lag=1)
    rows = counts.sum(axis=1)
    probabilities = np.zeros_like(counts)
    for position, total in enumerate(rows):
        if total > 0:
            probabilities[position] = counts[position] / total
    cost = 2.0 - probabilities - probabilities.T
    np.fill_diagonal(cost, 0.0)
    return np.clip(cost, 0.0, 2.0)


def substitution_costs(sequences, states, kind="constant", value=1.0, custom=""):
    """三种口径：常数 / TRATE / 自定义（计划 §3 IND05）。"""
    if kind == "trate":
        return transition_rate_costs(sequences, states)
    if kind == "custom":
        return parse_custom_costs(custom, states)
    matrix = np.full((len(states), len(states)), float(value), dtype=float)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def om_distance(left, right, cost, index, indel=1.0):
    """Needleman–Wunsch 最优匹配：cost 为替换代价矩阵，indel 为插入/删除代价。"""
    a = [index[code] for code in left]
    b = [index[code] for code in right]
    n, m = len(a), len(b)
    previous = np.arange(m + 1, dtype=float) * indel
    for i in range(1, n + 1):
        current = np.empty(m + 1, dtype=float)
        current[0] = i * indel
        row = cost[a[i - 1]]
        for j in range(1, m + 1):
            current[j] = min(previous[j - 1] + row[b[j - 1]],
                             previous[j] + indel,
                             current[j - 1] + indel)
        previous = current
    return float(previous[m])


def hamming_distance(left, right, cost, index):
    """HAM：同位置逐格代价相加（不等长时只比公共前缀，并在 notes 里写清楚）。"""
    total = 0.0
    for a, b in zip(left, right):
        total += cost[index[a], index[b]]
    return float(total)


def channel_streams(tracks, dimensions, compress=False):
    """{uid: {dim: [码]}} → [(uid, 每通道码表)]；时间栅格已经对齐。"""
    out = []
    for uid in sorted(tracks):
        streams = []
        for dimension in dimensions:
            values = [code for code in tracks[uid].get(dimension, []) if code]
            if compress:
                merged = []
                for code in values:
                    if not merged or merged[-1] != code:
                        merged.append(code)
                values = merged
            streams.append(values)
        out.append((uid, streams))
    return out


def multichannel_distance(a_streams, b_streams, indel=1.0, max_cells=400000):
    """多通道 CAT（MCO）：联合状态的最优匹配，逐通道代价取均值。

    各通道长度可以不同（缺码被去掉），匹配到第 i 个**有效**时点；超出 DP 规模上限时
    报 NaN 而不是静默给个错的数。
    """
    channels = min(len(a_streams), len(b_streams))
    if channels == 0:
        return 0.0
    left = [list(stream) for stream in a_streams[:channels]]
    right = [list(stream) for stream in b_streams[:channels]]
    n = max(len(stream) for stream in left)
    m = max(len(stream) for stream in right)
    if n * m > max_cells:
        return float("nan")

    def pair_cost(i, j):
        acc = 0.0
        for channel in range(channels):
            if i < len(left[channel]) and j < len(right[channel]):
                acc += 0.0 if left[channel][i] == right[channel][j] else 1.0
        return acc / channels

    previous = np.arange(m + 1, dtype=float) * indel
    for i in range(1, n + 1):
        current = np.empty(m + 1, dtype=float)
        current[0] = i * indel
        for j in range(1, m + 1):
            current[j] = min(previous[j - 1] + pair_cost(i - 1, j - 1),
                             previous[j] + indel,
                             current[j - 1] + indel)
        previous = current
    return float(previous[m])


def sequence_index(tracks, dimensions, times):
    """多通道 MD 的 sequence index：把各通道的 run 按发生时刻排成并行时间轴事件流。

    元素 = (通道, 码, 持续时点数)；空码不形成事件（它不是状态，是缺测）。
    ``times`` 省略时以**时点序号**充当时刻（多通道 MD 只关心事件先后，不关心秒数）。
    """
    elements = []
    for position, dimension in enumerate(dimensions):
        codes = tracks.get(dimension, [])
        run_code, run_start = None, 0
        for step in range(len(codes) + 1):
            code = codes[step] if step < len(codes) else None
            if code == run_code:
                continue
            if run_code:
                moment = float(times[run_start]) if times and run_start < len(times) \
                    else float(run_start)
                elements.append((moment, position, run_code, step - run_start))
            run_code, run_start = code, step
    elements.sort(key=lambda item: (item[0], item[1]))
    return [(dimensions[position], code, length) for _time, position, code, length in elements]


def index_distance(a, b, channels_of):
    """sequence index 距离（TraMineR seqMD 同口径）：

    同通道不同状态 2/k，不同通道 1/k + 1/k'，同一元素 0；indel = 1。k 为该状态出现的通道数。
    """
    n, m = len(a), len(b)
    previous = np.arange(m + 1, dtype=float)
    for i in range(1, n + 1):
        current = np.empty(m + 1, dtype=float)
        current[0] = float(i)
        channel_a, code_a, _len_a = a[i - 1]
        for j in range(1, m + 1):
            channel_b, code_b, _len_b = b[j - 1]
            if channel_a == channel_b and code_a == code_b:
                cost = 0.0
            elif channel_a == channel_b:
                cost = 2.0 / channels_of.get(code_a, 1)
            else:
                cost = 1.0 / channels_of.get(code_a, 1) + 1.0 / channels_of.get(code_b, 1)
            current[j] = min(previous[j - 1] + cost, previous[j] + 1.0, current[j - 1] + 1.0)
        previous = current
    return float(previous[m])


def build_distance_matrix(sequences, params, ctx=None):
    """统一入口：返回 (标签, 方阵, 说辞)。标签用 student_uid，单组内不歧义。"""
    method = (params.get("method") or "OM").upper()
    compress = bool(params.get("compress", True))
    raw_ratio = params.get("indel_ratio")
    ratio = 0.5 if raw_ratio in (None, "") else float(raw_ratio)
    labels = [seq.uid + " / " + seq.dimension for seq in sequences]
    if method in ("MCO", "MCM"):
        if ctx is None:
            raise ValueError("多通道距离需要 AnalysisContext")
        dimensions = sorted({seq.dimension for seq in sequences})
        students = {seq.uid for seq in sequences}
        tracks = aligned_channels(ctx, students, dimensions)
        uids = [uid for uid in sorted(tracks) if uid in students]
        if len(uids) < 2:
            raise ValueError("多通道距离需要 ≥ 2 名学生的对齐码轨")
        matrix = np.zeros((len(uids), len(uids)), dtype=float)
        indel = max(ratio, 1e-9)
        if method == "MCM":
            indexes = {uid: sequence_index(tracks[uid], dimensions, None) for uid in uids}
            channels_of = {}
            for dimension in dimensions:
                for code in ctx.codebook_of(dimension):
                    channels_of[code] = channels_of.get(code, 0) + 1
            for elements in indexes.values():
                for _dim, code, _length in elements:
                    channels_of.setdefault(code, 1)
            for i, j in itertools.combinations(range(len(uids)), 2):
                value = index_distance(indexes[uids[i]], indexes[uids[j]], channels_of)
                matrix[i, j] = matrix[j, i] = value
            return uids, matrix, "多通道 MD（sequence index）"
        streams = {uid: channel_streams({uid: tracks[uid]}, dimensions, compress)[0][1]
                   for uid in uids}
        for i, j in itertools.combinations(range(len(uids)), 2):
            value = multichannel_distance(streams[uids[i]], streams[uids[j]], indel)
            matrix[i, j] = matrix[j, i] = value
        return uids, matrix, "多通道 CAT（联合状态 OM）"
    states = states_of(sequences)
    if method in ("DTW", "SDTW"):
        streams = [token_stream(seq, "compressed" if compress else "raw")
                   for seq in sequences]
        matrices = [one_hot_matrix(values, states) for values in streams]
        gamma = float(params.get("soft_gamma") or 1.0) if method == "SDTW" else 0.0
        matrix = np.zeros((len(matrices), len(matrices)), dtype=float)
        for i, j in itertools.combinations(range(len(matrices)), 2):
            value = dtw_distance(matrices[i], matrices[j], gamma)
            matrix[i, j] = matrix[j, i] = value
        note = ("soft-DTW（γ=%g）" % gamma) if method == "SDTW" else \
            "DTW（one-hot 指示向量 + 欧氏局部代价，不做长度归一化）"
        return labels, matrix, note
    cost = substitution_costs(sequences, states,
                              kind=params.get("subst") or "constant",
                              value=float(params.get("cost_value") or 1.0),
                              custom=params.get("custom_cost") or "")
    indel = ratio * (float(np.nanmax(cost)) or 1.0)
    streams = []
    for seq in sequences:
        values = token_stream(seq, "compressed" if compress else "raw")
        streams.append(values)
    index = {state: position for position, state in enumerate(states)}
    matrix = np.zeros((len(streams), len(streams)), dtype=float)
    for i, j in itertools.combinations(range(len(streams)), 2):
        if method == "HAM":
            value = hamming_distance(streams[i], streams[j], cost, index)
        elif method == "DHD":
            unit = np.ones((len(states), len(states)))
            np.fill_diagonal(unit, 0.0)
            value = om_distance(streams[i], streams[j], unit, index, 1.0)
        else:
            value = om_distance(streams[i], streams[j], cost, index, max(indel, 1e-9))
        matrix[i, j] = matrix[j, i] = value
    note = {"OM": "最优匹配 OM", "HAM": "Hamming（逐时点）",
            "DHD": "动态 Hamming DHD",
            "DTW": "DTW（one-hot + 欧氏局部代价）",
            "SDTW": "soft-DTW"}.get(method, method)
    return labels, matrix, note


def distance_analysis(ctx, params):
    """IND05：序列间距离矩阵（OM / HAM / DHD / 多通道 CAT / 多通道 MD）。"""
    marker = _start()
    sequences = pick_sequences(ctx, params)
    if len(sequences) < 2:
        return blocked("IND05", "序列距离 OM/HAM/DHD", "至少需要 2 条有效序列才能算距离")
    try:
        labels, matrix, note = build_distance_matrix(sequences, params, ctx)
    except ValueError as error:
        return blocked("IND05", "序列距离 OM/HAM/DHD", str(error))
    frame = pd.DataFrame(np.round(matrix, 4), index=labels, columns=labels)
    frame.insert(0, "序列", labels)
    off = matrix[np.triu_indices(len(labels), 1)]
    tables = {"distance_matrix": frame.reset_index(drop=True),
              "pairs": pd.DataFrame([{
                  "a": labels[i], "b": labels[j], "d": fmt_round(matrix[i, j], 4)}
                  for i, j in itertools.combinations(range(len(labels)), 2)])
                  .sort_values("d", ascending=False).reset_index(drop=True)}
    notes = ["距离在%s上计算，%s" % ("压缩序列" if params.get("compress", True) else "原始序列", note),
             "indel 代价 = indel_ratio × 最大替换代价（TraMineR 默认 0.5）"]
    if np.isnan(off).any():
        notes.append("部分序列对太长，多通道 DP 已跳过（报 NaN）：请改压缩口径或减少维度")
    return am.result("IND05", "序列距离 OM/HAM/DHD", tables,
                     meta={"method": (params.get("method") or "OM").upper(),
                           "n_sequences": len(labels), "k_states": len(states_of(sequences)),
                           "mean_distance": fmt_round(float(np.nanmean(off)), 4),
                           "max_distance": fmt_round(float(np.nanmax(off)), 4)},
                     notes=notes, elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ IND07 序列聚类

def _choose(value, k):
    return math.comb(int(value), k) if value >= k else 0


def adjusted_rand_index(first, second):
    """ARI：对两个划分共现矩阵做随机基线校正，1 = 完全一致，0 = 随机水平。"""
    first = list(first)
    second = list(second)
    if len(first) != len(second) or not first:
        return 0.0
    pairs = {}
    for a, b in zip(first, second):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
    rows = {}
    cols = {}
    for (a, b), value in pairs.items():
        rows[a] = rows.get(a, 0) + value
        cols[b] = cols.get(b, 0) + value
    total = _choose(len(first), 2)
    if total <= 0:
        return 0.0
    index = sum(_choose(value, 2) for value in pairs.values())
    expected = sum(_choose(value, 2) for value in rows.values()) \
        * sum(_choose(value, 2) for value in cols.values()) / total
    max_index = 0.5 * (sum(_choose(value, 2) for value in rows.values())
                       + sum(_choose(value, 2) for value in cols.values()))
    denominator = max_index - expected
    if abs(denominator) < 1e-12:
        return 1.0 if abs(index - expected) < 1e-12 else 0.0
    return (index - expected) / denominator


def co_clustering_jaccard(first, second):
    """配对 Jaccard：同簇对的交集 / 并集（不校正随机基线，与 ARI 并列看）。"""
    first, second = list(first), list(second)
    if len(first) != len(second) or not first:
        return 0.0
    both = 0
    only = 0
    for i, j in itertools.combinations(range(len(first)), 2):
        same_a = first[i] == first[j]
        same_b = second[i] == second[j]
        if same_a and same_b:
            both += 1
        elif same_a or same_b:
            only += 1
    union = both + only
    return both / union if union else 1.0


def silhouette_scores(dist, labels):
    """轮廓系数（直接吃距离阵，不需原始坐标）：返回逐点值与均值。"""
    labels = list(labels)
    n = len(labels)
    unique = sorted(set(labels))
    if n < 3 or len(unique) < 2:
        return [0.0] * n, 0.0
    values = []
    for index in range(n):
        own = [position for position in range(n)
               if labels[position] == labels[index] and position != index]
        if not own:
            values.append(0.0)
            continue
        a = float(np.mean([dist[index, position] for position in own]))
        b = np.inf
        for cluster in unique:
            if cluster == labels[index]:
                continue
            others = [position for position in range(n) if labels[position] == cluster]
            if others:
                b = min(b, float(np.mean([dist[index, position] for position in others])))
        if not np.isfinite(b):
            values.append(0.0)
            continue
        denominator = max(a, b)
        values.append((b - a) / denominator if denominator > 0 else 0.0)
    return values, float(np.mean(values))


def ahc_clusters(dist, linkage="average", k=3):
    """层次聚类（scipy 的实现，与 R 的 hclust 同一算法族）：返回标签与合并矩阵。

    负相异度按 0 截断：soft-DTW 这类 soft-min 距离本身可取负，而 scipy 校验合并
    高度必须 ≥ 0（``Linkage 'Z' contains negative distances``）。截断是单调变换，
    不改变正值之间的大小关系，只把负值并到 0 —— 上报的距离阵仍是原始值。
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    matrix = np.array(dist, dtype=float, copy=True)          # 不回写调用方的阵
    np.fill_diagonal(matrix, 0.0)
    matrix = (matrix + matrix.T) / 2.0
    condensed = squareform(matrix)
    if condensed.size and not np.all(np.isfinite(condensed)):
        raise ValueError("距离阵含 NaN，无法建树：请改用压缩序列或减少维度")
    condensed = np.clip(condensed, 0.0, None)
    method = {"single": "single", "complete": "complete",
              "average": "average", "ward": "ward", "centroid": "centroid"}.get(
        linkage, "average")
    tree = linkage(condensed, method=method)
    labels = fcluster(tree, t=max(int(k), 2), criterion="maxclust")
    return [int(value) for value in labels], tree


def pam_clusters(dist, k, max_iter=60):
    """k-medoids (PAM)：Build 阶段取“总距离增益最大”贪心，再跑 Swap。自实现，不引额外包。"""
    matrix = np.asarray(dist, dtype=float)
    n = matrix.shape[0]
    k = max(min(int(k), n - 1), 2)
    medoids = [int(np.argmin(matrix.sum(axis=1)))]
    while len(medoids) < k:
        best, gain = -1, -np.inf
        for candidate in range(n):
            if candidate in medoids:
                continue
            nearest = matrix[:, medoids].min(axis=1)
            delta = np.maximum(nearest - matrix[:, candidate], 0.0).sum()
            if delta > gain:
                best, gain = candidate, delta
        if best < 0:
            break
        medoids.append(best)
    labels = _assign(matrix, medoids)
    for _ in range(max_iter):
        moved = False
        # 总代价 = 每个点到最近 medoid 的距离之和；matrix[i, medoids] 才是那一行的子阵
        base = float(matrix[:, medoids].min(axis=1).sum())
        for position, medoid in enumerate(medoids):
            members = [index for index in range(n)
                       if labels[index] == medoid and index != medoid]
            improvement, best = 0.0, medoid
            for candidate in members:
                trial = list(medoids)
                trial[position] = candidate
                cost = float(matrix[:, trial].min(axis=1).sum())
                if cost < base - improvement:
                    improvement, best = base - cost, candidate
            if best != medoid:
                medoids[position] = best
                base -= improvement
                moved = True
        labels = _assign(matrix, medoids)
        if not moved:
            break
    remap = {medoid: index + 1 for index, medoid in enumerate(sorted(medoids))}
    return [remap[medoid] for medoid in labels], sorted(medoids)


def _assign(matrix, medoids):
    sub = matrix[:, medoids]
    return [int(medoids[position]) for position in np.argmin(sub, axis=1)]


def bootstrap_stability(dist, labels, reps=50, seed=20240517):
    """bootstrap 稳定性：重抽序列→重新聚类→与全样本解比较 ARI / Jaccard。"""
    matrix = np.asarray(dist, dtype=float)
    n = matrix.shape[0]
    rng = np.random.default_rng(int(seed))
    target = [int(value) for value in labels]
    ari, jac, ok = [], [], 0
    for _ in range(max(int(reps), 0)):
        picked = rng.integers(0, n, size=n)
        if len(set(picked.tolist())) < 3:
            continue
        sub = matrix[np.ix_(picked, picked)]
        try:
            _tree, resampled = _cluster_like(target, sub)
        except Exception:                       # noqa: BLE001  子样本退化就跳过
            continue
        ok += 1
        ari.append(adjusted_rand_index([target[i] for i in picked], resampled))
        jac.append(co_clustering_jaccard([target[i] for i in picked], resampled))
    if not ok:
        return {"reps": 0, "ari": 0.0, "jaccard": 0.0}
    return {"reps": ok, "ari": float(np.mean(ari)), "ari_min": float(np.min(ari)),
            "jaccard": float(np.mean(jac))}


def _cluster_like(reference, sub_matrix):
    """按全样本解的簇数在子样本上重跑同一方法（这里统一用 AHC-average）。"""
    k = max(len(set(reference)), 2)
    return None, ahc_clusters(sub_matrix, "average", k)


def sequence_clustering(ctx, params):
    """IND07：在 IND05 的距离阵上做 AHC / k-medoids，报 silhouette、簇规模、bootstrap。"""
    marker = _start()
    sequences = pick_sequences(ctx, params)
    if len(sequences) < 4:
        return blocked("IND07", "序列聚类", "至少需要 4 条序列，簇结果才有意义")
    try:
        labels, matrix, note = build_distance_matrix(sequences, params, ctx)
        method = (params.get("cluster_method") or "ahc").lower()
        k = max(2, min(int(params.get("k") or 3), len(labels) - 1))
        if method == "pam":
            clusters, medoids = pam_clusters(matrix, k)
        else:
            clusters, tree = ahc_clusters(matrix, params.get("linkage") or "average", k)
            medoids = []
            tree_matrix = tree
    except ValueError as error:
        return blocked("IND07", "序列聚类", str(error))
    scores, mean_silhouette = silhouette_scores(matrix, clusters)
    rows = []
    for index, uid in enumerate(labels):
        rows.append({"序列": uid, "簇": clusters[index],
                     "silhouette": fmt_round(scores[index], 4),
                     "长度": len(valid_codes(sequences[index])) if index < len(sequences) else ""})
    detail = pd.DataFrame(rows)
    sizes = detail["簇"].value_counts().sort_index()
    medoid_of = {}
    if method == "pam" and medoids:
        for cluster in sorted(set(clusters)):
            members = [index for index in range(len(labels)) if clusters[index] == cluster]
            medoid_of[cluster] = labels[min(members, key=lambda i: sum(
                matrix[i, j] for j in members))]
    summary = pd.DataFrame([{
        "簇": int(cluster), "规模": int(sizes.get(cluster, 0)),
        "占比%": fmt_round(int(sizes.get(cluster, 0)) / len(labels) * 100.0),
        "簇内均距": fmt_round(float(np.mean([
            matrix[i, j] for i in range(len(labels)) for j in range(i + 1, len(labels))
            if clusters[i] == cluster and clusters[j] == cluster])))
        if int(sizes.get(cluster, 0)) > 1 else 0.0,
        "代表序列": medoid_of.get(cluster, ""),
    } for cluster in sorted(set(clusters))])
    tables = {"clusters": detail, "summary": summary,
              "distance_matrix": pd.DataFrame(np.round(matrix, 4), index=labels, columns=labels)
              .rename_axis("序列").reset_index()}
    meta = {"k": k, "method": method, "mean_silhouette": fmt_round(mean_silhouette, 4),
            "distance": note, "n": len(labels)}
    if params.get("stability", True):
        stability = bootstrap_stability(matrix, clusters, int(params.get("reps") or 50))
        tables["stability"] = pd.DataFrame([{
            "重复次数": stability["reps"],
            "ARI 均值": fmt_round(stability["ari"], 4),
            "ARI 最小值": fmt_round(stability.get("ari_min", 0.0), 4),
            "配对 Jaccard": fmt_round(stability["jaccard"], 4)}])
        meta["bootstrap_ari"] = stability["ari"]
        if stability["reps"] and stability["ari"] < 0.5:
            qa = [QAItem("warn", "IND07.unstable",
                         "bootstrap ARI = %.2f < 0.5：簇结构对抽样敏感，不建议报簇名" %
                         stability["ari"])]
        else:
            qa = []
    else:
        qa = []
    if mean_silhouette < 0.1:
        qa.append(QAItem("warn", "IND07.weak_silhouette",
                         "平均 silhouette = %.3f，分开度很弱" % mean_silhouette))
    offdiag = matrix[np.triu_indices(len(labels), 1)]
    if offdiag.size and float(np.nanmin(offdiag)) < 0:
        qa.append(QAItem("warn", "IND07.negative_distance",
                         "距离阵有 %d 个负值（最小 %.4f）：soft-DTW 的 soft-min 会让距离为负，"
                         "建树时已按 0 截断（表里的距离仍是原值）" %
                         (int((offdiag < 0).sum()), float(np.nanmin(offdiag)))))
    if "tree_matrix" in locals():
        tables["dendrogram"] = pd.DataFrame(tree_matrix,
                                           columns=["合并a", "合并b", "高度", "样本数"])
    return am.result("IND07", "序列聚类 AHC / k-medoids", tables, meta=meta, qa=qa,
                     notes=["距离：%s" % note,
                            "ward 只在欧氏距离上有意义；OM/HAM 距离请用 average/complete"],
                     elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ IND06 DTW

def one_hot_matrix(values, states):
    """码流 → n × k 指示矩阵：局部代价由行向量的欧氏距离给出（同码 0、异码 √2）。"""
    index = {state: position for position, state in enumerate(states)}
    matrix = np.zeros((max(len(values), 0), len(states)), dtype=float)
    for row, code in enumerate(values):
        position = index.get(code)
        if position is not None:
            matrix[row, position] = 1.0
    return matrix


def local_cost_matrix(left, right):
    """逐时点局部代价矩阵（one-hot 行之间的欧氏距离）。"""
    return np.sqrt(((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=2))


def soft_min(values, gamma):
    """soft-min（Cuturi–Douc）：−γ·log Σ exp(−v/γ)，恒 ≤ 硬 min，γ→0 时退化；只对有穷的值参与。"""
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return float("inf")
    if gamma is None or float(gamma) <= 0:
        return min(finite)
    gamma = float(gamma)
    scaled = -np.asarray(finite, dtype=float) / gamma
    highest = float(scaled.max())
    return float(-gamma * (highest + np.log(np.exp(scaled - highest).sum())))


def dtw_cost_matrix(left, right, gamma=0.0):
    """DTW 累积代价表：D[0,0]=c(0,0)，其余 c(i,j)+pool(左上、上、左)（越界视为 Inf）。"""
    n, m = len(left), len(right)
    table = np.full((n, m), np.inf, dtype=float)
    if not n or not m:
        return table
    cost = local_cost_matrix(left, right)
    table[0, 0] = cost[0, 0]
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            neighbours = []
            if i > 0 and j > 0:
                neighbours.append(table[i - 1, j - 1])
            if i > 0:
                neighbours.append(table[i - 1, j])
            if j > 0:
                neighbours.append(table[i, j - 1])
            table[i, j] = cost[i, j] + soft_min(neighbours, gamma)
    return table


def dtw_distance(left, right, gamma=0.0):
    """经典 / soft DTW 距离（**不做长度归一化**，与 r_refs/dtwclust_ref.R 同径）。

    soft（γ > 0）把累加表的 min 换成 soft-min，所以 soft-DTW ≤ 经典 DTW。
    """
    table = dtw_cost_matrix(left, right, gamma)
    if not table.size:
        return float("nan")
    return float(table[-1, -1])


def dtw_path(table):
    """自后向前回溯硬 min 的规整路径：[(序列A 下标, 序列B 下标)]。"""
    n, m = table.shape
    if not n or not m:
        return []
    i, j, path = n - 1, m - 1, []
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        options = []
        if i > 0 and j > 0:
            options.append((table[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            options.append((table[i - 1, j], i - 1, j))
        if j > 0:
            options.append((table[i, j - 1], i, j - 1))
        _value, i, j = min(options, key=lambda item: item[0])
    path.reverse()
    return path


def dtw_barycenter(matrices, states, iterations=3):
    """DBA 重心（plan §3 IND06）：以总距离最小的序列起手，逐轮按规整路径摊码质量。"""
    matrices = [matrix for matrix in matrices if len(matrix)]
    if not matrices:
        return np.zeros((0, len(states)), dtype=float)
    if len(matrices) == 1:
        return matrices[0].astype(float).copy()
    totals = [sum(dtw_distance(a, b) for b in matrices) for a in matrices]
    center = matrices[int(np.argmin(totals))].astype(float).copy()
    for _round in range(max(1, int(iterations))):
        acc = np.zeros(center.shape, dtype=float)
        hits = np.zeros(len(center), dtype=float)
        for matrix in matrices:
            table = dtw_cost_matrix(matrix, center)
            for i, j in dtw_path(table):
                acc[j] += matrix[i]
                hits[j] += 1.0
        keep = hits > 0
        if not keep.any():
            break
        center = acc[keep] / hits[keep][:, None]
    return center


def _distance_frame(labels, matrix):
    frame = pd.DataFrame(np.round(matrix, 4), index=labels, columns=labels)
    frame.insert(0, "序列", labels)
    return frame.reset_index(drop=True)


def _pair_frame(labels, matrix):
    return pd.DataFrame([{"a": labels[i], "b": labels[j], "d": fmt_round(matrix[i, j], 4)}
                         for i, j in itertools.combinations(range(len(labels)), 2)]) \
        .sort_values("d", ascending=False).reset_index(drop=True)


def dtw_analysis(ctx, params):
    """IND06：类别序列的 DTW 距离（one-hot 局部代价）+ soft-DTW + DBA 重心。"""
    marker = _start()
    sequences = pick_sequences(ctx, params)
    if len(sequences) < 2:
        return blocked("IND06", "DTW 序列距离", "至少需要 2 条有效序列才能算 DTW 距离")
    states = states_of(sequences)
    if len(states) < 2:
        return blocked("IND06", "DTW 序列距离", "状态不足 2 个：所有序列同一个码，距离恒为 0")
    on = "compressed" if params.get("compress", True) else "raw"
    streams = [token_stream(seq, on) for seq in sequences]
    matrices = [one_hot_matrix(values, states) for values in streams]
    labels = [seq.uid + " / " + seq.dimension for seq in sequences]
    matrix = np.zeros((len(matrices), len(matrices)), dtype=float)
    for i, j in itertools.combinations(range(len(matrices)), 2):
        value = dtw_distance(matrices[i], matrices[j])
        matrix[i, j] = matrix[j, i] = value
    tables = {"distance_matrix": _distance_frame(labels, matrix),
              "pairs": _pair_frame(labels, matrix)}
    meta = {"method": "DTW", "n_sequences": len(labels), "k_states": len(states),
            "stream": on,
            "mean_distance": fmt_round(float(np.nanmean(matrix[np.triu_indices(len(labels), 1)])), 4),
            "max_distance": fmt_round(float(np.nanmax(matrix)), 4)}
    notes = ["局部代价 = one-hot 行向量的欧氏距离（同码 0、异码 √2），累积代价**不做长度归一化**",
             "距离在%s序列上算" % ("压缩（连续同码合并）" if on == "compressed" else "原始（去缺码）")]
    if params.get("soft_dtw"):
        gamma = float(params.get("soft_gamma") or 1.0)
        soft = np.zeros((len(matrices), len(matrices)), dtype=float)
        for i, j in itertools.combinations(range(len(matrices)), 2):
            value = dtw_distance(matrices[i], matrices[j], gamma)
            soft[i, j] = soft[j, i] = value
        tables["soft_distance_matrix"] = _distance_frame(labels, soft)
        meta["soft_gamma"] = fmt_round(gamma, 4)
        notes.append("soft-DTW（γ=%g）：soft-min 可微，恒 ≤ 经典 DTW（γ→0 时收回硬 min）" % gamma)
    if params.get("barycenter", True):
        center = dtw_barycenter(matrices, states, int(params.get("ba_iterations") or 3))
        if len(center):
            frame = pd.DataFrame(np.round(center, 4), columns=list(states))
            frame.insert(0, "重心位置", [index + 1 for index in range(len(frame))])
            frame["主导码"] = [states[position] for position in center.argmax(axis=1)]
            tables["barycenter"] = frame
            tables["to_barycenter"] = pd.DataFrame({
                "序列": labels,
                "到重心距离": [fmt_round(dtw_distance(matrix_one, center), 4)
                                 for matrix_one in matrices]})
            meta["barycenter_length"] = int(len(center))
            notes.append("重心（DBA）长度 %d：把整批序列规整到同一条时间轴上" % len(center))
    return am.result("IND06", "DTW 序列距离", tables, meta=meta, notes=notes,
                     elapsed_ms=_ms(marker))


def dimension_barycenters(ctx, params):
    """{维度: (状态表, 重心矩阵)}：阶段 7 的 `dtw_to_centroid` 特征直接吃这个。"""
    out = {}
    for dimension, group in by_dimension(pick_sequences(ctx, params)).items():
        states = states_of(group)
        if len(states) < 2:
            continue
        on = "compressed" if params.get("compress", True) else "raw"
        matrices = [one_hot_matrix(token_stream(seq, on), states) for seq in group]
        center = dtw_barycenter(matrices, states, int(params.get("ba_iterations") or 3))
        if len(center):
            out[dimension] = (states, center)
    return out


# ------------------------------------------------------------------ IND08 HMM / MHMM

FLOOR = 1e-300            # 对数域下限：与 r_refs/seqhmm_fit.R 里的 1e-300 同一取值


def hmm_emission(codes, symbols, B):
    """码流 → n × S 发射概率矩阵（字母表外的码直接丢弃，与 R 侧 `stream %in% symbols` 同径）。"""
    index = {symbol: position for position, symbol in enumerate(symbols)}
    hits = [index[code] for code in codes if code in index]
    if not hits:
        return np.zeros((0, np.asarray(B).shape[0]), dtype=float)
    return np.asarray(B, dtype=float)[:, hits].T


def hmm_inference(emit, pi, A):
    """带缩放因子的前向后向 + 对数域 Viterbi（逐式照搬 r_refs/seqhmm_fit.R）。

    ``emit[t, s]`` = P(o_t | 状态 s)：单通道时取自 B 的一列，多通道时是各维独立发射的乘积。
    返回 ``{loglik, gamma, path, alpha, beta, scales, degenerate}``；首步就概率为 0 时
    ``degenerate=True`` 并把 logLik 记成 −Inf（与 R 一致，不拿一个似是而非的数蒙人）。
    """
    emit = np.asarray(emit, dtype=float)
    n, k = emit.shape
    if n == 0 or k == 0:
        return None
    pi = np.asarray(pi, dtype=float)
    A = np.asarray(A, dtype=float)
    alpha = np.zeros((n, k), dtype=float)
    beta = np.zeros((n, k), dtype=float)
    scales = np.zeros(n, dtype=float)
    alpha[0] = pi * emit[0]
    scales[0] = float(alpha[0].sum())
    if scales[0] <= 0:
        return {"loglik": float("-inf"), "gamma": np.zeros((0, k)), "path": [],
                "alpha": alpha, "beta": beta, "scales": scales, "degenerate": True}
    alpha[0] /= scales[0]
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ A) * emit[t]
        scales[t] = float(alpha[t].sum())
        alpha[t] = alpha[t] / scales[t] if scales[t] > 0 else 0.0
    beta[n - 1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (A @ (emit[t + 1] * beta[t + 1])) / max(scales[t + 1], FLOOR)
    loglik = float(np.log(np.maximum(scales, FLOOR)).sum())
    gamma = np.zeros((n, k), dtype=float)
    for t in range(n):
        weights = alpha[t] * beta[t]
        total = float(weights.sum())
        if total > 0:
            gamma[t] = weights / total
    log_emit = np.log(np.maximum(emit, FLOOR))
    log_A = np.log(np.maximum(A, FLOOR))
    delta = np.full((n, k), -np.inf, dtype=float)
    psi = np.zeros((n, k), dtype=int)
    delta[0] = np.log(np.maximum(pi, FLOOR)) + log_emit[0]
    for t in range(1, n):
        for j in range(k):
            values = delta[t - 1] + log_A[:, j]
            psi[t, j] = int(np.argmax(values))
            delta[t, j] = float(np.max(values)) + log_emit[t, j]
    path = [0] * n
    path[n - 1] = int(np.argmax(delta[n - 1]))
    for t in range(n - 2, -1, -1):
        path[t] = int(psi[t + 1, path[t + 1]])
    return {"loglik": loglik, "gamma": gamma, "path": path, "alpha": alpha,
            "beta": beta, "scales": scales, "degenerate": False}


def symbol_maps(channel_symbols):
    """每个通道的 {码: 列号}（空码不进表，所以缺测不会被当成一个状态）。"""
    return [{symbol: position for position, symbol in enumerate(symbols)}
            for symbols in channel_symbols]


def xhmm_emit(sequence_channels, maps, Bs):
    """各维独立发射 → ``(emit, hits)``：emit 是 n × S 矩阵，hits 是每通道对齐后的列号。

    某通道缺码 → 该通道这一行不乘；所有通道都缺的时点直接丢掉（它没有观测）。
    单通道时 ``emit`` 就是 B 按观测码取列转置，与 `hmm_emission` 等价。
    """
    states = np.asarray(Bs[0], dtype=float).shape[0]
    length = min((len(channel) for channel in sequence_channels), default=0)
    if not length:
        return np.zeros((0, states), dtype=float), [[] for _ in sequence_channels]
    emit = np.ones((length, states), dtype=float)
    live = np.zeros(length, dtype=float)
    raw = []
    for position, channel in enumerate(sequence_channels):
        index, matrix = maps[position], np.asarray(Bs[position], dtype=float)
        column = []
        for step in range(length):
            found = index.get(channel[step])
            if found is None:
                column.append(-1)
                continue
            emit[step] *= matrix[:, found]
            live[step] += 1.0
            column.append(found)
        raw.append(column)
    keep = [step for step in range(length) if live[step] > 0]
    if not keep:
        return np.zeros((0, states), dtype=float), [[] for _ in sequence_channels]
    hits = [[column[step] for step in keep] for column in raw]
    return emit[keep], hits


def xhmm_em_fit(series, channel_symbols, n_states, seed=20240517, max_iter=100, tol=1e-4):
    """Baum–Welch（多通道通用）：各维独立发射，logLik = Σ_序列 Σ_t ln c_t。

    ``series`` = [每条序列的【各通道码表】]；``channel_symbols`` = 每个通道的字母表。
    返回 ``(pi, A, [B_通道], info)``；初值由 seed 固定，同一数据永远同一个模型。
    """
    k = max(int(n_states), 1)
    maps = symbol_maps(channel_symbols)
    rng = np.random.default_rng(int(seed))
    pi = np.full(k, 1.0 / k, dtype=float)
    trans = np.full((k, k), 1.0 / k, dtype=float) + rng.random((k, k)) * 0.1
    A = trans / trans.sum(axis=1, keepdims=True)
    Bs = []
    for symbols in channel_symbols:
        raw = rng.random((k, max(len(symbols), 1))) + 0.1
        Bs.append(raw / raw.sum(axis=1, keepdims=True))
    prepared = []
    for sequence in series:
        cut = min((len(channel) for channel in sequence), default=0)
        prepared.append([list(channel[:cut]) for channel in sequence] if cut else [])
    n_obs = sum(len(channel[0]) for channel in prepared if channel)
    previous, iterations, loglik = 0.0, 0, 0.0
    for iteration in range(max(int(max_iter), 1)):
        iterations = iteration + 1
        acc_pi = np.zeros(k, dtype=float)
        acc_A = np.zeros((k, k), dtype=float)
        acc_B = [np.zeros_like(matrix) for matrix in Bs]
        loglik = 0.0
        for sequence in prepared:
            if not sequence:
                continue
            emit, hits = xhmm_emit(sequence, maps, Bs)
            if not len(emit):
                continue
            out = hmm_inference(emit, pi, A)
            if out is None or out["degenerate"]:
                continue
            loglik += out["loglik"]
            gamma, alpha, beta, scales = out["gamma"], out["alpha"], out["beta"], out["scales"]
            acc_pi += gamma[0]
            for step in range(len(emit) - 1):
                # ξ_t(i,j) = α_t(i)·A_ij·b_j(o_{t+1})·β_{t+1}(j) / c_{t+1}
                acc_A += (alpha[step][:, None] * A
                          * (emit[step + 1] * beta[step + 1])[None, :]) \
                    / max(scales[step + 1], FLOOR)
            for position, column in enumerate(hits):
                for step, symbol in enumerate(column):
                    if symbol >= 0:
                        acc_B[position][:, symbol] += gamma[step]
        if acc_pi.sum() <= 0:
            break
        pi = acc_pi / acc_pi.sum()
        A = acc_A / np.maximum(acc_A.sum(axis=1, keepdims=True), FLOOR)
        Bs = [acc / np.maximum(acc.sum(axis=1, keepdims=True), FLOOR) for acc in acc_B]
        converged = abs(loglik - previous) < float(tol)
        previous = loglik
        if converged:
            break
    return pi, A, Bs, {"loglik": float(loglik), "iterations": iterations,
                       "n_obs": int(n_obs)}


def hmm_em_fit(streams, symbols, n_states, seed=20240517, max_iter=100, tol=1e-4):
    """单通道离散 HMM 的 EM：只返 (pi, A, B)，手算单测直接对着它。"""
    pi, A, Bs, _info = xhmm_em_fit([[stream] for stream in streams], [list(symbols)],
                                   n_states, seed=seed, max_iter=max_iter, tol=tol)
    return pi, A, Bs[0]


def hmmlearn_fit(streams, symbols, n_states, seed=20240517, max_iter=100, tol=1e-4):
    """hmmlearn 的 CategoricalHMM（核心依赖）：装不上时抛 ImportError，上层回退自实现。"""
    from hmmlearn.hmm import CategoricalHMM        # noqa: WPS433  可选后端，用到才导
    index = {symbol: position for position, symbol in enumerate(symbols)}
    rows, lengths = [], []
    for stream in streams:
        picked = [index[code] for code in stream if code in index]
        if picked:
            rows.extend(picked)
            lengths.append(len(picked))
    if not rows:
        raise ValueError("观测全不在字母表内，hmmlearn 无从拟合")
    k = int(n_states)
    rng = np.random.default_rng(int(seed))
    model = CategoricalHMM(n_components=k, n_iter=int(max_iter), tol=float(tol),
                           random_state=int(seed), init_params="", params="ste")
    model.startprob_ = np.full(k, 1.0 / k)
    trans = np.full((k, k), 1.0 / k) + rng.random((k, k)) * 0.1
    model.transmat_ = trans / trans.sum(axis=1, keepdims=True)
    emis = rng.random((k, len(symbols))) + 0.1
    model.emissionprob_ = emis / emis.sum(axis=1, keepdims=True)
    model.fit(np.asarray(rows).reshape(-1, 1), lengths)
    return (np.asarray(model.startprob_, dtype=float), np.asarray(model.transmat_, dtype=float),
            np.asarray(model.emissionprob_, dtype=float))


def hmm_criteria(loglik, n_states, alphabet_sizes, n_obs):
    """参数量 m = (S−1) + S(S−1) + Σ_通道 S(V−1)；AIC = −2lnL+2m，BIC = −2lnL+m·lnN。"""
    free = int(n_states) - 1 + int(n_states) * (int(n_states) - 1) \
        + sum(int(n_states) * (int(size) - 1) for size in alphabet_sizes)
    base = float(loglik) if np.isfinite(loglik) else 0.0
    penalty_n = math.log(n_obs) if n_obs and n_obs > 1 else 0.0
    return free, -2.0 * base + 2.0 * free, -2.0 * base + free * penalty_n


def state_labels(n_states):
    """隐状态名：H1…Hk（避开学生号 S1，两套东西同名会很坑）。"""
    return ["H%d" % (position + 1) for position in range(int(n_states))]


def hmm_fit_candidate(series, channel_symbols, n_states, engine, seed, max_iter, tol):
    """按引擎拟合一个候选模型 → ``(pi, A, Bs, logLik, 观测数, EM 轮数, 实际引擎)``。"""
    maps = symbol_maps(channel_symbols)
    Bs, rounds, engine_used = None, 0, engine
    if len(channel_symbols) == 1 and engine == "hmmlearn":
        try:
            pi, A, single = hmmlearn_fit([item[0] for item in series], channel_symbols[0],
                                         n_states, seed=seed, max_iter=max_iter, tol=tol)
            Bs, engine_used = [single], "hmmlearn"
        except ImportError:
            engine_used = "em"            # 没装 hmmlearn：回退自实现，不拿半截结果骗人
    if Bs is None:
        pi, A, Bs, info = xhmm_em_fit(series, channel_symbols, n_states, seed=seed,
                                      max_iter=max_iter, tol=tol)
        rounds = int(info["iterations"])
    loglik, n_obs = 0.0, 0
    for item in series:
        emit, _hits = xhmm_emit(item, maps, Bs)
        n_obs += len(emit)
        if not len(emit):
            continue
        out = hmm_inference(emit, pi, A)
        if out is None or out["degenerate"]:
            continue
        loglik += out["loglik"]
    return pi, A, Bs, float(loglik), int(n_obs), rounds, engine_used


def _single_channel_series(group, on):
    """单通道形态：每条序列就是一个通道的码流，标签与 `sequences_frame` 一致。"""
    series = [[token_stream(seq, on)] for seq in group]
    labels = ["%s / %s" % (seq.uid, seq.dimension) for seq in group]
    return series, labels


def _mhmm_series(ctx, group, params, on):
    """多通道形态：选中维度当并行通道（时间栅格已对齐），返 (series, 字母表, 标签, 通道名)。"""
    channels = [ctx.dimension_key(item) for item in
                (params.get("dimensions") or ctx.dimensions)]
    students = sorted({seq.uid for seq in group})
    tracks = aligned_channels(ctx, set(students), channels)
    series, labels = [], []
    for uid in students:
        item = [list(tracks.get(uid, {}).get(dimension, [])) for dimension in channels]
        if min((len(channel) for channel in item), default=0) < 2:
            continue
        length = min(len(channel) for channel in item)
        series.append([channel[:length] for channel in item])
        labels.append("%s / 多通道" % uid)
    if not series:
        return [], [], [], []
    pools = []
    for position in range(len(channels)):
        pool = []
        for item in series:
            for code in item[position]:
                if code and code not in pool:
                    pool.append(code)
        pools.append(pool)
    keep = [position for position, pool in enumerate(pools) if pool]
    if len(keep) < 2:
        return [], [], [], labels
    return ([[item[position] for position in keep] for item in series],
            [pools[position] for position in keep],
            [labels[index] for index in range(len(series))],
            [channels[position] for position in keep])


def hmm_analysis(ctx, params):
    """IND08：离散 HMM / 多通道 MHMM（隐状态、A/B/π、posterior、logLik、AIC/BIC 选状态数）。"""
    marker = _start()
    model_kind = (params.get("model") or "hmm").lower()
    chosen = pick_sequences(ctx, params)
    if not chosen:
        return blocked("IND08", "隐马尔可夫模型",
                       "没有足够长的序列（有效码 < %d 个）" % am.MIN_SEQUENCE_LENGTH)
    on = "compressed" if params.get("compress", True) else "raw"
    engine = (params.get("engine") or "em").lower()
    max_iter = int(params.get("max_iter") or 100)
    tol = float(params.get("tol") or 1e-4)
    seed = int(params.get("seed") or 20240517)
    criterion = (params.get("criterion") or "BIC").upper()
    low = max(2, int(params.get("k_min") or 2))
    high = max(low, int(params.get("k_max") or low + 2))
    candidates = list(range(low, min(high, low + 5) + 1)) if params.get("select_states") \
        else [max(2, int(params.get("n_states") or 2))]
    selection, qa, engines = [], [], set()
    state_rows, emission_rows, transition_rows = [], [], []
    occupancy_rows, path_rows, posterior_rows, loglik_rows = [], [], [], []
    model_store = {}
    for dimension, group in by_dimension(chosen).items():
        if model_kind == "mhmm":
            series, channel_symbols, labels, channel_names = _mhmm_series(
                ctx, group, params, on)
        else:
            series, labels = _single_channel_series(group, on)
            channel_symbols, channel_names = [states_of(group)], [dimension]
        if not series or not channel_symbols or not channel_symbols[0]:
            continue
        fits = []
        floor = max(min((len(channel) for channel in item), default=0) for item in series)
        for n_states in candidates:
            if floor < n_states * 10:      # 计划 §3 门槛：序列长度 ≥ 状态数×10
                continue
            try:
                pi, A, Bs, loglik, n_obs, rounds, engine_used = hmm_fit_candidate(
                    series, channel_symbols, n_states, engine, seed, max_iter, tol)
            except Exception as error:                    # noqa: BLE001  拟合失败只记质检
                qa.append(QAItem("warn", "IND08.fit_failed",
                                 "%s：k=%d 拟合失败（%s）" % (ctx.label_of(dimension),
                                                            n_states, error)))
                continue
            engines.add(engine_used)
            free, aic, bic = hmm_criteria(loglik, n_states,
                                          [len(item) for item in channel_symbols],
                                          max(n_obs, 2))
            fits.append({"k": int(n_states), "pi": pi, "A": A, "Bs": Bs, "loglik": loglik,
                         "aic": aic, "bic": bic, "free": free, "n_obs": n_obs,
                         "rounds": rounds})
        if not fits:
            qa.append(QAItem("warn", "IND08.too_short",
                             "%s：最长序列只有 %d 步，不够「状态数×10」，本轮未拟合"
                             % (ctx.label_of(dimension), floor)))
            continue
        best = min(fits, key=lambda item: item["aic"] if criterion == "AIC" else item["bic"])
        for item in fits:
            selection.append({"维度": ctx.label_of(dimension), "状态数": item["k"],
                              "logLik": fmt_round(item["loglik"], 6), "参数数": item["free"],
                              "AIC": fmt_round(item["aic"], 4), "BIC": fmt_round(item["bic"], 4),
                              "观测数": item["n_obs"], "EM 轮数": item["rounds"],
                              "选用": int(item is best)})
        states = state_labels(best["k"])
        pi, A, Bs = best["pi"], best["A"], best["Bs"]
        model_store[dimension] = (states, channel_symbols, Bs, pi, A)
        maps = symbol_maps(channel_symbols)
        for i, state in enumerate(states):
            state_rows.append({"维度": ctx.label_of(dimension), "state": state,
                               "稳态分布": fmt_round(float(pi[i]), 4),
                               "主导发射码": channel_symbols[0][int(np.argmax(Bs[0][i]))],
                               "最大发射概率": fmt_round(float(np.max(Bs[0][i])), 4)})
            for j, target in enumerate(states):
                transition_rows.append({"维度": ctx.label_of(dimension), "from": state,
                                        "to": target, "p": fmt_round(float(A[i, j]), 6)})
            for position, matrix in enumerate(Bs):
                for j, symbol in enumerate(channel_symbols[position]):
                    emission_rows.append({"维度": ctx.label_of(dimension),
                                          "通道": ctx.label_of(channel_names[position]),
                                          "state": state, "symbol": symbol,
                                          "p": fmt_round(float(matrix[i, j]), 6)})
        for item, label in zip(series, labels):
            emit, hits = xhmm_emit(item, maps, Bs)
            out = hmm_inference(emit, pi, A)
            if out is None or out["degenerate"]:
                continue
            gamma, path = out["gamma"], out["path"]
            loglik_rows.append({"label": label, "logLik": fmt_round(out["loglik"], 6),
                                "时点数": int(len(emit))})
            row = {"label": label}
            means = gamma.mean(axis=0)
            for position, state in enumerate(states):
                row[state] = fmt_round(float(means[position]), 4)
            row["状态切换次数"] = int(sum(1 for left, right in zip(path[:-1], path[1:])
                                       if left != right))
            row["状态熵"] = fmt_round(am.entropy([int(value) for value in path]), 4)
            occupancy_rows.append(row)
            for step, state in enumerate(path):
                observed = "|".join(channel_symbols[position][column[step]]
                                    for position, column in enumerate(hits)
                                    if position < len(channel_symbols)
                                    and step < len(column) and column[step] >= 0)
                path_rows.append({"label": label, "position": step, "观测": observed,
                                  "state": states[state]})
            for step in range(len(gamma)):
                for position, state in enumerate(states):
                    posterior_rows.append({"label": label, "position": step, "state": state,
                                           "posterior": fmt_round(float(gamma[step, position]),
                                                                   6)})
    if not model_store:
        detail = "；".join(item.message for item in qa[:3])
        return blocked("IND08", "隐马尔可夫模型",
                       "没有维度能拟合 HMM：%s" % (detail or "序列都太短（长度 < 状态数×10）"))
    notes = []
    tables = {"model_selection": pd.DataFrame(selection),
              "states": pd.DataFrame(state_rows),
              "emission": pd.DataFrame(emission_rows),
              "transition": pd.DataFrame(transition_rows),
              "occupancy": pd.DataFrame(occupancy_rows),
              "loglik": pd.DataFrame(loglik_rows)}
    limit = int(params.get("max_rows") or 2000)
    if posterior_rows:
        frame = pd.DataFrame(posterior_rows)
        if len(frame) > limit:
            notes.append("逐时点后验只列前 %d 行（共 %d 行），统计量与对照不受影响"
                         % (limit, len(frame)))
            frame = frame.head(limit)
        tables["posterior"] = frame
    if path_rows:
        tables["viterbi"] = pd.DataFrame(path_rows[:limit])
    notes.append("拟合引擎：%s；推断引擎：缩放前向后向（与 r_refs/seqhmm_fit.R 逐式同径）"
                 % (" + ".join(sorted(engines))))
    notes.append("状态数为候选内按 %s 选取；参数量 m = (S−1) + S(S−1) + Σ S(V−1)" % criterion)
    if model_kind == "mhmm":
        notes.append("多通道 MHMM = 各维独立发射（观测概率为各通道之积）；"
                     "R 参照只覆盖单通道模型，本口径不做逐格对照")
    else:
        notes.append("单通道 HMM：每个维度一套模型，逐时点表贴 “学生 / 维度” 标签")
    return am.result("IND08", "隐马尔可夫模型 HMM / MHMM", tables,
                     meta={"model": model_kind, "engine": "+".join(sorted(engines)),
                           "criterion": criterion, "stream": on,
                           "states_per_dimension": {ctx.label_of(item[0]): len(item[1][0])
                                                    for item in model_store.items()},
                           "hmm_model": _hmm_model_tables(model_store)},
                     qa=qa, notes=notes, elapsed_ms=_ms(marker))


def _hmm_model_tables(model_store):
    """模型参数 → 桥接层的 model_pi / model_A / model_B（带 dimension 列，逐维取对应模型）。"""
    pi_rows, a_rows, b_rows = [], [], []
    for dimension, (states, channel_symbols, Bs, pi, A) in model_store.items():
        for i, state in enumerate(states):
            pi_rows.append({"dimension": dimension, "state": state, "pi": float(pi[i])})
            for j, target in enumerate(states):
                a_rows.append({"dimension": dimension, "from": state, "to": target,
                               "p": float(A[i, j])})
            for position, matrix in enumerate(Bs):
                for j, symbol in enumerate(channel_symbols[position]):
                    b_rows.append({"dimension": dimension, "channel": position,
                                   "state": state, "symbol": symbol,
                                   "p": float(matrix[i, j])})
    return {"pi": pd.DataFrame(pi_rows), "A": pd.DataFrame(a_rows),
            "B": pd.DataFrame(b_rows)}


# ------------------------------------------------------------------ IND09 变点 / 分段

def code_distribution(values, states):
    """码表 → 各状态频率（和为 1；空表给均匀分布，避免 Hellinger 除零）。"""
    counts = np.asarray([values.count(state) for state in states], dtype=float)
    total = float(counts.sum())
    if total <= 0:
        return np.full(len(states), 1.0 / max(len(states), 1), dtype=float)
    return counts / total


def hellinger_distance(first, second):
    """H(p,q) = √(½ Σ (√p − √q)²)：值域 [0,1]，两个分布相同为 0。"""
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(left) - np.sqrt(right)) ** 2)))


def window_hellinger(values, states, window=5):
    """滑窗分布距离轨迹：第 i 行 = 窗 [i, i+w) 与 [i+w, i+2w) 的 Hellinger 距离。"""
    rows = []
    length = len(values)
    window = max(int(window), 1)
    for start in range(0, max(length - 2 * window, 0) + 1):
        left = code_distribution(values[start:start + window], states)
        right = code_distribution(values[start + window:start + 2 * window], states)
        rows.append({"位置": start + window, "窗口": window,
                     "Hellinger": fmt_round(hellinger_distance(left, right), 4),
                     "左窗主导码": states[int(np.argmax(left))],
                     "右窗主导码": states[int(np.argmax(right))]})
    return pd.DataFrame(rows)


def multinomial_loglik(values, states):
    """多项式 MLE 的对数似然：Σ n_k ln(n_k / N)（n_k = 0 的项记 0）。"""
    if not values:
        return 0.0
    total = float(len(values))
    acc = 0.0
    for state in states:
        count = values.count(state)
        if count > 0:
            acc += count * math.log(count / total)
    return acc


def bic_split_gain(values, states, cut, penalty=1.0):
    """在 cut 处一分为二的 ΔBIC/2：越大越值得分。额外参数按 (K−1)·ln N 计。"""
    if cut <= 0 or cut >= len(values):
        return -np.inf
    left = values[:cut]
    right = values[cut:]
    gain = multinomial_loglik(left, states) + multinomial_loglik(right, states) \
        - multinomial_loglik(values, states)
    free = max(len({code for code in values}) - 1, 1)
    return gain - penalty * 0.5 * free * math.log(max(len(values), 2))


def find_changepoints(values, states, min_segment=5, penalty=1.0, max_points=6):
    """递归二分变点检测：每段取 ΔBIC 最大的切点，不显著（≤0）或段太短就停。"""
    cuts = []

    def search(lo, hi):
        segment = values[lo:hi]
        if hi - lo < 2 * max(int(min_segment), 1) or len(cuts) >= int(max_points):
            return
        best, gain = None, 0.0
        for offset in range(int(min_segment), hi - lo - int(min_segment) + 1):
            value = bic_split_gain(segment, states, offset, penalty)
            if value > gain:
                best, gain = offset, value
        if best is None:
            return
        cuts.append(lo + best)
        search(lo, lo + best)
        search(lo + best, hi)

    if len(values) >= 2 * max(int(min_segment), 1):
        search(0, len(values))
    return sorted(cuts)


def segmentation_analysis(ctx, params):
    """IND09：变点 / 分段（滑窗 Hellinger 诊断 + 多项式 BIC 递归二分）。"""
    marker = _start()
    chosen = pick_sequences(ctx, params)
    if not chosen:
        return blocked("IND09", "变点 / 分段", "没有足够长的序列可分段")
    min_segment = max(int(params.get("min_segment") or 5), 2)
    window = max(int(params.get("window") or 5), 1)
    penalty = float(params.get("penalty") or 1.0)
    rows, segment_rows, trace = [], [], []
    for seq in chosen:
        values = token_stream(seq, "compressed" if params.get("compress", True) else "raw")
        states = states_of([seq])
        label = "%s / %s" % (seq.uid, seq.dimension)
        if window and len(values) >= 2 * window:
            part = window_hellinger(values, states, window)
            part.insert(0, "label", label)
            trace.append(part)
        cuts = find_changepoints(values, states, min_segment, penalty) \
            if params.get("detect", True) else []
        bounds = [0] + cuts + [len(values)]
        for number, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
            segment = values[start:end]
            distribution = code_distribution(segment, states)
            segment_rows.append({"label": label, "段号": number,
                                 "起": start + 1, "止": end,
                                 "段长": len(segment),
                                 "段内熵": fmt_round(am.entropy(segment), 4),
                                 "归一化熵": fmt_round(am.normalized_entropy(segment), 4),
                                 "主导码": states[int(np.argmax(distribution))],
                                 "主导占比": fmt_round(float(np.max(distribution)), 4)})
        for position, cut in enumerate(cuts, 1):
            rows.append({"label": label, "变点序号": position, "位置": cut + 1,
                         "前窗Hellinger": "", "后窗Hellinger": ""})
        for row in rows[len(rows) - len(cuts):]:
            offset = row["位置"] - 1
            if window and offset >= window and offset + window <= len(values):
                left = code_distribution(values[offset - window:offset], states)
                right = code_distribution(values[offset:offset + window], states)
                row["前窗Hellinger"] = fmt_round(hellinger_distance(left, right), 4)
                row["后窗Hellinger"] = row["前窗Hellinger"]
    if not segment_rows:
        return blocked("IND09", "变点 / 分段", "序列太短：连一段都切不出来")
    tables = {"segments": pd.DataFrame(segment_rows)}
    if rows:
        tables["changepoints"] = pd.DataFrame(rows)
    if trace:
        tables["hellinger"] = pd.concat(trace, ignore_index=True)
    total_segments = len(segment_rows)
    counts = pd.DataFrame(segment_rows).groupby("label")["段号"].max()
    return am.result("IND09", "变点 / 分段", tables,
                     meta={"n_sequences": int(len(counts)),
                           "n_segments": int(total_segments),
                           "mean_segments_per_series": fmt_round(
                               float(np.mean(counts.tolist())), 4),
                           "min_segment": min_segment, "penalty": penalty},
                     notes=["分段准则：递归二分 + BIC 惩罚（每多一段罚 0.5·(K−1)·lnN）",
                            "Hellinger 滑窗只是诊断轨迹，不参与切点判定（两者口径独立可对照）",
                            "段长下限 = %d 个时点；变点上限 = 每条序列 %d 个" %
                            (min_segment, int(params.get("max_points") or 6))],
                     elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ 方法声明

P_MAX_LAG = ParamSpec("max_lag", "最大滞后期", "spin", default=3, minimum=1, maximum=5,
                      tooltip="lag=k 含义：前一个状态取第 t 个有效编码，后一个取第 t+k 个")
P_Z = ParamSpec("z_threshold", "|Z| 阈值", "dspin", default=1.96, minimum=0.5, maximum=5.0,
                tooltip="1.96 ⇔ 双侧 α=0.05")
P_LSA_SCOPE = ParamSpec("scope", "口径", "combo",
                        options=(("same", "同维（前状态→后状态）"),
                                 ("cross", "跨维（维度A_t → 维度B_t+k）"),
                                 ("both", "两种都要")), default="same")
P_FDR = ParamSpec("fdr", "BH-FDR 校正", "check", default=True)
P_SELF = ParamSpec("self_transition", "计入自转移", "check", default=True)
P_ORDER = ParamSpec("max_order", "最高阶数", "spin", default=3, minimum=1, maximum=5)
P_CRITERION = ParamSpec("criterion", "选阶准则", "combo",
                        options=(("BIC", "BIC（保守）"), ("AIC", "AIC（宽松）")), default="BIC")
P_EVAL = ParamSpec("evaluation", "预测口径", "combo",
                   options=(("backoff", "变阶回退"), ("strict", "严格看本阶上下文")),
                   default="backoff")
P_N = ParamSpec("max_n", "最长 n-gram", "spin", default=3, minimum=2, maximum=6)
P_MINE = ParamSpec("min_sup", "最小 support", "dspin", default=0.1, minimum=0.01, maximum=1.0,
                   tooltip="分母 = 参与本次分析的序列条数，不是窗口数")
P_STREAM = ParamSpec("stream", "方法流形态", "combo",
                     options=(("raw", "原始（去缺码）"), ("compressed", "连续相同码合并")),
                     default="raw")
P_ALGO = ParamSpec("algorithm", "挖掘算法", "combo",
                   options=(("both", "n-gram + PrefixSpan"), ("ngram", "仅 n-gram（连续）"),
                            ("span", "仅 PrefixSpan（可跳）")), default="both")
P_DIST_METHOD = ParamSpec("method", "距离", "combo",
                          options=(("OM", "最优匹配 OM"), ("HAM", "Hamming"),
                                   ("DHD", "动态 Hamming DHD"),
                                   ("MCO", "多通道 CAT"), ("MCM", "多通道 MD（序列索引）"),
                                   ("DTW", "动态时间规整 DTW"),
                                   ("SDTW", "soft-DTW")),
                          default="OM")
P_SUBST = ParamSpec("subst", "替换代价", "combo",
                    options=(("constant", "常数"), ("trate", "转移率 TRATE"),
                             ("custom", "自定义")), default="constant",
                    shows_for=("method", "OM"))
P_COST_VALUE = ParamSpec("cost_value", "常数代价", "dspin", default=1.0, minimum=0.01,
                         maximum=10.0, shows_for=("subst", "constant"))
P_CUSTOM_COST = ParamSpec("custom_cost", "自定义代价", "text", default="",
                          tooltip="形如 A,B=1.5; A,C=2（未给出的非对角格默认 1）",
                          shows_for=("subst", "custom"))
P_INDEL = ParamSpec("indel_ratio", "indel 比例", "dspin", default=0.5, minimum=0.0,
                    maximum=2.0,
                    tooltip="indel 代价 = 此比例 × 最大替换代价（TraMineR 默认 0.5）")
P_COMPRESS = ParamSpec("compress", "在压缩序列上算", "check", default=True)
P_CLUSTER_METHOD = ParamSpec("cluster_method", "聚类方法", "combo",
                             options=(("ahc", "层次聚类 AHC"), ("pam", "k-medoids PAM")),
                             default="ahc")
P_K = ParamSpec("k", "簇数 k", "spin", default=3, minimum=2, maximum=10)
P_LINKAGE = ParamSpec("linkage", "链接方式", "combo",
                      options=(("average", "平均"), ("complete", "完全"),
                               ("single", "最短"), ("ward", "Ward")), default="average",
                      shows_for=("cluster_method", "ahc"))
P_STABILITY = ParamSpec("stability", "bootstrap 稳定性", "check", default=True)
P_REPS = ParamSpec("reps", "重抽次数", "spin", default=50, minimum=10, maximum=500,
                   shows_for=("stability", True))
P_SOFT_DTW = ParamSpec("soft_dtw", "加算 soft-DTW", "check", default=False,
                       tooltip="soft-min 版本可微，对噪声不如硬 min 敏感；需 dtwclust 参照时一起对照")
P_SOFT_GAMMA = ParamSpec("soft_gamma", "soft-min γ", "dspin", default=1.0, minimum=0.01,
                         maximum=10.0, shows_for=("soft_dtw", True),
                         tooltip="γ→0 退化为经典 DTW；与 r_refs/dtwclust_ref.R 默认 1.0 一致")
P_BARYCENTER = ParamSpec("barycenter", "算 DBA 重心", "check", default=True)
P_BA_ITER = ParamSpec("ba_iterations", "重心迭代轮数", "spin", default=3, minimum=1, maximum=10,
                      shows_for=("barycenter", True))
P_HMM_MODEL = ParamSpec("model", "模型形态", "combo",
                        options=(("hmm", "单通道 HMM"), ("mhmm", "多通道 MHMM（各维独立发射）")),
                        default="hmm")
P_HMM_ENGINE = ParamSpec("engine", "拟合引擎", "combo",
                         options=(("em", "自实现 Baum–Welch"), ("hmmlearn", "hmmlearn CategoricalHMM")),
                         default="em",
                         tooltip="hmmlearn 只能拟合单通道；推断（前向后向/Viterbi）两路都是自实现")
P_HMM_STATES = ParamSpec("n_states", "隐状态数 k", "spin", default=3, minimum=2, maximum=8)
P_HMM_SELECT = ParamSpec("select_states", "自动选状态数", "check", default=True,
                         tooltip="逐 k 拟合，按 AIC/BIC 选最好的那个")
P_HMM_KMIN = ParamSpec("k_min", "状态数下限", "spin", default=2, minimum=2, maximum=8,
                       shows_for=("select_states", True))
P_HMM_KMAX = ParamSpec("k_max", "状态数上限", "spin", default=4, minimum=2, maximum=10,
                       shows_for=("select_states", True))
P_HMM_CRITERION = ParamSpec("criterion", "选状态数准则", "combo",
                            options=(("BIC", "BIC（保守）"), ("AIC", "AIC（宽松）")), default="BIC",
                            shows_for=("select_states", True))
P_HMM_ITER = ParamSpec("max_iter", "EM 最多轮数", "spin", default=100, minimum=10, maximum=1000)
P_HMM_TOL = ParamSpec("tol", "EM 收敛阈值", "dspin", default=0.0001, minimum=0.000001,
                      maximum=0.01,
                      tooltip="相邻两轮 logLik 变化小于此值即判收敛")
P_HMM_SEED = ParamSpec("seed", "初值随机种子", "spin", default=20240517, minimum=0,
                       maximum=999999999)
P_HMM_ROWS = ParamSpec("max_rows", "逐时点表最多行数", "spin", default=2000, minimum=200,
                       maximum=20000)
P_SEGMENT_WINDOW = ParamSpec("window", "Hellinger 滑窗长", "spin", default=5, minimum=2,
                             maximum=40)
P_MIN_SEGMENT = ParamSpec("min_segment", "最短段长", "spin", default=5, minimum=2, maximum=60)
P_PENALTY = ParamSpec("penalty", "BIC 惩罚倍率", "dspin", default=1.0, minimum=0.1, maximum=5.0,
                      tooltip="罚项 = penalty·0.5·(K−1)·lnN；调小更容易切出变点")
P_DETECT = ParamSpec("detect", "做变点检测", "check", default=True)


SPECS = [
    am.MethodSpec(
        id="IND01", name="滞后序列分析 LSA", group=GROUP_TRANSITION,
        question="哪些编码之间的先后衔接不偶然？",
        mode=MODE_INDIVIDUAL, compute=lag_sequence_analysis,
        params=(P_MAX_LAG, P_LSA_SCOPE, P_Z, P_FDR),
        r_reference="lsa_residual.R", unit="student×dimension",
        formula_note="E_ij = row_i·col_j/N（N 不含对角）；Z = (O−E)/√(E(1−row/N)(1−col/N))"),
    am.MethodSpec(
        id="IND02", name="一阶马尔可夫链", group=GROUP_TRANSITION,
        question="下一个状态在多大程度上由上一个决定？",
        mode=MODE_INDIVIDUAL, compute=markov_first,
        params=(P_SELF,), r_reference="markov_ref.R", unit="student×dimension",
        formula_note="P_ij = N_ij/Σ_j N_ij；转移熵率 = −Σ_i π_i Σ_j P_ij ln P_ij"),
    am.MethodSpec(
        id="IND03", name="高阶 / 变阶马尔可夫", group=GROUP_TRANSITION,
        question="一阶够了吗？多长记忆才能解释序列？",
        mode=MODE_INDIVIDUAL, compute=higher_order_markov,
        params=(P_ORDER, P_CRITERION, P_EVAL), requires=("length>=20",),
        unit="student×dimension",
        formula_note="BIC = −2lnL + m·lnN；上下文未观测时回退到已见过的最长上下文"),
    am.MethodSpec(
        id="IND04", name="n-gram / 频繁子序列", group=GROUP_TRANSITION,
        question="哪些行为模式反复出现？",
        mode=MODE_INDIVIDUAL, compute=frequent_patterns,
        params=(P_N, P_MINE, P_STREAM, P_ALGO), r_reference="traminer_seqefsub.R",
        unit="student×dimension",
        formula_note="support = 含该模式的序列数/总序列数；lift = confidence/support(后项)"),
    am.MethodSpec(
        id="IND05", name="序列距离 OM/HAM/DHD", group=GROUP_DISTANCE,
        question="两条序列有多不像？",
        mode=MODE_INDIVIDUAL, compute=distance_analysis,
        params=(P_DIST_METHOD, P_SUBST, P_COST_VALUE, P_CUSTOM_COST, P_INDEL, P_COMPRESS),
        r_reference="traminer_om.R", unit="student×dimension",
        formula_note="OM = Needleman–Wunsch；TRATE d(i,j)=2−p(i→j)−p(j→i)；indel=0.5×最大代价"),
    am.MethodSpec(
        id="IND07", name="序列聚类 AHC / k-medoids", group=GROUP_DISTANCE,
        question="序列能不能分堆？分得稳不稳？",
        mode=MODE_INDIVIDUAL, compute=sequence_clustering, requires=("students>=4",),
        params=(P_DIST_METHOD, P_SUBST, P_COST_VALUE, P_CUSTOM_COST, P_INDEL, P_COMPRESS,
                P_CLUSTER_METHOD, P_K, P_LINKAGE, P_STABILITY, P_REPS),
        r_reference="traminer_om.R", unit="student×dimension",
        formula_note="silhouette 直接由距离阵算；bootstrap 重抽单元 = 序列（学生）"),
    am.MethodSpec(
        id="IND06", name="DTW 序列距离", group=GROUP_DISTANCE,
        question="时间轴对不齐的两条轨迹，像不像？",
        mode=MODE_INDIVIDUAL, compute=dtw_analysis,
        params=(P_COMPRESS, P_SOFT_DTW, P_SOFT_GAMMA, P_BARYCENTER, P_BA_ITER),
        r_reference="dtwclust_ref.R", unit="student×dimension",
        formula_note="局部代价 = one-hot 行向量欧氏距离（同码 0、异码 √2）；累积代价不做长度归一化"),
    am.MethodSpec(
        id="IND08", name="隐马尔可夫模型 HMM / MHMM", group=GROUP_MODEL,
        question="看得见的编码背后，有几个隐藏的状态层？",
        mode=MODE_INDIVIDUAL, compute=hmm_analysis, requires=("length>=20",),
        params=(P_HMM_MODEL, P_HMM_ENGINE, P_HMM_SELECT, P_HMM_STATES, P_HMM_KMIN, P_HMM_KMAX,
                P_HMM_CRITERION, P_HMM_ITER, P_HMM_TOL, P_HMM_SEED, P_HMM_ROWS, P_COMPRESS),
        r_reference="seqhmm_fit.R", unit="student×dimension",
        formula_note="缩放前向后向：logLik = Σ ln c_t；AIC/BIC 参数量 = (S−1)+S(S−1)+Σ S(V−1)"),
    am.MethodSpec(
        id="IND09", name="变点 / 分段", group=GROUP_STATE,
        question="行为模式在哪个时刻换了档？",
        mode=MODE_INDIVIDUAL, compute=segmentation_analysis,
        params=(P_MIN_SEGMENT, P_PENALTY, P_SEGMENT_WINDOW, P_DETECT, P_COMPRESS),
        unit="student×dimension",
        formula_note="递归二分：ΔBIC = (ll_左+ll_右−ll_全) − 0.5·(K−1)·lnN；Hellinger 滑窗只做诊断"),
]
