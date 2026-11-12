# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""GRP01–GRP06 / GRP09：小组模式的组内聚合统计。

小组分析的第一条铁律：**独立分析单元是组，不是学生**。所以这里每张表都带 group_id，
比例的分母写在该行里（该时刻的有码成员数、该组的成员数），跨组比较一律用
proportion / 归一化熵 / mean pairwise sync 这类与组规模无关的量（文档 §7.1），
不拿求和直接比。

组状态矩阵的唯一取数口是 `GroupBundle.member_matrix(dimension)`，它返回
`(时间点, {成员: 与时间点等长的码表})`，缺码是空串而不是 0 —— 空串不参与任何分母。
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pandas as pd

import analysis_features as af
import analysis_methods as am
from analysis_methods import (GROUP_COMPARE, GROUP_NETWORK, GROUP_STATE,
                              GROUP_TRANSITION, MODE_GROUP, ParamSpec)
from analysis_stats_sequence import (blocked, lsa_residuals, matrix_frame,
                                     pooled_counts, states_of, yule_matrix)
from statistic_visualization_data import QAItem, fmt_round

GROUPS_UNITS = ("groups", "dimensions")
MIN_OVERLAP = 5           # 两条码轨重叠不足 5 个点时，同步度只是噪声


def _start():
    return time.perf_counter()


def _ms(marker):
    return int((time.perf_counter() - marker) * 1000)


# ------------------------------------------------------------------ 取数小件

def wanted_groups(ctx, params):
    return am.chosen(ctx.group_ids, params.get("groups"))


def wanted_dimensions(ctx, params):
    keys = [ctx.dimension_key(item) for item in (params.get("dimensions") or ctx.dimensions)]
    return [key for key in dict.fromkeys(keys)] or ctx.dimension_keys()


def member_tracks(ctx, group_id, dimension):
    """一个组 × 一个维度 → (时间点, {成员: 码表}, 成员顺序)。"""
    bundle = ctx.bundle(group_id)
    if bundle is None:
        return [], {}, []
    points, matrix = bundle.member_matrix(dimension)
    order = [member for member in bundle.members if member in matrix]
    return points, {member: matrix[member] for member in order}, order


def _pairs(values):
    return list(itertools.combinations(values, 2))


def valid_positions(*tracks):
    """若干条等长码轨上「全部有码」的位置下标。"""
    if not tracks:
        return []
    length = min(len(track) for track in tracks)
    return [index for index in range(length)
            if all(track[index] for track in tracks)]


# ------------------------------------------------------------------ 非参数跨组比较

def _ranks(values):
    """平均秩（并列取均值），返回 (秩数组, 并列校正项 Σ(t³−t))。"""
    array = np.asarray(list(values), dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=float)
    tie_correction = 0.0
    position = 0
    while position < len(array):
        end = position
        while end + 1 < len(array) and array[order[end + 1]] == array[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        size = end - position + 1
        tie_correction += size ** 3 - size
        position = end + 1
    return ranks, tie_correction


def kruskal_wallis(samples):
    """Kruskal–Wallis H（带并列校正）→ (H, df, p)。少于 2 组或总数 < 4 时给 None。"""
    groups = [[float(value) for value in sample if value is not None
               and np.isfinite(float(value))] for sample in samples]
    groups = [group for group in groups if group]
    if len(groups) < 2 or sum(len(group) for group in groups) < 4:
        return None
    pooled = [value for group in groups for value in group]
    ranks, tie_correction = _ranks(pooled)
    total = len(pooled)
    cursor = 0
    statistic = 0.0
    for group in groups:
        summed = float(ranks[cursor:cursor + len(group)].sum())
        statistic += summed ** 2 / len(group)
        cursor += len(group)
    statistic = 12.0 / (total * (total + 1)) * statistic - 3.0 * (total + 1)
    divisor = 1.0 - tie_correction / (total ** 3 - total)
    if divisor > 0:
        statistic /= divisor
    degrees = len(groups) - 1
    from analysis_stats_sequence import chi2_sf
    return {"H": round(statistic, 6), "df": degrees, "p": chi2_sf(statistic, degrees),
            "groups": [len(group) for group in groups]}


def mann_whitney_u(first, second):
    """Mann–Whitney U（正态近似 + 并列校正）→ (U, n1, n2, p, rank-biserial)。"""
    a = [float(value) for value in first if value is not None and np.isfinite(float(value))]
    b = [float(value) for value in second if value is not None and np.isfinite(float(value))]
    if not a or not b:
        return None
    pooled = a + b
    ranks, tie_correction = _ranks(pooled)
    sum_a = float(ranks[:len(a)].sum())
    u_a = sum_a - len(a) * (len(a) + 1) / 2.0
    u_b = len(a) * len(b) - u_a
    mean = len(a) * len(b) / 2.0
    n = len(pooled)
    sigma_sq = (len(a) * len(b) / 12.0) * ((n + 1) - tie_correction / (n * (n - 1))) \
        if n > 1 else 0.0
    z = (u_a - mean) / math.sqrt(sigma_sq) if sigma_sq > 0 else 0.0
    from analysis_stats_sequence import normal_two_sided_p
    p_value = normal_two_sided_p(z) if sigma_sq > 0 else 1.0
    return {"U": round(min(u_a, u_b), 4), "U_a": round(u_a, 4),
            "n1": len(a), "n2": len(b), "z": round(z, 4), "p": p_value,
            "rank_biserial": round(2.0 * u_a / (len(a) * len(b)) - 1.0, 4)}


def cliffs_delta(first, second):
    """Cliff's δ = (#(a>b) − #(a<b)) / (n1·n2)，量的是两组的偏移方向。"""
    a = [float(value) for value in first if value is not None]
    b = [float(value) for value in second if value is not None]
    if not a or not b:
        return 0.0
    greater = sum(1 for x in a for y in b if x > y)
    lesser = sum(1 for x in a for y in b if x < y)
    return round((greater - lesser) / float(len(a) * len(b)), 6)


def gini(values):
    """Gini 系数（非负值）：0 = 完全均衡，1 = 全集中在一个成员身上。"""
    array = np.asarray([abs(float(value)) for value in values if value is not None], dtype=float)
    if array.size < 2 or array.sum() <= 0:
        return 0.0
    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1)
    return float((2.0 * (index * array).sum()) / (n * array.sum()) - (n + 1.0) / n)


def distribution_entropy(values):
    """特征质量在成员间的分布均等度：shares_k = value_k/Σvalue，H = −Σ shares ln shares，
    再除 ln(n)。1 = 完全平均，0 = 全集中在一个成员（与 Gini 同口径，不是“值的熵”）。"""
    array = np.asarray([max(float(value), 0.0) for value in values], dtype=float)
    total = float(array.sum())
    if total <= 0 or array.size < 2:
        return 0.0
    shares = array[array > 0] / total
    raw = float(-(shares * np.log(shares)).sum())
    return raw / math.log(array.size)


def cross_group_compare(frame, value_column, group_column="group_id"):
    """跨组比较：每组一个分布 → Mann–Whitney 两两 + BH 校正 + 效应量（列名与 GRP06 对齐）。"""
    groups = sorted({str(value) for value in frame[group_column] if str(value)})
    if len(groups) < 2:
        return pd.DataFrame()
    series = {group: [float(value) for value in
                      frame[frame[group_column].astype(str) == group][value_column]
                      if value is not None and str(value) not in ("", "nan")]
              for group in groups}
    overall = kruskal_wallis([series[group] for group in groups])
    rows = []
    for left, right in _pairs(groups):
        result = mann_whitney_u(series[left], series[right])
        if result is None:
            continue
        rows.append({"特征": value_column, "组A": left, "组B": right,
                     "nA": result["n1"], "nB": result["n2"], "U": result["U"],
                     "p": result["p"],
                     "rank_biserial": result["rank_biserial"],
                     "cliffs_delta": cliffs_delta(series[left], series[right]),
                     "均值A": fmt_round(np.mean(series[left]), 4),
                     "均值B": fmt_round(np.mean(series[right]), 4)})
    out = pd.DataFrame(rows)
    if len(out):
        out["q(BH)"] = am.bh_fdr(out["p"].tolist())
    if overall is not None:
        out.attrs["kruskal"] = overall
    return out


# ------------------------------------------------------------------ GRP01 组状态分布

def group_state_distribution(ctx, params):
    """GRP01：`group_state_share(t)` —— 每个时刻全组的成员码构成。"""
    marker = _start()
    window = max(0, int(params.get("window") or 0))
    rows, summary = [], []
    for group_id in wanted_groups(ctx, params):
        points, tracks, order = None, None, None
        for dimension in wanted_dimensions(ctx, params):
            points, tracks, order = member_tracks(ctx, group_id, dimension)
            if not points:
                continue
            universe = states_of([_fake(dimension, track) for track in tracks.values()])
            for position, moment in enumerate(points):
                codes = [tracks[member][position] for member in order
                         if position < len(tracks[member]) and tracks[member][position]]
                for state in universe:
                    count = sum(1 for code in codes if code == state)
                    rows.append({"group_id": group_id, "dimension": ctx.label_of(dimension),
                                 "time": moment, "state": state, "成员数": count,
                                 "有码成员数": len(codes),
                                 "share": fmt_round(count / len(codes), 4) if codes else 0.0})
            frame = pd.DataFrame([item for item in rows if item["group_id"] == group_id
                                  and item["dimension"] == ctx.label_of(dimension)])
            if not len(frame):
                continue
            pivot = frame.pivot_table(index="time", columns="state", values="share",
                                      aggfunc="mean")
            if window > 1:
                pivot = pivot.rolling(window, min_periods=1).mean()
            for state in universe:
                column = pivot[state] if state in pivot.columns else pd.Series(dtype=float)
                dominant = int((frame[frame["state"] == state]
                                ["share"].astype(float) > 0.5).sum())
                summary.append({
                    "group_id": group_id, "dimension": ctx.label_of(dimension), "state": state,
                    "均值占比": fmt_round(float(column.mean()) if len(column) else 0.0, 4),
                    "峰值占比": fmt_round(float(column.max()) if len(column) else 0.0, 4),
                    "占主导的时刻数": dominant, "占主导比例": fmt_round(
                        dominant / max(len(points), 1), 4),
                })
    if not rows:
        return blocked("GRP01", "组状态分布", "没有可解析的组状态矩阵")
    detail = pd.DataFrame(rows)
    if window > 1:
        detail["平滑窗长"] = window
    return am.result("GRP01", "组状态分布", {
        "trajectory": detail,
        "summary": pd.DataFrame(summary),
    }, meta={"unit": "group×time", "window": window,
             "n_groups": len(wanted_groups(ctx, params)),
             "denominator": "该时刻的有码成员数"},
        notes=["share 的分母是该时刻**有码**成员数，缺码不进分母",
               "占主导 = 该码在该时刻占比 > 50%"],
        elapsed_ms=_ms(marker))


class _FakeSeq:
    """给 states_of 用的最小适配器：只要有 .codes。"""

    __slots__ = ("codes", "dimension")

    def __init__(self, dimension, codes):
        self.dimension = dimension
        self.codes = list(codes)


def _fake(dimension, codes):
    return _FakeSeq(dimension, codes)


# ------------------------------------------------------------------ GRP02 组熵轨迹

def group_entropy(ctx, params):
    """GRP02：每时刻成员构成的香农熵（ln）+ 归一化熵 + 轨迹汇总。"""
    marker = _start()
    rows, summary = [], []
    for group_id in wanted_groups(ctx, params):
        for dimension in wanted_dimensions(ctx, params):
            points, tracks, order = member_tracks(ctx, group_id, dimension)
            if not points:
                continue
            for position, moment in enumerate(points):
                codes = [tracks[member][position] for member in order
                         if position < len(tracks[member]) and tracks[member][position]]
                value = af.entropy_of(codes)
                rows.append({"group_id": group_id, "dimension": ctx.label_of(dimension),
                             "time": moment, "有码成员数": len(codes),
                             "出现状态数": len(set(codes)),
                             "H": fmt_round(value, 4),
                             "H_norm": fmt_round(af.normalized_entropy_of(codes), 4),
                             "最大可能熵": fmt_round(math.log(len(codes)) if len(codes) > 1
                                                     else 0.0, 4)})
            if rows:
                series = pd.DataFrame([item for item in rows if item["group_id"] == group_id
                                       and item["dimension"] == ctx.label_of(dimension)])
                pooled = [code for member in order for code in tracks[member] if code]
                summary.append({
                    "group_id": group_id, "dimension": ctx.label_of(dimension),
                    "成员数": len(order), "时刻数": len(points),
                    "均值熵": fmt_round(float(series["H"].astype(float).mean()), 4),
                    "峰值熵": fmt_round(float(series["H"].astype(float).max()), 4),
                    "轨迹熵(全时刻合并)": fmt_round(af.entropy_of(pooled), 4),
                    "归一化轨迹熵": fmt_round(af.normalized_entropy_of(pooled), 4),
                })
    if not rows:
        return blocked("GRP02", "组内熵", "没有可解析的组状态矩阵")
    return am.result("GRP02", "组内熵轨迹", {
        "trajectory": pd.DataFrame(rows), "summary": pd.DataFrame(summary),
    }, meta={"unit": "group×time", "log": "ln"},
        notes=["熵一律 ln；H_norm 的分母是 ln(该时刻有码成员数)，不是 ln(全组成员数)",
               "轨迹熵 = 把全组全时刻的码倒在一起算的边际熵，与均值熵不是同一个量"],
        elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ GRP03 同步度

def agreement(first, second, lag=0):
    """same-code agreement：比较 first[t] 与 second[t+lag]，只数两边都有码的位置。"""
    matched, same = 0, 0
    for index, value in enumerate(first):
        target = index + lag
        if target < 0 or target >= len(second):
            continue
        if not value or not second[target]:
            continue
        matched += 1
        same += 1 if value == second[target] else 0
    if matched < MIN_OVERLAP:
        return 0.0, matched
    return same / float(matched), matched


def chance_agreement(first, second, lag=0):
    """机会一致率：两边边际分布独立时的 Σ p_k·q_k（同步度要减掉它才有意义）。"""
    left = [code for code in first if code]
    right = [code for code in second if code]
    if not left or not right:
        return 0.0
    universe = set(left) | set(right)
    total = 0.0
    for state in universe:
        total += (left.count(state) / len(left)) * (right.count(state) / len(right))
    return total


def synchrony(ctx, params):
    """GRP03：两两同步度矩阵 + 组均值 + 置换检验 p。"""
    marker = _start()
    lag = int(params.get("lag") or 0)
    permutations = max(0, int(params.get("permutations") or 200))
    seed = int(params.get("seed") or 20240517)
    rows, cells, qa = [], [], []
    for group_id in wanted_groups(ctx, params):
        for dimension in wanted_dimensions(ctx, params):
            points, tracks, order = member_tracks(ctx, group_id, dimension)
            if len(order) < 2:
                continue
            observed = {}
            for left, right in _pairs(order):
                value, matched = agreement(tracks[left], tracks[right], lag)
                chance = chance_agreement(tracks[left], tracks[right], lag)
                observed[(left, right)] = value
                rows.append({"group_id": group_id, "dimension": ctx.label_of(dimension),
                             "成员A": left, "成员B": right, "lag": lag,
                             "agreement": fmt_round(value, 4),
                             "机会一致率": fmt_round(chance, 4),
                             "标准化同步度": fmt_round(
                                 (value - chance) / (1.0 - chance), 4) if chance < 1 else 0.0,
                             "重叠点数": matched})
            matrix = np.zeros((len(order), len(order)))
            for (left, right), value in observed.items():
                i, j = order.index(left), order.index(right)
                matrix[i, j] = matrix[j, i] = value
            mean_sync = float(np.mean([value for value in observed.values()])) if observed else 0.0
            p_value = permutation_p(tracks, order, lag, mean_sync, permutations, seed)
            cells.append({"group_id": group_id, "dimension": ctx.label_of(dimension),
                          "成员数": len(order), "组均值同步度": fmt_round(mean_sync, 4),
                          "置换检验 p": fmt_round(p_value, 4),
                          "重复次数": permutations})
            frame = pd.DataFrame(np.round(matrix, 4), index=order, columns=order)
            frame.insert(0, "成员", order)
            frame["group_id"] = group_id
            frame["dimension"] = ctx.label_of(dimension)
            cells.append({"__matrix__": frame})
            if p_value is not None and p_value > 0.05:
                qa.append(QAItem("info", "GRP03.not_significant",
                                 "%s / %s：组均值同步度 %.3f 未超置换分布（p=%.3f）"
                                 % (group_id, ctx.label_of(dimension), mean_sync, p_value)))
    detail = pd.DataFrame([row for row in rows if "__matrix__" not in row])
    summary = pd.DataFrame([row for row in cells if "__matrix__" not in row])
    matrices = [row["__matrix__"] for row in cells if "__matrix__" in row]
    if not len(detail):
        return blocked("GRP03", "同步度", "每组至少需要 2 名成员才有同步度可言")
    tables = {"pairs": detail, "summary": summary}
    if matrices:
        tables["sync_matrix"] = pd.concat(matrices, ignore_index=True)
    return am.result("GRP03", "组内同步度", tables,
                     meta={"lag": lag, "permutations": permutations, "unit": "group×pair",
                           "min_overlap": MIN_OVERLAP},
                     qa=qa,
                     notes=["agreement 只在两边都有码的位置上算，重叠 < %d 点记为不可比"
                            % MIN_OVERLAP,
                            "置换检验：组内各成员的码序列整体随机移位后重算组均值",
                            "lag>0 时比较的是 A(t) 与 B(t+lag)，属滞后同步"],
                     elapsed_ms=_ms(marker))


def permutation_p(tracks, order, lag, observed, repetitions, seed):
    """移位置换检验：给每个成员一个随机循环移位，重算组均值，看观测值排第几。"""
    if repetitions <= 0 or len(order) < 2:
        return None
    rng = np.random.default_rng(int(seed))
    count = 0
    for _ in range(repetitions):
        shifted = {}
        for member in order:
            values = list(tracks[member])
            offset = int(rng.integers(0, max(len(values), 1)))
            shifted[member] = values[offset:] + values[:offset] if offset else values
        values = [agreement(shifted[left], shifted[right], lag)[0]
                  for left, right in _pairs(order)]
        if values and float(np.mean(values)) >= observed - 1e-12:
            count += 1
    return (count + 1) / (repetitions + 1)


# ------------------------------------------------------------------ GRP04 领先-滞后

def cross_agreement_curve(first, second, max_lag):
    """CCF 的离散版本：lag = −k…+k 上的 agreement 与超额一致率。"""
    rows = []
    for lag in range(-abs(int(max_lag)), abs(int(max_lag)) + 1):
        value, matched = agreement(first, second, lag)
        chance = chance_agreement(first, second, lag)
        rows.append({"lag": lag, "agreement": value, "chance": chance,
                     "excess": value - chance, "重叠点数": matched})
    return rows


def lagged_mutual_information(first, second, lag):
    """MI(X_t ; Y_{t+lag})，ln 口径，只数两边都有码的位置。"""
    pairs = {}
    total = 0
    for index, value in enumerate(first):
        target = index + lag
        if target < 0 or target >= len(second) or not value or not second[target]:
            continue
        pairs[(value, second[target])] = pairs.get((value, second[target]), 0) + 1
        total += 1
    if total == 0:
        return 0.0
    rows = {}
    cols = {}
    for (a, b), count in pairs.items():
        rows[a] = rows.get(a, 0) + count
        cols[b] = cols.get(b, 0) + count
    mutual = 0.0
    for (a, b), count in pairs.items():
        p_ab = count / total
        mutual += p_ab * math.log(p_ab / ((rows[a] / total) * (cols[b] / total)))
    return mutual


def lead_lag(ctx, params):
    """GRP04：谁先动、谁跟着动 —— CCF(±k) + 滞后互信息 + lead-lag 矩阵。"""
    marker = _start()
    max_lag = max(1, min(int(params.get("max_lag") or 2), 15))
    rows, ranking, matrices = [], [], []
    for group_id in wanted_groups(ctx, params):
        for dimension in wanted_dimensions(ctx, params):
            points, tracks, order = member_tracks(ctx, group_id, dimension)
            if len(order) < 2:
                continue
            label = ctx.label_of(dimension)
            grid = np.full((len(order), len(order)), np.nan)
            tally = {member: 0 for member in order}
            compared = 0
            for left, right in _pairs(order):
                curve = [item for item in cross_agreement_curve(tracks[left], tracks[right],
                                                                max_lag)
                         if item["重叠点数"] >= MIN_OVERLAP]
                if not curve:
                    continue
                best = max(curve, key=lambda item: item["excess"])
                mutual = lagged_mutual_information(tracks[left], tracks[right], best["lag"])
                lead, follow = (left, right) if best["lag"] >= 0 else (right, left)
                compared += 1
                tally[lead] += 1
                rows.append({"group_id": group_id, "dimension": label,
                             "成员A": left, "成员B": right, "最优lag": best["lag"],
                             "agreement@lag": fmt_round(best["agreement"], 4),
                             "机会一致率": fmt_round(best["chance"], 4),
                             "超额一致率": fmt_round(best["excess"], 4),
                             "互信息@lag(ln)": fmt_round(mutual, 4),
                             "领先者": lead, "跟随者": follow,
                             "重叠点数": best["重叠点数"]})
                i, j = order.index(left), order.index(right)
                grid[i, j] = best["lag"]
                grid[j, i] = -best["lag"]
            for member in sorted(tally, key=lambda name: -tally[name]):
                ranking.append({"group_id": group_id, "dimension": label, "成员": member,
                                "领先次数": tally[member],
                                "领先指数": fmt_round(tally[member] / max(compared, 1), 4)})
            frame = pd.DataFrame(np.round(grid, 2), index=order, columns=order)
            frame.insert(0, "成员", order)
            frame["group_id"] = group_id
            frame["dimension"] = label
            matrices.append(frame)
    if not rows:
        return blocked("GRP04", "领先-滞后", "重叠点数不足，无法判读滞后关系")
    tables = {"pairs": pd.DataFrame(rows), "ranking": pd.DataFrame(ranking),
              "lead_lag_matrix": pd.concat(matrices, ignore_index=True)}
    return am.result("GRP04", "领先-滞后关系", tables,
                     meta={"max_lag": max_lag, "unit": "group×pair",
                           "min_overlap": MIN_OVERLAP},
                     notes=["最优 lag 取「超额一致率」（agreement − 机会一致率）最大的那个 lag，不是原始 agreement",
                            "lag>0：A 在 t、B 在 t+lag ⇒ A 领先 B；矩阵里 (i,j) 读作 i 领先 j 的步数",
                            "互信息仍是 ln 口径，与 IND 一致",
                            "领先指数 = 该成员的领先次数 / 本组内可比对数"],
                     elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ GRP05 组级转移 + LSA

def group_transitions(ctx, params):
    """GRP05：consensus 码与成员合并两套组级转移矩阵 + LSA 残差 + Yule's Q。"""
    marker = _start()
    max_lag = max(1, min(int(params.get("max_lag") or 1), 3))
    threshold = float(params.get("z_threshold") or 1.96)
    scope = params.get("scope") or "both"
    rows, matrices, qa = [], [], []
    for group_id in wanted_groups(ctx, params):
        bundle = ctx.bundle(group_id)
        for dimension in wanted_dimensions(ctx, params):
            flavors = {}
            if scope in ("consensus", "both") and bundle is not None:
                consensus = bundle.consensus_sequence(dimension)
                if consensus.valid:
                    flavors["consensus"] = [consensus]
            if scope in ("pooled", "both"):
                members = ctx.sequences(group_id=group_id, dimension=dimension)
                if members:
                    flavors["成员合并"] = members
            if not flavors:
                continue
            for flavor, sequences in flavors.items():
                states = states_of(sequences)
                for lag in range(1, max_lag + 1):
                    _states, counts = pooled_counts(sequences, states, lag=lag)
                    off, expected, residuals = lsa_residuals(counts)
                    q_matrix = yule_matrix(counts)
                    for i, j in _cell_pairs(len(states)):
                        rows.append({
                            "group_id": group_id, "口径": flavor,
                            "dimension": ctx.label_of(dimension), "lag": lag,
                            "from": states[i], "to": states[j],
                            "O": int(off[i, j]), "E": fmt_round(expected[i, j], 4),
                            "Z": fmt_round(residuals[i, j], 4),
                            "p": fmt_round(_normal_p(residuals[i, j]), 4),
                            "yules_Q": fmt_round(q_matrix[i, j]),
                            "significant": bool(abs(residuals[i, j]) > threshold)})
                    if lag == max_lag:
                        matrices.append({"group_id": group_id, "口径": flavor,
                                         "dimension": dimension,
                                         "frame": matrix_frame(states, residuals, "from")})
    if not rows:
        return blocked("GRP05", "组级转移 + LSA", "共识码与成员序列都不够长")
    frame = pd.DataFrame(rows)
    frame["q(BH)"] = am.bh_fdr(frame["p"].tolist())
    for item in matrices:
        item["frame"]["group_id"] = item["group_id"]
        item["frame"]["口径"] = item["口径"]
    tables = {"cells": frame,
              "residual_matrix": pd.concat([item["frame"] for item in matrices],
                                           ignore_index=True)}
    qa.append(QAItem("info", "GRP05.unit",
                     "组级转移的独立单元是组，跨组比较请用 q(BH) 与比例而非计数", False))
    return am.result("GRP05", "组级转移与 LSA", tables,
                     meta={"max_lag": max_lag, "scope": scope, "unit": "group",
                           "threshold": threshold},
                     qa=qa,
                     notes=["consensus = 每时刻全组该维度的众数码；成员合并 = 全组成员序列的转移计数相加",
                            "两套矩阵回答不同问题：共识看组作为整体的走向，合并看成员行为的共性"],
                     elapsed_ms=_ms(marker))


def _cell_pairs(size):
    return [(i, j) for i in range(size) for j in range(size) if i != j]


def _normal_p(z):
    from analysis_stats_sequence import normal_two_sided_p
    return normal_two_sided_p(z)


# ------------------------------------------------------------------ GRP06 异质性

def heterogeneity(ctx, params):
    """GRP06：成员特征向量的 variance / CV / Gini + 归一化熵，跨组再给非参数比较。"""
    marker = _start()
    families = params.get("features") or ("entropy", "switch_rate", "prop")
    rows, compared = [], []
    for group_id in wanted_groups(ctx, params):
        for dimension in wanted_dimensions(ctx, params):
            sequences = ctx.sequences(group_id=group_id, dimension=dimension)
            if not sequences:
                continue
            frame = af.feature_frame(sequences)
            if not len(frame):
                continue
            columns = [column for column in frame.columns
                       if any(str(column).startswith(prefix) for prefix in
                              _family_prefixes(families))]
            for column in columns:
                values = pd.to_numeric(frame[column], errors="coerce").dropna().tolist()
                if len(values) < 2:
                    continue
                mean = float(np.mean(values))
                variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
                rows.append({"group_id": group_id, "dimension": ctx.label_of(dimension),
                             "特征": column, "成员数": len(values),
                             "均值": fmt_round(mean, 4), "方差": fmt_round(variance, 4),
                             "标准差": fmt_round(math.sqrt(variance), 4),
                             "CV": fmt_round(math.sqrt(variance) / mean, 4) if mean else 0.0,
                             "Gini": fmt_round(gini(values), 4),
                             "分布熵(归一化)": fmt_round(distribution_entropy(values), 4)})
            if len(ctx.bundles) >= 2 and len(rows):
                base = pd.DataFrame(rows)
                for column in sorted({str(item) for item in base["特征"]}):
                    subset = base[base["特征"] == column]
                    result = cross_group_compare(subset, "均值", "group_id")
                    if len(result):
                        result.insert(0, "dimension", ctx.label_of(dimension))
                        compared.append(result)
    if not rows:
        return blocked("GRP06", "组内异质性", "成员特征算不出来：序列太短或维度未选")
    detail = pd.DataFrame(rows)
    tables = {"heterogeneity": detail}
    if compared:
        merged = pd.concat(compared, ignore_index=True)
        tables["cross_group"] = merged[~merged.duplicated(
            subset=["dimension", "特征", "组A", "组B"])]
    return am.result("GRP06", "组内异质性", tables,
                     meta={"unit": "group×feature", "features": list(families)},
                     notes=["CV = 样本标准差/均值（均值为 0 时记 0，不代表真无差异）",
                            "Gini 用成员特征值的绝对值算，跨组比较请用 CV / 归一化熵",
                            "跨组检验默认非参数（Mann–Whitney + BH），组数少时只看方向"],
                     elapsed_ms=_ms(marker))


def _family_prefixes(families):
    mapping = {"prop": ("prop_",), "freq": ("freq_",), "entropy": ("entropy",),
               "switch": ("switch_rate", "self_transition"), "run": ("run_",),
               "transition": ("transition_entropy", "top_transition")}
    out = []
    for item in families:
        out.extend(mapping.get(str(item), (str(item),)))
    return tuple(out) or ("prop_",)


# ------------------------------------------------------------------ GRP09 组级网络

def pooled_cooccurrence(ctx, params):
    """ENA 口径的加权共现：同一 (学生, 时刻) 上跨维度同时出现的编码对计数。"""
    rows, edges = [], {}
    min_edge = int(params.get("min_edge") or 0)
    for group_id in wanted_groups(ctx, params):
        bundle = ctx.bundle(group_id)
        if bundle is None:
            continue
        dimensions = wanted_dimensions(ctx, params)
        frames = {dimension: bundle.member_matrix(dimension) for dimension in dimensions}
        members = bundle.members
        for member in members:
            for position in range(len(frames[dimensions[0]][0]) if dimensions else 0):
                active = []
                for dimension in dimensions:
                    points, matrix = frames[dimension]
                    if position < len(points) and member in matrix \
                            and position < len(matrix[member]) and matrix[member][position]:
                        active.append((dimension, matrix[member][position]))
                for (dim_a, code_a), (dim_b, code_b) in _pairs(active):
                    if dim_a == dim_b:
                        continue
                    key = tuple(sorted(((dim_a, code_a), (dim_b, code_b))))
                    edges[(group_id, key)] = edges.get((group_id, key), 0) + 1
        for (source_group, (left, right)), weight in edges.items():
            if source_group != group_id or weight < min_edge:
                continue
            rows.append({"group_id": source_group,
                         "节点A": "%s:%s" % (ctx.label_of(left[0]), left[1]),
                         "节点B": "%s:%s" % (ctx.label_of(right[0]), right[1]),
                         "共现次数": int(weight)})
    return pd.DataFrame(rows)


def directed_network(ctx, params):
    """ONA 口径的有向时序网络：全组成员的滞后转移合并成一张加权有向图。"""
    lag = max(1, int(params.get("lag") or 1))
    rows = []
    for group_id in wanted_groups(ctx, params):
        for dimension in wanted_dimensions(ctx, params):
            sequences = ctx.sequences(group_id=group_id, dimension=dimension)
            if not sequences:
                continue
            states, counts = pooled_counts(sequences, lag=lag)
            index = {state: position for position, state in enumerate(states)}
            weights = {}
            for left in states:
                for right in states:
                    value = counts[index[left], index[right]]
                    if value > 0 and left != right:
                        weights[(left, right)] = float(value)
            total_weight = sum(weights.values())
            in_degree = {state: 0.0 for state in states}
            out_degree = {state: 0.0 for state in states}
            for (left, right), value in weights.items():
                out_degree[left] += value
                in_degree[right] += value
            for (left, right), value in sorted(weights.items(), key=lambda item: -item[1]):
                rows.append({"group_id": group_id, "dimension": ctx.label_of(dimension),
                             "source": left, "target": right, "权重": fmt_round(value, 2),
                             "占全网": fmt_round(value / total_weight, 4) if total_weight else 0.0,
                             "出度(source)": fmt_round(out_degree[left], 2),
                             "入度(target)": fmt_round(in_degree[right], 2),
                             "互惠": int((right, left) in weights)})
    return pd.DataFrame(rows)


def network_metrics(edges, directed=True):
    """节点/边数、密度、互惠率、度统计：纯图计数，不依赖 networkx。"""
    if not len(edges):
        return {}
    nodes = sorted(set(edges["source"]) | set(edges["target"])) if directed else \
        sorted(set(edges["节点A"]) | set(edges["节点B"]))
    size = len(nodes)
    count = len(edges)
    density = count / (size * (size - 1)) if directed and size > 1 else \
        count / (size * (size - 1) / 2.0) if size > 1 else 0.0
    out = {"节点数": size, "边数": count, "密度": fmt_round(density, 4)}
    if directed:
        present = {(str(row["source"]), str(row["target"])) for _i, row in edges.iterrows()}
        pairs = {tuple(sorted(pair)) for pair in present}
        reciprocated = sum(1 for pair in present if (pair[1], pair[0]) in present)
        out["互惠率"] = fmt_round(reciprocated / count, 4) if count else 0.0
        out["无向边对数"] = len(pairs)
        if "权重" in edges:
            values = edges.groupby("source")["权重"].sum().to_numpy(dtype=float)
            out["出度均值"] = fmt_round(float(np.mean(values)), 4)
            out["出度最大"] = fmt_round(float(np.max(values)), 4)
    return out


def group_network(ctx, params):
    """GRP09：全组成员合并的 ENA 共现网 + ONA 有向时序网。"""
    marker = _start()
    cooccurrence = pooled_cooccurrence(ctx, params)
    directed = directed_network(ctx, params)
    if not len(cooccurrence) and not len(directed):
        return blocked("GRP09", "组级 ENA/ONA", "没有可建网的编码共现或时序转移")
    tables, meta = {}, {}
    if len(cooccurrence):
        tables["ena_edges"] = cooccurrence.sort_values("共现次数", ascending=False) \
            .reset_index(drop=True)
        meta["ena"] = {"nodes": int(len(set(cooccurrence["节点A"]) |
                                        set(cooccurrence["节点B"]))),
                       "edges": int(len(cooccurrence))}
    if len(directed):
        tables["ona_edges"] = directed
        metrics = []
        for group_id, part in directed.groupby("group_id", sort=False):
            row = {"group_id": group_id}
            row.update(network_metrics(part))
            metrics.append(row)
        tables["ona_metrics"] = pd.DataFrame(metrics)
        meta["ona"] = {"edges": int(len(directed))}
    notes = ["ENA 共现单位 = 同一 (学生, 时刻) 上的跨维度编码对（同维度同时刻不构成对）",
             "ONA 是组内成员序列的滞后转移合并，不含跨组拼接",
             "网络坐标（TCA）与测地线路径在 IND10/11 一并给出，本卡先出边表与度统计"]
    return am.result("GRP09", "组级 ENA / ONA 网络", tables, meta=meta,
                     notes=notes, elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ 参数与方法声明

P_WINDOW = ParamSpec("window", "平滑窗长(时点)", "spin", default=0, minimum=0, maximum=60,
                     tooltip="0 = 不平滑；>1 时按该窗长做移动平均")
P_LAG = ParamSpec("lag", "滞后步数", "spin", default=0, minimum=0, maximum=30,
                  tooltip="比较 A(t) 与 B(t+lag)，分母只数两边都有码的位置")
P_PERM = ParamSpec("permutations", "置换次数", "spin", default=200, minimum=0, maximum=2000,
                   tooltip="0 = 不做置换检验（p 列留空）")
P_SEED = ParamSpec("seed", "随机种子", "spin", default=20240517, minimum=0, maximum=999999999)
P_MAX_LAG = ParamSpec("max_lag", "最大滞后", "spin", default=2, minimum=1, maximum=15)
P_Z = ParamSpec("z_threshold", "|Z| 阈值", "dspin", default=1.96, minimum=0.5, maximum=5.0)
P_SCOPE = ParamSpec("scope", "组级口径", "combo",
                    options=((("both"), "两套都要"), ("consensus", "仅共识码"),
                             ("pooled", "仅成员合并")), default="both")
P_FEATURES = ParamSpec("features", "特征族", "checklist",
                       options=am.FEATURE_FAMILY_CHOICES,
                       default=("prop", "entropy", "switch"))
P_MIN_EDGE = ParamSpec("min_edge", "最小边权", "spin", default=0, minimum=0, maximum=500)

SPECS = [
    am.MethodSpec(id="GRP01", name="组状态分布", group=GROUP_STATE,
                  question="每一时刻全组在做什么？", mode=MODE_GROUP,
                  compute=group_state_distribution, units=GROUPS_UNITS, params=(P_WINDOW,),
                  formula_note="group_state_share(t) = 该时刻取该码的成员数 / 该时刻有码成员数"),
    am.MethodSpec(id="GRP02", name="组内熵", group=GROUP_STATE,
                  question="组内意见/行为的分散程度如何随时间变化？", mode=MODE_GROUP,
                  compute=group_entropy, units=GROUPS_UNITS,
                  formula_note="H_t = −Σ p_k ln p_k（p 为该时刻成员构成），H_norm 分母 ln(有码成员数)"),
    am.MethodSpec(id="GRP03", name="同步度", group=GROUP_STATE,
                  question="谁和谁在做同一件事？", mode=MODE_GROUP,
                  compute=synchrony, units=GROUPS_UNITS, requires=("students>=2",),
                  params=(P_LAG, P_PERM, P_SEED),
                  formula_note="agreement = 同码位置数 / 两边都有码的位置数；置换检验用循环移位"),
    am.MethodSpec(id="GRP04", name="领先-滞后", group=GROUP_TRANSITION,
                  question="谁先动、谁跟着动？", mode=MODE_GROUP,
                  compute=lead_lag, units=GROUPS_UNITS, requires=("students>=2",),
                  params=(P_MAX_LAG,),
                  formula_note="最优 lag = 超额一致率(agreement − 机会一致率) 最大的滞后"),
    am.MethodSpec(id="GRP05", name="组级转移 + LSA", group=GROUP_TRANSITION,
                  question="组作为整体的行为走向是什么？", mode=MODE_GROUP,
                  compute=group_transitions, units=GROUPS_UNITS,
                  params=(P_MAX_LAG, P_Z, P_SCOPE), r_reference="lsa_residual.R",
                  formula_note="两套矩阵：consensus（每时刻众数码）与成员合并（转移计数相加）"),
    am.MethodSpec(id="GRP06", name="组内异质性", group=GROUP_COMPARE,
                  question="组内差异有多大、组间差异显著吗？", mode=MODE_GROUP,
                  compute=heterogeneity, units=GROUPS_UNITS, params=(P_FEATURES,),
                  formula_note="CV = s/mean；Gini 用成员特征值；跨组用 Mann–Whitney + BH"),
    am.MethodSpec(id="GRP09", name="组级 ENA / ONA", group=GROUP_NETWORK,
                  question="全组合起来是一张什么样的网？", mode=MODE_GROUP,
                  compute=group_network, units=GROUPS_UNITS,
                  params=(P_LAG, P_MIN_EDGE,), r_reference="rena_network.R",
                  formula_note="ENA 共现单位 = (学生, 时刻) 跨维度编码对；ONA = 组内成员转移合并"),
]
