# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""文档 §6.1 特征字典 V1 的唯一实现。

IND12–IND14、GRP06、GRP10 与导出都要吃同一份特征，口径必须只有一个出处：这里算
`freq_/prop_/run_*/switch_rate_/entropy_/transition_entropy_/self_transition_/
top_transition_`，后面几个族（`om_to_centroid`、`dtw_to_centroid`、`pattern_support_`、
`hmm_state_entropy`、`embedding_`）依赖距离/模型内核，由各自模块经 `register_feature`
挂进来，列名前缀仍在本文件登记，导出的 `feature_dictionary` 表因此永远是全量的。

单位口径（计划 §3 统一口径）：比例的分母一律是**该序列的有效时点数**，不是全组总计数。
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

FEATURE_PREFIXES = ("freq_", "prop_", "run_mean_", "run_median_", "run_max_", "run_count_",
                    "switch_rate_", "entropy_", "transition_entropy_", "self_transition_",
                    "top_transition_", "pattern_support_", "om_to_centroid", "dtw_to_centroid",
                    "hmm_state_entropy", "embedding_")

_BASE_COLUMNS = ("student_uid", "group_id", "student", "dimension", "n_steps", "n_valid",
                 "n_missing", "n_states")

_REGISTRY = {}


def register_feature(name, function):
    """把依赖重特征族挂进来（阶段 6/7 的 OM/DTW/HMM/embedding）。"""
    _REGISTRY[str(name)] = function
    return function


def extra_features():
    return dict(_REGISTRY)


def valid_codes(sequence):
    return [code for code in sequence.codes if code]


def run_lengths(values):
    """连续相同编码合并后的段长列表（空序列给空列表）。"""
    if not values:
        return []
    lengths, previous = [1], values[0]
    for code in values[1:]:
        if code == previous:
            lengths[-1] += 1
        else:
            lengths.append(1)
        previous = code
    return lengths


def switch_rate(values):
    """Σ I(S_t ≠ S_{t+1}) / (T−1)：T 为有效时点数，不足 2 个点时为 0。"""
    if len(values) < 2:
        return 0.0
    changes = sum(1 for left, right in zip(values[:-1], values[1:]) if left != right)
    return changes / float(len(values) - 1)


def transition_entropy(values):
    """条件熵 H(X_{t+1} | X_t) = −Σ_x p(x) Σ_y p(y|x) ln p(y|x)（ln 口径）。"""
    if len(values) < 2:
        return 0.0
    counts = {}
    totals = {}
    for left, right in zip(values[:-1], values[1:]):
        counts[(left, right)] = counts.get((left, right), 0) + 1
        totals[left] = totals.get(left, 0) + 1
    total = float(sum(totals.values()))
    entropy = 0.0
    for left, row_total in totals.items():
        p_left = row_total / total
        inner = 0.0
        for (source, _target), count in counts.items():
            if source != left:
                continue
            p_next = count / float(row_total)
            if p_next > 0:
                inner -= p_next * math.log(p_next)
        entropy += p_left * inner        # inner 已是非负的条件熵，加权求和不再取负
    return entropy


def self_transition(values):
    """Σ N_ii / Σ N_ij：自转移占全部转移的比例，无转移时为 0。"""
    if len(values) < 2:
        return 0.0
    same = sum(1 for left, right in zip(values[:-1], values[1:]) if left == right)
    return same / float(len(values) - 1)


def top_transition(values):
    """出现次数最多的转移（"A→B"）与其占比，并列时按字典序取第一个。"""
    if len(values) < 2:
        return "", 0.0
    counts = {}
    for left, right in zip(values[:-1], values[1:]):
        counts[(left, right)] = counts.get((left, right), 0) + 1
    total = float(sum(counts.values()))
    best = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return "%s→%s" % best[0], best[1] / total


def sequence_features(sequence, codes=None):
    """一条序列 → 一个特征字典（列名严格照 §6.1）。"""
    values = valid_codes(sequence)
    counts = pd.Series(values).value_counts() if values else pd.Series(dtype=int)
    universe = list(codes) if codes else list(counts.index)
    total = float(len(values))
    features = {
        "student_uid": sequence.uid,
        "group_id": sequence.group_id,
        "student": sequence.student,
        "dimension": sequence.dimension,
        "n_steps": int(sequence.length),
        "n_valid": int(len(values)),
        "n_missing": int(sequence.length - len(values)),
        "n_states": int(len(counts)),
        "entropy": round(entropy_of(values), 6),
        "entropy_norm": round(normalized_entropy_of(values), 6),
        "switch_rate": round(switch_rate(values), 6),
        "self_transition": round(self_transition(values), 6),
        "transition_entropy": round(transition_entropy(values), 6),
    }
    lengths = run_lengths(values)
    features["run_count"] = int(len(lengths))
    features["run_mean"] = round(float(np.mean(lengths)), 6) if lengths else 0.0
    features["run_median"] = round(float(np.median(lengths)), 6) if lengths else 0.0
    features["run_max"] = int(max(lengths)) if lengths else 0
    lead, share = top_transition(values)
    features["top_transition"] = lead
    features["top_transition_share"] = round(share, 6)
    for code in universe:
        count = int(counts.get(code, 0))
        features["freq_%s" % code] = count
        features["prop_%s" % code] = round(count / total, 6) if total else 0.0
    for name, function in _REGISTRY.items():
        try:
            features.update(function(sequence, values))
        except Exception:                       # noqa: BLE001  可选后端缺席不该拖垮整表
            continue
    return features


def entropy_of(values):
    counts = pd.Series(values).value_counts()
    total = float(counts.sum())
    if not total:
        return 0.0
    probabilities = counts.to_numpy(dtype=float) / total
    return float(-(probabilities * np.log(probabilities)).sum())


def normalized_entropy_of(values):
    unique = set(values)
    if len(unique) <= 1:
        return 0.0
    return entropy_of(values) / float(math.log(len(unique)))


def feature_frame(sequences, codes_by_dimension=None):
    """序列集合 → 特征表：列序稳定，缺码位补 0 以保证跨样本可比。"""
    rows = []
    for sequence in sequences:
        codes = (codes_by_dimension or {}).get(sequence.dimension)
        rows.append(sequence_features(sequence, codes))
    if not rows:
        return pd.DataFrame(columns=list(_BASE_COLUMNS))
    frame = pd.DataFrame(rows)
    ordered = [column for column in _BASE_COLUMNS if column in frame.columns]
    numeric = [column for column in frame.columns
               if column not in ordered and not is_text_feature(column)]
    # 身份列（student_uid / group_id / …）既是文本列又已在 ordered 里，不能再收一遍
    text = [column for column in frame.columns
            if is_text_feature(column) and column not in ordered]
    ordered += sorted(numeric) + sorted(text)
    rest = [column for column in frame.columns if column not in ordered]
    return frame[ordered + rest].fillna(0)


def is_text_feature(name):
    """纯文本列（不参与数值聚合）：维度标签是例外，它要参与分组。"""
    return name in ("student_uid", "group_id", "student", "dimension",
                    "top_transition") or bool(re.match(r"^pattern_support_", name))


def dictionary_table(frame):
    """特征字典说明书：列名 / 族 / 口径，随结果一起导出（FR-09 可追溯）。"""
    rows = []
    for column in frame.columns:
        family = next((prefix for prefix in FEATURE_PREFIXES if column.startswith(prefix)),
                      "base")
        rows.append({"column": column, "family": family,
                     "unit": "student×dimension 序列",
                     "denominator": "该序列的有效时点数" if column.startswith("prop_") else "—"})
    return pd.DataFrame(rows)
