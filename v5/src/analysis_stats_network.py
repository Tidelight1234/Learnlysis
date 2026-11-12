# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""网络层（阶段 6）：IND10 ENA/TCA、IND11 ONA、GRP07 角色发现、GRP08 交互网络 SNA。

这一层仍然只吃 ``AnalysisContext``，不碰控件；图指标统一走 networkx（核心依赖，
``requirements.txt`` 已固定版本），公式细节全部写进 ``formula_note`` 与导出说明。

口径约定（与 ``analysis_stats_sequence`` / ``analysis_stats_group`` 一致）：
- **共现** = 同一 (成员, 时点) 上跨维度同时出现的编码对，同维度同时刻不构成对；
- **时序网** = 同一序列内部的滞后转移，绝不跨组、也不跨学生的时间断点；
- **网络坐标（TCA）** = 行归一画像 − 参照组均值 → SVD，坐标即「相对参照组的主轴距离」；
- 熵一律 ln，比例必报分母，p 值给 BH-FDR 校正后的 q，未校正 p 也留着。
"""

from __future__ import annotations

import itertools
import math
import time

import networkx as nx
import numpy as np
import pandas as pd

import analysis_core as ac
import analysis_features as af
import analysis_methods as am
from analysis_methods import (GROUP_MODEL, GROUP_NETWORK, GROUP_STATE,
                              MODE_GROUP, MODE_INDIVIDUAL, ParamSpec)
from analysis_stats_sequence import (ahc_clusters, blocked, normal_two_sided_p,
                                     pam_clusters, pick_sequences, states_of,
                                     silhouette_scores, token_stream,
                                     valid_codes)
from statistic_visualization_data import QAItem, fmt_round

MIN_UNITS = 2             # 参照/焦点两侧至少各 2 个分析单元，否则对比只是噪声


def _start():
    return time.perf_counter()


def _ms(marker):
    return int((time.perf_counter() - marker) * 1000)


def wanted_groups(ctx, params):
    return am.chosen(ctx.group_ids, params.get("groups"))


def wanted_dimensions(ctx, params):
    keys = [ctx.dimension_key(item) for item in (params.get("dimensions") or ctx.dimensions)]
    return list(dict.fromkeys(keys)) or ctx.dimension_keys()


def wanted_uids(ctx, params):
    """选中的分析单元（学生号）；组模式没有 students 行时返回全部。"""
    return set(am.chosen(ctx.students, params.get("students")))


# ------------------------------------------------------------------ 共现网格与边表

def member_grid(ctx, params, dimensions=None):
    """(组 × 成员 × 时点 × 维度) 有码格：IND10 共现网与 GRP07 画像的共同出发点。

    返回的行口径与 ``analysis_r_bridge._member_grid`` 一致（R 侧读的就是这张
    grid.csv），差别只在这里额外带了完整 ``uid``，方便逐学生出结果表。
    """
    dims = dimensions or wanted_dimensions(ctx, params)
    uids = wanted_uids(ctx, params)
    rows = []
    for group_id in wanted_groups(ctx, params):
        bundle = ctx.bundle(group_id)
        if bundle is None:
            continue
        for member in bundle.members:
            uid = ac.student_uid(group_id, member)
            if uids and uid not in uids:
                continue
            for dimension in dims:
                _points, matrix = bundle.member_matrix(dimension)
                codes = matrix.get(member, [])
                for position, code in enumerate(codes):
                    if code:
                        rows.append({"group_id": str(group_id), "member": str(member),
                                     "uid": uid, "dimension": dimension,
                                     "position": position, "code": code})
    return rows, dims


def cooccurrence_rows(grid, ctx):
    """grid → 逐 (组, 成员) 的跨维度共现计数 {(group, member, node_a, node_b): 次数}。

    节点名 = 「维度显示名:码」（与 ``analysis_stats_group.pooled_cooccurrence``、
    ``r_refs/rena_network.R`` 的 cooccurrence 表同名同口径）。
    """
    buckets = {}
    for row in grid:
        key = (row["group_id"], row["member"], row["position"])
        buckets.setdefault(key, []).append((row["dimension"], row["code"]))
    edges = {}
    for (group_id, member, _position), active in buckets.items():
        for (dim_a, code_a), (dim_b, code_b) in itertools.combinations(active, 2):
            if dim_a == dim_b:
                continue
            left, right = sorted(((dim_a, code_a), (dim_b, code_b)))
            node_a = "%s:%s" % (ctx.label_of(left[0]), left[1])
            node_b = "%s:%s" % (ctx.label_of(right[0]), right[1])
            key = (group_id, member, node_a, node_b)
            edges[key] = edges.get(key, 0) + 1
    return edges


def cooccurrence_frame(ctx, params, dimensions=None):
    """逐成员 ENA 共现边表：group_id / member / uid / node_a / node_b / weight。"""
    grid, _dims = member_grid(ctx, params, dimensions)
    min_edge = int(params.get("min_edge") or 0)
    uid_of = {(row["group_id"], row["member"]): row["uid"] for row in grid}
    rows = []
    for (group_id, member, node_a, node_b), weight in sorted(
            cooccurrence_rows(grid, ctx).items(), key=lambda item: (-item[1], item[0])):
        if weight < min_edge:
            continue
        rows.append({"group_id": group_id, "member": member,
                     "uid": uid_of.get((group_id, member), group_id),
                     "node_a": node_a, "node_b": node_b, "weight": int(weight)})
    return pd.DataFrame(rows, columns=["group_id", "member", "uid",
                                       "node_a", "node_b", "weight"])


def dimension_pairs(ctx, params, dimensions=None):
    """维度对比：共现对按「维度 A × 维度 B」汇总，看哪两维的行为最常捆在一起。"""
    grid, dims = member_grid(ctx, params, dimensions)
    counts = {}
    for (group_id, member, node_a, node_b), weight in cooccurrence_rows(grid, ctx).items():
        key = tuple(sorted((node_a.split(":", 1)[0], node_b.split(":", 1)[0])))
        counts[key] = counts.get(key, 0) + weight
    total = float(sum(counts.values()))
    rows = [{"维度A": key[0], "维度B": key[1], "共现次数": int(weight),
             "占全部共现": fmt_round(weight / total, 4) if total else 0.0}
            for key, weight in sorted(counts.items(), key=lambda item: -item[1])]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ TCA 网络坐标

def profile_matrix(edges, index="uid", columns=("node_a", "node_b"), value="weight"):
    """边表 → 分析单元 × 节点 的加权画像（无向网：节点在两端各计一次权）。"""
    rows = list(pd.unique(edges[index]))
    nodes = []
    for column in columns:
        for node in edges[column]:
            if node not in nodes:
                nodes.append(node)
    row_index = {name: position for position, name in enumerate(rows)}
    col_index = {name: position for position, name in enumerate(nodes)}
    matrix = np.zeros((len(rows), len(nodes)), dtype=float)
    for record in edges.itertuples(index=False):
        weight = float(getattr(record, value))
        row = row_index[getattr(record, index)]
        matrix[row, col_index[getattr(record, columns[0])]] += weight
        matrix[row, col_index[getattr(record, columns[1])]] += weight
    return rows, nodes, matrix


def row_normalize(matrix):
    """行归一（每行的权重和为 1）：全零行原样保留，不参与比较。"""
    totals = matrix.sum(axis=1, keepdims=True)
    out = np.zeros_like(matrix, dtype=float)
    live = totals[:, 0] > 0
    out[live] = matrix[live] / totals[live]
    return out


def tca_coordinates(profile, reference=None, axes=2):
    """TCA / 网络坐标：中心化后 SVD，``reference`` 是参照行的布尔掩码。

    - ``reference`` 给了 → 减去参照组均值（ENA 的 focal vs reference 口径）；
    - 没给 → 减去总均值（等价于对画像做 PCA）。
    坐标 = U·Σ 的前 k 列；载荷 = Σ·Vᵀ 的前 k 行，两套坐标同一把尺子。
    """
    matrix = np.asarray(profile, dtype=float)
    if not matrix.size:
        return {"units": np.zeros((0, 0)), "nodes": np.zeros((0, 0)),
                "explained": [], "values": []}
    if reference is None or not np.any(reference):
        base = matrix.mean(axis=0)
    else:
        base = matrix[reference].mean(axis=0)
    centered = matrix - base
    # 中心化后秩最多 min(n, p) − 1：只 decompose 有效子空间，避免零奇异值污染解释率
    left = max(centered.shape[0] - 1, 1)
    right = max(centered.shape[1] - 1, 1)
    k = int(max(min(axes, left, right), 1))
    ucd, svd, vtr = np.linalg.svd(centered, full_matrices=False)
    total = float((svd ** 2).sum())
    keep = min(k, len(svd))
    coords = ucd[:, :keep] * svd[:keep]
    loadings = vtr[:keep].T * svd[:keep]
    return {"units": coords, "nodes": loadings,
            "values": [float(value) for value in svd[:keep]],
            "explained": [float(value * value / total) if total else 0.0
                          for value in svd[:keep]]}


def welch_contrast(focal, reference):
    """焦点 vs 参照：均值差 + 大样本 z（差 / 合并标准误）+ 双侧 p（正态近似）。"""
    focal = np.asarray(list(focal), dtype=float)
    reference = np.asarray(list(reference), dtype=float)
    if len(focal) < MIN_UNITS or len(reference) < MIN_UNITS:
        return {"n_focal": int(len(focal)), "n_reference": int(len(reference)),
                "差": 0.0, "z": 0.0, "p": 1.0}
    difference = float(focal.mean() - reference.mean())
    variance = float(focal.var(ddof=1) / len(focal) + reference.var(ddof=1) / len(reference))
    z = difference / math.sqrt(variance) if variance > 0 else 0.0
    return {"n_focal": int(len(focal)), "n_reference": int(len(reference)),
            "差": difference, "z": z, "p": normal_two_sided_p(z)}


# ------------------------------------------------------------------ 图指标


def build_graph(edges, source="source", target="target", weight="weight", directed=True):
    """边表 → networkx 图（自环丢掉：ONA/ENA 的自环只是重复编码，不是关系）。"""
    graph = nx.DiGraph() if directed else nx.Graph()
    if not len(edges):
        return graph
    for record in edges.itertuples(index=False):
        left, right = str(getattr(record, source)), str(getattr(record, target))
        value = float(getattr(record, weight))
        graph.add_node(left)
        graph.add_node(right)
        if left == right or value <= 0:
            continue
        graph.add_edge(left, right, weight=value)
    return graph


def weighted_reciprocity(graph):
    """加权互惠率 = Σ_{i<j} 2·min(w_ij, w_ji) / Σ_{i<j} (w_ij + w_ji)。"""
    total, mutual = 0.0, 0.0
    for left, right in itertools.combinations(sorted(graph.nodes()), 2):
        forward = graph.get_edge_data(left, right, {}).get("weight", 0.0) if graph.is_directed() else 0.0
        backward = graph.get_edge_data(right, left, {}).get("weight", 0.0) if graph.is_directed() else 0.0
        total += forward + backward
        mutual += 2.0 * min(forward, backward)
    return mutual / total if total > 0 else 0.0


def graph_metrics(graph, directed=True):
    """一张图的宏观指标：规模、密度、互惠、传递性、最大弱连通分量的平均最短路。"""
    nodes = graph.number_of_nodes()
    if not graph.number_of_edges() or nodes < 2:
        return {}
    undirected = graph.to_undirected()
    components = sorted(nx.connected_components(undirected), key=len, reverse=True)
    largest = components[0]
    span = len(largest)
    pairs = [length for source in largest for target, length in
             nx.single_source_shortest_path_length(undirected, source).items()
             if target in largest and target != source]
    out = {"节点数": int(nodes), "边数": int(graph.number_of_edges()),
           "密度": fmt_round(nx.density(graph), 4),
           "平均度": fmt_round(2.0 * graph.number_of_edges() / nodes, 4),
           "互惠率": fmt_round(nx.reciprocity(graph) if directed else 1.0, 4),
           "加权互惠率": fmt_round(weighted_reciprocity(graph) if directed else 1.0, 4),
           "传递性": fmt_round(nx.transitivity(undirected), 4),
           "弱连通分量数": int(len(components)),
           "最大分量节点数": int(span),
           "平均最短路径": fmt_round(float(np.mean(pairs)) if pairs else 0.0, 4)}
    if directed:
        out["出度均值"] = fmt_round(float(np.mean(
            [value for _node, value in graph.out_degree()])), 4)
        out["入度均值"] = fmt_round(float(np.mean(
            [value for _node, value in graph.in_degree()])), 4)
    return out


def degree_frame(graph, directed=True, label="节点"):
    """逐节点度表：出/入度（无权）、出/入强度（加权）、中心性与互惠伙伴数。"""
    rows = []
    for node in sorted(graph.nodes()):
        out_degree = graph.out_degree(node) if directed else graph.degree(node)
        in_degree = graph.in_degree(node) if directed else graph.degree(node)
        out_strength = sum(data.get("weight", 0.0) for _n, _t, data in graph.out_edges(node, data=True)) \
            if directed else sum(data.get("weight", 0.0) for _n, _t, data in graph.edges(node, data=True))
        in_strength = sum(data.get("weight", 0.0) for _n, _t, data in graph.in_edges(node, data=True)) \
            if directed else out_strength
        partners = set(graph.successors(node)) & set(graph.predecessors(node)) if directed else set()
        total = (out_degree + in_degree) if directed else out_degree
        rows.append({label: node,
                     "出度": int(out_degree), "入度": int(in_degree),
                     "出强度": fmt_round(out_strength, 2), "入强度": fmt_round(in_strength, 2),
                     "度中心性": fmt_round(total / max(graph.number_of_nodes() - 1, 1), 4),
                     "互惠伙伴数": int(len(partners))})
    frame = pd.DataFrame(rows)
    if len(frame):
        frame = frame.sort_values(["出强度", "入强度"], ascending=False).reset_index(drop=True)
    return frame


# ------------------------------------------------------------------ IND10 ENA

def ena_analysis(ctx, params):
    """IND10：个体层 ENA —— 逐学生加权共现网、网络坐标（TCA）、边权与维度对比。"""
    marker = _start()
    dimensions = wanted_dimensions(ctx, params)
    if len(dimensions) < 2:
        return blocked("IND10", "共现网络 ENA / TCA",
                       "ENA 的共现是跨维度编码对：至少需要 2 个维度，当前只选了 %d 个"
                       % len(dimensions))
    edges = cooccurrence_frame(ctx, params, dimensions)
    if not len(edges):
        return blocked("IND10", "共现网络 ENA / TCA",
                       "没有任何时刻出现「同一成员的多个维度都有码」，共现网建不起来")
    rows, nodes, matrix = profile_matrix(edges)
    profile = row_normalize(matrix)
    contrast_by = (params.get("contrast_by") or "none").lower()
    reference = None
    focal_label = ""
    group_of = {}
    for uid, group_id in zip(edges["uid"], edges["group_id"]):
        group_of.setdefault(uid, group_id)
    groups = [group_of.get(uid, "") for uid in rows]
    if contrast_by == "group":
        picked = [value for value in dict.fromkeys(groups) if value]
        if len(picked) < 2:
            return blocked("IND10", "共现网络 ENA / TCA",
                           "按组对比至少需要 2 个组，当前只有 1 组：请改用「整体主轴」")
        focal_label = picked[0]
        reference = np.asarray([value != focal_label for value in groups], dtype=bool)
    axes = max(2, min(int(params.get("axes") or 3), 5))
    tca = tca_coordinates(profile, reference, axes)
    coords = tca["units"]
    axis_names = ["坐标%d" % (number + 1) for number in range(coords.shape[1])]
    unit_rows = []
    for position, uid in enumerate(rows):
        record = {"uid": uid, "group_id": group_of.get(uid, ""),
                  "共现边数": int(len(edges[edges["uid"] == uid])),
                  "共现强度": fmt_round(float(profile[position].sum()), 2)}
        for number, name in enumerate(axis_names):
            record[name] = fmt_round(float(coords[position, number]), 4)
        unit_rows.append(record)
    load_rows = []
    for position, node in enumerate(nodes):
        record = {"node": node}
        for number, name in enumerate(axis_names):
            record[name] = fmt_round(float(tca["nodes"][position, number]), 4)
        load_rows.append(record)
    tables = {"edges": edges, "nodes": _node_summary(edges),
              "coordinates": pd.DataFrame(unit_rows),
              "loadings": pd.DataFrame(load_rows),
              "dimension_pairs": dimension_pairs(ctx, params, dimensions)}
    tables["tca"] = pd.DataFrame([{
        "轴": axis_names[number], "奇异值": fmt_round(tca["values"][number], 4),
        "解释方差%": fmt_round(tca["explained"][number] * 100.0, 2),
        "累计解释方差%": fmt_round(sum(tca["explained"][:number + 1]) * 100.0, 2)}
        for number in range(len(tca["values"]))])
    qa = []
    if contrast_by == "group" and reference is not None:
        contrast = []
        for number, node in enumerate(nodes):
            found = welch_contrast(profile[~reference, number], profile[reference, number])
            contrast.append({"node": node, "焦点组": focal_label, "n_focal": found["n_focal"],
                             "n_reference": found["n_reference"],
                             "差": fmt_round(found["差"], 4), "z": fmt_round(found["z"], 4),
                             "p": fmt_round(found["p"], 6)})
        frame = pd.DataFrame(contrast)
        if len(frame):
            frame["q(BH)"] = [fmt_round(value, 6) for value in
                              am.bh_fdr([float(value) for value in frame["p"]])]
            frame = frame.sort_values("q(BH)").reset_index(drop=True)
            tables["contrast"] = frame
            if int(params.get("min_q") or 0) and not len(
                    frame[pd.to_numeric(frame["q(BH)"], errors="coerce") <=
                        float(params.get("min_q"))]):
                qa.append(QAItem("warn", "IND10.no_signal",
                                 "按组对比没有任何节点在 q ≤ %g 上显著：两组共现结构差异不明显"
                                 % float(params["min_q"])))
    graph = build_graph(edges.rename(columns={"node_a": "source", "node_b": "target"}),
                        directed=False)
    metrics = graph_metrics(graph, directed=False)
    if metrics:
        tables["metrics"] = pd.DataFrame([metrics])
    explained = tca["explained"]
    if explained and explained[0] < 0.10:
        qa.append(QAItem("warn", "IND10.flat_axes",
                         "第一轴只解释 %.1f%% 方差：共现结构几乎没有主轴，坐标别过度解读"
                         % (explained[0] * 100.0)))
    meta = {"n_units": len(rows), "n_nodes": len(nodes), "n_edges": int(len(edges)),
            "axes": len(axis_names), "contrast_by": contrast_by,
            "axis1_explained": fmt_round(explained[0] * 100.0, 2) if explained else 0.0,
            "stream": "grid"}
    notes = ["共现单位 = 同一 (成员, 时点) 上的跨维度编码对；同维度同时刻不构成对",
             "画像先按行归一（每学生的共现强度和为 1），再减去%s → SVD 取前 %d 轴"
             % ("参照组均值（焦点 = %s）" % focal_label if reference is not None else "总均值",
                len(axis_names)),
             "坐标 = U·Σ 前 k 列，载荷 = Σ·Vᵀ 前 k 行，两者同一尺度可直接叠图"]
    if contrast_by == "group":
        notes.append("焦点/参照按组划分：坐标读作「相对参照组」的偏离，不是绝对水平")
    return am.result("IND10", "共现网络 ENA / TCA", tables, meta=meta, qa=qa,
                     notes=notes, elapsed_ms=_ms(marker))


def _node_summary(edges):
    """节点度汇总（无向）：强度、度数、参与的学生数。"""
    if not len(edges):
        return pd.DataFrame()
    rows = {}
    for record in edges.itertuples(index=False):
        for node in (record.node_a, record.node_b):
            found = rows.setdefault(node, {"weight": 0.0, "pairs": set(), "uids": set()})
            found["weight"] += float(record.weight)
            found["pairs"].add(tuple(sorted((record.node_a, record.node_b))))
            found["uids"].add(record.uid)
    out = [{"node": node, "强度": fmt_round(value["weight"], 2),
            "度数": int(len(value["pairs"])), "出现学生数": int(len(value["uids"]))}
           for node, value in rows.items()]
    return pd.DataFrame(out).sort_values("强度", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ IND11 ONA

def ona_edge_frame(sequences, ctx, lag=1, min_edge=0, compress=False):
    """逐 (学生 × 维度) 的有向时序边表：source / target / weight（自环剔除）。"""
    rows = []
    for sequence in sequences:
        values = token_stream(sequence, "compressed" if compress else "raw")
        if len(values) <= lag:
            continue
        weights = {}
        for position in range(len(values) - lag):
            left, right = values[position], values[position + lag]
            if left == right:
                continue
            weights[(left, right)] = weights.get((left, right), 0) + 1
        out_degree = {state: sum(value for (source, _t), value in weights.items() if source == state)
                      for state in {item[0] for item in weights}}
        in_degree = {state: sum(value for (_s, target), value in weights.items() if target == state)
                     for state in {item[1] for item in weights}}
        total = float(sum(weights.values()))
        for (left, right), value in sorted(weights.items(), key=lambda item: (-item[1], item[0])):
            if value < min_edge:
                continue
            rows.append({"uid": sequence.uid, "group_id": sequence.group_id,
                         "dimension": ctx.label_of(sequence.dimension),
                         "source": left, "target": right, "weight": int(value),
                         "占全网": fmt_round(value / total, 4) if total else 0.0,
                         "出度(source)": int(out_degree.get(left, 0)),
                         "入度(target)": int(in_degree.get(right, 0)),
                         "互惠": int((right, left) in weights)})
    return pd.DataFrame(rows)


def ona_analysis(ctx, params):
    """IND11：个体层 ONA —— 有向时序网络、边权、入/出度与图指标。"""
    marker = _start()
    lag = max(1, int(params.get("lag") or 1))
    min_edge = int(params.get("min_edge") or 0)
    sequences = pick_sequences(ctx, params)
    if not sequences:
        return blocked("IND11", "有向时序网络 ONA",
                       "没有足够长的序列（有效码 < %d 个）" % am.MIN_SEQUENCE_LENGTH)
    edges = ona_edge_frame(sequences, ctx, lag, min_edge,
                           bool(params.get("compress", False)))
    if not len(edges):
        return blocked("IND11", "有向时序网络 ONA",
                       "剔除自环后一条边都不剩：序列太单一或滞后太大")
    tables = {"edges": edges, "transitions": edges.groupby(
        ["source", "target"], as_index=False)["weight"].sum().sort_values(
        "weight", ascending=False).reset_index(drop=True)}
    metric_rows, node_rows = [], []
    for (uid, dimension), part in edges.groupby(["uid", "dimension"], sort=False):
        graph = build_graph(part)
        found = graph_metrics(graph, directed=True)
        if not found:
            continue
        found.update({"uid": uid, "dimension": dimension})
        metric_rows.append(found)
        frame = degree_frame(graph, directed=True)
        frame.insert(0, "dimension", dimension)
        frame.insert(0, "uid", uid)
        node_rows.append(frame)
    if not metric_rows:
        return blocked("IND11", "有向时序网络 ONA", "所有子图都只有一条边，指标无从谈起")
    tables["nodes"] = pd.concat(node_rows, ignore_index=True)
    tables["metrics"] = pd.DataFrame(metric_rows)
    pooled = build_graph(edges.groupby(["source", "target"], as_index=False)["weight"].sum())
    overall = graph_metrics(pooled, directed=True)
    if overall:
        tables["pooled_metrics"] = pd.DataFrame([overall])
    qa = []
    if len(edges["uid"].unique()) < 2:
        qa.append(QAItem("warn", "IND11.single_unit", "只有 1 个分析单元有边，网络比较无从谈起"))
    top = tables["transitions"].head(1)
    meta = {"lag": lag, "n_units": int(edges["uid"].nunique()),
            "n_edges": int(len(edges)), "n_nodes": int(len(set(edges["source"]) |
                                                           set(edges["target"]))),
            "top_edge": "%s→%s" % (top.iloc[0]["source"], top.iloc[0]["target"]) if len(top) else "",
            "min_edge": min_edge}
    notes = ["边 = 同一序列内部滞后 %d 步的编码转移，自环不计、跨组不接" % lag,
             "密度 = 边数 / N(N−1)（有向、不含自环）；互惠率 = 双向成对的弧占比",
             "平均最短路径只在最大弱连通分量内算（忽略方向与权重），分量外不可达"]
    return am.result("IND11", "有向时序网络 ONA", tables, meta=meta, qa=qa,
                     notes=notes, elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ GRP07 角色发现

ROLE_LABELS = {
    "entropy": "行为多样", "entropy_norm": "均衡投入", "switch_rate": "频繁切换",
    "self_transition": "原地保持", "transition_entropy": "转移难测",
    "n_valid": "编码密集", "n_steps": "在场时间长", "participation": "高参与",
    "co_strength": "共现活跃", "co_degree": "关联面广", "out_strength": "主动发起",
    "in_strength": "被动接收", "reciprocity": "互惠对等",
    "top_transition_share": "偏好集中", "max_run": "长时间专注", "run_max": "长专注段",
    "run_mean": "短轮次", "run_median": "轮次中位短", "n_states": "涉及码多",
    "n_missing": "缺码多",
}


def _numeric_feature_columns(frame):
    out = []
    for column in frame.columns:
        if af.is_text_feature(column):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any() and float(values.abs().sum() or 0.0) != 0.0:
            out.append(column)
    return out


def zscore_matrix(frame, columns):
    """逐列 z 分数（分母 = 全体标准差，σ=0 的列全给 0）：跨量纲特征的可比化。"""
    matrix = frame[list(columns)].to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    mean = matrix.mean(axis=0)
    sigma = matrix.std(axis=0, ddof=0)
    out = np.zeros_like(matrix)
    live = sigma > 1e-12
    if np.any(live):
        out[:, live] = (matrix[:, live] - mean[live]) / sigma[live]
    return out, list(columns)


def role_feature_table(ctx, params):
    """每个成员一行角色特征：轨迹特征（``analysis_features``）+ 共现/时序网络特征。"""
    dimensions = wanted_dimensions(ctx, params)
    uids = wanted_uids(ctx, params)
    sequences = [seq for seq in ctx.sequences() if seq.uid in uids
                 and ctx.dimension_key(seq.dimension) in dimensions]
    if not sequences:
        return pd.DataFrame()
    base = af.feature_frame(sequences, {item: ctx.codebook_of(item) for item in dimensions})
    numeric = _numeric_feature_columns(base)
    kept = [column for column in ["student_uid", "group_id", "dimension"] + numeric
            if column in base.columns]
    part = base[kept].copy()
    part["n_valid_total"] = part["n_valid"].fillna(0.0) if "n_valid" in part else 0.0
    part["n_steps_total"] = part["n_steps"].fillna(0.0) if "n_steps" in part else 0.0
    aggregated = part.groupby(["student_uid", "group_id"], as_index=False)[numeric].mean()
    totals = part.groupby(["student_uid", "group_id"], as_index=False)[
        ["n_valid_total", "n_steps_total"]].sum()
    aggregated = aggregated.merge(totals, on=["student_uid", "group_id"], how="left")
    aggregated["participation"] = np.where(
        aggregated["n_steps_total"] > 0,
        aggregated["n_valid_total"] / aggregated["n_steps_total"].replace(0, np.nan), 0.0)
    aggregated = aggregated.fillna({column: 0.0 for column in aggregated.columns
                                    if column not in ("student_uid", "group_id")})
    edges = cooccurrence_frame(ctx, dict(params, min_edge=0), dimensions)
    if len(edges):
        strength = edges.groupby("uid")["weight"].sum()
        spread = {}
        for record in edges.itertuples(index=False):
            found = spread.setdefault(record.uid, set())
            found.add(record.node_a)
            found.add(record.node_b)
        aggregated["co_strength"] = aggregated["student_uid"].map(strength).fillna(0.0)
        aggregated["co_degree"] = aggregated["student_uid"].map(
            {key: len(value) for key, value in spread.items()}).fillna(0)
    ona = ona_edge_frame(sequences, ctx, max(1, int(params.get("lag") or 1)))
    if len(ona):
        totals = ona.groupby("uid")["weight"].sum()
        mutual = ona[ona["互惠"] == 1].groupby("uid")["weight"].sum()
        aggregated["out_strength"] = aggregated["student_uid"].map(totals).fillna(0.0)
        aggregated["reciprocity"] = (aggregated["student_uid"].map(mutual).fillna(0.0) /
                                     aggregated["student_uid"].map(totals).replace(0, np.nan)
                                     ).fillna(0.0)
    return aggregated.rename(columns={"student_uid": "uid"})


ROLE_FEATURES = ["participation", "entropy", "entropy_norm", "switch_rate",
                 "self_transition", "transition_entropy", "run_mean", "run_max",
                 "top_transition_share", "co_strength", "co_degree", "out_strength",
                 "reciprocity"]


def role_analysis(ctx, params):
    """GRP07：角色发现 —— 轨迹 + 网络特征 z 分数 → AHC/PAM 聚类 + 簇画像 + silhouette。"""
    marker = _start()
    table = role_feature_table(ctx, params)
    if len(table) < 4:
        return blocked("GRP07", "角色发现", "至少需要 4 名成员，角色簇才有意义（当前 %d 名）"
                       % len(table))
    columns = [column for column in ROLE_FEATURES if column in table.columns]
    columns = [column for column in _numeric_feature_columns(table)
               if column in columns] or columns
    if len(columns) < 2:
        return blocked("GRP07", "角色发现", "可用特征不足 2 列：请先补充编码数据或减少筛选")
    matrix, columns = zscore_matrix(table, columns)
    distance = np.zeros((len(matrix), len(matrix)), dtype=float)
    for i, j in itertools.combinations(range(len(matrix)), 2):
        value = float(np.sqrt(((matrix[i] - matrix[j]) ** 2).sum()))
        distance[i, j] = distance[j, i] = value
    method = (params.get("cluster_method") or "ahc").lower()
    low, high = 2, max(2, int(params.get("k") or 3))
    k_choices = [max(2, int(params.get("k") or 3))]
    if params.get("select_k", True):
        low = max(2, int(params.get("k_min") or 2))
        high = max(low, int(params.get("k_max") or low + 2))
        k_choices = [value for value in range(low, min(high, len(table) - 1) + 1)
                     if value >= 2]
    trials, best = [], None
    for k in k_choices:
        try:
            if method == "pam":
                clusters, _medoids = pam_clusters(distance, k)
            else:
                clusters, _tree = ahc_clusters(distance, params.get("linkage") or "average", k)
        except ValueError as error:
            return blocked("GRP07", "角色发现", str(error))
        scores, mean_silhouette = silhouette_scores(distance, clusters)
        trials.append({"k": int(k), "silhouette": fmt_round(mean_silhouette, 4),
                       "方法": "PAM" if method == "pam" else "AHC-" + str(
                           params.get("linkage") or "average")})
        if best is None or mean_silhouette > best["silhouette"]:
            best = {"k": int(k), "silhouette": mean_silhouette, "labels": clusters,
                    "scores": scores, "distance": distance}
    if best is None:
        return blocked("GRP07", "角色发现",
                       "候选簇数 %s–%s 超出成员数（%d）：调低 k_min 或多选几名成员"
                       % (low, high, len(table)))
    labels = best["labels"]
    table = table.reset_index(drop=True)
    profile_rows = []
    for column in columns:
        values = pd.to_numeric(table[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        overall = float(values.mean())
        for cluster in sorted(set(labels)):
            picked = [position for position, value in enumerate(labels) if value == cluster]
            mean = float(values[picked].mean()) if picked else 0.0
            sigma = float(values.std(ddof=0)) or 1.0
            profile_rows.append({"特征": column, "簇": int(cluster),
                                 "簇均值": fmt_round(mean, 4), "全体均值": fmt_round(overall, 4),
                                 "偏离(σ)": fmt_round((mean - overall) / sigma, 3)})
    profile = pd.DataFrame(profile_rows)
    names = {}
    for cluster in sorted(set(labels)):
        part = profile[profile["簇"] == cluster].sort_values("偏离(σ)", ascending=False)
        top = [str(row["特征"]) for _i, row in part.head(2).iterrows()
               if float(row["偏离(σ)"]) > 0.2]
        low = [str(row["特征"]) for _i, row in part.tail(1).iterrows()
               if float(row["偏离(σ)"]) < -0.2]
        text = "·".join(ROLE_LABELS.get(item, item) for item in top) or "综合"
        names[cluster] = "%s%s型" % (text, ("（低" + ROLE_LABELS.get(low[0], low[0]) + "）")
                                     if low else "")
    role_rows = []
    for position, record in table.iterrows():
        role_rows.append({"uid": record["uid"], "group_id": record["group_id"],
                          "簇": int(labels[position]), "角色": names[labels[position]],
                          "silhouette": fmt_round(best["scores"][position], 4)})
    roles = pd.DataFrame(role_rows)
    sizes = roles["簇"].value_counts().sort_index()
    roles = roles.merge(sizes.rename("簇规模"), on="簇")
    qa = []
    if best["silhouette"] < 0.1:
        qa.append(QAItem("warn", "GRP07.weak_silhouette",
                         "平均 silhouette = %.3f：角色分开度很弱，建议按整组叙述而非命名角色"
                         % best["silhouette"]))
    tables = {"roles": roles, "profile": profile, "features": table,
              "model_selection": pd.DataFrame(trials),
              "cluster_summary": pd.DataFrame([{
                  "簇": int(cluster), "角色": names[cluster],
                  "规模": int(sizes.get(cluster, 0)),
                  "占比%": fmt_round(int(sizes.get(cluster, 0)) / len(roles) * 100.0, 2),
                  "成员": "、".join(roles[roles["簇"] == cluster]["uid"].tolist())}
                  for cluster in sorted(set(labels))])}
    if roles["group_id"].nunique() > 1:
        # 跨组看角色分布：同一簇在不同组里的占比差得开，才能说“角色”不是组间抽样噪声
        tables["group_cross"] = pd.crosstab(roles["group_id"], roles["簇"]).reset_index()
    notes = ["特征 = 轨迹特征（analysis_features 的熵/切换率/保持率…）+ 网络特征"
             "（共现强度/广度、出强度、互惠率），逐列 z 标准化后等权参与",
             "距离 = z 分数上的欧氏距离；簇数在 %s 中按 silhouette 选优（本次 k=%d）"
             % ("/".join(str(item["k"]) for item in trials), best["k"]),
             "角色名由该簇偏离最大的特征（σ 偏离 > 0.2）自动拼名，只做叙述起点不做定性标签"]
    return am.result("GRP07", "角色发现（轨迹 + 网络聚类）", tables,
                     meta={"k": int(best["k"]), "method": method,
                           "mean_silhouette": fmt_round(best["silhouette"], 4),
                           "n_members": int(len(table)), "features": columns,
                           "roles": {int(key): str(value) for key, value in names.items()}},
                     qa=qa, notes=notes, elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ GRP08 交互网络 SNA

EVENT_RATE_MINUTES = 60.0


def person_edges(ctx, params):
    """事件表 → PERSON→PERSON 有向边表（含每对的应答数与平均时长）。"""
    frame = ctx.events
    if not len(frame):
        return pd.DataFrame()
    picked_groups = {str(value) for value in wanted_groups(ctx, params)}
    part = frame[frame["group_id"].astype(str).isin(picked_groups)]
    part = part[(part["target_type"].astype(str) == "PERSON")
                & (part["actor_id"].astype(str) != "")
                & (part["target_id"].astype(str) != "")]
    part = part[part["actor_id"].astype(str) != part["target_id"].astype(str)]
    if not len(part):
        return pd.DataFrame()
    rows = []
    for (group_id, source, target), subset in part.groupby(
            ["group_id", "actor_id", "target_id"], sort=False):
        durations = pd.to_numeric(subset["duration"], errors="coerce").dropna()
        replies = subset[subset["reply_to"].astype(str) != ""]
        rows.append({"group_id": str(group_id), "source": str(source), "target": str(target),
                     "weight": int(len(subset)),
                     "事件类型数": int(subset["interaction_type"].astype(str).nunique()),
                     "应答次数": int(len(replies)),
                     "平均时长": fmt_round(float(durations.mean()), 3) if len(durations) else 0.0})
    frame = pd.DataFrame(rows)
    if len(frame):
        keys = list(zip(frame["group_id"], frame["source"], frame["target"]))
        index = set(keys)
        frame["互惠"] = [int((group, right, left) in index)
                      for group, left, right in keys]
        total = frame.groupby("group_id")["weight"].transform("sum")
        frame["占全网"] = (frame["weight"] / total).round(4)
    return frame


def reply_latency(ctx, params):
    """应答时延：带 ``reply_to`` 的事件与它指向的事件之间的时间差（同一组内匹配）。"""
    frame = ctx.events
    if not len(frame) or "reply_to" not in frame.columns:
        return pd.DataFrame()
    picked_groups = {str(value) for value in wanted_groups(ctx, params)}
    part = frame[frame["group_id"].astype(str).isin(picked_groups)].copy()
    if not len(part):
        return pd.DataFrame()
    part["timestamp"] = pd.to_numeric(part["timestamp"], errors="coerce")
    origins = {str(row.group_id) + "|" + str(row.event_id): row
               for row in part.itertuples() if str(row.event_id)}
    rows = []
    for record in part.itertuples():
        target = str(record.reply_to or "")
        origin = origins.get(str(record.group_id) + "|" + target) if target else None
        if origin is None:
            continue
        moment = pd.to_numeric(pd.Series([record.timestamp]), errors="coerce").iloc[0]
        base = pd.to_numeric(pd.Series([origin.timestamp]), errors="coerce").iloc[0]
        if pd.isna(moment) or pd.isna(base):
            continue
        rows.append({"group_id": str(record.group_id), "应答人": str(record.actor_id),
                     "被应答人": str(origin.actor_id), "时延": float(moment - base),
                     "轮次": str(record.turn_id or "")})
    return pd.DataFrame(rows)


def interaction_network(ctx, params):
    """GRP08：交互网络 SNA —— 事件计数、入/出度、密度、互惠、类型多样性熵、事件率、应答时延。"""
    marker = _start()
    if not ctx.has_events:
        return blocked("GRP08", "交互网络 SNA",
                       "需要 Interaction_Event 事件表：在「小组交互」页录入或随文件上传")
    picked_groups = [str(value) for value in wanted_groups(ctx, params)]
    events = ctx.events
    events = events[events["group_id"].astype(str).isin(picked_groups)]
    if not len(events):
        return blocked("GRP08", "交互网络 SNA", "所选组里没有事件记录")
    summary_rows, type_rows, node_rows, metric_rows = [], [], [], []
    edge_frame = person_edges(ctx, params)
    latency = reply_latency(ctx, params)
    for group_id, part in events.groupby("group_id", sort=False):
        actor = part["actor_id"].astype(str)
        senders = int((actor != "").sum())
        timestamps = pd.to_numeric(part["timestamp"], errors="coerce").dropna()
        span = float(timestamps.max() - timestamps.min()) if len(timestamps) > 1 else 0.0
        types = part["target_type"].astype(str).replace("", np.nan).dropna()
        interaction = part["interaction_type"].astype(str).replace("", np.nan).dropna()
        summary_rows.append({
            "group_id": str(group_id), "事件数": int(len(part)),
            "有 actor 的事件数": senders,
            "发言人数": int(actor[actor != ""].nunique()),
            "观测时长(秒)": fmt_round(span, 2),
            "事件率(%s分)" % int(EVENT_RATE_MINUTES): fmt_round(
                len(part) / (span / EVENT_RATE_MINUTES) if span > 0 else 0.0, 3),
            "target_type 熵": fmt_round(am.entropy(types.tolist()), 4),
            "target_type 归一化熵": fmt_round(am.normalized_entropy(types.tolist()), 4),
            "interaction_type 熵": fmt_round(am.entropy(interaction.tolist()), 4),
            "同窗多事件数": int(len(part) - part.drop_duplicates(
                ["group_id", "timestamp", "actor_id"]).shape[0]),
            "缺 interaction_type": int((part["interaction_type"].astype(str) == "").sum())})
        for value, count in interaction.value_counts().items():
            type_rows.append({"group_id": str(group_id), "interaction_type": str(value),
                              "次数": int(count),
                              "占比": fmt_round(count / float(len(part)), 4)})
        sub = edge_frame[edge_frame["group_id"] == str(group_id)] if len(edge_frame) \
            else edge_frame
        graph = build_graph(sub)
        metrics = graph_metrics(graph, directed=True)
        if metrics:
            metrics["group_id"] = str(group_id)
            metric_rows.append(metrics)
            frame = degree_frame(graph, directed=True)
            frame.insert(0, "group_id", str(group_id))
            node_rows.append(frame)
    if not summary_rows:
        return blocked("GRP08", "交互网络 SNA", "事件表为空：没有任何可统计的交互记录")
    tables = {"summary": pd.DataFrame(summary_rows),
              "types": pd.DataFrame(type_rows)}
    if len(edge_frame):
        tables["edges"] = edge_frame.sort_values("weight", ascending=False) \
            .reset_index(drop=True)
    if node_rows:
        tables["node_metrics"] = pd.concat(node_rows, ignore_index=True)
    if metric_rows:
        tables["metrics"] = pd.DataFrame(metric_rows)
    qa = []
    if not len(edge_frame):
        qa.append(QAItem("warn", "GRP08.no_person_edges",
                         "没有 PERSON→PERSON 的交互边：网络表为空，只能看事件计数"))
    if len(latency):
        values = latency["时延"].to_numpy(dtype=float)
        tables["latency"] = pd.DataFrame([{
            "group_id": str(group_id), "应答次数": int(len(part)),
            "平均时延": fmt_round(float(part["时延"].mean()), 3),
            "中位时延": fmt_round(float(part["时延"].median()), 3),
            "P90 时延": fmt_round(float(np.percentile(part["时延"].to_numpy(dtype=float), 90)), 3),
            "最长时延": fmt_round(float(part["时延"].max()), 3)}
            for group_id, part in latency.groupby("group_id", sort=False)])
        if float(np.mean(values)) < 0:
            qa.append(QAItem("warn", "GRP08.negative_latency",
                             "存在时延为负的应答链（%d 条）：时间戳顺序需要复核"
                             % int((values < 0).sum())))
    else:
        qa.append(QAItem("info", "GRP08.no_reply_chain",
                         "没有 reply_to 指向已记录事件的应答链，时延表为空"))
    meta = {"n_events": int(len(events)), "n_groups": int(len(summary_rows)),
            "n_person_edges": int(len(edge_frame)),
            "n_nodes": int(len(set(edge_frame["source"]) | set(edge_frame["target"])))
            if len(edge_frame) else 0,
            "replies": int(len(latency))}
    notes = ["边 = PERSON→PERSON 的事件计数（去掉自指与空 actor/target），密度按有向不含自环算",
             "事件率 = 事件数 / 观测时长（首末事件时间跨度），单位 %d 分钟"
             % int(EVENT_RATE_MINUTES),
             "target_type 多样性熵按 ln 口径，归一化分母 = ln(实际出现的类型数)",
             "应答时延 = 带 reply_to 的事件时间戳 − 被应答事件时间戳（同组内按 event_id 匹配）"]
    return am.result("GRP08", "交互网络 SNA", tables, meta=meta, qa=qa,
                     notes=notes, elapsed_ms=_ms(marker))


# ------------------------------------------------------------------ 参数与方法声明

P_MIN_EDGE = ParamSpec("min_edge", "最小边权", "spin", default=0, minimum=0, maximum=500,
                       tooltip="低于该权重的边不进结果表（0 = 全留）")
P_LAG = ParamSpec("lag", "滞后步数", "spin", default=1, minimum=1, maximum=15,
                  tooltip="转移取 t 与 t+lag；只在同一序列内部计数")
P_ONA_COMPRESS = ParamSpec("compress", "在压缩序列上算", "check", default=False,
                           tooltip="合并连续同码会抹掉自转移，ONA 默认用原始序列")
P_AXES = ParamSpec("axes", "保留轴数", "spin", default=3, minimum=2, maximum=5)
P_CONTRAST = ParamSpec("contrast_by", "对比口径", "combo",
                       options=(("none", "整体主轴（不减参照）"), ("group", "按组：焦点 vs 参照")),
                       default="none",
                       tooltip="选「按组」时，焦点 = 组选择里的第一组，其余为参照组")
P_MIN_Q = ParamSpec("min_q", "显著性提示阈值 q", "dspin", default=0.05, minimum=0.001,
                    maximum=0.5,
                    tooltip="仅用于质检提示：没有任何节点 q ≤ 阈值时提醒别过度解读")
P_CLUSTER_METHOD = ParamSpec("cluster_method", "聚类方法", "combo",
                             options=(("ahc", "层次聚类 AHC"), ("pam", "k-medoids PAM")),
                             default="ahc")
P_LINKAGE = ParamSpec("linkage", "链接方式", "combo",
                      options=(("average", "平均"), ("complete", "完全"),
                               ("single", "最短"), ("ward", "Ward")), default="average",
                      shows_for=("cluster_method", "ahc"))
P_K = ParamSpec("k", "簇数 k", "spin", default=3, minimum=2, maximum=8)
P_SELECT_K = ParamSpec("select_k", "自动选簇数", "check", default=True)
P_KMIN = ParamSpec("k_min", "簇数下限", "spin", default=2, minimum=2, maximum=8,
                   shows_for=("select_k", True))
P_KMAX = ParamSpec("k_max", "簇数上限", "spin", default=4, minimum=2, maximum=10,
                   shows_for=("select_k", True))

SPECS = [
    am.MethodSpec(
        id="IND10", name="共现网络 ENA / 网络坐标", group=GROUP_NETWORK,
        question="哪些行为总是同时出现？相对参照组，这个人偏在哪一侧？",
        mode=MODE_INDIVIDUAL, compute=ena_analysis,
        params=(P_MIN_EDGE, P_CONTRAST, P_AXES, P_MIN_Q),
        r_reference="rena_network.R", unit="student",
        formula_note="共现 = 同一(成员,时刻)跨维度编码对；坐标 = 行归一画像减参照均值后 SVD，"
                     "载荷 = Σ·Vᵀ；密度按无向不含自环"),
    am.MethodSpec(
        id="IND11", name="有向时序网络 ONA", group=GROUP_NETWORK,
        question="谁接着谁的话？网络的进出平衡吗？",
        mode=MODE_INDIVIDUAL, compute=ona_analysis,
        params=(P_LAG, P_MIN_EDGE, P_ONA_COMPRESS),
        unit="student×dimension",
        formula_note="边 = 序列内部 t→t+lag 的转移计数（自环剔除）；密度 = E/N(N−1)；"
                     "加权互惠 = Σ2·min(w_ij,w_ji)/Σ(w_ij+w_ji)"),
    am.MethodSpec(
        id="GRP07", name="角色发现（轨迹 + 网络聚类）", group=GROUP_MODEL,
        question="组里有哪几种稳定角色？谁属于哪一种？",
        mode=MODE_GROUP, compute=role_analysis, requires=("students>=4",),
        units=("groups", "dimensions"), params=(P_LAG, P_CLUSTER_METHOD, P_LINKAGE, P_K,
                                                P_SELECT_K, P_KMIN, P_KMAX),
        formula_note="特征逐列 z 标准化 → 欧氏距离 → AHC/PAM；k 在候选内按 silhouette 选优"),
    am.MethodSpec(
        id="GRP08", name="交互网络 SNA", group=GROUP_NETWORK,
        question="互动在谁身上聚集、来回得勤不勤、隔多久才有人回？",
        mode=MODE_GROUP, compute=interaction_network, requires=("events",),
        units=("groups",),
        formula_note="边权 = PERSON→PERSON 事件计数；事件率 = 事件数 / 首末事件跨度；"
                     "时延 = reply_to 事件时间戳 − 被应答时间戳"),
]
