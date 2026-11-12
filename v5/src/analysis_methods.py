# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""IND / GRP 方法注册表与分析上下文。

与 `statistic_visualization_methods.MethodSpec` 同一形态（UI 只读表，不写方法专有分支），
但多两件事：

1. ``mode`` 决定这个方法在「个体模式」还是「小组模式」工作台里可见；
2. ``requires`` 是一组机器可判的前置条件（组数 / 学生数 / 有无事件 / 有无 torch），
   不满足时不隐藏方法，选中就在参数卡下方写明原因并阻断计算 —— 与 statviz 的
   ``availability()`` 观感一致，用户永远能看到"为什么这个跑不了"。

``AnalysisContext`` 是取数的唯一入口：它持有一批 `GroupBundle`，并按需给出
序列、事件表、成员 × 时间点矩阵。上下文不做任何统计。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from statistic_visualization_data import (AnalysisResult, QAItem, fmt_round,
                                          new_analysis_id)
import interaction_model as im

MODE_INDIVIDUAL = "individual"
MODE_GROUP = "group"
MODE_BOTH = "both"

ENGINE_PYTHON = "python"
ENGINE_R_REF = "python+r"        # Python 主算，R 只做交叉校验
ENGINE_OPTIONAL = "optional"     # 依赖 torch / xgboost 等可装可不装

GROUP_STATE = "A 状态与轨迹"
GROUP_TRANSITION = "B 转移与序列"
GROUP_DISTANCE = "C 距离与聚类"
GROUP_NETWORK = "D 网络"
GROUP_MODEL = "E 概率模型"
GROUP_PREDICT = "F 预测与机器学习"
GROUP_COMPARE = "G 组间比较"

# requires 允许的写法：见 availability()
REQUIRES_DOC = ("students>=N", "groups>=N", "events", "outcome", "torch", "r",
                "person_edges", "length>=N")

# 特征族枚举：GRP06 / GRP10 / IND12 的「选哪些特征」共用一份，
# 前缀必须与 analysis_features 的列名一致（prop_* / freq_* / …）。
FEATURE_FAMILY_CHOICES = (("prop", "编码占比"), ("freq", "编码频次"),
                          ("entropy", "熵"), ("switch", "切换/自转移"),
                          ("run", "连续段"), ("transition", "转移"))

_MIN_SEQUENCE_LENGTH = 10        # 短于这个长度的序列不足以谈转移
MIN_SEQUENCE_LENGTH = _MIN_SEQUENCE_LENGTH   # 对外公开名，统计层按此过滤


# ------------------------------------------------------------------ 前置条件

def _parse_requirement(text):
    """``"students>=2"`` → ("students", 2)；``"events"`` → ("events", None)。"""
    if ">=" in text:
        key, _, value = text.partition(">=")
        try:
            return key.strip(), float(value)
        except ValueError:
            return text.strip(), None
    return text.strip(), None


def missing_requirements(spec, ctx):
    """返回 [(条件, 人话说明)]：空列表就是全部满足。"""
    out = []
    for text in spec.requires or ():
        key, threshold = _parse_requirement(text)
        if key == "students" and ctx.n_students < int(threshold or 1):
            out.append((text, "需要 ≥ %d 名独立学生，当前 %d 名" % (int(threshold or 1),
                                                                   ctx.n_students)))
        elif key == "groups" and ctx.n_groups < int(threshold or 2):
            out.append((text, "需要 ≥ %d 个组才能跨组比较，当前 %d 组"
                        % (int(threshold or 2), ctx.n_groups)))
        elif key == "events" and not ctx.has_events:
            out.append((text, "需要 Interaction_Event 事件表（在「小组交互」页录入或随文件上传）"))
        elif key == "outcome" and ctx.outcome is None:
            out.append((text, "需要组结果 outcome：批量上传文件里带 outcome 列，"
                              "或先用无监督方法的簇标签作伪标签"))
        elif key == "torch" and not ctx.has_torch:
            out.append((text, "未安装 torch：pip install torch --index-url "
                              "https://download.pytorch.org/whl/cpu"))
        elif key == "r" and not ctx.has_r:
            out.append((text, "未探测到 Rscript，本方法只做交叉校验用"))
        elif key == "person_edges" and not ctx.has_person_edges:
            out.append((text, "没有 PERSON→PERSON 的事件边，时序图模型无处建边"))
        elif key == "length" and ctx.max_sequence_length < int(threshold or 1):
            out.append((text, "最长序列只有 %d 步，不足 %d 步"
                        % (ctx.max_sequence_length, int(threshold or 1))))
    return out


# ------------------------------------------------------------------ 方法声明

@dataclass
class MethodSpec:
    id: str
    name: str
    group: str
    question: str
    mode: str = MODE_BOTH
    compute: object = None
    render: object = None
    params: tuple = ()
    units: tuple = ("students", "dimensions")   # 工作台自动补的选择控件（不写进 params）
    requires: tuple = ()
    engine: str = ENGINE_PYTHON
    r_reference: str = ""            # r_refs/ 下的脚本名，交叉校验用
    unit: str = "student"            # 独立分析单元：student | group
    formula_note: str = ""           # 口径说明，进导出与结论解释
    family: str = ""                 # 算法族模块名，由 all_methods() 在汇总时填写

    def title(self):
        return "%s %s" % (self.id, self.name)

    def visible_in(self, mode):
        return self.mode in (MODE_BOTH, mode)

    def availability(self, ctx):
        """(是否可计算, 说明)。不可计算也允许选中，说明就是给用户看的原因。"""
        if ctx is None or not ctx.usable:
            return False, "尚未装载可解析的编码数据"
        missing = missing_requirements(self, ctx)
        if missing:
            return False, "；".join(text for text, _note in missing)
        return True, ""

    def blocked_result(self, ctx, note):
        return AnalysisResult(method_id=self.id, method_name=self.name,
                              analysis_id=new_analysis_id(self.id),
                              qa=[QAItem("error", "%s.requires" % self.id, note, True)],
                              blocked=True, blocked_reason=note)


# ------------------------------------------------------------------ 分析上下文

class AnalysisContext:
    """一批 GroupBundle 的只读视图：方法层只通过它拿数，不碰主窗口控件。"""

    def __init__(self, bundles=None, codebook=None, mode=MODE_INDIVIDUAL):
        self.bundles = list(bundles or [])
        self.codebook = list(codebook or [])
        self.mode = mode
        self.progress = None              # 训练回调（DL 用），由工作台注入
        self.cancelled = False
        self._sequences = None

    # ---------- 基本盘 ----------
    @property
    def usable(self):
        return any(len(bundle.state_frame()) for bundle in self.bundles)

    @property
    def n_groups(self):
        return len(self.bundles)

    @property
    def group_ids(self):
        return [bundle.group_id for bundle in self.bundles]

    @property
    def groups(self):
        """工作台的「组」清单：与 group_ids 同序，只为让 checklist 按参数名取得到候选。"""
        return self.group_ids

    @property
    def students(self):
        """独立分析单元 = student_uid：四组各有 S1 时这里是 4 个样本。"""
        out = []
        for bundle in self.bundles:
            for uid in bundle.uids:
                if uid not in out:
                    out.append(uid)
        return out

    @property
    def n_students(self):
        return len(self.students)

    @property
    def dimensions(self):
        """维度显示名：按数据里的列顺序（D1/D2…）拿框架（中文名）逐位对齐。"""
        keys = self.dimension_keys()
        names = [str(name) for name, _codes in self.codebook]
        return [names[index] if index < len(names) else key
                for index, key in enumerate(keys)] or keys

    @property
    def features(self):
        """特征族候选：参数声明里已有 options，这里只作兼容兜底。"""
        return [value for value, _text in FEATURE_FAMILY_CHOICES]

    def _all_sequences(self):
        if self._sequences is None:
            self._sequences = []
            for bundle in self.bundles:
                self._sequences.extend(bundle.sequences())
        return self._sequences

    def dimension_keys(self):
        """数据里真实存在的维度键（`S1Encoding_1` → `D1`），按首次出现顺序。"""
        out = []
        for sequence in self._all_sequences():
            if sequence.dimension not in out:
                out.append(sequence.dimension)
        return out

    def dimension_key(self, value):
        """显示名 / 数据键 / 1‑基序号 → 数据键：三种写法都认，免得上层传错就查不到数。"""
        text = str(value)
        keys = self.dimension_keys()
        if text in keys:
            return text
        names = [str(name) for name, _codes in self.codebook]
        if text in names:
            index = names.index(text)
            if index < len(keys):
                return keys[index]
        if text.isdigit():
            index = int(text) - 1
            if 0 <= index < len(keys):
                return keys[index]
        return text

    def label_of(self, dimension):
        """结果表里的维度列：有中文名就写「D1 认知投入」，没有就只写 D1。"""
        key = self.dimension_key(dimension)
        names = [str(name) for name, _codes in self.codebook]
        index = key[1:] if key.startswith("D") and key[1:].isdigit() else None
        name = ""
        if index and index.isdigit() and int(index) - 1 < len(names):
            name = names[int(index) - 1]
        if not name and key in self.dimension_keys():
            position = self.dimension_keys().index(key)
            if position < len(names):
                name = names[position]
        return "%s %s" % (key, name) if name and name != key else key

    @property
    def events(self):
        return im.ensure_event_columns(pd.concat(
            [bundle.events for bundle in self.bundles if len(bundle.events)],
            ignore_index=True)) if any(len(b.events) for b in self.bundles) \
            else im.ensure_event_columns(pd.DataFrame())

    @property
    def has_events(self):
        return any(len(bundle.events) for bundle in self.bundles)

    @property
    def outcome(self):
        """只要有任意一组带 outcome 就算可用（跨组比较的分组变量）。"""
        for bundle in self.bundles:
            if bundle.outcome is not None:
                return "outcome"
        return None

    @property
    def has_person_edges(self):
        frame = self.events
        if not len(frame):
            return False
        return bool(len(frame[(frame["target_type"] == "PERSON")
                              & (frame["actor_id"].astype(str) != "")]))

    @property
    def max_sequence_length(self):
        sequences = self.sequences()
        return max((sequence.length for sequence in sequences), default=0)

    # ---------- 可选后端 ----------
    @property
    def has_torch(self):
        import importlib.util
        return importlib.util.find_spec("torch") is not None

    @property
    def has_r(self):
        try:
            import analysis_r_bridge as bridge
            return bool(bridge.probe().get("available"))
        except Exception:                # noqa: BLE001  桥接层没装/没探测都算不可用
            return False

    # ---------- 序列 ----------
    def sequences(self, dimension=None, student=None, group_id=None, min_length=0):
        """跨 bundle 的序列集合：**每条序列永远只属于一个组**（绝不跨组拼接）。"""
        wanted_dimension = self.dimension_key(dimension) if dimension is not None else None
        out = []
        for sequence in self._all_sequences():
            if wanted_dimension is not None and str(sequence.dimension) != wanted_dimension:
                continue
            if student is not None and str(sequence.uid) != str(student) \
                    and str(sequence.student) != str(student):
                # 工作台传的是 student_uid（分析单元），手写脚本传短号 S1 也要能命中
                continue
            if group_id is not None and str(sequence.group_id) != str(group_id):
                continue
            if len(sequence.valid) < max(int(min_length), MIN_SEQUENCE_LENGTH):
                continue
            out.append(sequence)
        return out

    def sequences_for(self, params, default_dimension=None):
        """按参数卡的选择取序列：students / dimensions 留空即全部。"""
        wanted_students = params.get("students") or self.students
        wanted_dims = params.get("dimensions") or ([default_dimension]
                                                   if default_dimension else self.dimensions)
        out = []
        for student in wanted_students:
            for dimension in wanted_dims:
                out.extend(self.sequences(dimension=dimension, student=student))
        return out

    def bundle(self, group_id):
        for bundle in self.bundles:
            if bundle.group_id == str(group_id):
                return bundle
        return self.bundles[0] if self.bundles else None

    # ---------- 事件切片 ----------
    def events_for(self, group_id=None, student=None):
        frame = self.events
        if not len(frame):
            return frame
        if group_id is not None:
            frame = frame[frame["group_id"].astype(str) == str(group_id)]
        if student is not None:
            frame = frame[frame["actor_id"].astype(str) == str(student)]
        return frame.copy()

    def codebook_of(self, dimension):
        key = self.dimension_key(dimension)
        names = [str(name) for name, _codes in self.codebook]
        position = names.index(key) if key in names else None
        if position is None:
            keys = self.dimension_keys()
            position = keys.index(key) if key in keys else None
        for index, (name, codes) in enumerate(self.codebook):
            if index == position or str(name) == str(dimension):
                return list(codes)
        codes = []
        for sequence in self.sequences(dimension=dimension):
            for code in sequence.valid:
                if code not in codes:
                    codes.append(code)
        return codes

    # ---------- 质检 ----------
    def quality(self):
        import analysis_core as ac
        return ac.bundle_quality(self.bundles)


# ------------------------------------------------------------------ 公共小件

def result(method_id, method_name, tables, meta=None, qa=None, notes=None, elapsed_ms=0):
    return AnalysisResult(method_id=method_id, method_name=method_name,
                          analysis_id=new_analysis_id(method_id), tables=tables,
                          meta=meta or {}, qa=list(qa or []), notes=list(notes or []),
                          elapsed_ms=elapsed_ms)


def blocked(spec, ctx, note):
    return spec.blocked_result(ctx, note)


def chosen(values, picked):
    """参数卡留空 = 全选；给了值就按给的顺序过滤掉不存在的项。"""
    picked = [str(item) for item in (picked or []) if str(item)]
    if not picked:
        return list(values)
    allowed = {str(value) for value in values}
    return [item for item in picked if item in allowed] or list(values)


def bh_fdr(p_values):
    """Benjamini–Hochberg q 值：单调化后截到 [0, 1]。输入顺序即输出顺序。"""
    values = np.asarray([0.0 if p is None or not np.isfinite(p) else float(p)
                         for p in p_values], dtype=float)
    n = len(values)
    if n == 0:
        return []
    order = np.argsort(values, kind="stable")
    ranked = values[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]      # 从后往前取最小 → 单调
    out = np.empty(n, dtype=float)
    out[order] = ranked
    return [round(float(min(max(value, 0.0), 1.0)), 6) for value in out]


def fisher_exact_two_by_two(table):
    """2×2 的 Fisher 双侧精确 p（无 scipy 依赖的超几何求和）。"""
    flat = []
    for row in table:
        if isinstance(row, (list, tuple)):
            flat.extend(row)
        else:
            flat.append(row)
    a, b, c, d = [float(value) for value in flat[:4]]
    n = a + b + c + d
    if n <= 0:
        return 1.0
    from math import lgamma
    def log_choose(total, pick):
        if pick < 0 or pick > total:
            return -np.inf
        return lgamma(total + 1) - lgamma(pick + 1) - lgamma(total - pick + 1)
    row1, col1 = a + b, a + c
    observed = log_choose(col1, a) + log_choose(n - col1, row1 - a) - log_choose(n, row1)
    lo = max(0, int(round(row1 + col1 - n)))
    hi = min(int(round(row1)), int(round(col1)))
    total = 0.0
    for value in range(lo, hi + 1):
        probability = log_choose(col1, value) + log_choose(n - col1, row1 - value) \
            - log_choose(n, row1)
        if probability <= observed + 1e-9:
            total += np.exp(probability)
    return float(min(max(total, 0.0), 1.0))


def yules_q(table):
    """Yule's Q = (ad − bc) / (ad + bc)：2×2 转移表的关联强度，退化时给 0。

    接受嵌套的 [[a, b], [c, d]] 或展平的 [a, b, c, d]（行优先）。
    """
    flat = []
    for row in table:
        if isinstance(row, (list, tuple)):
            flat.extend(row)
        else:
            flat.append(row)
    if len(flat) != 4:
        raise ValueError("Yule's Q 需要 2×2 表，收到 %d 个值" % len(flat))
    a, b, c, d = [float(value) for value in flat]
    numerator = a * d - b * c
    denominator = a * d + b * c
    if abs(denominator) < 1e-12:
        return 0.0
    return round(numerator / denominator, 6)


def entropy(codes, base=np.e):
    """香农熵，默认 ln（全项目统一口径，计划 §3 统一口径）。

    base 只换对数底、不改口径：H_base = H_ln / ln(base)，所以传 2 得到 bit、
    传默认值得到 nats，两者可比。
    """
    counts = pd.Series([code for code in codes if str(code) != ""]).value_counts()
    total = int(counts.sum())
    if not total:
        return 0.0
    probabilities = counts.to_numpy(dtype=float) / total
    raw = float(-(probabilities * np.log(probabilities)).sum())
    divisor = float(np.log(base)) if base and abs(float(base) - 1.0) > 1e-12 else 1.0
    return raw / divisor


def normalized_entropy(codes):
    """H / ln K，K = 实际出现的状态数：只剩一种状态时为 0，空序列时为 0。"""
    unique = {str(code) for code in codes if str(code) != ""}
    if len(unique) <= 1:
        return 0.0
    return round(entropy(codes) / float(np.log(len(unique))), 6)


def transition_counts(codes, lag=1):
    """滞后 lag 的转移计数 {from: {to: n}}：只数同一序列内部的转移。"""
    values = [code for code in codes if str(code) != ""]
    counts = {}
    for index in range(max(len(values) - lag, 0)):
        left, right = values[index], values[index + lag]
        counts.setdefault(left, {}).setdefault(right, 0)
        counts[left][right] += 1
    return counts


def state_order(counts_by_state, how="count"):
    """状态排序：频次降序（默认）或字典序，决定转移矩阵的行列顺序。"""
    if how == "alpha":
        return sorted(counts_by_state)
    totals = {state: sum(row.values()) for state, row in counts_by_state.items()}
    return sorted(counts_by_state, key=lambda state: (-totals[state], str(state).lower()))


__all__ = ["MethodSpec", "ParamSpec", "AnalysisContext", "result", "blocked", "chosen",
           "bh_fdr", "yules_q", "fisher_exact_two_by_two", "entropy",
           "normalized_entropy", "transition_counts", "state_order", "unit_specs",
           "chart_specs", "figure_scale", "CHART_PARAMS", "RENDER_ONLY_PARAMS",
           "LEGEND_OPTIONS", "FIGURE_SIZE_OPTIONS", "FIGURE_SCALES",
           "MODE_INDIVIDUAL", "MODE_GROUP", "MODE_BOTH", "ENGINE_PYTHON",
           "ENGINE_R_REF", "ENGINE_OPTIONAL", "REQUIRES_DOC", "missing_requirements",
           "MIN_SEQUENCE_LENGTH", "FEATURE_FAMILY_CHOICES", "GROUP_STATE",
           "GROUP_TRANSITION", "GROUP_DISTANCE",
           "GROUP_NETWORK", "GROUP_MODEL", "GROUP_PREDICT", "GROUP_COMPARE"]

# ParamSpec 与 statviz 共用一份，避免两张表长出两种参数形状
from statistic_visualization_methods import (   # noqa: E402  放在末尾防循环导入
    ParamSpec, LEGEND_OPTIONS)


# ------------------------------------------------------------------ 分析对象选择行

_UNIT_PARAMS = {
    "students": ("students", "分析学生", "独立分析单元 = 组号__学生；留空即全部"),
    "groups": ("groups", "分析组", "留空即全部组；跨组比较建议先逐组看组内口径"),
    "dimensions": ("dimensions", "编码维度", "留空即全部维度"),
}


def unit_spec(name):
    """``MethodSpec.units`` 里的一项 → 一行多选控件声明（工作台自动补，不写进 params）。"""
    found = _UNIT_PARAMS.get(name)
    if found is None:
        return None
    key, label, tip = found
    return ParamSpec(key, label, "checklist", tooltip=tip)


def unit_specs(spec):
    """一个方法要的分析对象选择行，按声明顺序。"""
    out = []
    for name in spec.units or ():
        param = unit_spec(name)
        if param is not None:
            out.append(param)
    return out


# ------------------------------------------------------------------ 图形呈现参数
#
# 图例档位直接复用 statviz 的 LEGEND_OPTIONS：两个工作台里「图形正下方」必须是
# 同一个行为，不能各定一套词表。这两行只改看不改算，所以工作台把它们排在
# 方法参数之后，并且值变了只重画图、不重算（见 RENDER_ONLY_PARAMS）。

FIGURE_SIZE_OPTIONS = (("fit", "适应预览区"), ("medium", "放大 1.4 倍"),
                       ("large", "放大 1.8 倍"), ("huge", "放大 2.4 倍"))
FIGURE_SCALES = {"fit": 1.0, "medium": 1.4, "large": 1.8, "huge": 2.4}

P_LEGEND_LOC = ParamSpec("legend_loc", "图例位置", "combo", LEGEND_OPTIONS, default="below",
                         tooltip="选“图形正下方 / 图形右侧”时会在图上单独留一条边距，"
                                 "不会压住坐标区")
P_FIGURE_SIZE = ParamSpec("figure_size", "图形尺寸", "combo", FIGURE_SIZE_OPTIONS,
                          default="fit",
                          tooltip="标签放不下时图本来就会自己撑大（预览区滚动看）；"
                                  "这一档再整体放大，字号与格子都会更松")

CHART_PARAMS = (P_LEGEND_LOC, P_FIGURE_SIZE)
RENDER_ONLY_PARAMS = frozenset(param.key for param in CHART_PARAMS)


def chart_specs():
    """工作台自动补在参数卡末尾的图形呈现行。"""
    return tuple(CHART_PARAMS)


def figure_scale(params):
    """「图形尺寸」→ 预览区尺寸的倍数；没选或选“适应预览区”都是 1.0。"""
    key = str((params or {}).get("figure_size") or "fit")
    return FIGURE_SCALES.get(key, 1.0)


# ------------------------------------------------------------------ 注册表入口
#
# 方法声明按算法族摊在各 stats 模块的 SPECS 里（谁写算法谁声明参数），这里只做汇总；
# 延迟 import 是为了打断 analysis_methods ↔ analysis_stats_* 的环。

_FAMILY_MODULES = ("analysis_stats_sequence", "analysis_stats_group",
                   "analysis_stats_network", "analysis_ml", "analysis_dl")
_SPECS = None


def all_methods(refresh=False):
    """全部方法声明；某个族还没落地时静默跳过（阶段 3–7 分批交付）。"""
    global _SPECS
    if _SPECS is None or refresh:
        import importlib
        collected = []
        for name in _FAMILY_MODULES:
            try:
                module = importlib.import_module(name)
            except ImportError:
                continue
            specs = getattr(module, "SPECS", [])
            # 把族模块名写进 spec：导出审计要记「这个方法属于哪个族」，而**不能**靠
            # inspect.getmodule(func) 去推 —— 核心模块被 Cython 编成 .pyd 后没有
            # 源文件，getmodule 会静默返回 None，审计里的族名就悄悄变成空。
            # 这里手上的 module 对象在 .py / .pyd 两种形态下都是真实的。
            for spec in specs:
                spec.family = getattr(module, "__name__", name)
            collected.extend(specs)
        _SPECS = collected
    return _SPECS


def methods_for(mode):
    return [spec for spec in all_methods() if spec.visible_in(mode)]


def method_by_id(method_id):
    for spec in all_methods():
        if spec.id == method_id:
            return spec
    return None


def method_groups(mode):
    """[(分组名, [MethodSpec])]：按声明顺序给下拉做 optgroup。"""
    order, buckets = [], {}
    for spec in methods_for(mode):
        if spec.group not in buckets:
            buckets[spec.group] = []
            order.append(spec.group)
        buckets[spec.group].append(spec)
    return [(name, buckets[name]) for name in order]
