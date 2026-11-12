# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""编码数据的统一长表模型、时间处理与自动质检。

内部数据模型严格遵循 V1.1 技术方案：**Student × Time × Dimension × Code**。
S1 / S2 是学生 ID（不是 channel），每名学生内部有 D1 / D2 / D3 三个同步编码维度。
所有分析模块都从 CodingDataset.long_df 取数，不重复解析 Excel。
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ 列名与常量

# 附录 A 的严格式：S1Encoding_1 → student_id=S1, dimension=D1
STRICT_COLUMN_RE = re.compile(r"^(S\d+)Encoding_(\d+)$")
# 兜底式：容忍 P01_Encoding_2 之类的变体，解析出来即记一条 QA 提示
LOOSE_COLUMN_RE = re.compile(r"^(.+?)[ _]?Encoding[_\s-]?(\d+)$")

TIME_ALIASES = ("time", "时间", "时刻", "timestamp", "seconds", "秒")
MISSING_TOKENS = {"", "-", "--", "na", "n/a", "nan", "none", "null", "/"}

LEVEL_INFO = "info"
LEVEL_WARN = "warn"
LEVEL_ERROR = "error"

LONG_COLUMNS = ["student_id", "time", "dimension", "code", "duration",
                "time_bin", "source_row", "source_column", "is_missing"]


def natural_key(text):
    """S1 < S2 < S10、D1 < D2 的自然序排序键。"""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(text))]


def dimension_key(text):
    """维度标识 D1/D2/... 的排序键（去掉前缀比较数字）。"""
    match = re.search(r"(\d+)", str(text))
    return int(match.group(1)) if match else 0


# ------------------------------------------------------------------ 质检条目

@dataclass
class QAItem:
    level: str
    code: str
    message: str
    blocking: bool = False


@dataclass
class SamplingInfo:
    """Time 列的采样特征。Δt 一律来自数据，代码里不出现常量 10。"""
    median_dt: float
    min_dt: float
    max_dt: float
    is_regular: bool
    n_positions: int
    t_min: float
    t_max: float

    @property
    def valid(self):
        return self.n_positions >= 2 and np.isfinite(self.median_dt)

    def describe(self):
        if not self.valid:
            return "有效时间点不足 2 个，无法建立时间轴"
        mode = "规则采样" if self.is_regular else "不规则采样"
        return (f"{self.n_positions} 个时间点 · Δt={_num(self.median_dt)} s · "
                f"范围 {_num(self.t_min)}–{_num(self.t_max)} s · {mode}")


@dataclass
class AnalysisResult:
    """每个方法统一返回：结果表 + 元数据 + 质检提示。"""
    method_id: str
    method_name: str
    analysis_id: str
    tables: dict = field(default_factory=dict)          # 表名 -> DataFrame
    meta: dict = field(default_factory=dict)            # 运行元数据
    qa: list = field(default_factory=list)              # list[QAItem]
    notes: list = field(default_factory=list)           # 图形/结论的解释限制
    blocked: bool = False
    blocked_reason: str = ""
    elapsed_ms: int = 0

    def table(self):
        """第一张表（多数方法只有一张）。"""
        for frame in self.tables.values():
            return frame
        return pd.DataFrame()


def new_analysis_id(method_id):
    return "%s_%s" % (time.strftime("%Y%m%d-%H%M%S"), method_id)


def _num(value, digits=2):
    """统一的数字文本口径：整数不带小数点，其余保留 digits 位。"""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return ("%." + str(digits) + "f") % number


def _round(value, digits=2):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return float("nan")
    return round(float(value), digits)


def fmt_num(value, digits=2):
    """对外公开的数值文本口径（统计层与绘图层统一用它，避免各自拼字符串）。"""
    return _num(value, digits)


def fmt_round(value, digits=2):
    """对外公开的数值取整（写进元数据 / JSON 用）。

    标量走 ``_round``；numpy 数组与 Series 逐元素取整后按同类型返回——结果表里整列
    四舍五入是常事，再引一个并列助手只会多出一个口径。
    """
    if isinstance(value, np.ndarray):
        return np.round(value.astype(float), digits)
    if isinstance(value, pd.Series):
        return pd.Series(np.round(value.to_numpy(dtype=float), digits), index=value.index)
    return _round(value, digits)


# ------------------------------------------------------------------ 时间处理

def parse_time_value(value):
    """单个时间值 → 秒。数字直接用；HH:MM:SS / MM:SS 文本折算；其余 None。"""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        delta = value - value.normalize()
        return delta.total_seconds()
    if hasattr(value, "hour") and hasattr(value, "minute"):        # datetime.time
        second = getattr(value, "second", 0) or 0
        micro = getattr(value, "microsecond", 0) or 0
        return value.hour * 3600 + value.minute * 60 + second + micro / 1e6
    if hasattr(value, "total_seconds"):                             # datetime.timedelta
        return float(value.total_seconds())
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) in (2, 3) and all(_is_number(part) for part in parts):
            total = 0.0
            for part in parts:
                total = total * 60 + float(part)
            return total
        return None
    return float(text) if _is_number(text) else None


def _is_number(text):
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return text != ""


def coerce_time_column(series):
    """时间列 → float 秒（不可解析为 NaN），并给出解析成功率。"""
    values = series.to_numpy(dtype=object, na_value=None)
    parsed = np.array([parse_time_value(v) for v in values], dtype=float)
    return pd.Series(parsed, index=series.index), int(np.isnan(parsed).sum())


def estimate_time_info(times):
    """Δt = median(diff(sorted unique Time))；差值不全等即标记不规则采样。"""
    values = np.asarray([t for t in times if pd.notna(t)], dtype=float)
    unique = np.unique(values)
    if unique.size < 2:
        return SamplingInfo(float("nan"), float("nan"), float("nan"),
                            False, int(unique.size),
                            float(unique[0]) if unique.size else float("nan"),
                            float(unique[0]) if unique.size else float("nan"))
    diffs = np.diff(unique)
    median_dt = float(np.median(diffs))
    is_regular = bool(np.allclose(diffs, median_dt, rtol=1e-9, atol=1e-9))
    return SamplingInfo(median_dt, float(diffs.min()), float(diffs.max()),
                        is_regular, int(unique.size),
                        float(unique[0]), float(unique[-1]))


def build_duration_map(positions, info: SamplingInfo):
    """时间点 → 该点代表的时间长度。代码里不出现常量 10。

    规则采样：全部等于 Δt。不规则采样：取到下一个时间点的间隔，末点沿用前一段间隔；
    只有 1 个时间点时退化为 Δt（无 Δt 则退化为 1.0，只为避免除零）。
    """
    positions = [float(t) for t in positions]
    if len(positions) < 2:
        fallback = info.median_dt if info.valid and info.median_dt > 0 else 1.0
        return {t: float(fallback) for t in positions}
    if info.is_regular:
        return {t: float(info.median_dt) for t in positions}
    diffs = np.diff(np.asarray(positions, dtype=float))
    durations = {positions[i]: float(diffs[i]) for i in range(len(diffs))}
    durations[positions[-1]] = float(diffs[-1])
    return durations


# ------------------------------------------------------------------ 宽表 → 长表

def norm_code(value):
    """单元格 → 分类编码字符串；空白 / NA / 占位符统一视为缺失（None）。"""
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return None
    return text


def parse_columns(columns, time_col=None):
    """自动解析列名 → {列名: (student_id, dimension)}，返回 (映射, 未解析列)。

    列名不一定是字符串：Excel 里空表头或重复表头会被 pandas 读成 int / None / float（真实
    数据里就出现过 `group_1 - 副本.xlsx` 的第 9 列叫 `1`）。这类列既不是时间列也不是编码列，
    直接跳过，不得让整个装载因一个 TypeError 崩掉。
    """
    mapping, unparsed = {}, []
    for col in columns:
        if col == time_col:
            continue
        if not isinstance(col, str):
            continue
        match = STRICT_COLUMN_RE.match(col)
        if not match:
            match = LOOSE_COLUMN_RE.match(col)
        if match:
            student = match.group(1).strip() or "S1"
            mapping[col] = (student, "D%d" % int(match.group(2)))
        elif "encoding" in str(col).lower():
            unparsed.append(col)
    return mapping, unparsed


def detect_time_column(df):
    """先按列名别名，再按内容可解析比例（>0.8）挑时间列。"""
    for col in df.columns:
        if str(col).strip().lower() in TIME_ALIASES:
            return col
    best, best_ratio = None, 0.0
    for col in df.columns:
        values, _ = coerce_time_column(df[col])
        ratio = float(values.notna().mean()) if len(values) else 0.0
        if ratio > best_ratio:
            best, best_ratio = col, ratio
    return best if best_ratio > 0.8 else None


def load_dataframe(path):
    """读 Excel / CSV；CSV 依次尝试 utf-8-sig 与 gbk。"""
    lowered = str(path).lower()
    if lowered.endswith(".csv"):
        for encoding in ("utf-8-sig", "gbk"):
            try:
                return pd.read_csv(path, encoding=encoding, dtype=object)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, dtype=object)
    return pd.read_excel(path, dtype=object)


def from_dataframe(df, source=""):
    """外部（如数据汇总页）已拼好的 DataFrame 走与文件读取完全等价的链路。"""
    return CodingDataset.build(df, source=source)


# ------------------------------------------------------------------ 时间窗口

def parse_bin_bounds(text):
    """自定义边界→升序去重的浮点列表。

    UI 里这是一个单行文本框，交回来的是字符串；支持 , ，、 空格 分号 五种分隔符，
    非法片段直接跳过（至少两个合法值才能成窗，否则回退到单窗口）。
    """
    if isinstance(text, (list, tuple, set)):
        items = list(text)
    else:
        items = re.split(r"[,，、;；\s]+", str(text or ""))
    bounds = []
    for item in items:
        try:
            bounds.append(float(str(item).strip()))
        except (TypeError, ValueError):
            continue
    return sorted(set(bounds))


def make_bin_edges(t_min, t_max, mode, value, step=None):
    """三种分箱模式：seconds 固定窗长 / count 等分窗数 / custom 自定义边界。

    step 为采样间隔（规则采样时传入）：等分窗数会把窗长向上对齐到整格，
    避免出现 311.67 s 这类踩不到数据点的边界。
    """
    span = (t_max - t_min) + (step or 0.0)          # 末点的整段也要落进窗口
    if mode == "seconds":
        width = max(float(value), 1e-9)
        edges = [t_min + width * i for i in range(int(np.ceil(span / width - 1e-9)) + 1)]
        if edges[-1] <= t_max:
            edges.append(edges[-1] + width)
        return [float(e) for e in edges]
    if mode == "count":
        bins = max(int(value), 1)
        width = span / bins
        if step:
            width = max(float(np.ceil(width / step - 1e-9)) * step, step)
        edges = [t_min + width * i for i in range(bins)] + [t_min + width * bins]
        while edges[-1] <= t_max:
            edges.append(edges[-1] + width)
        return [float(e) for e in edges]
    if mode == "custom":
        bounds = sorted({float(v) for v in (value or [])})
        if len(bounds) < 2:
            return [float(t_min), float(t_max) + 1.0]
        if bounds[0] > t_min:
            bounds.insert(0, float(t_min))
        if bounds[-1] <= t_max:
            bounds.append(float(t_max) + 1e-9)
        return bounds
    raise ValueError("未知的分箱模式：%s" % mode)


def bin_labels(edges):
    """边界 → (标签列表, (标签, 下界, 上界) 列表)。窗口右开，图上按真实边界标注。"""
    labels, bounds = [], []
    for idx in range(len(edges) - 1):
        low, high = edges[idx], edges[idx + 1]
        label = "T%02d [%s, %s)" % (idx + 1, _num(low), _num(high))
        labels.append(label)
        bounds.append((label, float(low), float(high)))
    return labels, bounds


def assign_bins(times, edges):
    """按右开区间（末区间闭合）把时间点落到窗口标签上。"""
    values = np.asarray(times, dtype=float)
    inner = np.asarray(edges[1:-1], dtype=float) if len(edges) > 2 else np.array([])
    idx = np.searchsorted(inner, values, side="right") if inner.size else np.zeros(values.shape, dtype=int)
    idx = np.clip(idx, 0, len(edges) - 2)
    labels, _ = bin_labels(edges)
    return [labels[i] for i in idx]


# ------------------------------------------------------------------ 数据集

class CodingDataset:
    """统一长表数据集：student_id / time / dimension / code / duration / time_bin。"""

    def __init__(self):
        self.source = ""
        self.raw_df = pd.DataFrame()
        self.time_col = None
        self.column_map = {}
        self.unparsed_columns = []
        self.long_df = pd.DataFrame(columns=LONG_COLUMNS)
        self.time_info = SamplingInfo(float("nan"), float("nan"), float("nan"),
                                      False, 0, float("nan"), float("nan"))
        self.qa = []
        self.students = []
        self.dimensions = []
        self.codes_by_dim = {}
        self.unit_missing = {}
        self.duration_by_time = {}
        self.dropped_bad_time = 0
        self.duplicate_rows = 0
        self.duplicates_confirmed = False
        self._bin_cache = {}
        self._bin_key = None
        self.bin_order = []
        self.bin_bounds = []
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---------------------------------------------------------------- 构建
    @classmethod
    def build(cls, raw_df, time_col=None, column_map=None, source=""):
        ds = cls()
        ds.source = source
        frame = raw_df.copy()
        frame.columns = [str(col).strip() for col in raw_df.columns]
        ds.raw_df = frame

        if time_col is None:
            time_col = detect_time_column(frame)
            if time_col is None:
                ds.qa.append(QAItem(LEVEL_ERROR, "time_missing",
                                    "未能识别时间列：请在下方手动指定", blocking=True))
                return ds
        ds.time_col = str(time_col)

        times, unparsed_time = coerce_time_column(frame[ds.time_col])
        if unparsed_time:
            ds.qa.append(QAItem(LEVEL_WARN, "time_unparsable",
                                "时间列有 %d 个值无法解析为秒数，相关行不参与时间分析"
                                % unparsed_time))

        auto_map, unparsed = parse_columns(list(frame.columns), ds.time_col)
        if column_map:
            ds.column_map = {str(k): tuple(v) for k, v in column_map.items()}
            ds.unparsed_columns = unparsed
        else:
            ds.column_map = auto_map
            ds.unparsed_columns = unparsed
            if auto_map:
                ds.qa.append(QAItem(LEVEL_INFO, "columns_auto_parsed",
                                    "按 ^(S\\d+)Encoding_(\\d+)$ 自动解析出 %d 个编码列"
                                    % len(auto_map)))
        if unparsed:
            ds.qa.append(QAItem(LEVEL_WARN, "columns_unparsed",
                                "以下列含 Encoding 但无法解析出学生/维度，已忽略：%s"
                                % "、".join(unparsed)))
        if not ds.column_map:
            ds.qa.append(QAItem(LEVEL_ERROR, "no_encoding_columns",
                                "没有可用的编码列（应形如 S1Encoding_1），请在下方手动指定映射",
                                blocking=True))
            return ds

        monotonic = times.dropna()
        if monotonic.size and bool((np.diff(monotonic.to_numpy()) < 0).any()):
            ds.qa.append(QAItem(LEVEL_INFO, "time_non_monotonic",
                                "时间列非单调递增，已按时间排序后分析"))

        ds._build_long(times)
        ds.time_info = estimate_time_info(ds.long_df["time"])
        if not ds.time_info.valid:
            ds.qa.append(QAItem(LEVEL_ERROR, "time_missing",
                                "有效时间点不足 2 个，时间轴类方法不可用", blocking=True))
        ds._attach_duration()
        ds._profile_units()
        return ds

    def _build_long(self, times):
        records = []
        source_rows = np.arange(2, len(self.raw_df) + 2, dtype=int)
        time_values = times.to_numpy(dtype=float)
        for col, (student, dimension) in self.column_map.items():
            records.append(pd.DataFrame({
                "student_id": student,
                "dimension": dimension,
                "source_column": col,
                "time": time_values,
                "source_row": source_rows,
                "code": [norm_code(v) for v in self.raw_df[col].to_numpy()],
            }))
        long_df = pd.concat(records, ignore_index=True)
        bad_time = int(long_df["time"].isna().sum())
        if bad_time:
            long_df = long_df[long_df["time"].notna()]
        # 空表上赋 Series 会把列留成 object dtype，后续 ~ 掩码就会变成「列名列表」而不是布尔筛选
        long_df["is_missing"] = long_df["code"].isna().to_numpy(dtype=bool)
        long_df = long_df.sort_values(["source_column", "time", "source_row"],
                                      kind="mergesort")
        duplicated = long_df.duplicated(subset=["student_id", "dimension", "time"],
                                        keep="first")
        self.duplicate_rows = int(duplicated.sum())
        if self.duplicate_rows:
            long_df = long_df[~duplicated]
            self.qa.append(QAItem(LEVEL_ERROR, "time_duplicate",
                                  "同一学生 × 维度 × 时间点存在 %d 条重复记录，已按原始行序保留首条；"
                                  "请确认源数据是否错行" % self.duplicate_rows, blocking=True))
        if bad_time:
            self.dropped_bad_time = bad_time
        long_df["duration"] = float("nan")
        long_df["time_bin"] = pd.NA
        columns = ["student_id", "time", "dimension", "code", "duration",
                   "time_bin", "source_row", "source_column", "is_missing"]
        self.long_df = long_df[columns].reset_index(drop=True)

    def _attach_duration(self):
        info = self.time_info
        if not info.valid:
            self.long_df["duration"] = float("nan")
            self.duration_by_time = {}
            return
        positions = np.unique(self.long_df["time"].to_numpy(dtype=float)).tolist()
        if not info.is_regular:
            diffs = np.diff(np.asarray(positions, dtype=float))
            self.qa.append(QAItem(
                LEVEL_WARN, "irregular_sampling",
                "采样间隔不一致（%s–%s s），Duration / Spell / Time-bin 改用相邻时间差"
                % (_num(diffs.min()), _num(diffs.max()))))
        self.duration_by_time = build_duration_map(positions, info)
        self.long_df["duration"] = self.long_df["time"].map(self.duration_by_time).astype(float)

    def _profile_units(self):
        long_df = self.long_df
        self.students = sorted(long_df["student_id"].unique().tolist(), key=natural_key)
        all_dims = sorted(long_df["dimension"].unique().tolist(), key=dimension_key)
        keep = []
        for dim in all_dims:
            part = long_df[long_df["dimension"] == dim]
            if part["is_missing"].all():
                self.qa.append(QAItem(LEVEL_WARN, "dimension_all_empty",
                                      "维度 %s 在所有学生下都没有有效编码，已跳过" % dim))
                continue
            keep.append(dim)
        self.dimensions = keep
        if not self.dimensions:
            self.qa.append(QAItem(LEVEL_ERROR, "all_dimensions_empty",
                                  "所有编码维度均为空，无法分析", blocking=True))
        self.codes_by_dim = {}
        for dim in self.dimensions:
            codes = sorted(long_df.loc[(long_df["dimension"] == dim)
                                       & (~long_df["is_missing"]), "code"].unique().tolist(),
                           key=natural_key)
            self.codes_by_dim[dim] = codes
            if len(codes) > 25:
                self.qa.append(QAItem(LEVEL_WARN, "too_many_states",
                                      "维度 %s 有 %d 种编码，图例可能过载" % (dim, len(codes))))
        self.unit_missing = {}
        for (student, dim), part in long_df.groupby(["student_id", "dimension"], sort=False):
            if dim not in self.dimensions:
                continue
            self.unit_missing[(student, dim)] = int(part["is_missing"].sum())
        missing_total = int(long_df["is_missing"].sum())
        if missing_total:
            rate = missing_total / max(len(long_df), 1) * 100
            self.qa.append(QAItem(LEVEL_WARN, "partial_missing",
                                  "共有 %d 个空编码（%.2f%%），不计入频次与分母"
                                  % (missing_total, rate)))
        if len(self.students) < 3 and self.students:
            self.qa.append(QAItem(
                LEVEL_WARN, "few_students",
                "当前只有 %d 名学生：%s。State Distribution / Mean Time 等跨学生方法"
                "仍可计算，但比例只能是 0/%s/100%%，不代表稳定群体分布"
                % (len(self.students), "、".join(self.students),
                   _num(100.0 / max(len(self.students), 1)))))

    # ---------------------------------------------------------------- 查询
    @property
    def usable(self):
        return len(self.long_df) > 0 and bool(self.dimensions)

    @property
    def blocking_qa(self):
        return [item for item in self.qa if item.blocking]

    def units(self, students=None, dimensions=None):
        """按选择返回 [(student, dimension)]，顺序固定为学生序 × 维度序。"""
        chosen_students = self._pick(self.students, students)
        chosen_dims = self._pick(self.dimensions, dimensions)
        return [(s, d) for s in chosen_students for d in chosen_dims]

    @staticmethod
    def _pick(all_items, chosen):
        if not chosen:
            return list(all_items)
        wanted = {str(c) for c in chosen}
        picked = [item for item in all_items if str(item) in wanted]
        return picked or list(all_items)

    def slice(self, students=None, dimensions=None, with_missing=False):
        """取长表切片（按学生、维度、时间稳定排序）。"""
        chosen_students = self._pick(self.students, students)
        chosen_dims = self._pick(self.dimensions, dimensions)
        frame = self.long_df[self.long_df["student_id"].isin(chosen_students)
                             & self.long_df["dimension"].isin(chosen_dims)]
        if not with_missing and len(frame):
            frame = frame[~frame["is_missing"].to_numpy(dtype=bool)]
        return frame.sort_values(["student_id", "dimension", "time"],
                                 kind="mergesort").reset_index(drop=True)

    def sequence(self, student, dimension):
        """某个 student × dimension 的时间有序编码列表（含 None，表示该位置缺失）。"""
        positions = self.time_positions()
        frame = self.long_df[(self.long_df["student_id"] == student)
                             & (self.long_df["dimension"] == dimension)]
        by_time = dict(zip(frame["time"].to_numpy(dtype=float), frame["code"]))
        return [by_time.get(t) for t in positions]

    def time_positions(self):
        if not hasattr(self, "_positions") or self._positions is None:
            self._positions = np.unique(self.long_df["time"].to_numpy(dtype=float)).tolist()
        return self._positions

    # ---------------------------------------------------------------- 分箱
    def ensure_bins(self, mode="count", value=6):
        """按参数生成（并缓存）time_bin 列；返回 (标签顺序, 边界表)。"""
        if not self.time_info.valid:
            self.bin_order, self.bin_bounds = [], []
            return self.bin_order, self.bin_bounds
        key = (mode, tuple(value) if isinstance(value, (list, tuple)) else value)
        cached = self._bin_cache.get(key)
        if cached is None:
            step = self.time_info.median_dt if self.time_info.is_regular else None
            edges = make_bin_edges(self.time_info.t_min, self.time_info.t_max,
                                   mode, value, step=step)
            labels, bounds = bin_labels(edges)
            assigned = assign_bins(self.long_df["time"].to_numpy(dtype=float), edges)
            self._bin_cache[key] = (labels, bounds, assigned)
            cached = self._bin_cache[key]
        labels, bounds, assigned = cached
        self.long_df["time_bin"] = assigned
        self.bin_order, self.bin_bounds = labels, bounds
        self.bin_mode, self.bin_value = mode, value
        return labels, bounds

    # ---------------------------------------------------------------- 元数据
    def meta(self, extra=None):
        data = {
            "source": self.source,
            "created_at": self.created_at,
            "time_column": self.time_col,
            "column_mapping": {col: {"student_id": stu, "dimension": dim}
                               for col, (stu, dim) in self.column_map.items()},
            "students": list(self.students),
            "dimensions": list(self.dimensions),
            "codes_by_dimension": {k: list(v) for k, v in self.codes_by_dim.items()},
            "sampling": {
                "median_dt": _round(self.time_info.median_dt, 6),
                "min_dt": _round(self.time_info.min_dt, 6),
                "max_dt": _round(self.time_info.max_dt, 6),
                "is_regular": bool(self.time_info.is_regular),
                "n_positions": int(self.time_info.n_positions),
                "t_min": _round(self.time_info.t_min, 6),
                "t_max": _round(self.time_info.t_max, 6),
            },
            "rows_long": int(len(self.long_df)),
            "dropped_bad_time_rows": int(self.dropped_bad_time),
            "duplicate_rows_removed": int(self.duplicate_rows),
            "missing_cells": int(self.long_df["is_missing"].sum()),
            "missing_rule": "空/NA 不计入频次，也不进入比例分母；missing N 与 missing% 单独报告",
        }
        if self.bin_bounds:
            data["time_bins"] = {"mode": self.bin_mode,
                                 "bounds": [[label, _round(low, 6), _round(high, 6)]
                                            for label, low, high in self.bin_bounds]}
        if extra:
            data.update(extra)
        return data

    def uniqueness_ratio(self, dimension, students=None):
        """完整序列的唯一比例，用于 high_sequence_uniqueness 判定。"""
        chosen = self._pick(self.students, students)
        sequences = [tuple(self.sequence(s, dimension)) for s in chosen]
        if not sequences:
            return 1.0
        return len(set(sequences)) / float(len(sequences))

    def qa_dicts(self):
        return [asdict(item) for item in self.qa]


def merge_qa(dataset_qa, result_qa):
    """数据集级 + 方法级质检合并去重：UI 质检卡与 run_audit 必须看到同一份清单。"""
    combined = list(dataset_qa or [])
    for item in result_qa or []:
        if not any(item.code == other.code and item.message == other.message
                   for other in combined):
            combined.append(item)
    return combined


def qa_summary(items):
    """(阻断, 警告, 提示) 计数，供质检卡汇总。"""
    errors = warns = infos = 0
    for item in items:
        level = getattr(item, "level", None) or item.get("level")
        if level == LEVEL_ERROR:
            errors += 1
        elif level == LEVEL_WARN:
            warns += 1
        else:
            infos += 1
    return errors, warns, infos
