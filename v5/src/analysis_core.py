# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""分析层的数据契约与装载：GroupBundle / 序列切分 / 编码框架解析。

「一组 = 一个文件」是本工具的事实边界：跨组的同类编码不能首尾相接成一条序列（组 A
的最后一个状态和组 B 的第一个状态之间没有任何因果），求和型指标也不能拿一组的人数
去当另一组的分母。所以装载阶段就把 group_id 钉死在 bundle 上，序列一律按 bundle
边界切，`student_uid = {group_id}__{student}` 保证四组各有 S1 时特征表里仍是四个样
本，而不是三个字段撞在一起。

- ``GroupBundle``     ：一组的全部事实（状态宽表 / CodingDataset / 事件表 / 编码框架）。
- ``load_bundles``    ：批量上传，group_id 取文件名（可覆写）。
- ``bundle_from_session``：从当前编码会话取数（数据汇总页 + 小组交互页）。
- ``concat_bundles``  ：多组合并成分析用长表，**严格按组边界切序列**。
- ``resolve_codebook``：当前会话框架 → 数据推断 → 导入框架 Excel，三档优先。
- ``sequences_of``    ：组内每个 (学生 × 维度) 一条符号序列，IND/GRP 方法共用的入口。
"""

import os
import re

import pandas as pd

from statistic_visualization_data import (CodingDataset, QAItem, from_dataframe,
                                          norm_code, parse_columns)
import interaction_model as im

STRICT_COLUMN_RE = re.compile(r"^(S\d+)Encoding_(\d+)$")
GROUP_ID_FORBIDDEN = re.compile(r"[^\w.\-]+")


def sanitize_group_id(text, fallback="group"):
    """主键安全化：group_id 会拼进 event_id 与 student_uid，不能带分隔符与空白。"""
    cleaned = GROUP_ID_FORBIDDEN.sub("_", str(text or "").strip()).strip("_")
    return cleaned or fallback


def student_uid(group_id, student):
    """跨组唯一的学生标识：四组各有 S1 时，特征表里必须是四个样本。"""
    return "%s__%s" % (group_id, student)


def _split_list(value):
    """Group_Info.members 可以塞逗号串也可以塞列表，两边都收。"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = re.split(r"[,\u3001;；/]", str(value))
    return [str(item).strip() for item in items if str(item).strip()]


def split_uid(uid):
    """`{group_id}__{student}` → (group_id, student)；不含分隔符时组号留空。"""
    text = str(uid or "")
    if "__" in text:
        group, _, student = text.partition("__")
        return group, student
    return "", text


# ------------------------------------------------------------------ 一条序列

class Sequence:
    """某个 (组 × 学生 × 维度) 的符号序列：时间升序，缺失编码以 "" 占位。

    ``codes`` 与 ``times`` 一一对应，长度相等；``valid`` 只统计非空编码，供「序列太
    短」这类门槛判定使用。跨组绝不共享一个 Sequence —— 那是 ``concat_bundles`` 的
    职责边界。
    """

    __slots__ = ("group_id", "student", "dimension", "times", "codes", "source")

    def __init__(self, group_id, student, dimension, times, codes, source=""):
        self.group_id = str(group_id)
        self.student = str(student)
        self.dimension = str(dimension)
        self.times = list(times)
        self.codes = [str(code or "") for code in codes]
        self.source = source

    @property
    def uid(self):
        return student_uid(self.group_id, self.student)

    @property
    def label(self):
        return "%s / %s / %s" % (self.uid, self.dimension, self.source or "state")

    @property
    def length(self):
        return len(self.codes)

    @property
    def valid(self):
        return [code for code in self.codes if code]

    def compressed(self):
        """连续相同编码合并（OM/HAM 与 run 型指标都以此为原始序列）。"""
        out = []
        for code in self.codes:
            if not out or out[-1] != code:
                out.append(code)
        return out

    def ngram_support(self, n):
        """返回所有长度 n 的窗口（含空位）：n-gram / PrefixSpan 的原料。"""
        n = max(int(n), 1)
        return [tuple(self.codes[i:i + n]) for i in range(0, max(self.length - n + 1, 0))]


# ------------------------------------------------------------------ bundle

class GroupBundle:
    """一组分析样本的全部事实。

    ``wide`` 是 `Time + S{i}Encoding_{j}` 形态的状态宽表，``dataset`` 是它经统一链路
    得到的长表（描述统计与 statviz 方法可直接复用），``events`` 是 InteractionEvent。
    状态表与事件表永远分开存：同一时刻允许多条事件（V09），合并就会丢数据。
    """

    def __init__(self, group_id, source_file="", wide=None, events=None,
                 codebook=None, meta=None, dataset=None):
        self.group_id = sanitize_group_id(group_id)
        self.source_file = str(source_file or "")
        self.wide = im.ensure_event_columns(pd.DataFrame()) if wide is None else pd.DataFrame(wide).copy()
        self.events = im.normalize_events(events, self.group_id) if events is not None \
            else im.ensure_event_columns(pd.DataFrame())
        self.codebook = list(codebook or [])
        self.meta = dict(meta or {})
        self.meta.setdefault("group_name", self.meta.get("group_name") or self.group_id)
        self._dataset = dataset
        if self._dataset is None and len(self.wide):
            self._dataset = from_dataframe(self.wide, source=self.source_file or self.group_id)
        if self._dataset is None:
            self._dataset = CodingDataset()

    # ---------- 基本盘 ----------
    @property
    def dataset(self):
        return self._dataset

    @property
    def members(self):
        """组成员：优先 Students/Group_Info 的登记，其次从长表里现读。"""
        listed = [str(value).strip() for value in self.meta.get("members", []) if str(value).strip()]
        if listed:
            return listed
        students = list(getattr(self._dataset, "students", []) or [])
        if students:
            return students
        return members_from_columns(self.wide.columns)

    @property
    def uids(self):
        return [student_uid(self.group_id, member) for member in self.members]

    def time_points(self):
        """观察到的时间点（升序、去重）：来自状态表，状态表空时用事件表的时刻。"""
        times = _time_column(self.wide)
        if times is not None:
            values = sorted({float(value) for value in times if pd.notna(value)})
            if values:
                return values
        stamps = [im.to_seconds(value) for value in self.events.get("timestamp", [])] \
            if len(self.events) else []
        return sorted({float(value) for value in stamps if value is not None})

    @property
    def n_times(self):
        return len(self.time_points())

    @property
    def has_events(self):
        return bool(len(self.events))

    @property
    def outcome(self):
        """组结果（FR 里的监督目标）：Group_Info 的 outcome 列，没有就留 None。"""
        value = self.meta.get("outcome")
        return None if value in (None, "") else value

    def event_context(self, interaction_types=None):
        """事件校验现场：成员 + 外部主体 + 对象字典 + 观察窗口。"""
        members = self.members
        actors = [str(value).strip() for value in self.meta.get("actors", []) if str(value).strip()]
        points = self.time_points()
        return im.EventContext(
            group_id=self.group_id, members=members, actors=actors,
            object_dict=self.meta.get("object_dict") or {},
            interaction_types=interaction_types or self.meta.get("interaction_types"),
            t_min=points[0] if points else 0.0,
            t_max=points[-1] if points else None,
            events=[dict(row) for row in self.events.to_dict("records")],
        )

    # ---------- 序列 ----------
    def state_frame(self):
        """长表视角的状态帧：student / time / dimension / code，按 (student, time) 排序。"""
        frame = getattr(self._dataset, "long_df", None)
        if frame is None or not len(frame):
            frame = melt_wide(self.wide, self.members)
        if not len(frame):
            return frame
        frame = frame.copy()
        frame["group_id"] = self.group_id
        frame["uid"] = [student_uid(self.group_id, value) for value in frame["student_id"]]
        return frame.sort_values(["uid", "dimension", "time"], kind="stable")

    def sequences(self, dimension=None, student=None, drop_empty=True):
        """组内 (学生 × 维度) 序列集合：这是 IND 全部方法与 GRP 组内方法的取数口。"""
        frame = self.state_frame()
        if not len(frame):
            return []
        out = []
        grouped = frame.groupby(["student_id", "dimension"], sort=False)
        for (member, dim), part in grouped:
            if student is not None and str(member) != str(student):
                continue
            if dimension is not None and str(dim) != str(dimension):
                continue
            part = part.sort_values("time", kind="stable")
            seq = Sequence(self.group_id, member, dim, list(part["time"]),
                           [norm_code(value) for value in part["code"]],
                           source=os.path.basename(self.source_file))
            if drop_empty and not seq.valid:
                continue
            out.append(seq)
        return out

    def consensus_sequence(self, dimension, times=None):
        """组级共识码：同一时刻全组该维度的众数码（GRP05 的两套矩阵之一）。"""
        frame = self.state_frame()
        if not len(frame) or dimension is None:
            return Sequence(self.group_id, "GROUP", dimension, [], [])
        part = frame[frame["dimension"] == str(dimension)]
        points = list(times if times is not None else self.time_points())
        codes, used = [], []
        for moment in points:
            bucket = [norm_code(value) for value in part[part["time"] == moment]["code"]
                      if norm_code(value)]
            codes.append(pd.Series(bucket).mode().iloc[0] if bucket else "")
            used.append(moment)
        return Sequence(self.group_id, "GROUP", dimension, used, codes, source="consensus")

    def member_matrix(self, dimension, times=None):
        """成员 × 时间点的码矩阵（GRP01/02/03/06 的取数口）：缺码留空串。"""
        frame = self.state_frame()
        points = list(times if times is not None else self.time_points())
        matrix = {member: [""] * len(points) for member in self.members}
        if not len(frame) or dimension is None:
            return points, matrix
        part = frame[frame["dimension"] == str(dimension)]
        index = {moment: position for position, moment in enumerate(points)}
        for _, row in part.iterrows():
            member = str(row["student_id"])
            position = index.get(float(row["time"]))
            if position is None:
                continue
            matrix.setdefault(member, [""] * len(points))[position] = norm_code(row["code"])
        return points, matrix

    # ---------- 存档 ----------
    def to_dict(self):
        return {
            "group_id": self.group_id,
            "source_file": self.source_file,
            "members": self.members,
            "meta": self.meta,
            "codebook": [{"dimension": name, "codes": list(codes)}
                         for name, codes in self.codebook],
            "wide": self.wide.to_dict("records"),
            "events": self.events.to_dict("records"),
        }

    @classmethod
    def from_dict(cls, payload):
        codebook = [(str(item.get("dimension", "")),
                     [str(code) for code in item.get("codes", [])])
                    for item in payload.get("codebook", [])]
        return cls(payload.get("group_id", ""), payload.get("source_file", ""),
                   wide=pd.DataFrame(payload.get("wide") or []),
                   events=pd.DataFrame(payload.get("events") or []),
                   codebook=codebook, meta=payload.get("meta") or {})


# ------------------------------------------------------------------ 宽表工具

def members_from_columns(columns):
    """从 `S{i}Encoding_{j}` 列名里提取成员号（保持首次出现顺序）。"""
    seen, out = set(), []
    for column in columns:
        match = STRICT_COLUMN_RE.match(str(column).strip())
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            out.append(match.group(1))
    return out


def _time_column(frame):
    if frame is None or not len(frame):
        return None
    for column in frame.columns:
        if str(column).strip().lower() in ("time", "时间", "timestamp", "秒"):
            return pd.to_numeric(frame[column], errors="coerce")
    return None


def melt_wide(wide, members=None):
    """`Time + S{i}Encoding_{j}` → 长表（无 CodingDataset 时的兜底路径）。"""
    rows = []
    if wide is None or not len(wide):
        return pd.DataFrame(columns=["student_id", "time", "dimension", "code"])
    times = _time_column(wide)
    mapping, _ = parse_columns(list(wide.columns))
    for position in range(len(wide)):
        moment = None if times is None else times.iloc[position]
        if moment is None or pd.isna(moment):
            continue
        for column, (member, dimension) in mapping.items():
            rows.append({"student_id": member, "time": float(moment),
                         "dimension": dimension,
                         "code": norm_code(wide[column].iloc[position])})
    return pd.DataFrame(rows, columns=["student_id", "time", "dimension", "code"])


def build_wide(records, members, dimensions, times=None):
    """从 (student, time, dimension) → code 的字典造状态宽表：造测试数据与读档共用。"""
    dimensions = list(dimensions) or ["D1"]
    columns = ["Time"] + ["%sEncoding_%d" % (member, index + 1)
                          for member in members for index in range(len(dimensions))]
    points = sorted(times if times is not None else
                    {key[1] for key in records}, key=float)
    frame = pd.DataFrame(index=range(len(points)), columns=columns)
    frame["Time"] = points
    for (member, moment, dimension), code in records.items():
        position = points.index(float(moment))
        try:
            index = dimensions.index(dimension)
        except ValueError:
            continue
        frame.iat[position, columns.index("%sEncoding_%d" % (member, index + 1))] = code
    return frame.fillna("")


# ------------------------------------------------------------------ 装载

def group_id_from_path(path, override=None):
    """组号默认取文件名去扩展名（`group_1.xlsx` → `group_1`），可被覆写。"""
    if override not in (None, ""):
        return sanitize_group_id(override)
    return sanitize_group_id(os.path.splitext(os.path.basename(str(path)))[0])


def load_bundle(path, group_id=None, with_events=True):
    """一个文件装成一个 bundle：状态 sheet + 可选事件 sheet + Group_Info 附带字段。"""
    identifier = group_id_from_path(path, group_id)
    wide = im.load_group_sheet(path)
    events = im.load_event_sheet(path) if with_events else pd.DataFrame()
    meta = {}
    members = members_from_columns(wide.columns)
    try:
        info = pd.read_excel(path, sheet_name="Group_Info", dtype=object)
        if len(info):
            row = info.iloc[0].to_dict()
            # cell_text 把 NaN 洗成空串，0 / 0.0 仍是真值文本，不会误删“观察起点 0”
            meta.update({key: value for key, value in row.items() if im.cell_text(value)})
            members = _split_list(row.get("members")) or members
        students = pd.read_excel(path, sheet_name="Students", dtype=object)
        if len(students) and "student_id" in students.columns:
            members = [im.cell_text(value) for value in students["student_id"]
                       if im.cell_text(value)] or members
    except ValueError:
        pass      # 老文件只有 Sheet1，没有 Group_Info / Students，属正常
    # 事件表只认 Interaction_Event sheet：拿状态表去当事件会满屏 V01/V03
    codebook = []
    try:
        book = pd.read_excel(path, sheet_name="Codebook", dtype=object)
        codebook = codebook_from_frame(book)
    except ValueError:
        pass
    meta.setdefault("source_file", str(path))
    return GroupBundle(identifier, str(path), wide=wide, events=events,
                       codebook=codebook, meta=dict(meta, members=members))


def load_bundles(paths, group_ids=None, with_events=True):
    """批量上传：一文件一组，第 i 个文件可用 group_ids[i] 覆写组号。"""
    paths = [path for path in (paths or []) if path]
    ids = list(group_ids or [])
    bundles = []
    for index, path in enumerate(paths):
        override = ids[index] if index < len(ids) else None
        bundles.append(load_bundle(path, override, with_events=with_events))
    return dedupe_group_ids(bundles)


def dedupe_group_ids(bundles):
    """组号撞车就地加后缀：撞了就会让 student_uid 混样本，宁可在装载期改名。"""
    seen = {}
    for bundle in bundles:
        base = bundle.group_id
        count = seen.get(base, 0)
        seen[base] = count + 1
        if count:
            bundle.group_id = "%s_%d" % (base, count + 1)
            bundle.meta["group_id_renamed_from"] = base
            bundle.events = im.normalize_events(bundle.events, bundle.group_id)
    return bundles


def state_frame_of(summary_tab):
    """取数据汇总页的 `Time + S{i}Encoding_{j}` 宽表。

    `build_analysis_data` 返的是 ``(列头, 行列表)``，不是 DataFrame；
    旧存档页可能只有 `table_dataframe`，两条路径都收。
    """
    builder = getattr(summary_tab, "build_analysis_data", None)
    if callable(builder):
        columns, rows = builder()
        if columns and rows:
            return pd.DataFrame(rows, columns=columns)
        return pd.DataFrame()
    frame_getter = getattr(summary_tab, "table_dataframe", None)
    if callable(frame_getter):
        return frame_getter()
    return pd.DataFrame()


def bundle_from_session(group_id="group_1", summary_tab=None, group_encoder=None,
                        encoder_pages=None, members=None, codebook=None, meta=None):
    """从当前编码会话取数：数据汇总页给状态表，小组交互页给事件表。

    参数都做成可注入的“页面对象”，是为了让离屏测试能造一个假 summary_tab 进来，
    也为了让主窗口不必把内部控件结构摊开给分析层。
    """
    wide = pd.DataFrame()
    if summary_tab is not None:
        wide = state_frame_of(summary_tab)
    events = pd.DataFrame()
    if group_encoder is not None:
        getter = getattr(group_encoder, "events_frame", None)
        if callable(getter):
            events = getter()
        collect = getattr(group_encoder, "collect_meta", None)
        if callable(collect):
            meta = dict(meta or {}, **collect())
    if codebook is None and encoder_pages:
        codebook = resolve_codebook(None, framework=framework_from_pages(encoder_pages))
    if members is None:
        members = members_from_columns(wide.columns) or list(members or [])
        if not members and encoder_pages:
            members = ["S%d" % (index + 1) for index in range(len(encoder_pages))]
    payload = dict(meta or {})
    payload.setdefault("members", members)
    payload.setdefault("source", "session")
    return GroupBundle(group_id, "", wide=wide, events=events,
                       codebook=codebook or [], meta=payload)


def framework_from_pages(pages):
    """从若干编码页收框架：[(维度名, [编码...]), ...]，同名维度合并编码。"""
    merged, order = {}, []
    for page in pages or []:
        panel = getattr(page, "dim_panel", None)
        if panel is None or not hasattr(panel, "framework"):
            continue
        for name, codes in panel.framework():
            key = str(name).strip()
            if not key:
                continue
            if key not in merged:
                merged[key] = []
                order.append(key)
            for code in codes:
                text = str(code).strip()
                if text and text not in merged[key]:
                    merged[key].append(text)
    return [(key, merged[key]) for key in order]


# ------------------------------------------------------------------ 编码框架

def codebook_from_frame(frame):
    """Codebook sheet（`dimension` / `code` / `label`）→ [(维度, [编码])]。"""
    if frame is None or not len(frame):
        return []
    columns = {str(column).strip().lower(): column for column in frame.columns}
    dim_key = next((columns[key] for key in ("dimension", "维度") if key in columns), None)
    code_key = next((columns[key] for key in ("code", "编码", "类型") if key in columns), None)
    if dim_key is None or code_key is None:
        return []
    out, order = {}, []
    for _, row in frame.iterrows():
        name = str(row[dim_key] or "").strip()
        code = str(row[code_key] or "").strip()
        if not name or not code:
            continue
        if name not in out:
            out[name] = []
            order.append(name)
        if code not in out[name]:
            out[name].append(code)
    return [(name, out[name]) for name in order]


def read_framework_excel(path):
    """导入框架 Excel：严格两列 `维度` / `类型`（与 dimension_panel 同一口径）。"""
    frame = pd.read_excel(path)
    columns = [str(column).strip() for column in frame.columns]
    if columns != ["维度", "类型"]:
        raise ValueError("框架文件必须包含且仅包含两列：维度 / 类型（当前为 %s）" % columns)
    out, order = {}, []
    for _, row in frame.iterrows():
        name = str(row["维度"]).strip()
        code = str(row["类型"]).strip()
        if not name:
            continue
        if name not in out:
            out[name] = []
            order.append(name)
        if code and code.lower() != "nan" and code not in out[name]:
            out[name].append(code)
    return [(name, out[name]) for name in order]


def resolve_codebook(bundle, framework=None, excel=None, infer_codes=True):
    """三档优先：调用方给的当前会话框架 → 导入的框架 Excel → 从数据推断。

    返回 `[(维度名, [编码])]`。维度名优先用框架里的中文名，数据推断出来的是
    `D1/D2`；下游表头与导出都以这份为准，所以谁提供了名字就在结果里保留谁。
    """
    if framework:
        return [(str(name).strip(), [str(code).strip() for code in codes if str(code).strip()])
                for name, codes in framework]
    if excel:
        return read_framework_excel(excel)
    if bundle is not None and bundle.codebook:
        return list(bundle.codebook)
    if bundle is None or not infer_codes:
        return []
    frame = bundle.state_frame()
    if not len(frame):
        return []
    out = []
    for dimension, part in frame.groupby("dimension", sort=False):
        codes = sorted({norm_code(value) for value in part["code"] if norm_code(value)})
        out.append((str(dimension), codes))
    return out


# ------------------------------------------------------------------ 合并

def concat_bundles(bundles, dimension=None):
    """多组 → 序列列表：**只在组内切序列，绝不把两组的头尾接起来**。

    返回的每个 Sequence 仍带自己的 group_id，跨组比较（GRP10）拿到的是「样本 = 组
    内序列」的集合，而不是被拼接污染的长链。
    """
    out = []
    for bundle in bundles or []:
        out.extend(bundle.sequences(dimension=dimension))
    return out


def concat_frames(bundles, columns=None):
    """长表级合并：加 group_id / uid 两列，供描述统计按组算分母。"""
    frames = []
    for bundle in bundles or []:
        frame = bundle.state_frame()
        if len(frame):
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns or [])
    merged = pd.concat(frames, ignore_index=True)
    return merged[columns] if columns else merged


def event_frame(bundles):
    """多组事件表纵向合并（列序照总表），group_id 天然区分了来源。"""
    frames = [bundle.events for bundle in bundles or [] if len(bundle.events)]
    if not frames:
        return im.ensure_event_columns(pd.DataFrame())
    return im.ensure_event_columns(pd.concat(frames, ignore_index=True))


# ------------------------------------------------------------------ 质检

def bundle_quality(bundles):
    """装载期质检：组数、成员数、时间点、事件与状态表的组号一致性。"""
    items = []
    bundles = list(bundles or [])
    if not bundles:
        return [QAItem("error", "core.no_data", "尚未装载任何组数据", True)]
    for bundle in bundles:
        if not bundle.members:
            items.append(QAItem("error", "core.no_members",
                                "%s：未识别到成员列（应为 S1Encoding_1 形态）" % bundle.group_id, True))
        if bundle.n_times < 2:
            items.append(QAItem("warn", "core.short_axis",
                                "%s：只有 %d 个时间点，转移/熵类结果不稳定"
                                % (bundle.group_id, bundle.n_times)))
        if len(bundle.events) and set(bundle.events["group_id"].astype(str)) - {bundle.group_id}:
            items.append(QAItem("warn", "core.event_group_mismatch",
                                "%s：事件表里混入了其他 group_id" % bundle.group_id))
        items.extend(im.validate_events(bundle.events, bundle.event_context()))
    if len(bundles) == 1:
        items.append(QAItem("info", "core.single_group",
                            "当前仅 1 组，跨组比较与组间检验只能作组内报告", False))
    return items


def group_count_note(bundles):
    """跨组方法的降级文案：不隐藏功能，选中就写清楚为什么只能组内看。"""
    count = len(bundles or [])
    if count >= 2:
        return ""
    return "当前仅 %d 组，只能作组内报告；跨组比较需 ≥ 2 组" % count
