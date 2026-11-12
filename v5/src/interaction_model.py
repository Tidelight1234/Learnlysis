# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""小组交互事件的数据契约：InteractionEvent 列定义、枚举词表、V01–V10 校验器。

个体编码记的是「某一时刻某人处于什么状态」（状态格），小组交互记的是「某一时刻谁
对谁/对什么做了什么」（离散事件）。两者不能合并成一张表：同一时刻允许多条事件并存
（V09），而状态格同一时刻只有一值。列名与枚举因此是硬契约，导入导出、R 参考层、
网络分析都按本文件的顺序与取值解释，任何一处改名都会让下游静默错位。

- ``EVENT_COLUMNS``   ：总表列序（13 列），表格与导出都用这一种顺序。
- ``TARGET_TYPES`` / ``INTERACTION_TYPES``：枚举词表；interaction_type 用户可就地编辑。
- ``EventContext``    ：一次录入/一批导入的判定现场（成员、主体、对象字典、已录事件）。
- ``validate_row`` / ``validate_events``：V01–V10，返回 ``QAItem`` 列表。
- ``new_event_id`` / ``make_template`` / ``write_workbook``：ID 生成与多 sheet 模板落盘。

阻断与否的分工照规格说明书：V01–V04、V06、V07、V10 违反即拒绝写入（会污染主键或
外键，留到分析阶段无法回溯）；V05、V08 只警告并计入数据质量报告（口径问题，删不删
由研究者判断）。V09 明确「同窗多事件不去重」，任何去重动作都是错。
"""

import os
import re

import pandas as pd

from statistic_visualization_data import QAItem

# ------------------------------------------------------------------ 列与枚举

#: Interaction_Event 总表列序：录入表格、Excel 导出、R 参考层共用
EVENT_COLUMNS = [
    "group_id", "event_id", "timestamp", "actor_id",
    "target_id", "target_type", "interaction_type",
    "reply_to", "turn_id", "duration", "source", "confidence", "notes",
]
#: 只读三列：由时间轴与组号推算，表格双击不给改
DERIVED_COLUMNS = ["group_id", "event_id", "timestamp"]
#: 话语原文所在列（个体编码页的「注释」在这里换成了它）
UTTERANCE_COLUMN = "utterance"

#: target_type 的 9 个对象类别 + 2 个特殊值
TARGET_TYPES = [
    "PERSON", "TASK", "QUESTION", "ANSWER", "MATERIAL",
    "TOOL", "SYSTEM", "GROUP", "SPACE",
]
NO_TARGET = "NO_TARGET"
SELF_TARGET = "SELF"
TARGET_TYPE_CHOICES = TARGET_TYPES + [NO_TARGET, SELF_TARGET]
#: 需要落到「本组对象字典」的类别（PERSON 走成员/主体字典，NO_TARGET/SELF 不填 target_id）
OBJECT_TARGET_TYPES = ["TASK", "QUESTION", "ANSWER", "MATERIAL", "TOOL", "SYSTEM",
                       "GROUP", "SPACE"]

#: interaction_type 默认 8 值：词表可编辑，但落表后不得残留表外的值
INTERACTION_TYPES = ["SPEAK", "ASK", "ANSWER", "EXPLAIN",
                     "REQUEST", "CONFIRM", "CHALLENGE", "DIRECT"]

EVENT_SOURCES = ["CODING", "TRANSCRIPT", "LOG", "MANUAL"]

#: 校验规则编号：UI 与质量报告都按这个编号说话
RULES = {
    "V01": "主键与必填列（group_id / event_id / timestamp / actor_id）不得为空",
    "V02": "actor_id 必须是已登记主体（组成员或教师/系统等其他主体）",
    "V03": "timestamp 必须可解析为非负秒数，且落在观察窗口内",
    "V04": "target_type 必须取自枚举词表",
    "V05": "target_id 与 target_type 一致（对象类须在对象字典内；SELF 即本人；NO_TARGET 留空）",
    "V06": "event_id 组内唯一且形如 {group_id}_E{n:03d}",
    "V07": "reply_to 只能指向本组内时间更早的事件",
    "V08": "duration 非负、confidence 落在 0–1",
    "V09": "同一时刻允许多条事件并存（不去重）",
    "V10": "interaction_type 必须取自当前词表",
}
#: 违反即拒绝写入的规则
BLOCKING_RULES = {"V01", "V02", "V03", "V04", "V06", "V07", "V10"}
#: 只警告、计入数据质量报告的规则
WARNING_RULES = {"V05", "V08"}

EVENT_ID_RE = re.compile(r"^(?P<group>.+)_E(?P<n>\d{3,})$")
TIME_ALIASES = ("timestamp", "time", "时间", "时刻", "秒")


# ------------------------------------------------------------------ 小工具

def blank_event(group_id="", **fields):
    """一列不缺的事件字典：UI 与读档都以此为底，避免下游 KeyError。"""
    row = {column: "" for column in EVENT_COLUMNS}
    row["group_id"] = group_id
    row["timestamp"] = None
    row["duration"] = None
    row["confidence"] = None
    row["source"] = "CODING"
    row.update(fields)
    return row


def codebook_rows(framework):
    """编码框架 → Codebook sheet 的行。

    兼容两种上游形状：编码页 ``[(维度名, [编码…])]`` 与已展平的
    ``[{'dimension','code','label'}]``。不猜维度，空框架就空表。
    """
    rows = []
    for item in framework or []:
        if isinstance(item, dict):
            rows.append({"dimension": str(item.get("dimension", "")),
                         "code": str(item.get("code", "")),
                         "label": str(item.get("label", ""))})
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            name, codes = item
            rows += [{"dimension": str(name), "code": str(code), "label": ""}
                     for code in (codes or [])]
    return rows


def new_event_id(group_id, number):
    """事件主键：`{group_id}_E{n:03d}`，序号从 1 起，不足三位补零。"""
    return "%s_E%03d" % (group_id, int(number))


def next_event_number(existing, group_id):
    """已有事件里的最大序号 + 1：删过行也不复用旧号，主键一旦发号就永不回收。"""
    highest = 0
    for event_id in existing or []:
        match = EVENT_ID_RE.match(str(event_id or ""))
        if not match or match.group("group") != str(group_id):
            continue
        highest = max(highest, int(match.group("n")))
    return highest + 1


def to_seconds(value):
    """时间列 → 秒：数字直接用，HH:MM:SS / MM:SS 折算，其余 None。

    小组事件的 timestamp 允许来自视频播放头（HH:MM:SS）或日志（数字秒），
    判定口径与状态表完全一致，两处各写一份迟早分叉，故直接复用数据层的解析。
    """
    from statistic_visualization_data import parse_time_value
    return parse_time_value(value)


def parse_confidence(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cell_text(value):
    """单元格→安全文本：Excel 读进来的空格是 NaN，`str(NaN)` 会给出字面量 'nan'。

    校验与下拉候选都拿它比字符串，不统一就会把“没填”当成“填了叫 nan 的值”。
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass                     # 列表/元格这类不可标量判空的，按文本走
    return str(value).strip()


# ------------------------------------------------------------------ 判定现场

class EventContext:
    """一次校验所依赖的现场：谁是成员、有哪些主体、对象字典、已录事件、观察窗口。

    所有集合都存成字符串，比较时一律 `str().strip()`：Excel 读进来的是 object，
    视频页给出的是 int，不统一就会把同一个 ID 判成两个。
    """

    def __init__(self, group_id="", members=None, actors=None, object_dict=None,
                 interaction_types=None, t_min=0.0, t_max=None, events=None):
        self.group_id = cell_text(group_id)
        self.members = [cell_text(m) for m in (members or []) if cell_text(m)]
        # 主体字典：成员之外的教师/系统/观察者也算合法 actor
        self.actors = [cell_text(a) for a in (actors or []) if cell_text(a)]
        self.object_dict = {cell_text(key): [cell_text(v) for v in values if cell_text(v)]
                            for key, values in (object_dict or {}).items()}
        self.interaction_types = [cell_text(t) for t in (interaction_types or [])
                                  if cell_text(t)] or list(INTERACTION_TYPES)
        self.t_min = float(t_min or 0.0)
        self.t_max = None if t_max is None else float(t_max)
        self.events = list(events or [])

    # ---------- 候选（UI 的字段选择器直接拿它填下拉） ----------
    def actor_choices(self):
        seen, out = set(), []
        for value in self.members + self.actors:
            if value not in seen:
                seen.add(value)
                out.append(value)
        return out

    def object_choices(self, target_type):
        if target_type == NO_TARGET:
            return []
        if target_type == SELF_TARGET:
            return list(self.members)
        if target_type == "PERSON":
            return self.actor_choices()
        return list(self.object_dict.get(target_type, []))

    def add_object(self, target_type, object_id):
        """就地新增对象：录了一个没登记过的任务号，不该为此去改模板。"""
        object_id = cell_text(object_id)
        if not object_id:
            return
        bucket = self.object_dict.setdefault(cell_text(target_type), [])
        if object_id not in bucket:
            bucket.append(object_id)

    def known_actor(self, actor_id):
        return cell_text(actor_id) in set(self.actor_choices())

    def existing_ids(self):
        return [cell_text(item) for item in self.events if cell_text(item)]

    def event_time(self, event_id):
        """已录事件的时间：reply_to 的前后判定要用。"""
        for item in self.events:
            if isinstance(item, dict) and cell_text(item.get("event_id")) == cell_text(event_id):
                return to_seconds(item.get("timestamp"))
        return None

    def earlier_event_ids(self, timestamp, exclude=""):
        """给定时刻上可被 reply_to 引用的事件（严格更早，排除自己）。"""
        out = []
        for item in self.events:
            if not isinstance(item, dict):
                continue
            event_id = cell_text(item.get("event_id"))
            if not event_id or event_id == cell_text(exclude):
                continue
            when = to_seconds(item.get("timestamp"))
            if when is None:
                continue
            current = to_seconds(timestamp)
            if current is None or when < current:
                out.append(event_id)
        return out


# ------------------------------------------------------------------ V01–V10

def _item(rule, message, level=None, blocking=None):
    level = level or ("error" if rule in BLOCKING_RULES else "warn")
    blocking = (rule in BLOCKING_RULES) if blocking is None else blocking
    return QAItem(level, rule, "[ %s ] %s" % (rule, message), blocking)


def _is_blank(value):
    """“没填”的判定：None / NaN / 空串算没填，0 / 0.0 是合法值（时间 0 秒就是观察起点）。"""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def validate_row(row, ctx):
    """单条事件的即时校验：返回按规则号排序的 QAItem 列表（空列表即放行）。"""
    items = []
    label = cell_text(row.get("event_id")) or cell_text(row.get("actor_id")) or "新事件"

    missing = [column for column in ("group_id", "event_id", "timestamp", "actor_id")
               if _is_blank(row.get(column))]
    if missing:
        items.append(_item("V01", "%s 缺少必填列：%s" % (label, "、".join(missing))))

    seconds = to_seconds(row.get("timestamp"))
    if seconds is None:
        if "V03" not in [item.code for item in items]:
            items.append(_item("V03", "%s 的时间 %r 无法解析为秒数"
                               % (label, row.get("timestamp"))))
    elif seconds < 0:
        items.append(_item("V03", "%s 的时间为负（%s）" % (label, seconds)))
    elif ctx.t_max is not None and seconds > ctx.t_max + 1e-9:
        items.append(_item("V03", "%s 的时间 %s 超出观察窗口上界 %s"
                           % (label, _num(seconds), _num(ctx.t_max))))
    elif seconds < ctx.t_min - 1e-9:
        items.append(_item("V03", "%s 的时间 %s 早于观察起点 %s"
                           % (label, _num(seconds), _num(ctx.t_min))))

    actor = cell_text(row.get("actor_id"))
    if actor and not ctx.known_actor(actor):
        items.append(_item("V02", "主体 %s 未登记（既不是组成员也不是已登录的其他主体）" % actor))

    target_type = cell_text(row.get("target_type"))
    if target_type and target_type not in TARGET_TYPE_CHOICES:
        items.append(_item("V04", "target_type %r 不在枚举内" % target_type))

    interaction = cell_text(row.get("interaction_type"))
    if interaction and interaction not in ctx.interaction_types:
        items.append(_item("V10", "interaction_type %r 不在当前词表内" % interaction))

    target_id = cell_text(row.get("target_id"))
    problems = []
    if target_type == NO_TARGET and target_id:
        problems.append("NO_TARGET 不应填 target_id")
    if target_type == SELF_TARGET and target_id and target_id != actor:
        problems.append("SELF 的 target_id 必须等于 actor_id")
    if target_type == "PERSON" and target_id and not ctx.known_actor(target_id):
        problems.append("PERSON 目标 %s 未登记" % target_id)
    if target_type in OBJECT_TARGET_TYPES:
        if not target_id:
            problems.append("%s 必须给出对象 ID" % target_type)
        elif ctx.object_dict.get(target_type) and target_id not in ctx.object_choices(target_type):
            # 只在已登录该类对象时才查一致性：没字典可查时报“不在字典内”是报个假错
            problems.append("对象 %s 不在 %s 字典内" % (target_id, target_type))
    if problems:
        items.append(_item("V05", "%s：%s" % (label, "；".join(problems))))

    event_id = cell_text(row.get("event_id"))
    if event_id:
        match = EVENT_ID_RE.match(event_id)
        if not match:
            items.append(_item("V06", "event_id %r 不符合 {group_id}_E{n:03d}" % event_id))
        elif ctx.group_id and match.group("group") != ctx.group_id:
            items.append(_item("V06", "event_id %r 的组号与当前组 %r 不一致"
                               % (event_id, ctx.group_id)))

    reply_to = cell_text(row.get("reply_to"))
    if reply_to:
        when = to_seconds(row.get("timestamp"))
        previous = ctx.event_time(reply_to)
        if previous is None:
            items.append(_item("V07", "reply_to 指向的 %r 不在本组已录事件内" % reply_to))
        elif when is not None and previous >= when:
            items.append(_item("V07", "reply_to 只能指向时间更早的事件（%s 在 %s 之后或同时）"
                               % (reply_to, label)))

    duration = to_seconds(row.get("duration"))
    if not _is_blank(row.get("duration")) and duration is None:
        items.append(_item("V08", "duration %r 不是数值" % row.get("duration")))
    elif duration is not None and duration < 0:
        items.append(_item("V08", "duration 不得为负（%s）" % _num(duration)))
    if not _is_blank(row.get("confidence")):
        confidence = parse_confidence(row.get("confidence"))
        if confidence is None or not 0.0 <= confidence <= 1.0:
            items.append(_item("V08", "confidence 必须落在 0–1（当前 %r）"
                               % row.get("confidence")))

    order = {"V01": 1, "V02": 2, "V03": 3, "V04": 4, "V05": 5,
             "V06": 6, "V07": 7, "V08": 8, "V09": 9, "V10": 10}
    return sorted(items, key=lambda item: order.get(item.code, 99))


def blocking_items(items):
    return [item for item in items if item.code in BLOCKING_RULES]


def warning_items(items):
    return [item for item in items if item.code in WARNING_RULES]


def validate_events(frame, ctx):
    """整表校验（导入、导出前、数据质量报告）：逐行 V01–V10 + 全局 V06/V09。"""
    items = []
    if frame is None or len(frame) == 0:
        return [QAItem("info", "V00", "暂无交互事件，组级网络类方法不可用", False)]
    frame = ensure_event_columns(frame)
    counts = {}
    ids = []
    for _, record in frame.iterrows():
        row = record.to_dict()
        items.extend(validate_row(row, ctx))
        event_id = cell_text(row.get("event_id"))
        if event_id:
            ids.append(event_id)
        counts[cell_text(row.get("timestamp"))] = counts.get(
            cell_text(row.get("timestamp")), 0) + 1
    duplicated = sorted({value for value in ids if ids.count(value) > 1})
    if duplicated:
        items.append(_item("V06", "event_id 重复：%s" % "、".join(duplicated[:8])))
    multi = {key: value for key, value in counts.items() if value > 1}
    if multi:
        # V09 是「允许多条」而不是「发现重复」：只报计数，不给阻断
        items.append(QAItem("info", "V09",
                            "有 %d 个时刻存在多条事件（按规格不去重，网络分析按时序建边）"
                            % len(multi), False))
    return items


# ------------------------------------------------------------------ 表与模板

def _is_missing(value):
    """单格是否缺失：None / NaN / NaT。空串不在此列（它就是“没填”的表示形式）。"""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def ensure_event_columns(frame):
    """补齐缺失列、抹掉多余列，并按总表列序返回副本（不改动入参）。

    空格统一成空串：Excel 读回来是 NaN，直接 `str()` 会得到字面量 'nan'，
    把“没填”伪装成一个真值，V07/V05 就会报假错。数值列的实体值照原样保留。
    """
    frame = pd.DataFrame(frame) if frame is not None else pd.DataFrame()
    out = pd.DataFrame(index=frame.index)
    columns = EVENT_COLUMNS + ([UTTERANCE_COLUMN] if UTTERANCE_COLUMN in frame.columns else [])
    for column in columns:
        if column in frame.columns:
            out[column] = ["" if _is_missing(value) else value
                           for value in frame[column]]
        else:
            out[column] = ""
    return out


def normalize_events(frame, group_id):
    """读进来的事件表 → 契约形态：时间折算成秒、组号补齐、主键缺号就现编。"""
    frame = ensure_event_columns(frame)
    frame["group_id"] = frame["group_id"].apply(
        lambda value: cell_text(value) or str(group_id))
    frame["timestamp"] = frame["timestamp"].apply(to_seconds)
    frame["duration"] = frame["duration"].apply(to_seconds)
    frame["confidence"] = frame["confidence"].apply(parse_confidence)
    used = [cell_text(value) for value in frame["event_id"] if cell_text(value)]
    number = next_event_number(used, group_id)
    filled = []
    for value in frame["event_id"]:
        text = cell_text(value)
        if not text:
            text = new_event_id(group_id, number)
            number += 1
        else:
            # 改了组号就要重发前缀：号码不回收，但前缀必须跟着当前组，否则 V06 永远红
            match = EVENT_ID_RE.match(text)
            if match and match.group("group") != str(group_id):
                text = new_event_id(group_id, int(match.group("n")))
        filled.append(text)
    frame["event_id"] = filled
    return frame


def reindex_event_ids(frame, group_id, start=1):
    """按当前时间序重排主键：只在「清空重录」或造测试数据时用，平时主键永不回收。"""
    out = ensure_event_columns(frame).reset_index(drop=True)
    if len(out):
        out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    out["group_id"] = group_id
    out["event_id"] = [new_event_id(group_id, start + index) for index in range(len(out))]
    return out


def template_sheets(group_id="group_1", group_name="", members=None,
                    interaction_types=None, codebook=None, behavior_state=None,
                    events=None):
    """多 sheet 模板：Group_Info / Students / Behavior_State / Interaction_Event / Codebook / Enums。

    Behavior_State 可以塞一张已编好的状态表（`Time + S{i}Encoding_{j}`），其余 sheet
    只有表头 —— 模板的用途就是让人照列名填，不替人猜。
    """
    members = [str(m).strip() for m in (members or []) if str(m).strip()] or ["S1"]
    group_info = pd.DataFrame([{
        "group_id": group_id, "group_name": group_name or group_id,
        "member_count": len(members), "observation_start": 0, "observation_end": None,
        "source_file": "",
        # 组结果：监督型方法（IND12 outcome 路径 / GRP10 / IND15）的标签列，缺了就只走伪标签
        "outcome": None,
    }])
    students = pd.DataFrame({
        "group_id": [group_id] * len(members),
        "student_id": members,
        "student_uid": ["%s__%s" % (group_id, member) for member in members],
        "role": [""] * len(members),
    })
    state = pd.DataFrame(behavior_state) if behavior_state is not None and len(behavior_state) \
        else pd.DataFrame(columns=["Time"] + ["%sEncoding_%d" % (member, 1) for member in members])
    event = ensure_event_columns(events if events is not None else pd.DataFrame())
    if len(event) == 0:
        event[UTTERANCE_COLUMN] = []
    book = pd.DataFrame(codebook or [], columns=["dimension", "code", "label"])
    if len(book) and "label" not in book.columns:
        book["label"] = ""
    enums = pd.DataFrame([
        {"field": "target_type", "value": value, "note": ""} for value in TARGET_TYPE_CHOICES
    ] + [
        {"field": "interaction_type", "value": value, "note": ""}
        for value in (interaction_types or INTERACTION_TYPES)
    ] + [
        {"field": "rule", "value": key, "note": text} for key, text in sorted(RULES.items())
    ])
    return {
        "Group_Info": group_info, "Students": students, "Behavior_State": state,
        "Interaction_Event": event, "Codebook": book, "Enums": enums,
    }


def write_workbook(sheets, path):
    """多 sheet 落盘：列宽与表头加粗走 summary_tab 的同一口径。"""
    from openpyxl.styles import Alignment, Font
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            for column in sheet.columns:
                letter = column[0].column_letter
                width = max(len(str(cell.value)) for cell in column) if column else 10
                sheet.column_dimensions[letter].width = min(max(width + 2, 10), 42)
                for cell in column:
                    cell.alignment = Alignment(horizontal="center")
            for cell in sheet["1:1"]:
                cell.font = Font(bold=True)
    return path


def load_event_sheet(path):
    """批量上传时读 Interaction_Event sheet；没有该 sheet 返回空表（不报错）。"""
    try:
        frame = pd.read_excel(path, sheet_name="Interaction_Event", dtype=object)
    except ValueError:
        return ensure_event_columns(pd.DataFrame())
    return ensure_event_columns(frame)


def load_group_sheet(path, sheet=0):
    """读状态表主 sheet：兼容「只有一个 Sheet1」的老文件与新的多 sheet 文件。"""
    try:
        return pd.read_excel(path, sheet_name="Behavior_State", dtype=object)
    except ValueError:
        return pd.read_excel(path, sheet_name=sheet, dtype=object)


def _num(value):
    from statistic_visualization_data import fmt_num
    return fmt_num(value)
