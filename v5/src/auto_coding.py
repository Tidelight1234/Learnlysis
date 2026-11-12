# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""Phase 2 自动编码：关键帧 + 转写 + 编码框架 → 多模态 LLM → 结构化「建议码」→ 人在环写回。

贯穿原则（见方案）：产出一律是**「AI 建议、待确认」**，绝不静默覆盖用户已录的编码；逐维度
校验建议码必须落在 ``resolve_codebook`` / ``dim_panel.framework()`` 的词表内，越界即标注、
不可写回。网络调用放 ``AiWorker``（QThread），取消走单一通路；BYO key = 用户自付，逐次显示
用量。写回经编码页现有的 ``_push_history`` 口径，故可撤销、会发脏标记，与手工记录同源。

管线复用功能二基建：``ai_client``（OpenAI 兼容 + ``user_multimodal`` 图片块）、``ai_worker``、
``ai_settings``（端点 / key / 模型）、``video_probe``（独立离线抽帧，绝不动直播播放器）。
"""

from __future__ import annotations

import json
import logging
import re

import pandas

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QTextEdit, QListWidget, QListWidgetItem, QTableWidget,
                             QTableWidgetItem, QHeaderView, QAbstractItemView)

import theme
import user_state
import ai_settings
import video_probe
from ai_client import OpenAICompatClient, system_message, user_multimodal
from ai_worker import AiWorker

#: 写回时在注释列打的标记：提醒用户这行是 AI 建议、仍需逐条确认
SUGGEST_MARKER = "AI 建议、待确认"
#: 从模型回复里抠 JSON 用的正则：优先 ```json 围栏，其次裸 {...}
_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)
#: 模型有时把结果包一层：{"codes": {...}} / {"result": {...}} 等，遇到就拆包
_WRAP_KEYS = ("codes", "编码", "result", "data", "dimensions", "维度")
#: 维度值是对象时，从中取编码的候选键
_CODE_KEYS = ("code", "编码", "value", "label", "值")


# ------------------------------------------------------------------ prompt 构造

#: 码本释义 Excel 的列名别名（宽松匹配，中文为主，英文列头也认）
_CODEBOOK_ALIASES = {
    "dimension": ("维度", "维度名", "dimension", "Dimension", "编码维度"),
    "code": ("类型", "编码", "码", "code", "Code", "编码名", "type", "Type"),
    "desc": ("描述", "释义", "含义", "说明", "定义", "description", "definition"),
    "example": ("示例", "例子", "例", "example", "例句"),
}
#: 单条释义/示例进 prompt 的字符上限：超长截断，防 prompt 爆炸
_NOTE_DESC_CAP = 200
_NOTE_EXAMPLE_CAP = 160


def read_codebook_excel(path):
    """读「码本释义 Excel」→ ``{维度名: {编码: {"desc", "example"}}}``。

    列名按 ``_CODEBOOK_ALIASES`` 宽松匹配：至少要有「维度/编码」两列（与编码页框架导入
    的「维度/类型」两列硬校验同源），描述/释义与示例列可选。描述与示例都为空的行跳过；
    同一码多行时非空字段后行补空。无有效列/无有效行时抛 ValueError（人话）给调用方展示。
    """
    frame = pandas.read_excel(path)
    lower = {str(col).strip().lower(): col for col in frame.columns}
    taken = {}
    for field, aliases in _CODEBOOK_ALIASES.items():
        for alias in aliases:
            key = alias.strip().lower()
            if key in lower and lower[key] not in taken.values():
                taken[field] = lower[key]
                break
    if "dimension" not in taken or "code" not in taken:
        raise ValueError("未找到「维度」与「编码」（或「类型」）列；现有列：%s"
                         % "、".join(str(col) for col in frame.columns))

    def _cell(row, field):
        col = taken.get(field)
        if col is None:
            return ""
        try:
            if pandas.isna(row[col]):
                return ""
        except (TypeError, ValueError):
            pass
        return str(row[col]).strip()

    notes = {}
    for _i, row in frame.iterrows():
        dim = _cell(row, "dimension")
        code = _cell(row, "code")
        if not dim or not code:
            continue
        desc = _cell(row, "desc")[:_NOTE_DESC_CAP]
        example = _cell(row, "example")[:_NOTE_EXAMPLE_CAP]
        if not desc and not example:
            continue
        entry = notes.setdefault(dim, {}).setdefault(code, {"desc": "", "example": ""})
        if desc and not entry["desc"]:
            entry["desc"] = desc
        if example and not entry["example"]:
            entry["example"] = example
    if not notes:
        raise ValueError("文件里没有带释义/示例的有效行（至少需要维度、编码、描述列）")
    return notes


def format_code_tokens(codes, notes=None):
    """把词表渲染成 prompt 里的码串：有释义的码附「码（释：…；例：…）」。

    共用于自动编码 prompt 与 AI 对话上下文（ai_context.framework_section），保证两处口径
    一致；notes 缺失或该码无释义时退化为裸码。
    """
    out = []
    for code in codes:
        token = str(code)
        note = (notes or {}).get(token)
        if isinstance(note, dict) and (note.get("desc") or note.get("example")):
            parts = []
            if note.get("desc"):
                parts.append("释：%s" % str(note["desc"])[:_NOTE_DESC_CAP])
            if note.get("example"):
                parts.append("例：%s" % str(note["example"])[:_NOTE_EXAMPLE_CAP])
            token = "%s（%s）" % (token, "；".join(parts))
        out.append(token)
    return out


def build_prompt(codebook, transcript="", time_seconds=None, extra_rules="",
                 code_notes=None):
    """拼出 ``(system_text, user_text)``。``codebook`` = ``[(维度名, [编码…])]``。

    system 里把框架、规则与「只返回 JSON」的硬约束讲清楚；user 给时刻、转写与图片说明。
    ``code_notes`` 是码本释义（维度→码→desc/example），在场时逐码附注并多一条判据规则。
    """
    notes = code_notes or {}
    framework_lines = []
    for name, codes in codebook or []:
        if codes:
            vocab = "、".join(format_code_tokens(codes, notes.get(str(name))))
        else:
            vocab = "（无编码）"
        framework_lines.append("- %s：%s" % (name, vocab))
    framework_block = "\n".join(framework_lines) or "- （尚无维度框架）"
    example = "{" + ", ".join('"%s": "%s"' % (name, (codes[0] if codes else ""))
                              for name, codes in (codebook or [])[:3]) + "}"
    annotated = any(notes.get(str(name)) for name, _codes in (codebook or []))
    note_rule = ("4. 编码后括号内是该码的释义与示例（如「释：…；例：…」），"
                 "请据语义判断归属，优先选释义最贴合当前画面/话语的码。\n" if annotated else "")
    system_text = (
        "你是协作学习行为编码助手。请严格依据下面的编码框架，为指定时刻的画面与话语，"
        "逐维度选出最贴切的一个编码。\n\n"
        "硬性规则：\n"
        "1. 每个维度只能从它自己的词表里选一个编码，绝不杜撰词表之外的码，也不要把维度名当编码。\n"
        "2. 画面/话语不足以判断某维度时，把该维度的值留空（\"\"），不要猜。\n"
        "3. 只输出一个 JSON 对象：键为维度名（与框架完全一致），值为编码字符串（不带括号里的"
        "释义与示例）。不要输出任何解释、前后缀或多余文字。\n"
        "%s"
        "   示例：%s\n\n"
        "编码框架（维度 → 可选编码）：\n%s"
    ) % (note_rule, example or "{}", framework_block)
    if annotated:
        system_text += ("\n\n词表说明：编码后的括号内是该码的释义与示例，"
                        "返回时只写编码本身。")
    if str(extra_rules or "").strip():
        system_text += "\n\n补充规则：\n%s" % str(extra_rules).strip()
    when = "t=%g 秒" % float(time_seconds) if time_seconds is not None else "当前时刻"
    transcript_text = str(transcript or "").strip() or "（无转写）"
    user_text = (
        "请为 %s 这一刻编码。\n"
        "话语转写：%s\n"
        "随附的关键帧是这一刻前后的画面。请按系统要求，只返回那个 JSON 对象。"
    ) % (when, transcript_text)
    return system_text, user_text


def coding_messages(codebook, data_uris=None, transcript="", time_seconds=None,
                    extra_rules="", code_notes=None):
    """把 prompt + 关键帧拼成发给 LLM 的 messages（system + 多模态 user）。"""
    system_text, user_text = build_prompt(codebook, transcript, time_seconds,
                                          extra_rules, code_notes)
    messages = [system_message(system_text)]
    images = [uri for uri in (data_uris or []) if uri]
    messages.append(user_multimodal(user_text, images))
    return messages


# ------------------------------------------------------------------ 结果解析

def _extract_json_object(text):
    """从模型回复里尽力抠出一个 JSON 对象：整段 → ```json 围栏 → 首个配平的 {...}。"""
    raw = str(text or "").strip()
    if not raw:
        return None
    candidates = [raw]
    fence = _FENCE_RE.search(raw)
    if fence:
        candidates.append(fence.group(1).strip())
    start = raw.find("{")
    if start >= 0:
        depth = 0
        for index in range(start, len(raw)):
            char = raw[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start:index + 1])
                    break
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _unwrap(obj):
    """模型把结果包了一层（{"codes": {...}}）时拆包；否则原样返回。"""
    if not isinstance(obj, dict):
        return {}
    for key in _WRAP_KEYS:
        inner = obj.get(key)
        if isinstance(inner, dict):
            return inner
    return obj


def _code_from_value(value):
    """维度值可能是字符串，也可能是 {"code": "...", "reason": "..."}；统一取出编码字符串。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in _CODE_KEYS:
            if key in value and isinstance(value[key], str):
                return value[key].strip()
    return ""


def _find_key(obj, name):
    """在 obj 里按维度名找键：先精确，再去空白/大小写归一。"""
    if name in obj:
        return name
    target = str(name).strip().lower()
    for key in obj:
        if str(key).strip().lower() == target:
            return key
    return None


def parse_suggestions(text, codebook):
    """解析模型回复为逐维度建议码，并按框架词表校验。

    返回 ``{"ok", "error", "raw", "dims"}``，``dims`` = ``[{"dimension", "dim_index",
    "code", "in_vocab", "vocab", "reason"}]``（按框架顺序，模型没答的维度 code=""）。
    ``in_vocab`` 为假表示越界（杜撰 / 拼错），一律不可写回，只作标注。
    """
    obj = _extract_json_object(text)
    if obj is None:
        return {"ok": False, "error": "未能从回复中解析出 JSON 对象（模型没按要求返回）",
                "raw": str(text or ""), "dims": []}
    obj = _unwrap(obj)
    dims = []
    for index, (name, codes) in enumerate(codebook or []):
        vocab = [str(code).strip() for code in (codes or [])]
        key = _find_key(obj, name)
        code = _code_from_value(obj.get(key)) if key is not None else ""
        reason = ""
        if key is not None and isinstance(obj.get(key), dict):
            reason = str(obj[key].get("reason") or obj[key].get("理由") or "").strip()
        in_vocab = bool(code) and any(code == item for item in vocab)
        dims.append({"dimension": str(name), "dim_index": index, "code": code,
                     "in_vocab": in_vocab, "vocab": vocab, "reason": reason})
    return {"ok": True, "error": "", "raw": str(text or ""), "dims": dims}


# ------------------------------------------------------------------ 写回编码页

def find_row_at_time(encoder, seconds, tolerance=0.5):
    """编码页里时间 ≈ ``seconds`` 的行号；找不到返回 None。"""
    try:
        times = encoder.computed_row_times()
    except Exception:                            # noqa: BLE001 编码器状态异常时当作没这行
        return None
    target = float(seconds)
    for index, value in enumerate(times):
        try:
            if abs(float(value) - target) <= tolerance:
                return index
        except (TypeError, ValueError):
            continue
    return None


def apply_suggestions(encoder, seconds, dim_codes, marker=SUGGEST_MARKER):
    """把某时刻、若干维度的建议码写入编码页：可撤销、标注待确认、**绝不覆盖已有人工码**。

    ``dim_codes`` = ``[(dim_index, code)]``（调用方应只传 ``in_vocab`` 的码）。该时刻没有行则
    新建一行（注释即 marker）；已有行则**只填空的维度单元格**，非空的一律跳过（skipped）；若
    目标维度全被人工码占满，则**不压栈、不打标记、不发脏**直接返回（避免把没写入的行误标为
    「待确认」）。真有空位可写才 ``_push_history()``，故可撤销并发脏标记。返回
    ``{"written", "skipped", "row_index", "note"}``。
    """
    cards = getattr(encoder, "dimension_cards", [])
    n_dims = len(cards)
    if n_dims == 0:
        return {"written": 0, "skipped": 0, "row_index": None,
                "note": "编码页还没有维度框架，无法写回"}
    valid = [(int(i), str(c)) for i, c in (dim_codes or [])
             if 0 <= int(i) < n_dims and str(c).strip()]
    if not valid:
        return {"written": 0, "skipped": 0, "row_index": None,
                "note": "没有可写回的建议码（可能都越界或为空）"}
    row_index = find_row_at_time(encoder, seconds)
    written = skipped = 0
    if row_index is None:
        encoder._push_history()                  # 新建行必写：可撤销 + 脏标记
        values = [""] * n_dims + [marker]
        for dim_index, code in valid:
            values[dim_index] = code
            written += 1
        if not encoder._record_at_time(values, float(seconds)):
            encoder.history.pop()                # 被拒（如早于起始时间）：回退刚压的栈，别留空快照
            return {"written": 0, "skipped": 0, "row_index": None,
                    "note": "该时间点无法写入（可能早于起始时间）"}
        encoder.update_table()
        return {"written": written, "skipped": 0,
                "row_index": find_row_at_time(encoder, seconds), "note": ""}
    row = encoder.selected_data[row_index]
    while len(row) < n_dims + 1:
        row.append("")
    writable = [(i, c) for i, c in valid if not str(row[i] or "").strip()]
    skipped = len(valid) - len(writable)         # 已有人工码的维度：绝不覆盖
    if not writable:                             # 全被人工码占满：不压栈、不打标记、不发脏，避免误导
        return {"written": 0, "skipped": skipped, "row_index": row_index,
                "note": "该时刻这些维度已有人工编码，未覆盖（建议仅供参考）"}
    encoder._push_history()                      # 确有可写的空位才压栈：可撤销 + 脏标记
    for dim_index, code in writable:
        row[dim_index] = code
        written += 1
    comment = str(row[-1] or "")
    if marker not in comment:
        row[-1] = ("%s %s" % (comment, marker)).strip() if comment else marker
    encoder.update_table()
    return {"written": written, "skipped": skipped, "row_index": row_index, "note": ""}


def run_auto_code(client, codebook, data_uris=None, transcript="", time_seconds=None,
                  cancel=None, extra_rules="", code_notes=None):
    """同步跑一次自动编码（供 worker / 测试直接调）：发多模态请求 → 解析 → 校验。

    返回 ``parse_suggestions`` 的结果再并上 ``usage``。异常向上抛 ``AiError`` 由调用方处理。
    """
    messages = coding_messages(codebook, data_uris, transcript, time_seconds,
                               extra_rules, code_notes)
    result = client.chat(messages, cancel=cancel, stream=False)
    parsed = parse_suggestions(result.get("text", ""), codebook)
    parsed["usage"] = result.get("usage") or {}
    parsed["cancelled"] = bool(result.get("cancelled"))
    return parsed


# ------------------------------------------------------------------ 人在环复核 UI

class AutoCodingDialog(QDialog):
    """「AI 辅助编码」对话框：选时刻 → 抽帧 + 转写 → 多模态请求 → 复核表 → 应用选中到编码页。

    复核表逐维度列出建议码，越界的标红且不可勾选；只有勾选的、词表内的码才写回，且写回时
    只填空单元格、注释打上「AI 建议、待确认」。全程可撤销。
    """

    COL_APPLY, COL_TIME, COL_DIM, COL_CODE, COL_IN_VOCAB = 0, 1, 2, 3, 4

    def __init__(self, encoder, parent=None, transport=None, settings=None):
        super().__init__(parent or encoder)
        self.encoder = encoder
        self._transport = transport
        self._settings = settings or ai_settings.settings
        self._worker = None
        self._queue = []
        self._review = []                        # 与复核表行一一对应
        self.setWindowTitle("AI 辅助编码（Phase 2 · 多模态建议）")
        self.resize(720, 620)
        self._build_ui()
        self._refresh_backend_status()
        self._refresh_time_points()
        self._refresh_codebook_status()

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        intro = QLabel("抽取关键帧 + 你的话语转写，交给多模态模型按编码框架给「建议码」；"
                       "建议需你逐条勾选确认后才写回，绝不自动覆盖已有编码。")
        intro.setProperty("hint", True)
        intro.setWordWrap(True)
        root.addWidget(intro)

        self.video_label = QLabel()
        self.video_label.setProperty("section", True)
        self.video_label.setWordWrap(True)
        root.addWidget(self.video_label)

        self.backend_notice = QLabel()
        self.backend_notice.setProperty("hint", True)
        self.backend_notice.setWordWrap(True)
        root.addWidget(self.backend_notice)

        cb_row = QHBoxLayout()
        cb_row.addWidget(self._section_label("码本释义（可选：每个编码的描述/释义/示例）"))
        cb_row.addStretch(1)
        self.import_codebook_btn = QPushButton("导入码本 Excel")
        self.import_codebook_btn.setProperty("btn", "ghost")
        theme.bind_icon(self.import_codebook_btn, "import", 16)
        self.import_codebook_btn.clicked.connect(self._on_import_codebook)
        self.clear_codebook_btn = QPushButton("清除")
        self.clear_codebook_btn.setProperty("btn", "ghost")
        self.clear_codebook_btn.clicked.connect(self._on_clear_codebook)
        cb_row.addWidget(self.import_codebook_btn)
        cb_row.addWidget(self.clear_codebook_btn)
        root.addLayout(cb_row)
        self.codebook_label = QLabel()
        self.codebook_label.setProperty("hint", True)
        self.codebook_label.setWordWrap(True)
        root.addWidget(self.codebook_label)

        pick_row = QHBoxLayout()
        pick_row.addWidget(self._section_label("选择要编码的时刻（可多选；默认当前时间点）"))
        pick_row.addStretch(1)
        self.refresh_times_btn = QPushButton("刷新时刻")
        self.refresh_times_btn.setProperty("btn", "ghost")
        self.refresh_times_btn.clicked.connect(self._refresh_time_points)
        pick_row.addWidget(self.refresh_times_btn)
        root.addLayout(pick_row)

        self.time_list = QListWidget()
        self.time_list.setMaximumHeight(120)
        root.addWidget(self.time_list)

        root.addWidget(self._section_label("话语转写（可留空；此阶段由你粘贴，或改用能直接听音频的模型）"))
        self.transcript_edit = QTextEdit()
        self.transcript_edit.setFixedHeight(64)
        self.transcript_edit.setPlaceholderText("把这一刻学生说的话粘到这里……")
        root.addWidget(self.transcript_edit)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("生成建议码")
        self.run_btn.setProperty("btn", "primary")
        theme.bind_icon(self.run_btn, "wand", 16)
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setProperty("btn", "ghost")
        theme.bind_icon(self.stop_btn, "stop", 16)
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.stop_btn)
        run_row.addStretch(1)
        root.addLayout(run_row)

        root.addWidget(self._section_label("复核（只有勾选且词表内的码才会写回）"))
        self.review_table = QTableWidget(0, 5)
        self.review_table.setHorizontalHeaderLabels(["采用", "时间", "维度", "建议码", "词表内"])
        self.review_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.review_table.verticalHeader().setVisible(False)
        header = self.review_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(self.COL_APPLY, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_IN_VOCAB, QHeaderView.ResizeToContents)
        root.addWidget(self.review_table, 1)

        apply_row = QHBoxLayout()
        self.apply_btn = QPushButton("应用选中到编码页")
        self.apply_btn.setProperty("btn", "primary")
        theme.bind_icon(self.apply_btn, "robot", 16)
        self.apply_btn.clicked.connect(self._on_apply)
        self.clear_btn = QPushButton("清空复核")
        self.clear_btn.setProperty("btn", "ghost")
        self.clear_btn.clicked.connect(self._clear_review)
        apply_row.addWidget(self.apply_btn)
        apply_row.addWidget(self.clear_btn)
        apply_row.addStretch(1)
        root.addLayout(apply_row)

        self.status_label = QLabel("就绪")
        self.status_label.setProperty("hint", True)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        self.close_btn = QPushButton("关闭")
        self.close_btn.setProperty("btn", "secondary")
        self.close_btn.clicked.connect(self.accept)
        close_row.addWidget(self.close_btn)
        root.addLayout(close_row)

    def _section_label(self, text):
        label = QLabel(text)
        label.setProperty("section", True)
        return label

    # ---------- 数据面 ----------
    def _codebook(self):
        try:
            return self.encoder.dim_panel.framework()
        except Exception:                        # noqa: BLE001
            return []

    def _video_path(self):
        player = getattr(self.encoder, "video_player", None)
        return str(getattr(player, "current_file_path", "") or "")

    def _client(self):
        provider = self._settings.provider()
        return OpenAICompatClient(
            base_url=provider.get("base_url"), api_key=provider.get("api_key"),
            model=provider.get("model"), temperature=provider.get("temperature", 0.2),
            max_tokens=provider.get("max_tokens", 512), transport=self._transport)

    # ---------- 码本释义 ----------
    def _notes(self):
        """从设置取已导入的码本释义；取不到（老版本/无该方法）就当没导入。"""
        try:
            return self._settings.codebook_notes() or {}
        except Exception:                        # noqa: BLE001
            return {}

    def _refresh_codebook_status(self):
        notes = self._notes()
        if not notes:
            self.codebook_label.setText("未导入码本释义：AI 只能凭码名猜含义；"
                                        "导入带释义的码本可明显提高建议码准确率。")
            self.clear_codebook_btn.setEnabled(False)
            return
        dims, codes = ai_settings.count_notes(notes)
        self.codebook_label.setText("已载入码本释义：%d 个维度、%d 个编码带释义/示例。"
                                    % (dims, codes))
        self.clear_codebook_btn.setEnabled(True)

    def _on_import_codebook(self, file_path=None):
        """导入码本释义 Excel。``file_path`` 缺省时弹文件对话框；传入则直接读（供脚本/重开用）。"""
        path = file_path
        if not path:
            path = user_state.open_file_name(
                self, user_state.BUCKET_FRAMEWORK, "选择码本释义 Excel",
                "Excel Files (*.xlsx *.xls);;All Files (*)")
        if not path:
            return False
        try:
            notes = read_codebook_excel(path)
        except Exception as exc:                 # noqa: BLE001 读不出列/空文件 → 人话说明
            self.codebook_label.setText("导入失败：%s" % exc)
            return False
        try:
            self._settings.set_codebook_notes(notes)
        except Exception as exc:                 # noqa: BLE001 存不下也只记状态，不改旧值
            self.codebook_label.setText("码本已解析但保存失败：%s" % exc)
            return False
        self._refresh_codebook_status()
        return True

    def _on_clear_codebook(self):
        try:
            self._settings.set_codebook_notes({})
        except Exception:                        # noqa: BLE001
            pass
        self._refresh_codebook_status()

    def _refresh_backend_status(self):
        path = self._video_path()
        if path:
            self.video_label.setText("视频：%s" % path)
        else:
            self.video_label.setText("视频：未加载（可仅凭转写文本编码，或在编码页先加载视频）")
        ok, backends = video_probe.frames_available()
        if ok:
            have = "、".join(name for name, present in backends.items() if present)
            self.backend_notice.setText("抽帧后端：%s（就绪）" % have)
        else:
            self.backend_notice.setText("抽帧后端：不可用 —— " + video_probe.install_hint())

    def _refresh_time_points(self):
        self.time_list.clear()
        seconds = []
        try:
            seconds = [float(t) for t in self.encoder.computed_row_times()]
        except Exception:                        # noqa: BLE001
            seconds = []
        current = None
        try:
            current = self.encoder.time_ctrl.parse_timepoint()
        except Exception:                        # noqa: BLE001
            current = None
        candidates = list(dict.fromkeys(([current] if current is not None else []) + seconds))
        if not candidates:
            candidates = [0.0]
        for value in candidates:
            item = QListWidgetItem("t = %g 秒" % value)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if value == current else Qt.Unchecked)
            item.setData(Qt.UserRole, float(value))
            self.time_list.addItem(item)
        if current is None:
            self.time_list.item(0).setCheckState(Qt.Checked)

    def _checked_times(self):
        out = []
        for index in range(self.time_list.count()):
            item = self.time_list.item(index)
            if item.checkState() == Qt.Checked:
                out.append(float(item.data(Qt.UserRole)))
        return out

    # ---------- 运行 ----------
    def _on_run(self):
        codebook = self._codebook()
        if not codebook:
            self.status_label.setText("编码页还没有维度框架，先导入/添加维度再来。")
            return
        client = self._client()
        ok, note = client.ready()
        if not ok:
            self.status_label.setText("AI 未配置：%s（到「AI 助手」页填 base_url / key / model）" % note)
            return
        times = self._checked_times()
        if not times:
            self.status_label.setText("请先在时刻列表里勾选至少一个时间点。")
            return
        self._queue = list(times)
        self._set_busy(True)
        self.status_label.setText("排队 %d 个时刻，逐个请求……" % len(times))
        self._drain()

    def _drain(self):
        """处理队列里的下一个时刻；队列空则收工。"""
        if not self._queue:
            self._set_busy(False)
            self.status_label.setText("完成：复核表里勾选需要的建议码后点「应用选中到编码页」。")
            return
        seconds = self._queue.pop(0)
        transcript = self.transcript_edit.toPlainText()
        path = self._video_path()
        uris = []
        note = ""
        if path:
            uri, note = video_probe.extract_frame_uri(path, seconds)
            if uri:
                uris.append(uri)
        if not uris and note:
            self.status_label.setText("t=%g 秒抽帧不可用（%s），改用纯转写请求。" % (seconds, note))
        codebook = self._codebook()
        messages = coding_messages(codebook, uris, transcript, seconds,
                                   code_notes=self._notes())
        self._pending_seconds = seconds
        self._worker = AiWorker(self._client(), messages, parent=self, stream=False)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result):
        seconds = getattr(self, "_pending_seconds", None)
        parsed = parse_suggestions(result.get("text", ""), self._codebook())
        if not parsed["ok"]:
            self.status_label.setText("t=%s 秒：%s" % (seconds, parsed["error"]))
        else:
            got = sum(1 for d in parsed["dims"] if d["in_vocab"])
            bad = sum(1 for d in parsed["dims"] if d["code"] and not d["in_vocab"])
            self._append_review(seconds, parsed["dims"])
            self.status_label.setText("t=%s 秒：词表内 %d 条、越界 %d 条已标注。" % (seconds, got, bad))
        self._show_usage(result.get("usage"))
        self._cleanup_worker()
        self._drain()

    def _on_error(self, message):
        self.status_label.setText("请求失败：%s" % message)
        self._cleanup_worker()
        self._queue = []                         # 出错即停队列，不把错误刷一屏
        self._set_busy(False)

    def _cleanup_worker(self):
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def _on_stop(self):
        self._queue = []
        if self._worker is not None:
            self._worker.cancel()
        self._set_busy(False)
        self.status_label.setText("已停止。")

    def _set_busy(self, busy):
        self.run_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)

    def _show_usage(self, usage):
        if not isinstance(usage, dict):
            return
        total = usage.get("total_tokens")
        if total is None:
            return
        tag = "（估算）" if usage.get("estimated") else ""
        self.status_label.setText("%s 本次用量%s：约 %s tokens（BYO key 自付）。"
                                  % (self.status_label.text(), tag, total))

    # ---------- 复核表 ----------
    def _clear_review(self):
        self.review_table.setRowCount(0)
        self._review = []

    def _append_review(self, seconds, dims):
        for dim in dims:
            if not dim["code"]:
                continue                         # 模型留空的维度不进复核表
            row = self.review_table.rowCount()
            self.review_table.insertRow(row)
            in_vocab = bool(dim["in_vocab"])
            apply_item = QTableWidgetItem()
            apply_item.setFlags(apply_item.flags() | Qt.ItemIsUserCheckable)
            apply_item.setCheckState(Qt.Checked if in_vocab else Qt.Unchecked)
            if not in_vocab:
                apply_item.setFlags(apply_item.flags() & ~Qt.ItemIsUserCheckable)
            self.review_table.setItem(row, self.COL_APPLY, apply_item)
            self.review_table.setItem(row, self.COL_TIME, QTableWidgetItem("%g 秒" % seconds))
            self.review_table.setItem(row, self.COL_DIM, QTableWidgetItem(dim["dimension"]))
            code_item = QTableWidgetItem(dim["code"])
            self.review_table.setItem(row, self.COL_CODE, code_item)
            flag = QTableWidgetItem("是" if in_vocab else "否·越界")
            self.review_table.setItem(row, self.COL_IN_VOCAB, flag)
            if not in_vocab:
                danger = QColor(theme.tokens().get("danger", "#E5484D"))
                for col in range(5):
                    cell = self.review_table.item(row, col)
                    if cell is not None:
                        cell.setForeground(danger)
                tip = "词表：%s" % "、".join(dim["vocab"])
                if dim.get("reason"):
                    tip += "\n模型理由：%s" % dim["reason"]
                code_item.setToolTip(tip)
            self._review.append({"seconds": seconds, "dim_index": dim["dim_index"],
                                 "code": dim["code"], "in_vocab": in_vocab})

    def _checked_review(self):
        """复核表里勾选的、词表内的行 → 按时刻分组 ``{seconds: [(dim_index, code)]}``。"""
        grouped = {}
        for row, entry in enumerate(self._review):
            item = self.review_table.item(row, self.COL_APPLY)
            if item is None or item.checkState() != Qt.Checked:
                continue
            if not entry["in_vocab"]:
                continue
            grouped.setdefault(entry["seconds"], []).append((entry["dim_index"], entry["code"]))
        return grouped

    def _on_apply(self):
        grouped = self._checked_review()
        if not grouped:
            self.status_label.setText("没有勾选任何词表内的建议码。")
            return
        total_written = total_skipped = 0
        for seconds, dim_codes in grouped.items():
            outcome = apply_suggestions(self.encoder, seconds, dim_codes)
            total_written += outcome["written"]
            total_skipped += outcome["skipped"]
            if outcome["note"]:
                self.status_label.setText("t=%g 秒：%s" % (seconds, outcome["note"]))
        self.status_label.setText("已写回 %d 个编码（跳过已有人工码 %d 个），注释标注「%s」，可撤销。"
                                  % (total_written, total_skipped, SUGGEST_MARKER))

    # ---------- 关闭时收尾 ----------
    def reject(self):
        self._on_stop()
        super().reject()

    def accept(self):
        self._on_stop()
        super().accept()
