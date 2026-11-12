# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""「小组交互」编码页：把 InteractionEvent 的 13 个字段变成可录的表单。

这一页与个体编码页共享同一套录入语义（三个时间胶囊、时间轴骨架、记录/撤销/删行），
但左栏不是维度编辑面板而是字段选择器，注释框换成话语框：

- 维度面板 → 字段选择器：actor / target_type / target_id / interaction_type / reply_to
  / turn_id / duration / confidence 八个字段，候选全部来自主体字典与本组对象字典，
  不给自由文本手输（手输会造出 ` stu1 ` 与 `stu1` 两个主体）。
- 注释框 → 话语框：记该时刻的学生对话原文，落到 `utterance` 列，与 `notes` 并存。
- 同一时刻允许多条事件（V09）：记录走 ``overwrite=False``，绝不覆盖已有行。
- 阻断类违例（V01–V04、V06、V07、V10）当场拒绝写入并在状态栏说明是哪条规则；V05/V08
  放行但计入数据质量报告，删不删由研究者判断。

页面对象只认注入进来的 provider（成员、框架、汇总表），不伸手进主窗口的内部结构。
"""

import copy

from pandas import DataFrame
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (QAbstractItemDelegate, QAbstractItemView, QComboBox,
                             QDialog, QDoubleSpinBox, QFormLayout,
                             QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QListWidget, QMainWindow, QMessageBox,
                             QMenu, QPushButton, QScrollArea, QSplitter,
                             QStyledItemDelegate, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

import theme
import interaction_model as im
import user_state
from analysis_core import sanitize_group_id
from analysis_params import MISSING_TEXT
from timeline_axis import (HISTORY_LIMIT, POPUP_MIN_WIDTH, TimeAxis, TimeControls,
                           apply_column_widths, format_seconds)

#: 表格里的只读列：group_id 来自 Group_Info 卡，event_id 由发号器定，timestamp 由时间轴推算
READONLY_FIELDS = ("group_id", "event_id", "timestamp")
#: 表头文案：契约列名不动，表头给中文提示（导出仍用英文列名）
HEADERS = {
    "group_id": "组号", "event_id": "事件ID", "timestamp": "时间",
    "actor_id": "主体", "target_id": "对象ID", "target_type": "对象类型",
    "interaction_type": "交互类型", "reply_to": "应答", "turn_id": "轮次",
    "duration": "时长", "source": "来源", "confidence": "置信度",
    "notes": "备注", "utterance": "话语",
}


class EventCellDelegate(QStyledItemDelegate):
    """表格内编辑：能枚举的字段一律走下拉，自由文本只留给 turn_id / notes / 话语。"""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.owner = owner

    def createEditor(self, parent, option, index):
        field = self.owner.column_field(index.column())
        if field in im.EVENT_COLUMNS and field not in READONLY_FIELDS \
                and field in ("actor_id", "target_type", "target_id",
                              "interaction_type", "reply_to"):
            combo = QComboBox(parent)
            combo.setProperty("cellEditor", True)
            # 词表可编辑：interaction_type / target_id 允许就地敲一个新值
            combo.setEditable(field in ("interaction_type", "target_id", "reply_to"))
            combo.setInsertPolicy(QComboBox.NoInsert)
            view = QListWidget()
            view.setObjectName("cellPopup")
            view.setMinimumWidth(POPUP_MIN_WIDTH)
            combo.setView(view)
            theme.bind_combo_popup(combo)
            combo.addItems(self.owner.choices_for(field, index.row()))
            combo.setCurrentText(str(index.data(Qt.DisplayRole) or ""))
            combo.setMinimumContentsLength(8)
            if combo.lineEdit() is not None:
                combo.lineEdit().selectAll()
            combo.activated.connect(lambda _=None, c=combo: self._commit_when_closed(c))
            return combo
        if field == "confidence":
            spin = QDoubleSpinBox(parent)
            spin.setDecimals(2)
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.05)
            try:
                spin.setValue(float(index.data(Qt.DisplayRole) or 1.0))
            except (TypeError, ValueError):
                spin.setValue(1.0)
            return spin
        editor = QLineEdit(str(index.data(Qt.DisplayRole) or ""), parent)
        if field == "duration":
            editor.setPlaceholderText("秒 或 00:00:10")
        return editor

    def _commit_when_closed(self, editor):
        """Qt 在展开/收起弹窗瞬间也会发 activated，只在弹窗已关（用户真选完）时提交"""
        view = editor.view()
        if view is not None and view.isVisible():
            return
        self.commitData.emit(editor)
        self.closeEditor.emit(editor, QAbstractItemDelegate.SubmitModelCache)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            value = editor.currentText().strip()
        elif isinstance(editor, QDoubleSpinBox):
            value = editor.value()
        elif isinstance(editor, QLineEdit):
            value = editor.text().strip()
        else:
            super().setModelData(editor, model, index)
            return
        model.setData(index, value, Qt.EditRole)


class QualityDialog(QDialog):
    """数据质量报告：把 V01–V10 的命中逐条列出，阻断与警告分色。"""

    def __init__(self, parent, title, items):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(620, 380)
        lay = QVBoxLayout(self)
        note = QLabel("共 %d 条；【阻断】会在录入时就被拒绝，【警告】已放行、由研究者判断是否修正"
                      % len(items))
        note.setProperty("hint", True)
        note.setWordWrap(True)
        lay.addWidget(note)
        box = QListWidget()
        for item in items:
            level = getattr(item, "level", "info")
            prefix = {"error": "【阻断】", "warn": "【警告】", "info": "【提示】"}.get(level, "【提示】")
            entry = QListWidgetItem("%s %s" % (prefix, getattr(item, "message", str(item))))
            entry.setProperty("qaLevel", level)
            box.addItem(entry)
        if not items:
            box.addItem("未发现任何违例")
        lay.addWidget(box)
        close = QPushButton("关闭")
        close.setProperty("btn", "secondary")
        close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        lay.addLayout(row)


class GroupInteractionEncoder(QWidget):
    """小组交互编码页（内层标签页「小组交互」）。"""

    #: 现场改动通知：事件行、组号/组名、主体与本组对象字典、交互词表任一变了都发。
    #: 主窗口据这一个信号触发自动保存，不必逐控件去接（见 MainWindow._bind_content_signals）。
    content_changed = Signal()

    def __init__(self, members_provider=None, framework_provider=None,
                 summary_provider=None, parent=None):
        super().__init__(parent)
        # 数据来源全部可注入：测试传假对象，主窗口传真页面
        self.members_provider = members_provider or (lambda: [])
        self.framework_provider = framework_provider or (lambda: [])
        self.summary_provider = summary_provider or (lambda: None)
        self.group_id = "group_1"
        self.group_name = "group_1"
        self.members = []
        self.actors = []
        self.object_dict = {}
        self.interaction_types = list(im.INTERACTION_TYPES)
        #: 观察窗口上界（秒）：默认不限——随录随延；接入视频时长或 Group_Info 时再定
        self.observation_end = None
        self.event_rows = []          # list[dict]：一行一条事件
        self.row_times = []           # 平行表：None=顺序行（时间由格位推算）
        self.row_positions = []       # 平行表：None=固定时间点行
        self.history = []
        self._loading = False
        self._warned = []             # 放行的 V05/V08 警告，进数据质量报告
        self.time_ctrl = TimeControls()
        # 骨架算法与编码页同一份，只是行容器换成本页的事件行
        self.axis = TimeAxis(self, rows_attr="event_rows")
        self.initUI()
        self.refresh_members()
        self.update_table()

    # ---------------------------------------------------------------- 时间轴接口
    def get_start_seconds(self):
        return self.time_ctrl.start_seconds()

    def interval_seconds(self):
        return self.time_ctrl.interval()

    # ---------------------------------------------------------------- UI
    def initUI(self):
        layout = QHBoxLayout(self)
        left = QVBoxLayout()

        # ---- Group_Info 卡 ----
        info = QWidget()
        info.setProperty("card", True)
        grid = QGridLayout(info)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        self.group_id_edit = QLineEdit(self.group_id)
        self.group_id_edit.setPlaceholderText("group_1")
        self.group_name_edit = QLineEdit(self.group_name)
        self.members_label = QLabel("—")
        self.members_label.setProperty("hint", True)
        self.members_label.setWordWrap(True)
        self.span_label = QLabel("—")
        self.span_label.setProperty("hint", True)
        self.actor_edit = QLineEdit()
        self.actor_edit.setPlaceholderText("教师 / 系统等其他主体，逗号分隔")
        self.actor_edit.editingFinished.connect(self._on_actors_edited)
        self.group_id_edit.editingFinished.connect(self._on_group_id_edited)
        self.group_name_edit.editingFinished.connect(self._on_group_name_edited)
        for row, (caption, widget) in enumerate([
                ("组号", self.group_id_edit), ("组名", self.group_name_edit),
                ("成员", self.members_label), ("观察区间", self.span_label),
                ("其他主体", self.actor_edit)]):
            label = QLabel(caption)
            label.setProperty("hint", True)
            grid.addWidget(label, row, 0)
            grid.addWidget(widget, row, 1)
        grid.setColumnStretch(1, 1)

        # ---- 字段选择器（取代维度编辑面板） ----
        fields = QWidget()
        fields.setProperty("card", True)
        form = QFormLayout(fields)
        form.setContentsMargins(14, 12, 14, 12)
        form.setSpacing(6)
        self.actor_combo = self._combo("主体 actor_id")
        self.target_type_combo = self._combo("对象类型 target_type")
        self.target_type_combo.addItems(im.TARGET_TYPE_CHOICES)
        self.target_type_combo.currentTextChanged.connect(self._on_target_type_changed)
        self.target_id_combo = self._combo("对象ID target_id", editable=True)
        self.interaction_combo = self._combo("交互类型 interaction_type", editable=True)
        self.interaction_combo.addItems(self.interaction_types)
        self.reply_combo = self._combo("应答 reply_to", editable=True)
        self.turn_edit = QLineEdit()
        self.turn_edit.setPlaceholderText("T1")
        self.duration_edit = QLineEdit()
        self.duration_edit.setPlaceholderText("留空=不填；可写 12 或 00:00:12")
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(1.0)
        add_object = QPushButton("新增对象")
        add_object.setProperty("btn", "ghost")
        add_object.clicked.connect(self._add_object)
        target_row = QWidget()
        target_lay = QHBoxLayout(target_row)
        target_lay.setContentsMargins(0, 0, 0, 0)
        target_lay.addWidget(self.target_id_combo, 1)
        target_lay.addWidget(add_object)
        edit_type = QPushButton("编辑词表")
        edit_type.setProperty("btn", "ghost")
        edit_type.clicked.connect(self._edit_interaction_types)
        interaction_row = QWidget()
        interaction_lay = QHBoxLayout(interaction_row)
        interaction_lay.setContentsMargins(0, 0, 0, 0)
        interaction_lay.addWidget(self.interaction_combo, 1)
        interaction_lay.addWidget(edit_type)
        for caption, widget in [
                ("主体 actor_id", self.actor_combo),
                ("对象类型 target_type", self.target_type_combo),
                ("对象ID target_id", target_row),
                ("交互类型 interaction_type", interaction_row),
                ("应答 reply_to", self.reply_combo),
                ("轮次 turn_id", self.turn_edit),
                ("时长 duration", self.duration_edit),
                ("置信度 confidence", self.confidence_spin)]:
            form.addRow(caption, widget)

        # ---- 话语框（取代注释框）+ 备注 ----
        self.utterance_edit = QLineEdit()
        self.utterance_edit.setPlaceholderText("记录该时刻的学生对话原文…")
        self.utterance_edit.setFixedHeight(56)
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("备注 notes（可选）")

        # ---- 按钮区 ----
        buttons = QWidget()
        box = QHBoxLayout(buttons)
        box.setContentsMargins(0, 0, 0, 0)
        self.record_btn = QPushButton("记录")
        self.undo_btn = QPushButton("撤销")
        self.clear_btn = QPushButton("清空")
        self.output_btn = QPushButton("输出")
        self.report_btn = QPushButton("数据质量报告")
        for btn in (self.record_btn, self.undo_btn, self.clear_btn,
                    self.output_btn, self.report_btn):
            btn.setProperty("btn", "secondary")
            box.addWidget(btn, 1)
        theme.bind_icon(self.output_btn, "export", 16)
        self.record_btn.clicked.connect(self.record_event)
        self.undo_btn.clicked.connect(self.undo)
        self.clear_btn.clicked.connect(self.clear_events)
        self.output_btn.clicked.connect(self.output)
        self.report_btn.clicked.connect(self.show_quality_report)

        self.time_ctrl.params_changed.connect(self.update_span_label)
        self.time_ctrl.timepoint_edited.connect(self._refresh_reply_choices)

        middle = QWidget()
        middle_lay = QVBoxLayout(middle)
        middle_lay.setContentsMargins(0, 0, 0, 0)
        middle_lay.addWidget(self.time_ctrl)
        middle_lay.addWidget(fields)
        middle_lay.addWidget(self.utterance_edit)
        middle_lay.addWidget(self.note_edit)
        middle_lay.addWidget(buttons)
        middle_lay.addStretch(1)
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setFrameShape(QScrollArea.NoFrame)
        scroller.setWidget(middle)
        left.addWidget(info)
        left.addWidget(scroller, 1)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(430)

        # ---- 右侧事件表 ----
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.DoubleClicked
                                   | QAbstractItemView.EditKeyPressed)
        self.table.setItemDelegate(EventCellDelegate(self, self.table))
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_row_menu)
        self.table.installEventFilter(self)
        self.table.viewport().setToolTip(
            "双击单元格修改事件字段；选中行后 Delete 或右键菜单删行；同一时刻可录多条事件")
        self.table.itemChanged.connect(self._on_item_changed)
        table_host = QWidget()
        table_host.setProperty("card", True)
        table_lay = QVBoxLayout(table_host)
        table_lay.setContentsMargins(14, 12, 14, 12)
        table_lay.addWidget(self.table)
        self.table_slot = QWidget()
        slot = QHBoxLayout(self.table_slot)
        slot.setContentsMargins(12, 11, 10, 10)
        slot.addWidget(table_host)
        theme.apply_card_shadow(self.table_slot)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(self.table_slot)
        self.splitter.setSizes([460, 620])
        layout.addWidget(self.splitter)
        self._rebuild_columns()

    def _combo(self, name, editable=False):
        combo = QComboBox()
        combo.setProperty("i18nData", True)
        theme.bind_combo_popup(combo)
        combo.setEditable(editable)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.setToolTip(name)
        return combo

    # ---------------------------------------------------------------- 成员与候选
    def refresh_members(self):
        """成员回显：编码个体页的标题 → S1/S2…，页多一页这里就长一个候选。"""
        listed = []
        for item in self.members_provider() or []:
            if isinstance(item, (tuple, list)):
                student = str(item[0])
                title = str(item[1]) if len(item) > 1 and item[1] else str(item[0])
            else:
                student = title = str(item)
            student = student.strip()
            if student and student not in [pair[0] for pair in listed]:
                listed.append((student, title))
        self.members = [student for student, _title in listed]
        self.members_label.setText(
            "、".join("%s（%s）" % pair for pair in listed) if listed else "尚无编码个体页")
        self._fill_choices()
        self.update_span_label()

    def context(self):
        """校验现场（快照）：写候选字典请用 ``register_object`` / ``register_actor``。

        窗口上界用 ``observation_end``（没定就是不限）而不是已录行数：第 N 条本身的
        时间也算窗口内，拿 N-1 行当边界会把合法的第 N 条踢成 V03。
        """
        return im.EventContext(
            group_id=self.group_id, members=self.members, actors=self.actors,
            object_dict=self.object_dict, interaction_types=self.interaction_types,
            t_min=self.get_start_seconds(),
            t_max=(None if self.observation_end is None else float(self.observation_end)),
            events=[dict(row) for row in self.event_rows],
        )

    def register_object(self, target_type, object_id):
        """就地新增对象：录了一个没登记过的任务号，不该为此去改模板。"""
        object_id = str(object_id or "").strip()
        if not object_id:
            return
        bucket = self.object_dict.setdefault(str(target_type), [])
        if object_id not in bucket:
            bucket.append(object_id)
            self.content_changed.emit()
        self._fill_choices()

    def _fill_choices(self):
        ctx = self.context()
        self._populate(self.actor_combo, ctx.actor_choices(), keep=True)
        self._populate(self.target_id_combo,
                       ctx.object_choices(self.target_type_combo.currentText()), keep=True)
        self._populate(self.interaction_combo, self.interaction_types, keep=True)
        self._refresh_reply_choices()

    def _refresh_reply_choices(self):
        """reply_to 只列时间更早的事件（V07）：跟随播放头时全部已录事件都更早。

        这一格是「可留空」字段，必须显式回收：可编辑下拉 addItems 会顺手选中第一项，
        不清回去就等于替用户做了一个他没做过的应答决定（下一条事件被静默挂到别处）。
        """
        moment = self.time_ctrl.parse_timepoint() if self.time_ctrl.edited else None
        ids = [str(row.get("event_id", "")) for row in self._rows_with_time()
               if moment is None or (row.get("timestamp") is not None
                                     and row["timestamp"] < moment)]
        self._populate(self.reply_combo, ids, keep=True)
        if self.reply_combo.currentText().strip() and self.reply_combo.currentText() not in ids:
            self.reply_combo.setCurrentText("")

    def _populate(self, combo, values, keep=False):
        """刷候选。可编辑下拉 addItems 会顺手选中第一项：原本留空就得继续留空。"""
        current = combo.currentText() if keep else ""
        combo.blockSignals(True)
        combo.clear()
        combo.addItems([str(value) for value in values if str(value).strip()])
        if current or combo.isEditable():
            combo.setCurrentText(current)
        combo.blockSignals(False)

    def _on_target_type_changed(self, target_type):
        """对象候选跟着 target_type 换：PERSON 给成员，对象类给本组对象字典。"""
        self._populate(self.target_id_combo, self.context().object_choices(target_type))
        self.target_id_combo.setEnabled(target_type != im.NO_TARGET)

    def _on_actors_edited(self):
        self.actors = [value.strip() for value in
                       self.actor_edit.text().replace("，", ",").split(",") if value.strip()]
        self.content_changed.emit()
        self._fill_choices()

    def _on_group_name_edited(self):
        self.group_name = self.group_name_edit.text().strip() or self.group_id
        self.content_changed.emit()

    def _on_group_id_edited(self):
        """改组号 = 换一次实验编号：已录事件的 group_id 与主键一起重发，旧号不复用。"""
        text = sanitize_group_id(self.group_id_edit.text(), self.group_id)
        if not text or text == self.group_id:
            self.group_id_edit.setText(self.group_id)
            return
        self.group_id = text
        self.group_id_edit.setText(text)
        self.content_changed.emit()
        numbers = []
        for row in self.event_rows:
            match = im.EVENT_ID_RE.match(str(row.get("event_id", "")))
            numbers.append(int(match.group("n")) if match else None)
        cursor = im.next_event_number(
            [im.new_event_id(text, value) for value in numbers if value is not None], text)
        for index, row in enumerate(self.event_rows):
            row["group_id"] = text
            if numbers[index] is None:
                numbers[index] = cursor
                cursor += 1
            row["event_id"] = im.new_event_id(text, numbers[index])
        self.update_table()

    def _add_object(self):
        from PySide6.QtWidgets import QInputDialog
        target_type = self.target_type_combo.currentText()
        if target_type in (im.NO_TARGET, im.SELF_TARGET):
            self._hint("%s 不需要对象 ID" % target_type)
            return
        text, ok = QInputDialog.getText(self, "新增对象",
                                        "为 %s 登记一个对象 ID：" % target_type)
        if ok and text.strip():
            self.register_object(target_type, text.strip())
            self.target_id_combo.setCurrentText(text.strip())

    def _edit_interaction_types(self):
        from PySide6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(
            self, "交互类型词表", "逗号分隔（当前 %d 项）：" % len(self.interaction_types),
            text=", ".join(self.interaction_types))
        if not ok:
            return
        values = [value.strip() for value in text.replace("，", ",").split(",") if value.strip()]
        if values:
            self.interaction_types = values
            self.content_changed.emit()
            self._fill_choices()

    def update_span_label(self):
        points = self.axis.computed_row_times()
        start = self.get_start_seconds()
        unit = self.interval_seconds()
        if not points:
            self.span_label.setText("%s 起 · 间隔 %d s · 尚无事件" % (format_seconds(start), unit))
            return
        self.span_label.setText("%s – %s · %d 条 · 间隔 %d s"
                                % (format_seconds(start), format_seconds(points[-1]),
                                   len(points), unit))

    # ---------------------------------------------------------------- 记录
    def current_event(self):
        """表单当前值 → 事件字段（group_id / event_id / timestamp 三个推算列不在内）。"""
        return {
            "actor_id": self.actor_combo.currentText().strip(),
            "target_type": self.target_type_combo.currentText().strip(),
            "target_id": self.target_id_combo.currentText().strip(),
            "interaction_type": self.interaction_combo.currentText().strip(),
            "reply_to": self.reply_combo.currentText().strip(),
            "turn_id": self.turn_edit.text().strip(),
            "duration": self.duration_edit.text().strip(),
            "source": "CODING",
            "confidence": self.confidence_spin.value(),
            "notes": self.note_edit.text().strip(),
            im.UTTERANCE_COLUMN: self.utterance_edit.text().strip(),
        }

    def target_time(self):
        """本次记录要落在哪个时刻：手动编辑过就按它，顺序追加时由骨架尾部推算。"""
        if not self.time_ctrl.edited:
            return None
        return self.time_ctrl.parse_timepoint()

    def record_event(self):
        """记录：同刻不覆盖（V09），违例当场拒收并说明是哪条规则。"""
        moment = self.target_time()
        if self.time_ctrl.edited and moment is None:
            QMessageBox.warning(self, "无效时间", "当前时间请按 时:分:秒 格式输入。")
            return
        if moment is not None and not self.axis.can_record_at(moment):
            # 早于起始时间的事件无处安放：它是时间轴原点，格位不往负方向走
            self._hint("记录时间不能早于起始时间 " + format_seconds(self.get_start_seconds()))
            return
        row = im.blank_event(self.group_id,
                             event_id=im.new_event_id(self.group_id, self._next_number()),
                             **self.current_event())
        row["timestamp"] = moment if moment is not None else self._tail_time()
        items = im.validate_row(row, self.context())
        blocking = im.blocking_items(items)
        if blocking:
            self._hint("拒绝写入：" + blocking[0].message)
            return
        for item in im.warning_items(items):
            if item.message not in [seen.message for seen in self._warned]:
                self._warned.append(item)
        self._push_history()
        if moment is None:
            self.axis.append(row)
        else:
            self.axis.record(row, moment, overwrite=False)   # V09：同窗多事件不去重
            self.time_ctrl.reset_to_playback()
        self.utterance_edit.clear()
        self.note_edit.clear()
        self.update_table()
        # 应答是逐条决定，不跟着话语一起顺延：刷完候选后手动收回空白
        self.reply_combo.setCurrentText("")

    def _next_number(self):
        return im.next_event_number([row.get("event_id", "") for row in self.event_rows],
                                    self.group_id)

    def _tail_time(self):
        """顺序追加时那一行的实际时间：由骨架尾部格位推出，仅用于校验与展示。"""
        times = self.axis.computed_row_times()
        if not times:
            return float(self.get_start_seconds())
        if self.axis.skeleton():
            head = self.axis.head_slot() or 0
            return float(self.get_start_seconds()
                         + (self.axis.tail_grid_slot() - head) * self.interval_seconds())
        return float(times[-1] + self.interval_seconds())

    # ---------------------------------------------------------------- 撤销栈
    def _push_history(self):
        """现场四件套：事件行 + 行时间 + 格位 + 已放行警告（撤销要连报告一起回退）。"""
        self.history.append((copy.deepcopy(self.event_rows), list(self.row_times),
                             list(self.row_positions), list(self._warned)))
        if len(self.history) > HISTORY_LIMIT:
            self.history.pop(0)
        self.content_changed.emit()

    def _restore(self, snapshot):
        rows, times, positions, warned = snapshot
        self.event_rows = rows
        self.row_times = times
        self.row_positions = positions
        self._warned = warned
        self.axis.align_row_states()
        self.update_table()

    def undo(self):
        if not self.history:
            return
        self._restore(self.history.pop())
        self._hint("已撤销上一次事件改动")

    # ---------------------------------------------------------------- 表格
    def column_fields(self):
        return list(im.EVENT_COLUMNS) + [im.UTTERANCE_COLUMN]

    def column_field(self, column):
        fields = self.column_fields()
        return fields[column] if 0 <= column < len(fields) else ""

    def choices_for(self, field, row_index=None):
        ctx = self.context()
        if field == "actor_id":
            return ctx.actor_choices()
        if field == "target_type":
            return im.TARGET_TYPE_CHOICES
        if field == "target_id":
            target_type = self._row_value(row_index, "target_type") or \
                self.target_type_combo.currentText()
            return ctx.object_choices(target_type)
        if field == "interaction_type":
            return self.interaction_types
        if field == "reply_to":
            moment = self._row_value(row_index, "timestamp")
            return [str(row.get("event_id", "")) for row in self._rows_with_time()
                    if moment is None or (row.get("timestamp") is not None
                                          and str(row.get("event_id")) !=
                                          str(self._row_value(row_index, "event_id"))
                                          and row["timestamp"] < float(moment))]
        return []

    def _row_value(self, row_index, field):
        if row_index is None or not (0 <= row_index < len(self.event_rows)):
            return ""
        row = dict(self.event_rows[row_index])
        if field == "timestamp":
            times = self.axis.computed_row_times()
            return times[row_index] if row_index < len(times) else row.get("timestamp")
        return row.get(field, "")

    def _rebuild_columns(self):
        fields = self.column_fields()
        self.table.setColumnCount(len(fields))
        self.table.setHorizontalHeaderLabels([HEADERS.get(field, field) for field in fields])
        header = self.table.horizontalHeader()
        header.setMinimumSectionSize(56)
        header.setSectionResizeMode(QHeaderView.Stretch)

    def events_frame(self):
        """当前事件表（含推算出的 timestamp）：分析层与存档都从这里取。"""
        return im.ensure_event_columns(DataFrame(self._rows_with_time()))

    def _rows_with_time(self):
        times = self.axis.computed_row_times()
        out = []
        for index, row in enumerate(self.event_rows):
            payload = dict(row)
            payload["timestamp"] = times[index] if index < len(times) else row.get("timestamp")
            out.append(payload)
        return out

    def update_table(self):
        """按时间轴重建事件表：行序与时间列全从骨架推导，三个推算列锁编辑。"""
        self._loading = True
        try:
            self.axis.sort_rows_by_time()
            fields = self.column_fields()
            rows = self._rows_with_time()
            self.table.setRowCount(len(rows))
            for position, row in enumerate(rows):
                for column, field in enumerate(fields):
                    value = row.get(field, "")
                    if field == "timestamp":
                        text = MISSING_TEXT if value in (None, "") else format_seconds(value)
                    elif field == "confidence":
                        try:
                            text = "%g" % float(value)
                        except (TypeError, ValueError):
                            text = ""
                    else:
                        text = "" if value is None else str(value)
                    item = QTableWidgetItem(text)
                    if field in READONLY_FIELDS:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(position, column, item)
            apply_column_widths(self.table, 0)
        finally:
            self._loading = False
        self.update_span_label()
        self._fill_choices()

    def _on_item_changed(self, item):
        """表格里改字段：改完立刻重校验，阻断类违例说明原因（值留着，让用户自己改回）。"""
        if self._loading:
            return
        field = self.column_field(item.column())
        if field in READONLY_FIELDS or field not in self.column_fields():
            return
        position = item.row()
        if position >= len(self.event_rows):
            return
        value = item.text().strip()
        self._push_history()
        if field == "confidence":
            parsed = im.parse_confidence(value)
            if parsed is None or not 0.0 <= parsed <= 1.0:
                self._hint("confidence 必须落在 0–1（当前 %r）：V08" % value)
                parsed = 1.0 if parsed is None else parsed
            self.event_rows[position]["confidence"] = parsed
            item.setText("%g" % self.event_rows[position]["confidence"])
        else:
            if self.event_rows[position].get(field, "") == value:
                return
            self.event_rows[position][field] = value
        if field in ("actor_id", "target_type", "target_id", "interaction_type", "reply_to"):
            items = im.blocking_items(im.validate_row(self._merged_row(position), self.context()))
            if items:
                self._hint("该行现在不合法：" + items[0].message)
        self._fill_choices()

    def _merged_row(self, position):
        row = dict(self.event_rows[position])
        times = self.axis.computed_row_times()
        row["timestamp"] = times[position] if position < len(times) else row.get("timestamp")
        return row

    def selected_rows(self):
        return sorted({index.row() for index in self.table.selectedIndexes()})

    def eventFilter(self, obj, event):
        if obj is self.table and event.type() == QEvent.KeyPress \
                and event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.selected_rows():
            self.delete_selected_rows()
            return True
        return super().eventFilter(obj, event)

    def delete_selected_rows(self):
        """删事件行：格位不回收（与编码页同语义），骨架上的隐藏格留着等补录。"""
        rows = self.selected_rows()
        if not rows:
            return
        self._push_history()
        self.axis.drop_rows(rows)
        self.update_table()

    def show_row_menu(self, position):
        if not self.event_rows:
            return
        if not self.selected_rows():
            index = self.table.indexAt(position)
            if index.isValid():
                self.table.selectRow(index.row())
        menu = theme.bind_menu_popup(QMenu(self))
        action = menu.addAction("删除选中事件")
        action.setEnabled(bool(self.selected_rows()))
        action.triggered.connect(self.delete_selected_rows)
        menu.exec(self.table.viewport().mapToGlobal(position))

    def clear_events(self):
        self._push_history()
        self.event_rows = []
        self.row_times = []
        self.row_positions = []
        self._warned = []
        self.axis.align_row_states()
        self.update_table()

    # ---------------------------------------------------------------- 质检 / 输出
    def quality_items(self):
        """报告 = 现场放行过的警告 + 整表重跑一遍 V01–V10。"""
        items = list(self._warned)
        items.extend(im.validate_events(self.events_frame(), self.context()))
        return items

    def show_quality_report(self):
        QualityDialog(self, "小组交互 · 数据质量报告", self.quality_items()).exec()

    def collect_meta(self):
        """给分析层的现场信息：主体字典、对象字典、词表、组名。"""
        return {
            "group_name": self.group_name,
            "members": list(self.members),
            "actors": list(self.actors),
            "object_dict": {key: list(values) for key, values in self.object_dict.items()},
            "interaction_types": list(self.interaction_types),
        }

    def output(self):
        """导出多 sheet 数据包：Group_Info / Students / Behavior_State / Interaction_Event / Codebook / Enums。"""
        path = user_state.save_file_name(self, user_state.BUCKET_EXPORT,
                                        "导出小组数据包", "",
                                        "Excel 文件 (*.xlsx);;所有文件 (*)")
        if not path:
            return
        state = self.summary_provider()
        sheets = im.template_sheets(
            group_id=self.group_id, group_name=self.group_name, members=self.members,
            interaction_types=self.interaction_types,
            codebook=im.codebook_rows(self.framework_provider()),
            behavior_state=state, events=self.events_frame())
        try:
            im.write_workbook(sheets, path)
        except Exception as exc:      # noqa: BLE001  落盘失败只提示，不炸页面
            QMessageBox.critical(self, "错误", "导出失败：%s" % exc)
            return
        QMessageBox.information(self, "成功", "已导出小组数据包")

    def _hint(self, text, msec=6000):
        """轻提示走主窗口状态栏：录入节奏中不弹模态窗挡手。"""
        win = self.window()
        if isinstance(win, QMainWindow):
            win.statusBar().showMessage(text, msec)

    # ---------------------------------------------------------------- 存档
    def to_dict(self):
        return {
            "group_id": self.group_id,
            "group_name": self.group_name,
            "group_events": copy.deepcopy(self.event_rows),
            # 事件行本身存的是内容；行时间与格位另存，读档才能还原隐藏格
            "group_row_times": list(self.row_times),
            "group_row_positions": list(self.row_positions),
            "actors": list(self.actors),
            "object_dict": {key: list(values) for key, values in self.object_dict.items()},
            "event_codebook": list(self.interaction_types),
            "start_seconds": self.get_start_seconds(),
            "time_unit": self.time_ctrl.unit_edit.text(),
        }

    def from_dict(self, payload):
        """读档：缺键一律按「旧存档没有小组页」处理，不猜数据。"""
        payload = payload or {}
        if not any(key in payload for key in
                   ("group_id", "group_events", "actors", "object_dict", "event_codebook")):
            return
        self.group_id = sanitize_group_id(payload.get("group_id"), self.group_id)
        self.group_name = str(payload.get("group_name") or self.group_id)
        self.group_id_edit.setText(self.group_id)
        self.group_name_edit.setText(self.group_name)
        self.actors = [str(value).strip() for value in payload.get("actors", [])
                       if str(value).strip()]
        self.actor_edit.setText(", ".join(self.actors))
        self.object_dict = {str(key): [str(v).strip() for v in values if str(v).strip()]
                            for key, values in (payload.get("object_dict") or {}).items()}
        types = [str(value).strip() for value in payload.get("event_codebook", [])
                 if str(value).strip()]
        self.interaction_types = types or list(im.INTERACTION_TYPES)
        if "start_seconds" in payload:
            self.time_ctrl.set_start_seconds(float(payload.get("start_seconds") or 0))
        self.time_ctrl.unit_edit.setText(str(payload.get("time_unit", "1") or "1"))
        rows = []
        for item in (payload.get("group_events") or []):
            fields = {key: value for key, value in item.items()
                      if key not in ("timestamp", "group_id")}
            restored = im.blank_event(self.group_id, **fields)
            restored["group_id"] = self.group_id
            rows.append(restored)
        self.event_rows = rows
        self.row_times = list(payload.get("group_row_times", []))[:len(rows)]
        self.row_times += [None] * (len(rows) - len(self.row_times))
        self.row_positions = list(payload.get("group_row_positions", []))[:len(rows)]
        self.row_positions += [None] * (len(rows) - len(self.row_positions))
        self.axis.align_row_states()
        self.history = []
        self._warned = []
        self.time_ctrl.set_edited(False)
        self.refresh_members()
        self.update_table()
