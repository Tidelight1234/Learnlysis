# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""AI「框架设计师」工作台：问题支架、文献参考、草稿、质检、导出与应用。"""

from __future__ import annotations

import copy
import logging
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QScrollArea,
)

import framework_export
import framework_library
import framework_prompt
import framework_quality
import theme
import user_state
from ai_worker import AiWorker
from framework_schema import (active_version, add_version, clean_dimension, clean_project,
                              compare_versions, new_project, new_version, now,
                              project_to_state_framework)
from framework_store import store


BRIEF_FIELDS = (
    ("research_question", "研究问题", "multi"),
    ("primary_goal", "主要目标", ("描述", "比较", "解释", "预测", "改进")),
    ("analysis_level", "分析层级", ("个体", "成员之间", "小组", "组间")),
    ("participants", "参与者与样本", "line"),
    ("task_context", "任务与研究情境", "multi"),
    ("setting", "活动设置", "line"),
    ("data_modalities", "数据模态（逗号分隔）", "line_list"),
    ("observable_evidence", "能可靠观察的证据", "multi"),
    ("unit", "分析/切分单位", ("固定时间窗", "完整话轮", "语义单元", "动作", "事件", "活动阶段")),
    ("temporal_rule", "起止与时间粒度", "multi"),
    ("coding_mode", "编码结构", ("状态与事件并用", "仅状态", "仅事件")),
    ("theory_focus", "理论关注（逗号分隔）", "line_list"),
    ("required_theory", "必须采用的理论", "line"),
    ("comparison", "比较条件或结果变量", "line"),
    ("out_of_scope", "明确排除的行为", "multi"),
    ("uncertainty_rule", "模糊/缺失处理", "multi"),
    ("reliability_plan", "试编与一致性计划", "multi"),
    ("ethics_constraints", "伦理与隐私边界", "multi"),
)

TABLE_HEADERS = ("维度", "类型", "层级", "编码", "决策", "操作定义", "可观察证据",
                 "纳入规则", "排除规则", "正例", "反例", "不确定规则", "来源谱系")


def _split(text):
    return [value.strip() for value in str(text or "").replace("；", ";").split(";") if value.strip()]


class FrameworkDesignerDialog(QDialog):
    """一个可独立离线使用的设计工作台；AI 诊断/起草只是可选增强。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("框架设计师")
        self.resize(1180, 780)
        self.setMinimumSize(900, 640)
        self._project = new_project()
        self._client_factory = None
        self._session_provider = None
        self._targets_provider = None
        self._applier = None
        self._worker = None
        self._worker_kind = None
        self._repair_attempted = False
        self._build()
        self._reload_projects(select_latest=True)

    # -------------------------------------------------------------- injections
    def set_client_factory(self, factory):
        self._client_factory = factory

    def set_session_provider(self, provider):
        self._session_provider = provider

    def set_framework_targets_provider(self, provider):
        self._targets_provider = provider
        self._reload_targets()

    def set_framework_applier(self, applier):
        self._applier = applier

    # -------------------------------------------------------------- layout
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        head = QHBoxLayout()
        head.addWidget(QLabel("框架项目"))
        self.project_combo = QComboBox()
        self.project_combo.currentIndexChanged.connect(self._project_changed)
        # 本页所有下拉都走其他模块同一套弹层配方，否则圆角列表外会露出方形原生边框；
        # 半径由 theme.POPUP_RADIUS 单点提供，调用点不传数字。
        theme.bind_combo_popup(self.project_combo)
        head.addWidget(self.project_combo, 1)
        self.new_btn = QPushButton("新建")
        self.new_btn.setProperty("btn", "secondary")
        self.new_btn.clicked.connect(self._new_project)
        self.delete_btn = QPushButton("删除")
        self.delete_btn.setProperty("btn", "ghost")
        self.delete_btn.clicked.connect(self._delete_project)
        head.addWidget(self.new_btn)
        head.addWidget(self.delete_btn)
        root.addLayout(head)

        title_row = QHBoxLayout()
        title_row.addWidget(QLabel("项目名称"))
        self.title_edit = QLineEdit()
        title_row.addWidget(self.title_edit, 1)
        self.save_btn = QPushButton("保存项目")
        self.save_btn.setProperty("btn", "primary")
        self.save_btn.clicked.connect(self._save_project)
        title_row.addWidget(self.save_btn)
        root.addLayout(title_row)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._brief_tab(), "1 研究问题")
        self.tabs.addTab(self._references_tab(), "2 文献依据")
        self.tabs.addTab(self._draft_tab(), "3 框架草稿")
        self.tabs.addTab(self._quality_tab(), "4 质检")
        self.tabs.addTab(self._delivery_tab(), "5 导出与应用")
        root.addWidget(self.tabs, 1)

        self.status = QLabel("框架设计结果均为待研究者确认的建议，不会自动改动编码数据。")
        self.status.setProperty("notice", True)
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.button(QDialogButtonBox.Close).setText("关闭")
        close_box.rejected.connect(self.reject)
        root.addWidget(close_box)

    def _brief_tab(self):
        host = QWidget()
        layout = QVBoxLayout(host)
        intro = QLabel("先说明要研究什么、能观察到什么以及如何切分。信息不足时，AI 只做诊断，不生成定稿。")
        intro.setProperty("notice", True)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        form = QFormLayout(content)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.brief_widgets = {}
        for key, label, kind in BRIEF_FIELDS:
            if kind == "multi":
                widget = QPlainTextEdit()
                widget.setFixedHeight(58)
            elif isinstance(kind, tuple):
                widget = QComboBox()
                widget.addItems(kind)
                theme.bind_combo_popup(widget)
            else:
                widget = QLineEdit()
            self.brief_widgets[key] = widget
            form.addRow(label, widget)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        return host

    def _references_tab(self):
        host = QWidget()
        layout = QVBoxLayout(host)
        note = QLabel("内置卡片只包含项目团队转述的摘要和可核查元数据；使用或发表时仍需阅读并引用原文。")
        note.setProperty("notice", True)
        note.setWordWrap(True)
        layout.addWidget(note)
        splitter = QSplitter(Qt.Horizontal)
        self.reference_list = QListWidget()
        for card in framework_library.all_cards():
            item = QListWidgetItem("%s（%s）" % (card["title_zh"], card["year"]))
            item.setData(Qt.UserRole, card["id"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.reference_list.addItem(item)
        self.reference_list.currentItemChanged.connect(self._show_reference)
        self.reference_detail = QPlainTextEdit()
        self.reference_detail.setReadOnly(True)
        splitter.addWidget(self.reference_list)
        splitter.addWidget(self.reference_detail)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        return host

    def _draft_tab(self):
        host = QWidget()
        layout = QVBoxLayout(host)
        actions = QHBoxLayout()
        self.offline_btn = QPushButton("按所选文献生成起始草稿")
        self.offline_btn.setProperty("btn", "secondary")
        self.offline_btn.clicked.connect(self._offline_draft)
        self.diagnose_btn = QPushButton("AI 设计诊断")
        self.diagnose_btn.setProperty("btn", "secondary")
        self.diagnose_btn.clicked.connect(self._ai_diagnosis)
        self.ai_draft_btn = QPushButton("AI 生成结构化草稿")
        self.ai_draft_btn.setProperty("btn", "primary")
        self.ai_draft_btn.clicked.connect(self._ai_draft)
        actions.addWidget(self.offline_btn)
        actions.addWidget(self.diagnose_btn)
        actions.addWidget(self.ai_draft_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        history = QHBoxLayout()
        history.addWidget(QLabel("历史版本"))
        self.version_combo = QComboBox()
        theme.bind_combo_popup(self.version_combo)
        history.addWidget(self.version_combo, 1)
        load_version = QPushButton("载入所选版本")
        load_version.setProperty("btn", "ghost")
        load_version.clicked.connect(self._load_selected_version)
        compare_version = QPushButton("与工作稿比较")
        compare_version.setProperty("btn", "ghost")
        compare_version.clicked.connect(self._compare_selected_version)
        history.addWidget(load_version)
        history.addWidget(compare_version)
        layout.addLayout(history)

        self.draft_table = QTableWidget(0, len(TABLE_HEADERS))
        self.draft_table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.draft_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.draft_table.setAlternatingRowColors(True)
        self.draft_table.horizontalHeader().setStretchLastSection(True)
        self.draft_table.setColumnWidth(0, 125)
        self.draft_table.setColumnWidth(3, 120)
        self.draft_table.setColumnWidth(5, 230)
        layout.addWidget(self.draft_table, 1)

        row_actions = QHBoxLayout()
        add_row = QPushButton("添加编码行")
        add_row.setProperty("btn", "ghost")
        add_row.clicked.connect(lambda: self._append_row({}))
        delete_row = QPushButton("删除所选行")
        delete_row.setProperty("btn", "ghost")
        delete_row.clicked.connect(self._delete_rows)
        accept_row = QPushButton("接受所选")
        accept_row.setProperty("btn", "secondary")
        accept_row.clicked.connect(lambda: self._set_decision("accepted"))
        reject_row = QPushButton("拒绝所选")
        reject_row.setProperty("btn", "ghost")
        reject_row.clicked.connect(lambda: self._set_decision("rejected"))
        save_version = QPushButton("保存为新版本")
        save_version.setProperty("btn", "primary")
        save_version.clicked.connect(self._save_version)
        for button in (add_row, delete_row, accept_row, reject_row):
            row_actions.addWidget(button)
        row_actions.addStretch(1)
        row_actions.addWidget(save_version)
        layout.addLayout(row_actions)
        return host

    def _quality_tab(self):
        host = QWidget()
        layout = QVBoxLayout(host)
        self.qa_summary = QLabel("尚未运行质检")
        self.qa_summary.setProperty("notice", True)
        layout.addWidget(self.qa_summary)
        self.qa_text = QPlainTextEdit()
        self.qa_text.setReadOnly(True)
        layout.addWidget(self.qa_text, 1)
        run = QPushButton("运行本地质检")
        run.setProperty("btn", "primary")
        run.clicked.connect(self._run_quality)
        layout.addWidget(run)

        pilot_title = QLabel("小样本试编与分歧记录")
        pilot_title.setProperty("section", True)
        layout.addWidget(pilot_title)
        pilot_form = QGridLayout()
        self.pilot_sample = QLineEdit()
        self.pilot_sample.setPlaceholderText("片段编号或时间范围")
        self.pilot_issue = QLineEdit()
        self.pilot_issue.setPlaceholderText("编码分歧或发现的问题")
        self.pilot_resolution = QLineEdit()
        self.pilot_resolution.setPlaceholderText("讨论结论 / 框架修改决定")
        pilot_form.addWidget(QLabel("片段"), 0, 0)
        pilot_form.addWidget(self.pilot_sample, 0, 1)
        pilot_form.addWidget(QLabel("分歧"), 1, 0)
        pilot_form.addWidget(self.pilot_issue, 1, 1)
        pilot_form.addWidget(QLabel("处理"), 2, 0)
        pilot_form.addWidget(self.pilot_resolution, 2, 1)
        add_pilot = QPushButton("记录本次试编")
        add_pilot.setProperty("btn", "secondary")
        add_pilot.clicked.connect(self._add_pilot_log)
        pilot_form.addWidget(add_pilot, 2, 2)
        layout.addLayout(pilot_form)
        self.pilot_list = QListWidget()
        self.pilot_list.setMaximumHeight(100)
        layout.addWidget(self.pilot_list)
        return host

    def _delivery_tab(self):
        host = QWidget()
        layout = QVBoxLayout(host)
        note = QLabel("导出完整研究设计包；应用时只把已接受的状态维度投影到编码页，事件维度保持独立。")
        note.setProperty("notice", True)
        note.setWordWrap(True)
        layout.addWidget(note)
        export = QPushButton("导出完整包（Markdown / Excel / JSON / BibTeX）")
        export.setProperty("btn", "secondary")
        export.clicked.connect(self._export_bundle)
        layout.addWidget(export)
        layout.addSpacing(20)
        apply_row = QHBoxLayout()
        apply_row.addWidget(QLabel("应用目标"))
        self.target_combo = QComboBox()
        theme.bind_combo_popup(self.target_combo)
        apply_row.addWidget(self.target_combo, 1)
        self.apply_btn = QPushButton("检查并应用状态框架")
        self.apply_btn.setProperty("btn", "primary")
        self.apply_btn.clicked.connect(self._apply_framework)
        apply_row.addWidget(self.apply_btn)
        layout.addLayout(apply_row)
        self.apply_preview = QPlainTextEdit()
        self.apply_preview.setReadOnly(True)
        self.apply_preview.setPlaceholderText("运行质检后，这里将显示可应用的状态框架。")
        layout.addWidget(self.apply_preview, 1)
        layout.addStretch(1)
        return host

    # -------------------------------------------------------------- projects
    def _reload_projects(self, select_id=None, select_latest=False):
        values = store.projects()
        if not values:
            value = new_project()
            store.upsert(value)
            values = [value]
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        for value in values:
            self.project_combo.addItem(value["title"], value["project_id"])
        wanted = select_id or (values[-1]["project_id"] if select_latest else values[0]["project_id"])
        index = self.project_combo.findData(wanted)
        self.project_combo.setCurrentIndex(index if index >= 0 else 0)
        self.project_combo.blockSignals(False)
        self._load_project(store.get(self.project_combo.currentData()) or values[0])

    def _project_changed(self, _index):
        value = store.get(self.project_combo.currentData())
        if value:
            self._load_project(value)

    def _load_project(self, project):
        self._project = clean_project(project)
        self.title_edit.setText(self._project["title"])
        brief = self._project.get("research_brief") or {}
        for key, _label, kind in BRIEF_FIELDS:
            widget = self.brief_widgets[key]
            value = brief.get(key, "")
            if kind == "multi":
                widget.setPlainText(str(value or ""))
            elif isinstance(kind, tuple):
                index = widget.findText(str(value or ""))
                widget.setCurrentIndex(index if index >= 0 else 0)
            else:
                widget.setText("、".join(value) if isinstance(value, list) else str(value or ""))
        selected = set(self._project.get("selected_references") or [])
        for index in range(self.reference_list.count()):
            item = self.reference_list.item(index)
            item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in selected else Qt.Unchecked)
        version = active_version(self._project)
        self._load_dimensions((version or {}).get("dimensions") or [])
        self._refresh_versions()
        self._refresh_pilot_log()
        self._run_quality(silent=True)

    def _collect_brief(self):
        brief = {}
        for key, _label, kind in BRIEF_FIELDS:
            widget = self.brief_widgets[key]
            if kind == "multi":
                value = widget.toPlainText().strip()
            elif isinstance(kind, tuple):
                value = widget.currentText()
            else:
                value = widget.text().strip()
            if kind == "line_list":
                value = [part.strip() for part in value.replace("，", ",").replace("、", ",").split(",") if part.strip()]
            brief[key] = value
        return brief

    def _selected_reference_ids(self):
        return [self.reference_list.item(index).data(Qt.UserRole)
                for index in range(self.reference_list.count())
                if self.reference_list.item(index).checkState() == Qt.Checked]

    def _sync_project(self):
        self._project["title"] = self.title_edit.text().strip() or "未命名框架项目"
        self._project["research_brief"] = self._collect_brief()
        self._project["selected_references"] = self._selected_reference_ids()

    def _save_project(self):
        self._sync_project()
        store.upsert(self._project)
        self._reload_projects(select_id=self._project["project_id"])
        self.status.setText("项目已保存。框架表格只有在“保存为新版本”后才进入版本历史。")

    def _new_project(self):
        value = new_project()
        store.upsert(value)
        self._reload_projects(select_id=value["project_id"])

    def _delete_project(self):
        if QMessageBox.question(self, "删除框架项目", "确定删除当前框架项目？此操作不可撤销。",
                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        store.remove(self._project["project_id"])
        self._reload_projects(select_latest=True)

    # -------------------------------------------------------------- references
    def _show_reference(self, item, _previous=None):
        card = framework_library.card(item.data(Qt.UserRole)) if item else None
        if not card:
            self.reference_detail.clear()
            return
        text = [card["title_zh"], card["title_original"], "",
                "作者：%s" % "、".join(card["authors"]),
                "年份：%s" % card["year"], "DOI：%s" % card["doi"], "",
                card["summary"], "", "分析单位：%s" % card["unit_guidance"], "",
                "优势：\n- " + "\n- ".join(card["strengths"]), "",
                "限制与迁移提醒：\n- " + "\n- ".join(card["limitations"])]
        self.reference_detail.setPlainText("\n".join(text))

    def _selected_references(self):
        return [framework_library.card(value) for value in self._selected_reference_ids()
                if framework_library.card(value)]

    # -------------------------------------------------------------- draft table
    def _append_row(self, values):
        row = self.draft_table.rowCount()
        self.draft_table.insertRow(row)
        data = [values.get("dimension", ""), values.get("kind", "state"),
                values.get("level", "individual"), values.get("label", ""),
                values.get("decision", "proposed"), values.get("definition", ""),
                values.get("evidence", ""), values.get("include", ""),
                values.get("exclude", ""), values.get("positive", ""), values.get("negative", ""),
                values.get("uncertainty", ""), values.get("sources", "")]
        for column, value in enumerate(data):
            self.draft_table.setItem(row, column, QTableWidgetItem(str(value or "")))

    def _load_dimensions(self, dimensions):
        self.draft_table.setRowCount(0)
        for dim in dimensions or []:
            for code in dim.get("codes") or []:
                sources = ";".join("%s:%s" % (value.get("reference_id") or "new",
                                                value.get("relation") or "new_by_researcher")
                                   for value in code.get("source_links") or [])
                self._append_row({
                    "dimension": dim.get("name"), "kind": dim.get("kind"), "level": dim.get("level"),
                    "label": code.get("label"), "decision": code.get("decision"),
                    "definition": code.get("definition"),
                    "evidence": ";".join(code.get("observable_evidence") or []),
                    "include": ";".join(code.get("include_when") or []),
                    "exclude": ";".join(code.get("exclude_when") or []),
                    "positive": ";".join(code.get("positive_examples") or []),
                    "negative": ";".join(code.get("negative_examples") or []),
                    "uncertainty": code.get("uncertainty_rule"), "sources": sources})

    def _cell(self, row, column):
        item = self.draft_table.item(row, column)
        return item.text().strip() if item else ""

    def _table_dimensions(self):
        grouped, order = {}, []
        for row in range(self.draft_table.rowCount()):
            name = self._cell(row, 0) or "未命名维度"
            key = (name, self._cell(row, 1) or "state", self._cell(row, 2) or "individual")
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            sources = []
            for token in _split(self._cell(row, 12)):
                ref, sep, relation = token.partition(":")
                if ref == "new":
                    ref = ""
                sources.append({"reference_id": ref, "relation": relation if sep else "new_by_researcher",
                                "source_concept": self._cell(row, 3), "adaptation_note": ""})
            grouped[key].append({
                "label": self._cell(row, 3), "decision": self._cell(row, 4) or "proposed",
                "definition": self._cell(row, 5), "observable_evidence": _split(self._cell(row, 6)),
                "include_when": _split(self._cell(row, 7)), "exclude_when": _split(self._cell(row, 8)),
                "positive_examples": _split(self._cell(row, 9)),
                "negative_examples": _split(self._cell(row, 10)),
                "uncertainty_rule": self._cell(row, 11), "source_links": sources})
        return [clean_dimension({"name": name, "kind": kind, "level": level,
                                 "multi_label": kind == "event", "unit": self._collect_brief().get("unit"),
                                 "codes": grouped[(name, kind, level)]}, index + 1)
                for index, (name, kind, level) in enumerate(order)]

    def _delete_rows(self):
        for row in sorted({index.row() for index in self.draft_table.selectedIndexes()}, reverse=True):
            self.draft_table.removeRow(row)

    def _set_decision(self, decision):
        rows = {index.row() for index in self.draft_table.selectedIndexes()}
        if not rows:
            QMessageBox.information(self, "选择编码", "请先选择一行或多行编码。")
            return
        for row in rows:
            self.draft_table.setItem(row, 4, QTableWidgetItem(decision))

    def _offline_draft(self):
        selected = self._selected_reference_ids()
        if not selected:
            QMessageBox.information(self, "选择文献", "请先在“文献依据”中勾选至少一个参考框架。")
            return
        self._sync_project()
        dimensions = framework_library.starter_dimensions(selected)
        self._load_dimensions(dimensions)
        self.status.setText("已生成离线起始草稿。请逐条补充边界并接受或拒绝，再保存为新版本。")
        self.tabs.setCurrentIndex(2)

    def _save_version(self):
        dimensions = self._table_dimensions()
        if not dimensions:
            QMessageBox.information(self, "保存版本", "当前没有可保存的编码。")
            return
        self._sync_project()
        self._project, version = add_version(
            self._project, dimensions, label="框架版本 %d" % (len(self._project.get("framework_versions") or []) + 1),
            reason="研究者在框架设计工作台保存", created_by="researcher_with_ai_suggestion")
        store.upsert(self._project)
        self.status.setText("已保存版本 %s；旧版本仍保留。" % version["version_id"])
        self._reload_projects(select_id=self._project["project_id"])

    def _refresh_versions(self):
        self.version_combo.clear()
        versions = self._project.get("framework_versions") or []
        for index, version in enumerate(versions, 1):
            label = "%d. %s · %s" % (index, version.get("label", "框架版本"),
                                      version.get("created_at", ""))
            self.version_combo.addItem(label, version.get("version_id"))
        active = self._project.get("active_version_id")
        selected = self.version_combo.findData(active)
        if selected >= 0:
            self.version_combo.setCurrentIndex(selected)

    def _selected_version(self):
        wanted = self.version_combo.currentData()
        return next((value for value in self._project.get("framework_versions") or []
                     if value.get("version_id") == wanted), None)

    def _load_selected_version(self):
        version = self._selected_version()
        if not version:
            QMessageBox.information(self, "历史版本", "当前还没有保存过框架版本。")
            return
        self._load_dimensions(version.get("dimensions") or [])
        self.status.setText("已把历史版本载入为工作稿；原版本未被修改。保存时会形成新版本。")

    def _compare_selected_version(self):
        version = self._selected_version()
        if not version:
            QMessageBox.information(self, "版本比较", "当前还没有可比较的历史版本。")
            return
        working = new_version(self._table_dimensions(), "当前工作稿")
        diff = compare_versions(version, working)
        text = ["新增：%d" % len(diff["added"]), "删除：%d" % len(diff["removed"]),
                "修改：%d" % len(diff["changed"]), ""]
        for label, key in (("新增", "added"), ("删除", "removed"), ("修改", "changed")):
            if diff[key]:
                text.append("%s\n- %s" % (label, "\n- ".join(diff[key])))
        QMessageBox.information(self, "版本差异", "\n".join(text))

    # -------------------------------------------------------------- AI
    def _current_framework(self):
        try:
            snapshot = self._session_provider() if self._session_provider else {}
            pages = snapshot.get("encoder_pages") or []
            for item in pages:
                page = item[-1] if isinstance(item, (tuple, list)) else item
                framework = page.dim_panel.framework()
                if framework:
                    return framework
        except Exception:
            pass
        return []

    def _client(self):
        if self._client_factory is None:
            return None, "AI 客户端尚未配置。"
        client = self._client_factory()
        ok, note = client.ready()
        return (client, "") if ok else (None, note)

    def _ai_diagnosis(self):
        self._sync_project()
        if not self._project["research_brief"].get("research_question"):
            QMessageBox.information(self, "补充研究问题", "请先填写研究问题。")
            return
        client, note = self._client()
        if client is None:
            QMessageBox.warning(self, "AI 未配置", note)
            return
        if not self._confirm_ai_payload("设计诊断"):
            return
        messages = framework_prompt.diagnosis_messages(
            self._project, self._selected_references(), self._current_framework())
        self._launch_worker(client, messages, "diagnosis")

    def _ai_draft(self):
        self._sync_project()
        if not self._project["research_brief"].get("research_question"):
            QMessageBox.information(self, "补充研究问题", "请先填写研究问题。")
            return
        if not self._selected_reference_ids():
            QMessageBox.information(self, "选择文献", "请先勾选至少一个参考框架。")
            return
        client, note = self._client()
        if client is None:
            QMessageBox.warning(self, "AI 未配置", note)
            return
        if not self._confirm_ai_payload("结构化草稿"):
            return
        self._repair_attempted = False
        messages = framework_prompt.draft_messages(
            self._project, self._selected_references(), self._current_framework())
        self._launch_worker(client, messages, "draft")

    def _confirm_ai_payload(self, purpose):
        brief = self._project.get("research_brief") or {}
        refs = self._selected_reference_ids()
        current = self._current_framework()
        message = ("本轮将发送给你配置的模型：\n"
                   "- 用途：%s\n- 研究问题：%s\n- 所选参考卡：%s\n"
                   "- 当前编码框架：%s\n\n"
                   "不会发送 API Key、原视频、完整文件路径或未选择的对话历史。是否继续？" % (
                       purpose, brief.get("research_question") or "（未填写）",
                       "、".join(refs) or "（无）", "有" if current else "无"))
        return QMessageBox.question(self, "确认发送内容", message,
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) == QMessageBox.Yes

    def _launch_worker(self, client, messages, kind):
        if self._worker is not None:
            return
        self._worker_kind = kind
        self._set_busy(True)
        self.status.setText("AI 正在%s……" % ("诊断设计" if kind == "diagnosis" else "生成结构化草稿"))
        self._worker = AiWorker(client, messages, parent=self, stream=True)
        self._worker.done.connect(self._ai_done)
        self._worker.error.connect(self._ai_error)
        self._worker.start()

    def _ai_done(self, result):
        kind = self._worker_kind
        self._worker = None
        self._worker_kind = None
        self._set_busy(False)
        text = (result or {}).get("text", "")
        if kind == "diagnosis":
            dialog = QDialog(self)
            dialog.setWindowTitle("AI 设计诊断（建议稿）")
            dialog.resize(760, 560)
            layout = QVBoxLayout(dialog)
            box = QPlainTextEdit(text)
            box.setReadOnly(True)
            layout.addWidget(box)
            buttons = QDialogButtonBox(QDialogButtonBox.Close)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            dialog.exec()
            self.status.setText("诊断完成；诊断不会自动修改框架。")
            return
        try:
            dimensions = framework_prompt.parse_draft(text, self._selected_reference_ids())
        except Exception as exc:
            # 未知引用是内容安全问题，不能当作“格式错误”洗掉；纯格式错误只允许修复一次。
            if (kind == "draft" and not self._repair_attempted
                    and "未提供的文献 ID" not in str(exc)):
                client, note = self._client()
                if client is not None:
                    self._repair_attempted = True
                    self._launch_worker(
                        client, framework_prompt.repair_messages(
                            text, self._selected_reference_ids()), "draft_repair")
                    self.status.setText("首轮草稿格式无效，正在进行一次仅限格式的修复……")
                    return
            QMessageBox.warning(self, "草稿解析失败", "%s\n\n模型原始输出已保留在本次对话结果中，未写入项目。" % exc)
            self.status.setText("AI 草稿未通过本地结构校验，项目未改动。")
            return
        self._load_dimensions(dimensions)
        self.status.setText("AI 草稿已通过结构校验。请逐条审阅、编辑并接受或拒绝。")

    def _ai_error(self, message):
        self._worker = None
        self._worker_kind = None
        self._set_busy(False)
        QMessageBox.warning(self, "AI 调用失败", str(message))
        self.status.setText("AI 调用失败；离线设计功能仍可继续使用。")

    def _set_busy(self, busy):
        for button in (self.offline_btn, self.diagnose_btn, self.ai_draft_btn, self.save_btn):
            button.setEnabled(not busy)

    # -------------------------------------------------------------- QA/export/apply
    def _working_project(self):
        self._sync_project()
        value = copy.deepcopy(self._project)
        dimensions = self._table_dimensions()
        if dimensions:
            version = active_version(value)
            if version:
                version["dimensions"] = dimensions
            else:
                value, _ = add_version(value, dimensions, "未保存工作稿", "质检临时版本")
        return value

    def _run_quality(self, _checked=False, silent=False):
        project = self._working_project()
        issues = framework_quality.inspect(project, framework_library.ids())
        stats = framework_quality.summary(issues)
        self.qa_summary.setText("阻断 %d 项 · 警告 %d 项 · %s" % (
            stats["blocks"], stats["warnings"], "可进入应用确认" if stats["passed"] else "尚不可应用"))
        lines = []
        for issue in issues:
            prefix = "阻断" if issue["level"] == "block" else "警告"
            lines.append("[%s] %s\n  位置：%s" % (prefix, issue["message"], issue.get("path") or "—"))
        self.qa_text.setPlainText("\n\n".join(lines) if lines else "未发现阻断或警告。仍建议完成双人试编与一致性检查。")
        framework = project_to_state_framework(project)
        self.apply_preview.setPlainText("\n".join("%s：%s" % (name, "、".join(codes))
                                                   for name, codes in framework)
                                        or "没有已接受、可应用的状态维度。")
        if not silent:
            self.tabs.setCurrentIndex(3)
        return project, issues, stats

    def _refresh_pilot_log(self):
        self.pilot_list.clear()
        for entry in self._project.get("pilot_log") or []:
            self.pilot_list.addItem("%s｜%s｜%s" % (
                entry.get("sample", "未标片段"), entry.get("issue", ""),
                entry.get("resolution", "")))

    def _add_pilot_log(self):
        sample = self.pilot_sample.text().strip()
        issue = self.pilot_issue.text().strip()
        resolution = self.pilot_resolution.text().strip()
        if not sample or not issue:
            QMessageBox.information(self, "试编记录", "请至少填写片段和发现的分歧/问题。")
            return
        self._sync_project()
        self._project.setdefault("pilot_log", []).append({
            "sample": sample, "issue": issue, "resolution": resolution,
            "framework_version_id": self._project.get("active_version_id", ""),
            "created_at": now(),
        })
        store.upsert(self._project)
        self.pilot_sample.clear()
        self.pilot_issue.clear()
        self.pilot_resolution.clear()
        self._refresh_pilot_log()
        self.status.setText("试编分歧已记录。若因此调整码本，请另存一个新版本。")

    def _references_for(self, project):
        return [framework_library.card(value) for value in project.get("selected_references") or []
                if framework_library.card(value)]

    def _export_bundle(self):
        project, issues, _stats = self._run_quality(silent=True)
        folder = user_state.pick_directory(self, user_state.BUCKET_EXPORT, "选择框架设计导出目录")
        if not folder:
            return
        refs = self._references_for(project)
        try:
            framework_export.write_markdown(os.path.join(folder, "framework_design.md"), project, refs, issues)
            framework_export.write_json(os.path.join(folder, "framework.json"), project)
            framework_export.write_excel(os.path.join(folder, "codebook.xlsx"), project, refs)
            framework_export.write_bibtex(os.path.join(folder, "references.bib"), refs)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        QMessageBox.information(self, "导出完成", "已导出到：\n%s" % folder)

    def _reload_targets(self):
        if not hasattr(self, "target_combo"):
            return
        self.target_combo.clear()
        self.target_combo.addItem("新建编码页（推荐）", "__new__")
        try:
            targets = list(self._targets_provider() or []) if self._targets_provider else []
        except Exception:
            targets = []
        for target in targets:
            label = target.get("label", target.get("id", "编码页"))
            suffix = "（已有 %d 条记录）" % target.get("row_count", 0) if target.get("row_count") else ""
            self.target_combo.addItem(label + suffix, target.get("id"))

    def _apply_framework(self):
        project, issues, stats = self._run_quality(silent=True)
        if not stats["passed"]:
            QMessageBox.warning(self, "尚不可应用", "仍有 %d 项阻断问题，请先在“质检”中处理。" % stats["blocks"])
            self.tabs.setCurrentIndex(3)
            return
        framework = project_to_state_framework(project)
        if not framework:
            QMessageBox.information(self, "没有状态框架", "当前没有已接受的状态维度；事件维度不会写入状态编码页。")
            return
        if self._applier is None:
            QMessageBox.warning(self, "无法应用", "主窗口尚未提供框架应用入口。")
            return
        target_id = self.target_combo.currentData()
        summary = "\n".join("- %s：%s" % (name, "、".join(codes)) for name, codes in framework)
        if QMessageBox.question(
                self, "确认应用框架",
                "将以下已接受的状态框架应用到“%s”：\n\n%s\n\n事件维度不会进入状态表。是否继续？" % (
                    self.target_combo.currentText(), summary),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            ok, message = self._applier(target_id, framework)
        except Exception as exc:
            ok, message = False, str(exc)
        if ok:
            # 让现有「编码助手 / AI 辅助编码」获得同一份操作定义；仍只保存释义与首个正例，
            # 完整来源谱系和负例继续留在框架项目中。
            try:
                from ai_settings import settings
                notes = {}
                version = active_version(project) or {"dimensions": []}
                for dimension in version.get("dimensions") or []:
                    if dimension.get("kind") != "state":
                        continue
                    codes = {}
                    for code in dimension.get("codes") or []:
                        if code.get("decision") != "accepted":
                            continue
                        details = [code.get("definition") or ""]
                        if code.get("include_when"):
                            details.append("纳入：" + "；".join(code["include_when"]))
                        if code.get("exclude_when"):
                            details.append("排除：" + "；".join(code["exclude_when"]))
                        codes[code.get("label", "")] = {
                            "desc": " ".join(value for value in details if value),
                            "example": (code.get("positive_examples") or [""])[0],
                        }
                    if codes:
                        notes[dimension.get("name", "维度")] = codes
                settings.set_codebook_notes(notes)
            except Exception as exc:
                logging.warning("框架已应用，但码本释义同步失败：%s", exc)
            self._project["status"] = "candidate"
            store.upsert(self._project)
            QMessageBox.information(self, "应用完成", message or "状态框架已应用。")
            self._reload_targets()
        else:
            QMessageBox.warning(self, "未应用", message or "操作已取消。")

    def closeEvent(self, event):
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(1500)
        super().closeEvent(event)
