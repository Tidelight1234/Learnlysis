# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""AI 助手外层标签页：智能体档案 / 对话 / 模型设置三区。

一块自成一体的 QWidget（照 ModeAnalysisWorkbench 的左右分栏范式），主窗口只需
``addTab`` 并注入一个 ``context_provider`` 回调；本模块绝不反向引用主窗口的内部控件。

- 左栏（滚动）：**智能体档案区**（下拉选择 + 新建/编辑/删除向导 + 任务模板）与
  **设置区**（base_url / model / api_key / 采样参数 / 测试连接）。
- 右栏：**对话区**（自绘气泡 + 流式逐字追加 + 每轮 token 用量 + 发送/停止/清空/导出）。

安全口径：api_key 走密码态输入框 + 「显示」切换 + 「本次不保存」勾选，落盘前由
ai_settings 过一层 DPAPI（见 secure_store.py）；网络调用一律放进 AiWorker(QThread)，
UI 线程绝不阻塞；停止只有一条通路（``AiWorker.cancel()``）。
样式全部走 theme 的 property 选择器（aiBubble / aiRole / card / btn / saveState 等），
图标走 icon_provider，无一处内联 setStyleSheet。
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QLineEdit, QPlainTextEdit, QComboBox, QCheckBox,
    QDoubleSpinBox, QSpinBox, QListWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QScrollArea, QDialog, QDialogButtonBox, QMessageBox,
)

import theme
import user_state
import ai_context
from ai_settings import settings, CONTEXT_KEYS, DEFAULT_AGENTS
from ai_client import OpenAICompatClient, user_message
from ai_worker import AiWorker

#: 上下文开关的中文标签（与 ai_settings.CONTEXT_KEYS 一一对应）
CONTEXT_LABELS = {
    "data": "数据规模",
    "framework": "编码框架",
    "methods": "方法清单",
    "results": "当前结果",
}

#: 新建向导的预置模板：(显示名, 系统提示, 上下文开关, 任务模板)
ROLE_PRESETS = tuple(
    (item["name"], item["role"], item["context"], item["templates"])
    for item in DEFAULT_AGENTS
)

# 左栏最宽自然内容是「本次不保存 API Key…」整句复选框 + 卡片左右边距（~390px），
# 340 会把它挤到横向裁切；加宽到 420 让左栏所有控件不开横向滚动即可完整显示。
LEFT_PANEL_W = 420

# 顶边距与「统计及可视化」/IND/GRP 三个模块同一档（它们的根布局都是 12）：
# 外层标签页与下面的卡片之间要有一条能看出是“页边”的缝，不能直接贴上去。
TOP_GAP = 12
# 两卡之间固定 12 px（计入“左栏总高 = 右栏总高”的账）；卡片内部基础间距 8 px，
# 富余高度全由上卡的模板清单吸收（见 _build_left 的 stretch 与 _build_agent_card）。
CARD_GAP = 12
CARD_SPACING = 8


# ------------------------------------------------------------------ 自适应高的标签与清单

class _AutoHeightLabel(QLabel):
    """宽度定死、高度只听文字需要的说明标签。

    左栏是定宽的（LEFT_PANEL_W），卡里的说明文字过去只有两种下场：被高度上限截断，
    或被 Qt 的 height-for-width 顶高整栏。这里自己量——每次宽度或文本变了，就把高
    钉成「按当前宽排版真正需要的高」。`heightForWidth` 已含 QSS 的 padding/border
    （实测 notice 标签 374px 宽报 92 = 4 行 × 19 + 上下 16，刷白底逐行验墨不截），
    所以不用再自己补边框账。窗口缩放时随 resizeEvent 自愈。
    """

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self._syncing = False

    def sync_height(self):
        """量不出高（还没排版 / 宽为 0）就原样不动，等下一次 resizeEvent 再算。"""
        width = self.width()
        if width <= 0 or self._syncing:
            return
        need = self.heightForWidth(width)
        if need <= 0 or need == self.height():
            return
        self._syncing = True                  # setFixedHeight 会回过头来触发 resizeEvent
        try:
            self.setFixedHeight(need)
        finally:
            self._syncing = False

    def showEvent(self, event):
        """QSS 的字号要 polish 之后才落到 font() 上，构造期量出来的高不算数。"""
        super().showEvent(event)
        self.sync_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.sync_height()


class _TemplateList(QListWidget):
    """任务模板清单：自然高只按自己声明的最小高给，富余照样由布局拉伸。

    Qt 默认塞给 QListWidget 的 sizeHint 高 192，比这张卡真正用得上的（最小 110）凭空
    多出 80 px —— 左栏「两卡自然高 ≤ 视口就不出滚动条」的账正是被它撑破的。
    """

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setHeight(self.minimumHeight())
        return hint


# ------------------------------------------------------------------ 对话气泡

class _MessageBubble(QWidget):
    """一条消息气泡：角色名 + 正文 + 用量。正文可选可复制，流式时逐块 append。"""

    _ROLE_NAMES = {"user": "你", "assistant": "AI 助手", "error": "提示"}

    def __init__(self, role, text="", parent=None):
        super().__init__(parent)
        self.role = role
        # 自绘 QWidget 子类要让 QSS 的 background 落画，必须显式开 StyledBackground
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("aiBubble", role)
        self._text = text or ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 10)
        lay.setSpacing(3)
        self.role_label = QLabel(self._ROLE_NAMES.get(role, role))
        self.role_label.setProperty("aiRole", True)
        self.body = QLabel(self._text or "…")
        self.body.setProperty("aiBody", True)
        self.body.setWordWrap(True)
        self.body.setTextFormat(Qt.PlainText)
        self.body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.usage_label = QLabel("")
        self.usage_label.setProperty("aiUsage", True)
        self.usage_label.hide()
        lay.addWidget(self.role_label)
        lay.addWidget(self.body)
        lay.addWidget(self.usage_label)

    def text(self):
        return self._text

    def set_text(self, text):
        self._text = text or ""
        self.body.setText(self._text or "…")

    def append(self, piece):
        if not piece:
            return
        self._text += piece
        self.body.setText(self._text)

    def set_usage(self, usage, cancelled=False):
        usage = usage or {}
        total = usage.get("total_tokens")
        if total is None and not cancelled:
            self.usage_label.hide()
            return
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        bits = []
        if total is not None:
            bits.append("tokens %d（入 %s / 出 %s）%s"
                        % (total, prompt if prompt is not None else "?",
                           completion if completion is not None else "?",
                           "［估算］" if usage.get("estimated") else ""))
        if cancelled:
            bits.append("已停止")
        self.usage_label.setText(" · ".join(bits))
        self.usage_label.show()

    def as_error(self, message):
        """把一条（通常为空的）气泡就地改成错误态：换底色 + 文案。"""
        self.role = "error"
        self.setProperty("aiBubble", "error")
        self.role_label.setText(self._ROLE_NAMES["error"])
        theme.refresh_style(self)
        self.set_text(message)


# ------------------------------------------------------------------ 新建/编辑向导

class AgentWizardDialog(QDialog):
    """智能体档案向导：新建时可选预置模板，编辑时预填现有档案。"""

    def __init__(self, parent=None, agent=None):
        super().__init__(parent)
        self._agent = agent
        self.setWindowTitle("编辑智能体" if agent else "新建智能体")
        self.setMinimumWidth(540)
        self._build()
        if agent:
            self._load(agent)
        else:
            self._apply_preset(0)

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)
        card = QWidget()
        card.setProperty("card", True)
        theme.apply_card_shadow(card)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)

        if self._agent is None:
            preset_row = QHBoxLayout()
            preset_row.addWidget(self._label("从模板开始"))
            self.preset_combo = QComboBox()
            for name, _role, _ctx, _tpl in ROLE_PRESETS:
                self.preset_combo.addItem(name)
            self.preset_combo.addItem("空白")
            self.preset_combo.currentIndexChanged.connect(self._apply_preset)
            # 下拉弹层必须显式施加圆角配方：容器窗体自带方形原生边框，不施加就会在
            # 圆角列表外露出一圈黑线框矩形（容器可能被 Qt 重建，故用绑定器持续施加）。
            theme.bind_combo_popup(self.preset_combo)
            preset_row.addWidget(self.preset_combo, 1)
            lay.addLayout(preset_row)

        lay.addWidget(self._label("名称"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如：分析规划师")
        lay.addWidget(self.name_edit)

        lay.addWidget(self._label("角色 / 系统提示"))
        self.role_edit = QPlainTextEdit()
        self.role_edit.setPlaceholderText("定义这个智能体的身份、任务与回答风格……")
        self.role_edit.setFixedHeight(120)
        lay.addWidget(self.role_edit)

        lay.addWidget(self._label("附带上下文"))
        ctx_row = QHBoxLayout()
        ctx_row.setSpacing(14)
        self.context_checks = {}
        for key in CONTEXT_KEYS:
            box = QCheckBox(CONTEXT_LABELS[key])
            self.context_checks[key] = box
            ctx_row.addWidget(box)
        ctx_row.addStretch(1)
        lay.addLayout(ctx_row)

        lay.addWidget(self._label("任务快捷模板（每行一条）"))
        self.templates_edit = QPlainTextEdit()
        self.templates_edit.setFixedHeight(72)
        self.templates_edit.setPlaceholderText("点击即可填入对话框的常用提问")
        lay.addWidget(self.templates_edit)

        outer.addWidget(card)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("保存")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    @staticmethod
    def _label(text):
        label = QLabel(text)
        label.setProperty("section", True)
        return label

    def _apply_preset(self, index):
        if index < 0 or index >= len(ROLE_PRESETS):
            self.name_edit.setText("")
            self.role_edit.setPlainText("")
            for box in self.context_checks.values():
                box.setChecked(False)
            self.templates_edit.setPlainText("")
            return
        name, role, context, templates = ROLE_PRESETS[index]
        self.name_edit.setText(name)
        self.role_edit.setPlainText(role)
        for key, box in self.context_checks.items():
            box.setChecked(bool(context.get(key)))
        self.templates_edit.setPlainText("\n".join(templates))

    def _load(self, agent):
        self.name_edit.setText(agent.get("name", ""))
        self.role_edit.setPlainText(agent.get("role", ""))
        context = agent.get("context") or {}
        for key, box in self.context_checks.items():
            box.setChecked(bool(context.get(key)))
        self.templates_edit.setPlainText("\n".join(agent.get("templates", [])))

    def values(self):
        templates = [line.strip() for line in
                     self.templates_edit.toPlainText().splitlines() if line.strip()]
        context = {key: box.isChecked() for key, box in self.context_checks.items()}
        return {
            "name": self.name_edit.text().strip() or "新智能体",
            "role": self.role_edit.toPlainText().strip(),
            "context": context,
            "templates": templates,
        }


# ------------------------------------------------------------------ 主模块

class AiAssistantModule(QWidget):
    """AI 助手页：档案 / 对话 / 设置三区。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._context_provider = None
        self._transport = None            # 注入的假传输：冒烟测试不连真网
        self._worker = None
        self._worker_kind = None          # "chat" | "test"
        self._framework_dialog = None
        self._framework_targets_provider = None
        self._framework_applier = None
        self._history = []                # 已完成的对话轮（不持久化，见方案假设 3）
        self._stream_bubble = None
        self._current_text = ""
        self.initUI()
        self._reload_agents()
        self._load_settings_into_form()
        self._add_bubble("assistant",
                         "你好，我是 AI 助手。先在左侧「模型与密钥」填入你的 API 信息并测试连接，"
                         "再选一个智能体开始对话。所有调用走你自己的 key、由你自付费用。")

    # ---------- 对外注入 ----------
    def set_context_provider(self, provider):
        """注入「取当前工作现场」的回调：返回 ai_context 认的会话快照 dict。"""
        self._context_provider = provider

    def set_transport(self, transport):
        """注入假传输（payload -> Iterable[str]）：冒烟测试用，默认 None 走 urllib。"""
        self._transport = transport

    def set_framework_workflow(self, targets_provider, applier):
        """注入框架设计工作台的目标列表与应用回调；仍不暴露主窗口控件。"""
        self._framework_targets_provider = targets_provider
        self._framework_applier = applier

    # ---------- 布局 ----------
    def initUI(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, TOP_GAP, 0, 0)
        root.setSpacing(12)
        root.addWidget(self._build_left())
        root.addWidget(self._build_right(), 1)

    def _build_left(self):
        host = QWidget()
        host.setFixedWidth(LEFT_PANEL_W)
        outer = QVBoxLayout(host)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        column = QVBoxLayout(inner)
        column.setContentsMargins(0, 0, 8, 0)
        column.setSpacing(CARD_GAP)
        # 高度账，两项都是确定值（卡里的说明文字自己钉高，既不会被压也不会虚占）：
        #   内层自然高 = 上卡自然高 + CARD_GAP + 下卡自然高
        #   · 视口 ≥ 它：内层 = 视口，富余全由上卡的模板清单吸收（下面那个 stretch=1），
        #     于是下卡底 = 视口底 = 右侧对话卡底 ——「模型与密钥」与「对话」严格齐平；
        #   · 视口 < 它：整栏出纵向滚动条，两卡内容与按钮都完整可滚到，一处不裁。
        # 下卡永远不参与富余分配（stretch=0）：它一被压，「测试连接 / 保存设置」就没了。
        column.addWidget(self._build_agent_card(), 1)
        column.addWidget(self._build_settings_card(), 0)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.left_scroll = scroll
        return host

    def _card(self, title):
        card = QWidget()
        card.setProperty("card", True)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(CARD_SPACING)
        if title:
            label = QLabel(title)
            label.setProperty("section", True)
            lay.addWidget(label)
        theme.apply_card_shadow(card)
        return card, lay

    # ---------- 档案区 ----------
    def _build_agent_card(self):
        card, lay = self._card("智能体")
        self.agent_combo = QComboBox()
        theme.bind_combo_popup(self.agent_combo)
        self.agent_combo.currentIndexChanged.connect(self._on_agent_selected)
        lay.addWidget(self.agent_combo)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self.new_agent_btn = QPushButton("新建")
        self.new_agent_btn.setProperty("btn", "dashed")
        theme.bind_icon(self.new_agent_btn, "wand", 15)
        self.new_agent_btn.clicked.connect(self._on_new_agent)
        self.edit_agent_btn = QPushButton("编辑")
        self.edit_agent_btn.setProperty("btn", "ghost")
        theme.bind_icon(self.edit_agent_btn, "edit", 15)
        self.edit_agent_btn.clicked.connect(self._on_edit_agent)
        self.del_agent_btn = QPushButton("删除")
        self.del_agent_btn.setProperty("btn", "ghost")
        theme.bind_icon(self.del_agent_btn, "trash", 15)
        self.del_agent_btn.clicked.connect(self._on_delete_agent)
        btn_row.addWidget(self.new_agent_btn)
        btn_row.addWidget(self.edit_agent_btn)
        btn_row.addWidget(self.del_agent_btn)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        self.context_chips = _AutoHeightLabel("")
        self.context_chips.setProperty("hint", True)
        lay.addWidget(self.context_chips)

        self.role_preview = _AutoHeightLabel("")
        self.role_preview.setProperty("hint", True)
        lay.addWidget(self.role_preview)

        lay.addWidget(self._hint("任务模板（双击填入对话框）"))
        self.template_list = _TemplateList()
        # 不定死高度：卡片被拉长时由清单吃掉富余（多出来的是空白行，不是控件间距）
        self.template_list.setMinimumHeight(110)
        self.template_list.itemDoubleClicked.connect(self._on_template_picked)
        lay.addWidget(self.template_list, 1)

        # 卡底主操作：摆在任务模板清单的正下方。只服务「框架设计师」这一个角色
        # （_refresh_agent_detail 按 id 显隐），它出现与否本来就该改变本卡的高。
        self.framework_designer_btn = QPushButton("进入框架设计工作台")
        self.framework_designer_btn.setProperty("btn", "primary")
        theme.bind_icon(self.framework_designer_btn, "wand", 15, color="#FFFFFF")
        self.framework_designer_btn.clicked.connect(self._open_framework_designer)
        self.framework_designer_btn.hide()
        lay.addWidget(self.framework_designer_btn)
        return card

    def _build_settings_card(self):
        card, lay = self._card("模型与密钥")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(CARD_SPACING)
        grid.setColumnStretch(1, 1)
        # 行距等分只在这张卡真的拿到富余时起效（正常情况下不会，见 _build_left）：
        # 宁可拉开行距，也不让五行表单整块贴顶、底下留一大块空白。
        for row in range(5):
            grid.setRowStretch(row, 1)

        grid.addWidget(self._field_label("Base URL"), 0, 0)
        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("https://api.openai.com/v1")
        grid.addWidget(self.base_url_edit, 0, 1)

        grid.addWidget(self._field_label("模型"), 1, 0)
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("gpt-4o-mini / deepseek-chat / qwen-plus")
        grid.addWidget(self.model_edit, 1, 1)

        grid.addWidget(self._field_label("API Key"), 2, 0)
        key_row = QHBoxLayout()
        key_row.setSpacing(6)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("sk-……")
        self.key_vis_btn = QPushButton("显示")
        self.key_vis_btn.setProperty("btn", "ghost")
        self.key_vis_btn.setCheckable(True)
        self.key_vis_btn.toggled.connect(self._on_toggle_key_visible)
        key_row.addWidget(self.key_edit, 1)
        key_row.addWidget(self.key_vis_btn)
        grid.addLayout(key_row, 2, 1)

        grid.addWidget(self._field_label("温度"), 3, 0)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setDecimals(2)
        grid.addWidget(self.temp_spin, 3, 1)

        grid.addWidget(self._field_label("Max tokens"), 4, 0)
        self.maxtok_spin = QSpinBox()
        self.maxtok_spin.setRange(1, 32768)
        self.maxtok_spin.setSingleStep(128)
        grid.addWidget(self.maxtok_spin, 4, 1)
        lay.addLayout(grid, 0)      # 表单按自然高排；本卡不参与富余分配（按钮不得被顶出视口）

        self.no_save_key_check = QCheckBox("本次不保存 API Key（仅内存使用，退出即丢）")
        lay.addWidget(self.no_save_key_check)

        warn = _AutoHeightLabel("Key 经 Windows DPAPI 加密后存于 data/ai_settings.json，只有当前 Windows "
                                "账户能解开；绝不进存档/导出/日志。共用电脑请勾选「本次不保存」。")
        warn.setProperty("notice", True)
        lay.addWidget(warn)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self.test_btn = QPushButton("测试连接")
        self.test_btn.setProperty("btn", "secondary")
        theme.bind_icon(self.test_btn, "link", 15)
        self.test_btn.clicked.connect(self._on_test_connection)
        self.save_settings_btn = QPushButton("保存设置")
        self.save_settings_btn.setProperty("btn", "primary")
        theme.bind_icon(self.save_settings_btn, "save", 15, color="#FFFFFF")
        self.save_settings_btn.clicked.connect(self._on_save_settings)
        action_row.addWidget(self.test_btn)
        action_row.addWidget(self.save_settings_btn)
        lay.addLayout(action_row)

        self.status_label = _AutoHeightLabel("未测试")
        self.status_label.setProperty("saveState", "idle")
        lay.addWidget(self.status_label)
        return card

    # ---------- 对话区 ----------
    def _build_right(self):
        card, lay = self._card(None)
        head = QHBoxLayout()
        title = QLabel("对话")
        title.setProperty("section", True)
        head.addWidget(title)
        head.addStretch(1)
        self.clear_btn = QPushButton("清空")
        self.clear_btn.setProperty("btn", "ghost")
        theme.bind_icon(self.clear_btn, "close", 14)
        self.clear_btn.clicked.connect(self._on_clear)
        self.export_btn = QPushButton("导出")
        self.export_btn.setProperty("btn", "ghost")
        theme.bind_icon(self.export_btn, "export", 14)
        self.export_btn.clicked.connect(self.export_conversation)
        head.addWidget(self.clear_btn)
        head.addWidget(self.export_btn)
        lay.addLayout(head)

        self.msg_scroll = QScrollArea()
        self.msg_scroll.setWidgetResizable(True)
        self.msg_scroll.setFrameShape(QScrollArea.NoFrame)
        self.msg_host = QWidget()
        self.msg_lay = QVBoxLayout(self.msg_host)
        self.msg_lay.setContentsMargins(2, 2, 8, 2)
        self.msg_lay.setSpacing(10)
        self.msg_lay.addStretch(1)
        self.msg_scroll.setWidget(self.msg_host)
        lay.addWidget(self.msg_scroll, 1)

        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("向智能体提问……（Ctrl+Enter 发送）")
        self.input.setFixedHeight(84)
        lay.addWidget(self.input)
        send_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self.input)
        send_shortcut.activated.connect(self._on_send)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.chat_status = QLabel("")
        self.chat_status.setProperty("hint", True)
        input_row.addWidget(self.chat_status, 1)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setProperty("btn", "secondary")
        theme.bind_icon(self.stop_btn, "stop", 15)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        self.send_btn = QPushButton("发送")
        self.send_btn.setProperty("btn", "primary")
        theme.bind_icon(self.send_btn, "send", 15, color="#FFFFFF")
        self.send_btn.clicked.connect(self._on_send)
        input_row.addWidget(self.stop_btn)
        input_row.addWidget(self.send_btn)
        lay.addLayout(input_row)
        return card

    @staticmethod
    def _field_label(text):
        label = QLabel(text)
        label.setProperty("hint", True)
        return label

    @staticmethod
    def _hint(text):
        label = QLabel(text)
        label.setProperty("hint", True)
        return label

    # ---------- 档案 CRUD ----------
    def _reload_agents(self):
        self.agent_combo.blockSignals(True)
        self.agent_combo.clear()
        for agent in settings.agents():
            self.agent_combo.addItem(agent["name"], agent["id"])
        active = settings.active_agent_id()
        index = self.agent_combo.findData(active)
        self.agent_combo.setCurrentIndex(index if index >= 0 else 0)
        self.agent_combo.blockSignals(False)
        self._refresh_agent_detail()

    def _on_agent_selected(self, index):
        agent_id = self.agent_combo.itemData(index)
        if agent_id:
            settings.set_active_agent(agent_id)
        self._refresh_agent_detail()

    def _refresh_agent_detail(self):
        agent = settings.active_agent() or {}
        role = str(agent.get("role") or "").strip()
        self.role_preview.setText(role or "（未设置系统提示）")
        self.role_preview.sync_height()          # setText 非虚函数，换完文字立刻重量高
        context = agent.get("context") or {}
        attached = [CONTEXT_LABELS[key] for key in CONTEXT_KEYS if context.get(key)]
        self.context_chips.setText("附带上下文：%s" % ("、".join(attached) if attached else "无"))
        self.context_chips.sync_height()
        self.framework_designer_btn.setVisible(agent.get("id") == "framework_designer")
        self.template_list.clear()
        for template in agent.get("templates", []):
            self.template_list.addItem(template)

    def _open_framework_designer(self):
        """框架设计是可审计工作流，用独立工作台承载，不塞进普通聊天气泡。"""
        from framework_designer import FrameworkDesignerDialog
        dialog = FrameworkDesignerDialog(self)
        dialog.set_client_factory(self._make_client)
        dialog.set_session_provider(self._snapshot)
        dialog.set_framework_targets_provider(self._framework_targets_provider)
        dialog.set_framework_applier(self._framework_applier)
        self._framework_dialog = dialog
        dialog.exec()
        self._framework_dialog = None

    def _on_new_agent(self):
        dialog = AgentWizardDialog(self)
        if dialog.exec() == QDialog.Accepted:
            settings.add_agent(**dialog.values())
            self._reload_agents()

    def _on_edit_agent(self):
        agent = settings.active_agent()
        if not agent:
            return
        dialog = AgentWizardDialog(self, agent=agent)
        if dialog.exec() == QDialog.Accepted:
            settings.update_agent(agent["id"], **dialog.values())
            self._reload_agents()

    def _on_delete_agent(self):
        agent = settings.active_agent()
        if not agent:
            return
        answer = QMessageBox.question(
            self, "删除智能体", "确定删除「%s」？此操作不可撤销。" % agent["name"],
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            settings.remove_agent(agent["id"])
            self._reload_agents()

    def _on_template_picked(self, item):
        self.input.setPlainText(item.text())
        self.input.setFocus()

    # ---------- 设置 ----------
    def _load_settings_into_form(self):
        provider = settings.provider()
        self.base_url_edit.setText(str(provider.get("base_url") or ""))
        self.model_edit.setText(str(provider.get("model") or ""))
        self.key_edit.setText(str(provider.get("api_key") or ""))
        try:
            self.temp_spin.setValue(float(provider.get("temperature", 0.3)))
        except (TypeError, ValueError):
            self.temp_spin.setValue(0.3)
        try:
            self.maxtok_spin.setValue(int(provider.get("max_tokens", 1024)))
        except (TypeError, ValueError):
            self.maxtok_spin.setValue(1024)
        self.no_save_key_check.setChecked(not provider.get("persist_key", True))

    def _on_toggle_key_visible(self, checked):
        self.key_edit.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)
        self.key_vis_btn.setText("隐藏" if checked else "显示")

    def _form_provider(self):
        return {
            "base_url": self.base_url_edit.text().strip(),
            "model": self.model_edit.text().strip(),
            "api_key": self.key_edit.text().strip(),
            "temperature": self.temp_spin.value(),
            "max_tokens": self.maxtok_spin.value(),
        }

    def _on_save_settings(self):
        provider = self._form_provider()
        provider["persist_key"] = not self.no_save_key_check.isChecked()
        settings.set_provider(**provider)
        note = "设置已保存" if provider["persist_key"] else "设置已保存（本次未保存 Key）"
        self._set_status(note, "saved")

    def _set_status(self, text, level="idle"):
        self.status_label.setText(text)
        self.status_label.setProperty("saveState", level)
        theme.refresh_style(self.status_label)
        self.status_label.sync_height()         # 状态文案长短不一，重量一次

    def _make_client(self):
        provider = self._form_provider()
        return OpenAICompatClient(
            provider["base_url"], provider["api_key"], provider["model"],
            temperature=provider["temperature"], max_tokens=provider["max_tokens"],
            transport=self._transport)

    def _on_test_connection(self):
        client = self._make_client()
        ok, note = client.ready()
        if not ok:
            self._set_status(note, "error")
            return
        self._set_status("正在测试连接……", "pending")
        self._launch_worker(client, [user_message("ping")], kind="test",
                            stream=False, max_tokens=5)

    # ---------- 对话 ----------
    def _snapshot(self):
        if self._context_provider is None:
            return {}
        try:
            snapshot = self._context_provider()
        except Exception as exc:                # noqa: BLE001 取现场失败不该打断对话
            logging.warning("AI 上下文快照获取失败：%s", exc)
            return {}
        return snapshot if isinstance(snapshot, dict) else {}

    def _add_bubble(self, role, text=""):
        bubble = _MessageBubble(role, text)
        self.msg_lay.insertWidget(self.msg_lay.count() - 1, bubble)
        self._scroll_to_bottom()
        return bubble

    def _scroll_to_bottom(self):
        bar = self.msg_scroll.verticalScrollBar()
        QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    def _on_send(self):
        if self._worker is not None:
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        client = self._make_client()
        ok, note = client.ready()
        if not ok:
            self._add_bubble("error", note)
            return
        agent = settings.active_agent()
        snapshot = self._snapshot()
        self._current_text = text
        self._add_bubble("user", text)
        self.input.clear()
        messages = ai_context.build_messages(agent, text, snapshot, self._history)
        self._launch_worker(client, messages, kind="chat", stream=True)

    def _launch_worker(self, client, messages, kind, stream=True, **overrides):
        self._worker_kind = kind
        self._set_busy(True)
        if kind == "chat":
            self._stream_bubble = self._add_bubble("assistant", "")
            self.chat_status.setText("正在生成……")
        else:
            self._stream_bubble = None
        self._worker = AiWorker(client, messages, parent=self, stream=stream, **overrides)
        self._worker.token.connect(self._on_token)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_token(self, piece):
        if self._stream_bubble is not None:
            self._stream_bubble.append(piece)
            self._scroll_to_bottom()

    def _on_done(self, result):
        result = result or {}
        kind, self._worker_kind = self._worker_kind, None
        self._worker = None
        self._set_busy(False)
        if kind == "test":
            self._set_status("连接成功：模型可正常应答", "saved")
            return
        bubble = self._stream_bubble
        self._stream_bubble = None
        text = result.get("text", "")
        cancelled = bool(result.get("cancelled"))
        if bubble is not None:
            if not text and not bubble.text():
                bubble.set_text("（模型未返回内容）")
            bubble.set_usage(result.get("usage"), cancelled)
        self.chat_status.setText("已停止" if cancelled else "完成")
        self._history.append({"role": "user", "content": self._current_text})
        self._history.append({"role": "assistant", "content": text})
        self._scroll_to_bottom()

    def _on_error(self, message):
        kind, self._worker_kind = self._worker_kind, None
        self._worker = None
        self._set_busy(False)
        if kind == "test":
            self._set_status(message, "error")
            return
        bubble = self._stream_bubble
        self._stream_bubble = None
        if bubble is not None and not bubble.text():
            bubble.as_error(message)
        else:
            self._add_bubble("error", message)
        self.chat_status.setText("出错")

    def _on_stop(self):
        if self._worker is not None:
            self._worker.cancel()
            self.chat_status.setText("正在停止……")

    def _set_busy(self, busy):
        self.input.setEnabled(not busy)
        self.send_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.test_btn.setEnabled(not busy)
        self.save_settings_btn.setEnabled(not busy)

    def _on_clear(self):
        if self._worker is not None:
            return
        self._history = []
        while self.msg_lay.count() > 1:
            item = self.msg_lay.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.chat_status.setText("")
        self._add_bubble("assistant", "对话已清空。开始新的一轮提问吧。")

    # ---------- 导出（文件菜单转发到此公开方法） ----------
    def export_conversation(self):
        """把当前对话导出成 Markdown；走 user_state 的保存对话框包装器。"""
        if not self._history:
            QMessageBox.information(self, "导出对话", "当前没有可导出的对话内容。")
            return
        path = user_state.save_file_name(
            self, user_state.BUCKET_EXPORT, "导出 AI 对话 / 分析规划",
            suggested="ai_conversation.md",
            name_filter="Markdown (*.md);;文本文件 (*.txt)")
        if not path:
            return
        agent = settings.active_agent() or {}
        lines = ["# AI 助手对话记录", "",
                 "- 智能体：%s" % agent.get("name", ""),
                 "- 模型：%s" % self.model_edit.text().strip(), ""]
        for turn in self._history:
            who = "你" if turn.get("role") == "user" else "AI 助手"
            lines += ["## %s" % who, "", str(turn.get("content", "")), ""]
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        QMessageBox.information(self, "导出完成", "已导出到：\n%s" % path)
