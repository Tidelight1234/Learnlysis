# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""数据编码模块 DataEncoderModule：维度部分用 dimension_panel 里的通用面板。

三个时间胶囊、时间轴骨架（行时间/格位两个平行表）与表格量宽三件事一律取自
 timeline_axis，与「小组交互」页共用同一份实现，不在本页另写一套。
"""

import copy
import re

import pandas as pd
from PySide6.QtWidgets import (QWidget, QSplitter, QPushButton, QLineEdit, QVBoxLayout,
                             QHBoxLayout, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMessageBox, QMenu, QAbstractItemView, QComboBox,
                             QListView, QStyledItemDelegate, QAbstractItemDelegate, QMainWindow)
from PySide6.QtCore import Qt, QEvent, QTimer, Signal
from openpyxl.styles import Alignment, Font

import theme
import user_state
from language import translate_widget_tree
from dimension_panel import DimensionPanel
from coding_assist import CodingSuggestBar
from timeline_axis import (FIXED_COL_COUNT, HISTORY_LIMIT, POPUP_MIN_WIDTH, TimeAxis,
                           TimeControls, apply_column_widths, content_column_width,
                           fixed_column_widths, format_seconds, make_time_field, parse_time)

# 页标题里的成员序号：「编码个体3」→ S3（小组交互页的 actor 候选、分析列名同出一源）
STUDENT_TITLE_RE = re.compile(r"(?:个体|学生|coded individual|student|s)\s*(\d+)", re.IGNORECASE)


class CodeCellDelegate(QStyledItemDelegate):
    """表格单元格编辑器：维度列用编码下拉（可选可输），注释列用文本框。

    时间/计时两列的 item 不带 ItemIsEditable，Qt 不会为它们创建编辑器，
    因此这里只需处理可编辑的两类列。
    """

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.owner = owner

    def createEditor(self, parent, option, index):
        card = self.owner.dimension_card_at(index.column())
        if card is None:
            return super().createEditor(parent, option, index)
        combo = QComboBox(parent)
        combo.setProperty("cellEditor", True)  # 命中紧凑 QSS，不撑高行
        combo.setEditable(True)  # 允许临时填入框架外的编码
        combo.setInsertPolicy(QComboBox.NoInsert)
        # 下拉列表沿用弹层菜单的圆角样式（objectName 命中 QSS 规则）；
        # 单元格往往只有几十字宽，弹窗需自带最小宽度才看得清编码项
        view = QListView()
        view.setObjectName("cellPopup")
        view.setSpacing(2)
        view.setMinimumWidth(POPUP_MIN_WIDTH)
        combo.setView(view)
        theme.bind_combo_popup(combo)
        combo.addItems(card.code_values())
        current = index.data(Qt.DisplayRole) or ""
        if current and combo.findText(current) < 0:
            combo.insertItem(0, current)
        combo.setCurrentText(current)
        combo.setMinimumContentsLength(6)
        combo.lineEdit().selectAll()
        # 选中下拉项即提交并收起编辑器（与 QItemEditorFactory 默认行为一致）
        combo.activated.connect(lambda _=None, c=combo: self.commit_when_popup_closed(c))
        return combo

    def commit_when_popup_closed(self, editor):
        """Qt 在展开/收起弹窗瞬间也会发 activated，只在弹窗已关（用户真选完）时提交"""
        view = editor.view()
        if view is not None and view.isVisible():
            return
        self.commitData.emit(editor)
        self.closeEditor.emit(editor, QAbstractItemDelegate.SubmitModelCache)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText().strip(), Qt.EditRole)
        else:
            super().setModelData(editor, model, index)


class DataEncoderModule(QWidget):
    #: 现场改动通知：记录/删行/改单元格/撤销/清空 都会推一份历史，推的那一刻就是真改动。
    #: 时间胶囊、维度框架、注释框这些没进撤销栈的改动由主窗口直接接控件信号。
    content_changed = Signal()

    def __init__(self, group_name="", tabmodule=None):
        super().__init__()
        self.group_name = group_name
        self.selected_data = []
        # 撤销栈：每次改动前压一份现场，栈里有几条就能撤销几次
        self.history = []
        self.row_times = []  # 每行时间：None=顺序行（按格位推算），int=固定秒数（当前时间点插入的行）
        # 时间轴骨架上的“第几格”：只有顺序行占格（固定时间点行为 None），删行不回收，当隐藏格用
        self.row_positions = []
        # 时间轴骨架算法：不持数据，按属性名现读现写上面三个平行表（见 timeline_axis.TimeAxis）
        self.axis = TimeAxis(self)
        self.tabmodule = tabmodule
        self.video_player = None
        self._loading_table = False
        # 撤销恢复期间屏蔽 framework_replaced 的默认收尾（它会抹掉行内容）
        self._restoring = False
        self.time_ctrl = None
        self.initUI()

    # ---------- 页身份：小组交互页据此认成员 ----------
    def tab_title(self):
        """本页在内层标签栏上的标题；不在标签栏里（测试单页）时退回 group_name。"""
        if self.tabmodule is not None:
            index = self.tabmodule.indexOf(self)
            if index >= 0:
                return self.tabmodule.tabText(index)
        return self.group_name

    def student_id(self):
        """页标题 → 成员号 ``S{n}``；标题里没有数字就原样回显标题。"""
        title = (self.tab_title() or "").strip()
        match = STUDENT_TITLE_RE.search(title)
        return "S%s" % match.group(1) if match else title

    def initUI(self):
        # 主布局
        main_layout = QHBoxLayout(self)
        # 左侧区域：维度区域 → 按钮区域一 → 注释输入框 → 按钮区域二
        left_layout = QVBoxLayout()
        # ---- 第一部分：维度编辑（与聚类/滞后共用同一块面板：白底卡片 + 导入框架 / 添加维度 / 清空）----
        self.dim_panel = DimensionPanel(self, title="维度编辑",
                                        sources_provider=self._framework_sources,
                                        clear_tooltip="清空编码框架与已记录的编码数据")
        self.import_btn = self.dim_panel.import_btn
        self.add_dimension_btn = self.dim_panel.add_dimension_btn
        self.clear_btn = self.dim_panel.clear_btn
        self.dim_panel.dimension_added.connect(self._on_dimension_added)
        self.dim_panel.dimension_removed.connect(self._on_dimension_removed)
        self.dim_panel.dimension_renamed.connect(self._on_dimension_renamed)
        self.dim_panel.framework_replaced.connect(self._on_framework_replaced)
        # ---- 按钮区域一与注释之间：起始时间 / 当前时间 / 时间间隔（可编辑胶囊）----
        self.time_ctrl = TimeControls()
        self.start_time_edit = self.time_ctrl.start_edit
        # 当前编码时间点：默认跟随视频播放时间，用户编辑后参与记录的插入/覆盖
        self.timepoint_edit = self.time_ctrl.timepoint_edit
        self.time_unit_entry = self.time_ctrl.unit_edit
        self.time_ctrl.params_changed.connect(self.validate_time_unit)
        self.time_ctrl.params_changed.connect(self.update_table)
        # ---- 第三部分：注释输入框（加高）----
        self.comment_entry = QLineEdit()
        self.comment_entry.setPlaceholderText("输入注释...")
        self.comment_entry.setFixedHeight(56)
        # ---- 第四部分：按钮区域二（记录 / 撤销 / 输出）----
        btn2_widget = QWidget()
        self.action_buttons_widget = btn2_widget
        btn2_layout = QHBoxLayout(btn2_widget)
        btn2_layout.setContentsMargins(0, 0, 0, 0)
        self.continue_btn = QPushButton('记录')
        self.continue_btn.setProperty("btn", "secondary")
        self.undo_btn = QPushButton('撤销')
        self.undo_btn.setProperty("btn", "secondary")
        self.output_btn = QPushButton('输出')
        self.output_btn.setProperty("btn", "secondary")
        theme.bind_icon(self.output_btn, "export", 16)
        for btn in (self.continue_btn, self.undo_btn, self.output_btn):
            btn2_layout.addWidget(btn, 1)
        # 右侧结果区域：只剩表格，删行改由表格自身承担（选中行 + Delete / 右键菜单）
        self.result_widget = QWidget()
        self.result_widget.setProperty("card", True)
        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(14, 12, 14, 12)
        result_layout.setSpacing(8)
        self.result_table = QTableWidget()
        self._normal_row_height = self.result_table.verticalHeader().defaultSectionSize()
        self.result_table.setColumnCount(FIXED_COL_COUNT + 2)
        self.result_table.setHorizontalHeaderLabels(["时间", "计时", "维度1", "注释"])
        self.result_table.horizontalHeader().setMinimumSectionSize(40)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # 整行选中 + 多选，配合 Delete 键完成删行
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # 双击进入编辑：维度列弹编码下拉，注释列弹文本框（时间/计时列 item 不可编辑）
        self.result_table.setEditTriggers(QAbstractItemView.DoubleClicked
                                          | QAbstractItemView.EditKeyPressed)
        self.result_table.setItemDelegate(CodeCellDelegate(self, self.result_table))
        self.result_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.result_table.customContextMenuRequested.connect(self.show_row_menu)
        self.result_table.installEventFilter(self)
        self.result_table.viewport().setToolTip("双击单元格可修改编码/注释；选中行后按 Delete 键或右键菜单可删除该行")
        result_layout.addWidget(self.result_table)
        self.result_widget.setLayout(result_layout)
        # 卡片外再套一层留白槽：白底不再顶在 splitter 的分界线上（横排时左边、纵排时上边
        # 各让开 10px 以上）。槽还顺带替投影腾出绘制空间——QGraphicsDropShadowEffect 的
        # 输出会被裁进控件自身矩形，挂在铺满整格的卡片上等于没阴影，挂在槽上才画得出来。
        self.result_slot = QWidget()
        slot_layout = QHBoxLayout(self.result_slot)
        slot_layout.setContentsMargins(12, 11, 10, 10)
        slot_layout.addWidget(self.result_widget)
        theme.apply_card_shadow(self.result_slot)
        # 组装左侧布局：维度面板（含三按钮）→ 时间点 → 注释输入框 → 按钮区域二
        left_widget = QWidget()
        left_layout.addWidget(self.dim_panel, 1)
        left_layout.addWidget(self.time_ctrl)
        left_layout.addWidget(self.comment_entry)
        # Phase 1 编码辅助：按已录序列给「下一步建议」，点选即填入维度勾选态，不自动写表
        self.suggest_bar = CodingSuggestBar(self)
        left_layout.addWidget(self.suggest_bar)
        # Phase 2 AI 辅助编码入口：抽关键帧 + 转写交多模态模型给建议码，逐条复核后写回
        self.auto_code_btn = QPushButton("AI 辅助编码（视频多模态）")
        self.auto_code_btn.setProperty("btn", "dashed")
        self.auto_code_btn.setToolTip("需在「AI 助手」页配置 OpenAI 兼容 API；抽帧需可选依赖 "
                                      "opencv-python 或 ffmpeg，缺席时对话框会写明降级原因")
        theme.bind_icon(self.auto_code_btn, "wand", 16)
        self.auto_code_btn.clicked.connect(self._open_auto_coding)
        left_layout.addWidget(self.auto_code_btn)
        # Phase 3 本地训练入口：在你已标注的编码上离线训练轻量分类头，预测仍作建议码写回
        self.local_train_btn = QPushButton("本地训练辅助编码（离线）")
        self.local_train_btn.setProperty("btn", "dashed")
        self.local_train_btn.setToolTip("用已录编码在本地训练分类头（不联网）；视听重后端 mediapipe/"
                                        "faster-whisper 缺席时对话框会写明安装命令，仍可用序列特征训练")
        theme.bind_icon(self.local_train_btn, "robot", 16)
        self.local_train_btn.clicked.connect(self._open_local_train)
        left_layout.addWidget(self.local_train_btn)
        left_layout.addWidget(btn2_widget)
        left_widget.setLayout(left_layout)
        # 主布局
        encodersplitter = QSplitter(Qt.Horizontal)
        # 将两个控件添加到分割器中
        encodersplitter.addWidget(left_widget)
        encodersplitter.addWidget(self.result_slot)

        # 设置分割器的拉伸因子（可选）
        # 编码区左列适度加宽；宽度由 splitter 约束在标签页内容区边界内
        left_widget.setMinimumWidth(280)
        self.result_widget.setMinimumWidth(100)
        encodersplitter.setSizes([440, 560])
        main_layout.addWidget(encodersplitter)
        # 初始化信号连接（添加维度由通用面板自己接）
        self.continue_btn.clicked.connect(self.continue_)
        self.undo_btn.clicked.connect(self.undo)
        self.clear_btn.clicked.connect(self.clear)
        self.output_btn.clicked.connect(self.output)
        # 初始化模块（时间胶囊的两个 textChanged 已由 TimeControls 转发 params_changed）
        self.result_table.itemChanged.connect(self.on_item_changed)
        # 无需确认即可开始编码：按钮可用性只由“有没有维度 / 有没有数据”决定
        self.update_table_columns()
        self.update_action_state()
        # 分割器引用：横/纵布局切换与宽度重分都靠它
        self.splitter = encodersplitter
        # 编码区排列方向：False=横（左编码右表格），True=纵（上编码下表格）
        self._stacked = False
        self.splitter.splitterMoved.connect(self.on_splitter_moved)
        self.set_ai_coding_visible(user_state.state.coding_ai_visible())
        self.set_coding_font_size(user_state.state.coding_font_size())
        self._install_free_layout(main_layout, left_widget)

    def set_ai_coding_visible(self, visible):
        """只切换三个辅助编码入口，不改动编码数据。"""
        self._ai_block_visible = bool(visible)
        for widget in (self.suggest_bar, self.auto_code_btn, self.local_train_btn):
            widget.setVisible(bool(visible))
        if hasattr(self, '_layout_blocks'):
            self._arrange_free_layout()

    LAYOUT_NAMES = {'dimensions': '维度编辑', 'controls': '时间、注释及按钮',
                    'ai': 'AI 编码辅助', 'table': '表格'}

    def _install_free_layout(self, main_layout, old_left):
        """Move the existing widgets into four movable regions without recreating them."""
        main_layout.removeWidget(self.splitter)
        self.splitter.setParent(None)
        old_left.layout().removeWidget(self.dim_panel)
        old_left.layout().removeWidget(self.time_ctrl)
        old_left.layout().removeWidget(self.comment_entry)
        old_left.layout().removeWidget(self.suggest_bar)
        old_left.layout().removeWidget(self.auto_code_btn)
        old_left.layout().removeWidget(self.local_train_btn)
        old_left.layout().removeWidget(self.action_buttons_widget)
        old_left.setParent(None)
        self.result_slot.setParent(None)
        groups = {
            'dimensions': [self.dim_panel],
            'controls': [self.time_ctrl, self.comment_entry, self.action_buttons_widget],
            'ai': [self.suggest_bar, self.auto_code_btn, self.local_train_btn],
            'table': [self.result_slot],
        }
        self._layout_blocks = {}
        for key, widgets in groups.items():
            panel = QWidget()
            panel.setMinimumWidth(180)
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(5, 5, 5, 5)
            title = QPushButton(self.LAYOUT_NAMES[key] + '  ▾')
            title.setContextMenuPolicy(Qt.CustomContextMenu)
            title.clicked.connect(lambda _checked=False, block=key, button=title:
                                  self._show_block_menu(block, button))
            layout.addWidget(title)
            for widget in widgets:
                layout.addWidget(widget, 1 if widget in (self.dim_panel, self.result_slot) else 0)
            if key == 'ai':
                layout.addStretch()
            self._layout_blocks[key] = panel
        state = user_state.state.coding_layout()
        self._block_order = list(state['order'])
        self._span_block = state['span']
        self._span_side = state['span_side']
        self._free_root = None
        self._free_layout_host = QWidget()
        self._free_host_layout = QVBoxLayout(self._free_layout_host)
        self._free_host_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._free_layout_host)
        self._arrange_free_layout()
        self._update_controls_min_width()

    def _show_block_menu(self, block, button):
        menu = theme.bind_menu_popup(QMenu(button))
        for target in self._block_order:
            if target != block and (self._ai_block_visible or target != 'ai'):
                action = menu.addAction('与%s交换位置' % self.LAYOUT_NAMES[target])
                action.triggered.connect(lambda _checked=False, other=target:
                                         self.swap_layout_blocks(block, other))
        if block != 'ai':
            menu.addSeparator()
            action = menu.addAction('隐藏 AI 时占两格')
            action.triggered.connect(lambda _checked=False: self.set_span_block(block))
        menu.addSeparator()
        for side, label in (('start', '合并在上方／左侧'), ('end', '合并在下方／右侧')):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self._span_side == side)
            action.triggered.connect(lambda _checked=False, value=side:
                                     self.set_span_side(value))
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    def swap_layout_blocks(self, first, second):
        if (first == second or first not in self._block_order or second not in self._block_order
                or (not self._ai_block_visible and 'ai' in (first, second))):
            return
        if self._ai_block_visible:
            a, b = self._block_order.index(first), self._block_order.index(second)
            self._block_order[a], self._block_order[b] = self._block_order[b], self._block_order[a]
        else:
            others = [key for key in self._block_order
                      if key not in ('ai', self._span_block)]
            shown = ([self._span_block] + others if self._span_side == 'start'
                     else others + [self._span_block])
            a, b = shown.index(first), shown.index(second)
            shown[a], shown[b] = shown[b], shown[a]
            self._span_block = shown[0] if self._span_side == 'start' else shown[-1]
            positions = (i for i, key in enumerate(self._block_order) if key != 'ai')
            for key in shown:
                self._block_order[next(positions)] = key
        self._save_free_layout()
        self._arrange_free_layout()

    def _save_free_layout(self):
        user_state.state.set_coding_layout(
            self._block_order, self._span_block, self._span_side)

    def set_span_side(self, side):
        if side in ('start', 'end'):
            self._span_side = side
            self._save_free_layout()
            self._arrange_free_layout()

    def set_span_block(self, block):
        if block in ('dimensions', 'controls', 'table'):
            self._span_block = block
            self._save_free_layout()
            self._arrange_free_layout()

    def reset_free_layout(self):
        self._block_order = ['dimensions', 'table', 'controls', 'ai']
        self._span_block = 'table'
        self._span_side = 'end'
        self._save_free_layout()
        self._arrange_free_layout()

    def _arrange_free_layout(self):
        if not hasattr(self, '_layout_blocks'):
            return
        old = self._free_root
        ratios = ([s.sizes() for s in self._free_rows], self._free_root.sizes()) if old else None
        if old:
            self._free_host_layout.removeWidget(old)
            for panel in self._layout_blocks.values():
                panel.setParent(None)
            old.setParent(None)
            old.deleteLater()
        visible = self._ai_block_visible
        # Hidden AI: the selected block occupies the chosen edge; the other two
        # retain their saved relative order in the opposite row or column.
        others = [x for x in self._block_order if x not in ('ai', self._span_block)]
        order = (list(self._block_order) if visible else
                 ([self._span_block] + others if self._span_side == 'start'
                  else others + [self._span_block]))
        vertical_span = not visible and self._stacked
        root = QSplitter(Qt.Horizontal if vertical_span else Qt.Vertical)
        root.setProperty('encoderGrid', True)
        root.setHandleWidth(8)
        rows = []
        for row_index in range(2):
            row = QSplitter(Qt.Vertical if vertical_span else Qt.Horizontal)
            row.setProperty('encoderGrid', True)
            row.setHandleWidth(8)
            if visible:
                keys = order[:2] if row_index == 0 else order[2:]
            elif self._span_side == 'start':
                keys = order[:1] if row_index == 0 else order[1:]
            else:
                keys = order[:2] if row_index == 0 else order[2:]
            for key in keys:
                row.addWidget(self._layout_blocks[key])
            root.addWidget(row)
            rows.append(row)
        self._free_root, self._free_rows = root, rows
        self.splitter = root
        self._free_host_layout.addWidget(root)
        if ratios and visible:
            for row, sizes in zip(rows, ratios[0]):
                if len(sizes) == row.count():
                    row.setSizes(sizes)
            root.setSizes(ratios[1])
        else:
            root.setSizes([3, 2])
        self.dim_panel.schedule_relayout_cards()

    def _update_controls_min_width(self):
        """Keep time values and action labels readable at every coding font size."""
        time_fields = (self.start_time_edit, self.timepoint_edit, self.time_unit_entry)
        for field in time_fields[:2]:
            field.setMinimumWidth(field.fontMetrics().horizontalAdvance('00:00:00') + 32)
        self.time_unit_entry.setMinimumWidth(
            self.time_unit_entry.fontMetrics().horizontalAdvance('0000') + 32)
        time_width = sum(field.minimumWidth() for field in time_fields) + 16
        buttons = (self.continue_btn, self.undo_btn, self.output_btn)
        button_width = sum(max(button.sizeHint().width(),
                               button.fontMetrics().horizontalAdvance(button.text()) + 44)
                           for button in buttons) + 16
        self._layout_blocks['controls'].setMinimumWidth(max(180, time_width, button_width) + 10)

    def set_coding_font_size(self, size):
        """预设或 10–20px 自定义字号，同步调整编码区密度。"""
        size = user_state.state._valid_coding_size(size)
        self.dim_panel.setStyleSheet(theme.coding_dimension_style(size))
        self.result_table.setStyleSheet(theme.coding_table_style(size))
        self.time_ctrl.setStyleSheet(theme.coding_input_style(size))
        self.comment_entry.setStyleSheet(theme.coding_input_style(size))
        self.comment_entry.setFixedHeight(
            56 if size == "normal" else theme.coding_density_value(size, 48, 56))
        action_style = theme.coding_action_button_style(size)
        self.action_buttons_widget.setStyleSheet(action_style)
        for button in (self.auto_code_btn, self.local_train_btn):
            button.setStyleSheet(action_style)
        self.result_table.setProperty("codingDensity", size)
        self.result_table.horizontalHeader().setMinimumSectionSize(
            theme.coding_density_value(size, 32, 40))
        self.result_table.verticalHeader().setDefaultSectionSize(
            self._normal_row_height if size == "normal" else
            theme.coding_density_value(size, 28, self._normal_row_height))
        self.dim_panel.set_compact(size)
        if hasattr(self, '_layout_blocks'):
            self._update_controls_min_width()
        # 字号与内边距样式会异步改变滚动区可用宽度；待样式几何稳定后再强制排一次。
        self.dim_panel.schedule_relayout_cards()
        self.action_buttons_widget.updateGeometry()
        self.apply_column_widths()

    def eventFilter(self, obj, event):
        """表格内的 Delete/Backspace 直接删除选中行（替代原“删除行”按钮）"""
        if obj is self.result_table:
            if event.type() == QEvent.KeyPress \
                    and event.key() in (Qt.Key_Delete, Qt.Key_Backspace) \
                    and self.selected_table_rows():
                self.delete_selected_rows()
                return True
            if event.type() == QEvent.Resize and not self._loading_table:
                # 分割器/窗口改变宽度后重新均分编码列，否则列宽会停在旧值
                self.apply_column_widths()
        return super().eventFilter(obj, event)

    # ---------- 三个时间胶囊：实体在 TimeControls，本页只留旧名转发 ----------
    def parse_time(self, text):
        """文本 → 总秒数，非法返回 None（与「小组交互」页共用同一份解析）"""
        return parse_time(text)
    def format_seconds(self, seconds):
        """总秒数 → HH:MM:SS：三个时间胶囊与表格时间列共用这一种写法（支持超过一小时）"""
        return format_seconds(seconds)
    def _make_time_field(self, text):
        return make_time_field(text, self)
    def parse_start_seconds(self):
        """起始时间胶囊 → 总秒数；空或半截输入返回 None"""
        return self.time_ctrl.parse_start_seconds()
    def get_start_seconds(self):
        """时间轴原点：非法或输入中时沿用上一次的合法值，不让表格跳回 0"""
        return self.time_ctrl.start_seconds()
    def set_start_seconds(self, seconds):
        self.time_ctrl.set_start_seconds(seconds)
    def validate_start_time(self):
        self.time_ctrl.validate_start()
    def parse_interval_seconds(self):
        """时间间隔胶囊 → 秒数；空/0/超上限返回 None（0 会让全表挤在同一时刻）"""
        return self.time_ctrl.parse_interval_seconds()
    def interval_seconds(self):
        """时间轴步长：非法或输入中时沿用上一次的合法值，所以恒在 1..INTERVAL_MAX"""
        return self.time_ctrl.interval()
    def validate_time_unit(self):
        self.time_ctrl.validate_unit()
    @property
    def _timepoint_edited(self):
        """手动编辑当前时间的标记：实体归 TimeControls，旧属性名留给主窗口与测试"""
        return self.time_ctrl.edited
    @_timepoint_edited.setter
    def _timepoint_edited(self, value):
        self.time_ctrl.set_edited(value)
    def _hint(self, text, msec=5000):
        """轻提示走主窗口状态栏：录入节奏中不弹模态窗挡手"""
        win = self.window()
        if isinstance(win, QMainWindow):
            win.statusBar().showMessage(text, msec)

    # ---------- 当前时间胶囊与播放头的联动（跟随策略在 TimeControls） ----------
    def set_video_player(self, player):
        """接入视频模块：当前时间默认跟随播放头动态更新，不影响表格。"""
        self.video_player = player
        self.time_ctrl.set_video_player(player)
    def show_timepoint(self, seconds):
        self.time_ctrl.show_timepoint(seconds)
    def _on_timepoint_text_edited(self, _text):
        self.time_ctrl.mark_timepoint_edited()
    def _reset_timepoint(self):
        """记录后当前时间回到默认（跟随视频）状态，不再影响表格"""
        self.time_ctrl.reset_to_playback()

    # ---------- 时间轴骨架：顺序行各占一格，格位 → 行时间（算法在 TimeAxis） ----------
    def align_row_states(self):
        """行时间/格位两个平行表补齐到行数，并维持“顺序行各占一格、固定时间点行不占”的不变式"""
        self.axis.align_row_states()
    def computed_row_times(self):
        """逐行计算表格时间：顺序行 = 起始时间 +（格位 - 表头格位）× 间隔，固定时间点行保持原值"""
        return self.axis.computed_row_times()
    def header_row_index(self):
        """起始时间行的行号：占用骨架表头那一格的顺序行（它的时间就是起始时间）"""
        return self.axis.header_row_index()
    def header_row_locked(self, rows=None):
        """起始时间行的删除保护：它是时间轴原点，删掉会让其余行整体重锚

        唯一放行的情况是选区覆盖全表（等价于清空表格数据）；部分选区里带着表头行就不删。
        """
        if rows is None:
            rows = self.selected_table_rows()
        return self.axis.header_row_locked(rows)

    # ---------- 横/纵布局：只换 splitter 方向，控件与信号不重建 ----------
    def on_splitter_moved(self):
        """分割器移动后让维度卡片按新宽度重排编码行"""
        for card in self.dimension_cards:
            card.update_block_layout()
    def is_stacked(self):
        """当前编码区是否上下排列。"""
        return self._stacked
    def set_stacked_layout(self, stacked):
        """横/纵布局切换：横=左编码右表格，纵=上编码下表格。

        只改承载两栏内容的 splitter 方向：QSplitter 换向会原样保留子件，控件、
        信号连接、表格列宽逻辑都不用重建。换向后按新的轴向重分两栏尺寸，再走
        一次 on_splitter_moved 让维度卡片按新宽度重排编码行。
        """
        stacked = bool(stacked)
        if stacked == self._stacked:
            return
        self._stacked = stacked
        if hasattr(self, '_layout_blocks'):
            self._block_order = (['dimensions', 'controls', 'ai', 'table'] if stacked else
                                 ['dimensions', 'table', 'controls', 'ai'])
            self._save_free_layout()
            self._arrange_free_layout()
            return
        self.splitter.setOrientation(Qt.Vertical if stacked else Qt.Horizontal)
        self._rebalance_split()
    def _rebalance_split(self):
        """按当前轴向重分两栏尺寸；页面尚未显示（拿不到尺寸）时下一拍再试。"""
        span = self.splitter.height() if self._stacked else self.splitter.width()
        if span <= 1:
            QTimer.singleShot(0, self._rebalance_split)
            return
        # 纵排时上栏要装下维度面板 + 两排按钮 + 时间行 + 注释，占多数空间
        first = int(span * (0.62 if self._stacked else 0.44))
        self.splitter.setSizes([first, max(span - first, 1)])
        self.on_splitter_moved()
        # 换向和 setSizes() 只提交 Qt 布局请求；此刻卡片常仍报告换向前的宽度。
        # 小/自定义小字号每行可容纳更多编码，旧宽度算出的列数尤其容易被缓存。
        # 下一事件循环待新几何生效后强制清缓存重排，确保横/纵切换真正自适应。
        self.dim_panel.schedule_relayout_cards()

    # ---------- 存档：只存推出表格所需的最小事实，不存表格快照也不存撤销栈 ----------
    def to_dict(self):
        """将模块状态转换为字典"""
        data = {
            "dimension_cards": [],  # 下面按通用面板的框架现填
            "selected_data": copy.deepcopy(self.selected_data),
            "time_unit": self.time_unit_entry.text(),
            "start_seconds": self.get_start_seconds(),
            "row_times": list(self.row_times),  # 固定时间点行的时间（None=顺序行）
            "row_positions": list(self.row_positions),  # 时间轴骨架格位：隐藏格、固定时间点行（None）靠它还原
            "comment": self.comment_entry.text(),  # 保存注释输入框内容
        }
        # 表格内容（含时间/计时列）完全由 selected_data + row_times + row_positions + 时间参数推导，不再另存一份
        # 框架直接取通用面板的 framework()，不必再伸手进卡片拼一遍编码行
        data["dimension_cards"] = [{"name": name, "codes": list(codes)}
                                   for name, codes in self.dim_panel.framework()]
        return data

    def from_dict(self, data):
        """从字典恢复模块状态"""
        # 存档里的空编码行不恢复（既不能勾选记录，又会在下拉里多出空白候选）
        items = [(str(mod.get("name", "")),
                  [str(code) for code in mod.get("codes", []) if str(code).strip()])
                 for mod in data.get("dimension_cards", [])]
        self.dim_panel.apply_framework(items)

        if "start_seconds" in data:
            self.set_start_seconds(data["start_seconds"])
        else:
            # 兼容旧存档：startmin/startsec（分:秒）
            try:
                legacy = int(data.get("startmin", 0) or 0) * 60 + int(data.get("startsec", 0) or 0)
            except (ValueError, TypeError):
                legacy = 0
            self.set_start_seconds(legacy)
        self.time_unit_entry.setText(data.get("time_unit", "1"))
        self.comment_entry.setText(data.get("comment", ""))  # 恢复注释输入框内容

        self.selected_data = [list(row) for row in data.get("selected_data", [])]
        # 恢复行时间覆盖与格位（旧存档无这些字段时由 align_row_states 按行序补齐）
        self.row_times = list(data.get("row_times", []))[:len(self.selected_data)]
        self.row_times += [None] * (len(self.selected_data) - len(self.row_times))
        stored = list(data.get("row_positions", []))[:len(self.selected_data)]
        if not stored and self.row_times:
            # 更早的存档没记格位：那时顺序行的时间就是「起始 + 行序号 × 间隔」，按行号还原骨架
            stored = [i if t is None else None for i, t in enumerate(self.row_times)]
        # 骨架只收顺序行：旧存档里固定时间点行也占了格位，读进来一并抹掉
        self.row_positions = [None if t is not None else p
                              for t, p in zip(self.row_times, stored)]
        self.align_row_states()
        self._timepoint_edited = False
        # 列名与行内容按当前维度重建，旧存档里的表格列快照直接忽略
        self.normalize_row_widths()
        self.update_table_columns()
        self.update_table()
        # 读档等于换了整套现场，之前的撤销栈不再适用
        self.history = []
        self.update_action_state()

    # ---------- 维度框架：增删、导入、复制统一交给通用面板 ----------
    @property
    def dimension_cards(self):
        """维度卡片列表（旧代码与测试仍按此名访问，实体由通用面板持有）"""
        return self.dim_panel.cards
    def _framework_sources(self):
        """可复制的框架来源：其余编码标签页（聚类/滞后等由主窗口追加）。"""
        sources = []
        if self.tabmodule:
            for i in range(self.tabmodule.count()):
                widget = self.tabmodule.widget(i)
                if widget is self or not isinstance(widget, DataEncoderModule):
                    continue
                if widget.dim_panel.has_framework():
                    sources.append((self.tabmodule.tabText(i), widget.dim_panel.framework()))
        return sources
    def _on_dimension_added(self, _card):
        self.normalize_row_widths()
        self.update_table_columns()
        self.update_table()
        self.update_action_state()

    def _on_dimension_removed(self, index):
        for row in self.selected_data:
            if index < len(row) - 1:  # 注释列之前的维度列才删，避免误删注释
                row.pop(index)
        self.normalize_row_widths()
        self.update_table_columns()
        self.update_table()
        self.update_action_state()

    def _on_dimension_renamed(self):
        # 维度名即表格列名，随时改名都要重建表头
        self.update_table_columns()
        self.update_table()

    def _on_framework_replaced(self):
        """导入/复制/读档换了整套框架：旧的逐列编码值不再适用"""
        if self._restoring:
            return  # 撤销恢复自己会重画，不能被这份收尾抹掉刚恢复的行内容
        self.reset_dimension_cells()
        self.update_table_columns()
        self.update_table()
        self.update_action_state()

    def normalize_row_widths(self):
        """保证每行宽度 = 维度列数 + 注释列，避免维度增删或旧存档导致列错位"""
        target = len(self.dimension_cards) + 1
        for row in self.selected_data:
            while len(row) < target:
                row.insert(max(len(row) - 1, 0), "")
            if len(row) > target:
                del row[target - 1:-1]  # 多余的维度列就地删掉，保留末列注释
    def dimension_card_at(self, col):
        """表格列号 → 维度卡片（非维度列返回 None）"""
        index = col - FIXED_COL_COUNT
        if 0 <= index < len(self.dimension_cards):
            return self.dimension_cards[index]
        return None
    def reset_dimension_cells(self):
        """框架被整体替换（导入/复制）后清掉旧编码值，仅保留注释列"""
        for row in self.selected_data:
            comment = row[-1] if row else ""
            row[:] = [""] * len(self.dimension_cards) + [comment]
    def update_action_state(self):
        """无确认步骤：有维度即可记录，有东西可清才能清空，栈里还有现场就能撤销"""
        has_data = bool(self.selected_data)
        self.continue_btn.setEnabled(bool(self.dimension_cards))
        # 「清空」连编码框架一起清：只剩框架没数据时也得能点
        self.clear_btn.setEnabled(has_data or bool(self.dimension_cards))
        self.output_btn.setEnabled(has_data)
        self.undo_btn.setEnabled(bool(self.history))

    # ---------- 记录：默认追加，或按“当前时间”插入/覆盖（入表规则在 TimeAxis） ----------
    def continue_(self):
        """记录一行；手动时间成功入表后推进一个间隔供下一次记录。"""
        self.sync_table_to_data()
        current_row = [card.get_selected_entry() for card in self.dimension_cards]
        current_row.append(self.comment_entry.text())
        if self._timepoint_edited:
            # 用户编辑过当前时间：按该时间点插入或覆盖表格行
            t = self.time_ctrl.parse_timepoint()
            if t is None:
                QMessageBox.warning(self, "无效时间", "当前时间请按 时:分:秒 格式输入。")
                return
            if not self.axis.can_record_at(t):
                # 早于起始时间的记录无处安放（它是时间轴原点），保留编辑态让用户改完重试
                self._hint("记录时间不能早于起始时间 " +
                           format_seconds(self.get_start_seconds()))
                return
            self._push_history()
            self.axis.record(current_row, t)  # 起点规则上面已放行，这里必定入表
            self.time_ctrl.advance_after_record(t)
        else:
            # 默认追加：永远接在时间轴骨架尾部，从此跟着起始时间/间隔一起走
            self._push_history()
            self.axis.append(current_row)
        self.update_table()
        self.comment_entry.clear()
    def _can_record_at(self, t):
        """记录时间不得早于起始时间：它是时间轴原点，格位不往负方向走"""
        return self.axis.can_record_at(t)
    def _record_at_time(self, values, t):
        """指定时间点记录：与某行时间重合则覆盖、踩在格点上则按顺序行入骨架，否则插固定时间点行"""
        return self.axis.record(values, t)

    # ---------- Phase 1 编码辅助：下一步建议（只预填勾选态，不写表） ----------
    def apply_suggestion(self, dim_index, code):
        """把建议码填入某维度卡片的勾选态（等价于用户点该编码块），不写表、不动撤销栈。

        建议只是「预选」：真正入表仍由用户点「记录」走 ``continue_()``，故脏标记与
        撤销口径原样不变。维度号越界或该码不在框架词表内时返回 False。
        """
        cards = self.dimension_cards
        if not (0 <= dim_index < len(cards)):
            return False
        card = cards[dim_index]
        if card.select_value(code):
            self._hint("已填入建议码 %s（%s）——确认后点「记录」写入" % (code, card.name))
            return True
        return False

    def _refresh_suggestions(self):
        """数据变动后重算建议条；面板尚未建好（早于 initUI 的调用）时静默跳过。"""
        bar = getattr(self, "suggest_bar", None)
        if bar is not None:
            bar.refresh()

    # ---------- Phase 2 AI 辅助编码：多模态抽帧给建议码，人在环复核后写回 ----------
    def _open_auto_coding(self):
        """打开「AI 辅助编码」对话框（懒导入 auto_coding，未配置/无抽帧依赖时在框内降级说明）。"""
        if not self.dimension_cards:
            self._hint("先导入或添加编码维度，再用 AI 辅助编码")
            return
        try:
            from auto_coding import AutoCodingDialog
        except Exception as exc:                 # noqa: BLE001 依赖缺失不该崩编码页
            self._hint("AI 辅助编码不可用：%s" % exc)
            return
        AutoCodingDialog(self).exec()

    # ---------- Phase 3 本地训练：在已标注编码上离线训练分类头，预测作建议码写回 ----------
    def _open_local_train(self):
        """打开「本地训练辅助编码」对话框（懒导入 auto_coding_train，视听重后端缺席时框内标注安装命令）。"""
        if not self.dimension_cards:
            self._hint("先导入或添加编码维度，再用本地训练辅助编码")
            return
        try:
            from auto_coding_train import LocalTrainDialog
        except Exception as exc:                 # noqa: BLE001 依赖缺失不该崩编码页
            self._hint("本地训练辅助编码不可用：%s" % exc)
            return
        LocalTrainDialog(self).exec()

    # ---------- 撤销栈：改动前先存一份现场（行数据 + 行时间 + 格位 + 维度框架） ----------
    def _push_history(self):
        """把当前现场压栈；超过上限丢最旧一条，避免长会话下快照无限增长。"""
        self.history.append((copy.deepcopy(self.selected_data), list(self.row_times),
                             list(self.row_positions), self.dim_panel.framework()))
        if len(self.history) > HISTORY_LIMIT:
            self.history.pop(0)
        self.content_changed.emit()
    def _restore_history(self, snap):
        """回到某个现场：先换框架（连带重建表头），再覆盖行数据、行时间与格位。"""
        data, times, positions, framework = snap
        self._restoring = True
        try:
            self.dim_panel.apply_framework(framework)
        finally:
            self._restoring = False
        self.selected_data = copy.deepcopy(data)
        self.row_times = list(times)
        self.row_positions = list(positions)
        self.normalize_row_widths()
        self.update_table_columns()
        self.update_table()
    def undo(self):
        """撤销上一个操作：记录 / 删行 / 改单元格 / 清空 都回到改动前的现场"""
        if not self.history:
            return
        self._restore_history(self.history.pop())
        self.comment_entry.clear()

    # ---------- 删行：顺序行留隐藏格，固定时间点行不留痕迹 ----------
    def selected_table_rows(self):
        """表格当前选中的行号（去重、升序）"""
        return sorted({index.row() for index in self.result_table.selectedIndexes()})
    def delete_selected_rows(self):
        """删除选中的行（支持多选）：两种行的“删”后果不同，但都进撤销栈

        删顺序行只是在骨架上留个隐藏格：其余行的时间与计时原样不动，改间隔时那一格
        照样跟着走，在那个时间上重新记录即补回；固定时间点行本来就不在骨架上，抹掉即
        回到没记过它的样子。起始时间行受保护（见 header_row_locked）。
        """
        rows = self.selected_table_rows()
        if not rows or not self.selected_data:
            return
        if self.header_row_locked(rows):
            self._hint("起始时间行不能单独删除（它决定全表的时间基准）；要清掉全部记录请全选后删除或用「清空」")
            return
        self.sync_table_to_data()
        self._push_history()
        self.axis.drop_rows(rows)
        self.update_table()
    def show_row_menu(self, pos):
        """表格右键菜单：删行入口（取代原工具栏的插入/删除行按钮）"""
        if not self.selected_data:
            return
        if not self.selected_table_rows():
            index = self.result_table.indexAt(pos)
            if index.isValid():
                self.result_table.selectRow(index.row())
        # 菜单窗体需在原生窗体创建前去掉阴影方框 + 圆角裁剪，否则圆角外残留底层矩形
        menu = theme.bind_menu_popup(QMenu(self))
        rows = self.selected_table_rows()
        locked = self.header_row_locked(rows)  # 判定一次，菜单文案与可用性共用
        delete_action = menu.addAction("起始时间行不可删除" if locked else "删除选中行")
        delete_action.setEnabled(bool(rows) and not locked)
        delete_action.triggered.connect(self.delete_selected_rows)
        menu.exec(self.result_table.viewport().mapToGlobal(pos))

    # ---------- 表格重建：列名/列宽/行内容都从时间轴与维度现推 ----------
    def update_table_columns(self):
        columns = self.column_names()
        self.result_table.setColumnCount(len(columns))
        self.result_table.setHorizontalHeaderLabels(columns)
        self.apply_column_widths()
        translate_widget_tree(self.result_table)
    def column_names(self):
        """表头列名：时间 / 计时 / 各维度 / 注释（表头与导出共用一份定义）"""
        return ["时间", "计时"] + [card.name for card in self.dimension_cards] + ["注释"]
    def display_rows(self):
        """逐行的 (时间文本, 距起始时间的秒数, 行数据)：重建表格与导出都从这里取"""
        return self.axis.display_rows()
    def fixed_column_widths(self):
        """时间/计时两列的宽度：量宽口径与「小组交互」页共用（见 timeline_axis）"""
        return fixed_column_widths(self.result_table, FIXED_COL_COUNT)
    def content_column_width(self, col):
        """单列按内容需要的宽度（模块级函数的实例别名）"""
        return content_column_width(self.result_table, col)
    def apply_column_widths(self):
        """时间/计时定宽，其余列先均分剩余空间，内容装不下时再按内容放宽。

        均分份额是地板（不低于 CODE_COL_MIN_WIDTH），内容宽是天花板（到 CONTENT_COL_MAX
        为止）；算法在 timeline_axis，两页同一口径。
        """
        apply_column_widths(self.result_table, FIXED_COL_COUNT)
    def sort_rows_by_time(self):
        """行按当前计算时间排列：内容随行一起移动，顺序行之间相对次序不变"""
        self.axis.sort_rows_by_time()
    def update_table(self):
        """按当前时间轴重建整个表格：行序、时间列、计时列全从计算结果推导"""
        # 重建期间屏蔽 itemChanged 回写，避免把中间态写回 selected_data
        self._loading_table = True
        table = self.result_table
        try:
            self.sort_rows_by_time()
            body_cols = table.columnCount() - FIXED_COL_COUNT
            table.setRowCount(len(self.selected_data))  # 空行就够用，不必逐行 insertRow
            for idx, (tstr, elapsed, row) in enumerate(self.display_rows()):
                cells = list(row)[:body_cols]
                cells += [""] * (body_cols - len(cells))  # 行比列短补空，不让上轮的残值留在尾列
                for col, value in enumerate([tstr, elapsed] + cells):
                    item = QTableWidgetItem(str(value))
                    if col < FIXED_COL_COUNT:
                        # 新建的 item 默认带 ItemIsEditable，只需抹掉时间/计时两列
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    table.setItem(idx, col, item)
            self.apply_column_widths()  # 行内容变了，按内容的列宽也要跟着重算
        finally:
            self._loading_table = False
        self.update_action_state()
        self._refresh_suggestions()
    def on_item_changed(self, item):
        """单元格直改（下拉选编码 / 改注释）即时回写行数据，不会被下次重建抹掉"""
        if self._loading_table or item.column() < FIXED_COL_COUNT:
            return
        row_index = item.row()
        col_index = item.column() - FIXED_COL_COUNT
        if row_index >= len(self.selected_data):
            return
        row = self.selected_data[row_index]
        while len(row) <= col_index:
            row.append("")
        if row[col_index] == item.text():
            return  # 同值的重复通知不该占一条撤销记录
        self._push_history()
        row[col_index] = item.text()
        # 改完立刻按新内容重量列宽：否则双击写进去的长注释要等下一次记录/拖动分割器
        # 才撑得开，看上去像被吞了。量宽带去重缓存，五千行也只十几毫秒，编辑一次无感
        self.apply_column_widths()
        self._refresh_suggestions()
    def sync_table_to_data(self):
        """将表格当前内容同步到self.selected_data（行时间与格位保持原样，仅对齐长度）"""
        table = self.result_table
        rows = []
        for row in range(table.rowCount()):
            cells = []
            for col in range(FIXED_COL_COUNT, table.columnCount()):  # 跳过时间、计时两列
                item = table.item(row, col)
                cells.append(item.text() if item else "")  # 缺项补空串，免得列错位
            rows.append(cells)
        self.selected_data = rows
        self.align_row_states()

    # ---------- 导出 / 清空 ----------
    def output(self):
        """输出表格内容到Excel文件"""
        # 同步表格内容到 self.selected_data
        self.sync_table_to_data()

        # 列名（含时间、计时）与行内容一致重建，不依赖表格快照
        columns = self.column_names()
        body_cols = len(columns) - FIXED_COL_COUNT
        if not self.selected_data:
            df = pd.DataFrame(columns=columns)
        else:
            full_data = []
            for tstr, elapsed, row_data in self.display_rows():
                cells = [str(v) for v in row_data][:body_cols]
                cells += [""] * (body_cols - len(cells))
                # 时间、计时两列插到每行最前
                full_data.append([tstr, elapsed] + cells)
            df = pd.DataFrame(full_data, columns=columns)
        # 选择保存路径（起点用上次导出所在的目录）
        file_path = user_state.save_file_name(
            self, user_state.BUCKET_EXPORT, "保存数据", "",
            "Excel 文件 (*.xlsx);;所有文件 (*)")
        if not file_path:
            return

        # 导出到 Excel
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='数据', index=False)

                worksheet = writer.sheets['数据']
                for col in worksheet.columns:
                    column = col[0].column_letter
                    width = max(len(str(cell.value)) for cell in col)
                    worksheet.column_dimensions[column].width = width + 4
                    for cell in col:
                        cell.alignment = Alignment(horizontal='center')

                for cell in worksheet["1:1"]:
                    cell.font = Font(bold=True)
                    cell.alignment = Alignment(horizontal='center')

            QMessageBox.information(self, "成功", "数据已成功导出！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")
    def clear(self):
        """清空：编码框架与已记录数据一起清掉（聚类/滞后的「清空」只清框架）"""
        self._push_history()
        self.dim_panel.clear_dimensions()
        self.selected_data = []
        self.row_times = []
        self.row_positions = []
        self.result_table.setRowCount(0)
        self.comment_entry.clear()
        self._reset_timepoint()
        # clear_dimensions 不发逐条信号，表头得自己收回到“时间/计时/注释”
        self.update_table_columns()
        self.update_action_state()
