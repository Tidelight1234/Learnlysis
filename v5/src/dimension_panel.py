# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""通用「维度编辑」面板：维度卡片列表 + 导入框架 / 添加维度 / 清空 三个按钮。

编码框架的形状在各处都是一致的（维度名 + 若干编码），差别只在使用方式：个体编码页要勾选
某个编码去记录，而「统计及可视化」与 IND/GRP 两个工作台只消费框架本身（由 `analysis_core`
从数据契约解析，不再嵌一个编辑面板）。所以本面板当前只被 `data_encoder.py` 持有，但布局、
自适应列数、导入框架（先问是否复制、否则读 Excel）仍只在这里维护一次，别处要同样的编辑
体验时直接复用本文件，不要另写一份。
点「添加维度」会先弹框收维度名（默认给个 维度N 建议名）；维度名也能随时改——
点标题条上的铅笔或直接双击标题条，改名后由 `dimension_renamed` 通知宿主同步列名。

框架的对外表示统一为 [(维度名, [编码, ...]), ...]，模块之间复制即传这个列表。
"""

import pandas as pd
from PySide6.QtWidgets import (
    QWidget, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QLabel,
    QScrollArea, QCheckBox, QFrame, QSizePolicy, QGridLayout, QMessageBox,
    QInputDialog, QMenu)
from PySide6.QtCore import QEvent, Qt, QTimer, Signal

import theme
import user_state

# 单个编码块的最小可读宽度（含边距）：每行能塞几个由面板实际宽度除以它推导
BLOCK_MIN_WIDTH = 150
# 宽度还没定下来时临时用的列数：只作占位，不写进缓存，布局到位后会再排一次
FALLBACK_PER_ROW = 2


def read_framework_excel(path):
    """读「维度 / 类型」两列的框架表，按维度出现顺序聚成 [(维度名, [编码...])]。

    空编码（未填的 类型 单元格）直接跳过：框架里留一行空编码既没法勾选记录，
    又会在表格下拉里多出一个空白候选。
    """
    df = pd.read_excel(path).fillna('')
    if list(df.columns) != ['维度', '类型']:
        raise ValueError("Excel 文件必须包含 '维度' 和 '类型' 两列")
    df['维度'] = df['维度'].astype(str)
    df['类型'] = df['类型'].astype(str)
    sequence = []
    for _, row in df.iterrows():
        name = row['维度'].strip()
        code = row['类型'].strip()
        if not code:
            continue
        if sequence and sequence[-1][0] == name:
            sequence[-1][1].append(code)
        else:
            sequence.append([name, [code]])
    return [(name, codes) for name, codes in sequence]


class DimensionCard(QWidget):
    """一个维度：灰底圆角标题条 + 编码块网格（每行数量按宽度自适应）+ 虚线添加按钮。"""

    removed = Signal()
    renamed = Signal()

    def __init__(self, owner=None, name=None, selectable=True):
        super().__init__()
        self.owner = owner
        # 编码页要勾选某个编码做记录；聚类/滞后只维护框架，不画那个勾选框
        self.selectable = selectable
        if name:
            self.name = name
        else:
            count = len(owner.cards) if owner is not None else 0
            self.name = f"维度{count + 1}"
        self.blocks = []
        self.selected_block = None
        # 已应用的每行列数与块数：未变化时跳过重排，避免 resize 反复抖动卡死
        self._applied_per_row = None
        self._applied_count = -1
        self.initUI()

    def initUI(self):
        # 整体布局：卡片本身透明，白底由外层维度面板提供
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        # 维度头部：灰底圆角条（名称 + 改名铅笔 + 折叠箭头 + 删除叉）
        header_frame = QFrame()
        header_frame.setProperty("dimHead", True)
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(12, 8, 6, 8)
        header_layout.setSpacing(8)
        self.label = QLabel(self.name)
        self.label.setProperty("section", True)
        self.label.setMinimumWidth(100)
        self.label.setToolTip("双击此处或点右侧铅笔按钮可重命名维度")
        # 双击标题条时在这个输入框里改名
        self.edit_entry = QLineEdit()
        self.edit_entry.setMaximumWidth(160)
        self.edit_entry.setVisible(False)
        self.edit_entry.editingFinished.connect(self.save_label)
        self.dropdown_indicator = QLabel(self)
        theme.bind_icon(self.dropdown_indicator, "chevron-down", 18, "ink3")
        # 维度改名铅笔（与双击标题条同一条路，只是把入口摆到明面上）
        self.edit_dim_btn = QPushButton()
        self.edit_dim_btn.setFixedSize(26, 26)
        self.edit_dim_btn.setProperty("btn", "ghost")
        self.edit_dim_btn.setToolTip("重命名维度")
        theme.bind_icon(self.edit_dim_btn, "edit", 14)
        self.edit_dim_btn.clicked.connect(self.edit_label)
        self.add_code_icon_btn = QPushButton()
        self.add_code_icon_btn.setFixedSize(26, 26)
        self.add_code_icon_btn.setProperty("btn", "ghost")
        self.add_code_icon_btn.setToolTip("添加编码")
        theme.bind_icon(self.add_code_icon_btn, "plus", 14)
        self.add_code_icon_btn.clicked.connect(self._add_code_from_header)
        # 维度删除叉
        self.remove_dim_btn = QPushButton()
        self.remove_dim_btn.setFixedSize(26, 26)
        self.remove_dim_btn.setProperty("btn", "ghost")
        theme.bind_icon(self.remove_dim_btn, "close", 14)
        self.remove_dim_btn.clicked.connect(self.removed.emit)
        # 头部右侧的动作钮：点它们不该被当成折叠（见 handle_header_click）
        self.header_actions = (self.add_code_icon_btn, self.edit_dim_btn, self.remove_dim_btn)
        header_layout.addWidget(self.label)
        header_layout.addWidget(self.edit_entry)
        header_layout.addStretch()
        header_layout.addWidget(self.add_code_icon_btn)
        header_layout.addWidget(self.edit_dim_btn)
        header_layout.addWidget(self.dropdown_indicator)
        header_layout.addWidget(self.remove_dim_btn)
        # 编码块容器
        self.block_container = QWidget()
        self.block_layout = QGridLayout()
        self.block_layout.setContentsMargins(0, 0, 0, 0)
        self.block_layout.setSpacing(8)
        self.block_container.setLayout(self.block_layout)
        # 维度内虚线添加编码按钮
        self.add_code_btn = QPushButton("添加编码")
        self.add_code_btn.setProperty("btn", "dashed")
        theme.bind_icon(self.add_code_btn, "plus", 14, "purple_ink")
        # 点标题条折叠/展开编码区；双击改名（叉按钮自身除外）
        header_frame.mousePressEvent = self.handle_header_click
        header_frame.mouseDoubleClickEvent = self.edit_label
        layout.addWidget(header_frame)
        layout.addWidget(self.block_container)
        layout.addWidget(self.add_code_btn)
        # clicked 会带 checked 布尔值，而 add_block 的首参是编码文本，必须隔一层 lambda
        self.add_code_btn.clicked.connect(lambda: self.add_block())
        self.set_display_mode(getattr(self.owner, "_display_mode", "standard"))
        # 首次建卡即带一个空编码行（按框架批量建卡时会被清掉）
        self.add_block()

    # ---------- 名称 ----------
    def set_name(self, name):
        """设置维度名（导入/复制框架时用），不触发 renamed。"""
        self.name = name
        self.label.setText(name)
        self.edit_entry.setText(name)

    def edit_label(self, event=None):
        """进入维度名编辑态（铅笔按钮与双击标题条共用）"""
        if not self.edit_entry.isHidden():
            return
        self.label.setVisible(False)
        self.edit_entry.setText(self.label.text())
        self.edit_entry.setVisible(True)
        self.edit_entry.setFocus()
        self.edit_entry.selectAll()

    def save_label(self):
        """保存维度名称；改名由宿主同步表头（维度名即表格列名）"""
        new_text = self.edit_entry.text().strip()
        self.edit_entry.setVisible(False)
        self.label.setVisible(True)
        if new_text and new_text != self.name:
            self.name = new_text
            self.label.setText(new_text)
            self.renamed.emit()

    # ---------- 编码块 ----------
    def add_block(self, text=""):
        """添加新块：输入框 +（可选）勾选框 + 删除叉"""
        block_frame = QFrame()
        block_frame.setProperty("codeRow", True)
        block_frame.installEventFilter(self)
        block_layout = QHBoxLayout(block_frame)
        block_layout.setContentsMargins(10, 5, 6, 5)
        block_layout.setSpacing(8)
        density = getattr(self.owner, "_coding_density", "normal")
        if density != "normal":
            block_layout.setContentsMargins(
                *(theme.coding_density_value(density, a, b)
                  for a, b in ((6, 10), (3, 5), (4, 6), (3, 5))))
            block_layout.setSpacing(theme.coding_density_value(density, 5, 8))
        entry = QLineEdit()
        entry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        entry.setText(text)
        block_layout.addWidget(entry)
        checkbox = None
        if self.selectable:
            # 勾选框只在编码页出现：勾中的编码就是本次记录写进该维度的值
            checkbox = QCheckBox()
            checkbox.clicked.connect(self.select_block)
            block_layout.addWidget(checkbox)
        delete_btn = QPushButton()
        delete_btn.setFixedSize(24, 24)
        delete_btn.setProperty("btn", "ghost")
        theme.bind_icon(delete_btn, "close", 12)
        delete_btn.clicked.connect(self.delete_block)
        block_layout.addWidget(delete_btn)
        # 新建、导入和读档都会走这里；编码行必须立即继承卡片当前模式。
        compact = self._display_mode == 'compact'
        if checkbox is not None:
            checkbox.setVisible(not compact)
        delete_btn.setVisible(not compact)
        self.blocks.append({'entry': entry, 'checkbox': checkbox,
                            'frame': block_frame, 'delete_btn': delete_btn})
        entry.installEventFilter(self)
        entry.textChanged.connect(self.relayout)
        self._update_block_appearance(self.blocks[-1])
        # 按当前宽度重新排布（每行编码数量自适应）
        self.update_block_layout()

    def delete_block(self):
        """编码行上的叉按钮触发：直接删去该编码"""
        btn = self.sender()
        block = next((b for b in self.blocks if b['delete_btn'] is btn), None)
        if block is not None:
            self._drop_block(block)

    def remove_block(self):
        """删除最后一个块"""
        if self.blocks:
            self._drop_block(self.blocks[-1])

    def clear_blocks(self):
        """清空全部编码块（按框架重建时用）"""
        while self.blocks:
            self._drop_block(self.blocks[-1])

    def _drop_block(self, block):
        if self.selected_block is block:
            self.selected_block = None
        self.blocks.remove(block)
        # 立即从布局摘除（单个控件，开销小），避免残留空格；再异步销毁
        self.block_layout.removeWidget(block['frame'])
        block['frame'].setParent(None)
        block['frame'].deleteLater()
        self.update_block_layout()

    def code_values(self):
        """本维度的编码候选列表（表格下拉用），去重、跳过空行"""
        values = []
        for block in self.blocks:
            text = block['entry'].text().strip()
            if text and text not in values:
                values.append(text)
        return values

    # 聚类/滞后的老名字：三块面板对外只一套 API，旧调用处不必改名
    add_code = add_block
    remove_code = remove_block
    get_codes = code_values

    def get_selected_entry(self):
        """获取选中块的文本"""
        if self.selected_block is not None and self.selected_block in self.blocks:
            return self.selected_block['entry'].text()
        return ""

    def select_block(self, _checked=None):
        """选择块：按信号发送者定位，避免删除后索引失效"""
        checkbox = self.sender()
        block = next((b for b in self.blocks if b['checkbox'] is checkbox), None)
        if block is None:
            return
        self._select_block(block)

    def _select_block(self, block):
        previous = self.selected_block
        if self.selected_block is block:
            self.selected_block = None
            block['checkbox'].setChecked(False)
        else:
            if self.selected_block is not None:
                self.selected_block['checkbox'].setChecked(False)
            self.selected_block = block
            block['checkbox'].setChecked(True)
        for item in (previous, block):
            if item is not None:
                self._update_block_appearance(item)

    def select_value(self, value):
        """按编码文本选中对应块（供编码辅助「点选填入」用）。

        等价于用户亲手点该块的勾选框：``clicked`` 信号只由真实点击发出，程序化
        ``setChecked`` 不会再触发 ``select_block``，故这里直接改勾选态与 ``selected_block``
        不会双重处理。找不到对应编码（或本卡片不可选）时返回 False。
        """
        target = str(value or "").strip()
        block = next((b for b in self.blocks
                      if b['entry'].text().strip() == target and b['checkbox'] is not None), None)
        if block is None:
            return False
        if self.selected_block is not None and self.selected_block is not block:
            self.selected_block['checkbox'].setChecked(False)
        self.selected_block = block
        block['checkbox'].setChecked(True)
        for item in self.blocks:
            self._update_block_appearance(item)
        return True

    def reset(self):
        """清掉勾选态与输入内容"""
        self.selected_block = None
        for block in self.blocks:
            block['entry'].clear()
            if block['checkbox'] is not None:
                block['checkbox'].setChecked(False)

    # ---------- 每行编码数量自适应 ----------
    def _codes_per_row(self):
        """按可用宽度推导每行编码块数量；宽度还没定下来时返回 0。"""
        # The card may still have its old, oversized width after a splitter change.
        # The scroll viewport is the actual space available to the user.
        viewport = self.owner.scroll_area.viewport() if self.owner is not None else None
        width = viewport.width() if viewport is not None and viewport.isVisible() else 0
        width = width or self.block_container.width() or self.width()
        if width <= 0:
            return 0
        density = getattr(self.owner, "_coding_density", "normal")
        target = theme.coding_density_value(density, 124, BLOCK_MIN_WIDTH)
        minimum = max((self._block_width(block) for block in self.blocks), default=0)
        spacing = self.block_layout.horizontalSpacing()
        cell = max(target, minimum)
        return max(1, (width + spacing) // (cell + spacing))

    def _block_width(self, block):
        """Measure the complete label plus the controls visible in this mode."""
        frame = block['frame']
        layout = frame.layout()
        margins = layout.contentsMargins()
        entry = block['entry']
        label_width = entry.fontMetrics().horizontalAdvance(entry.text() or "编码") + 12
        controls = []
        compact = getattr(self, '_display_mode', 'standard') == 'compact'
        if block['checkbox'] is not None and not compact:
            controls.append(block['checkbox'].sizeHint().width())
        if not compact:
            controls.append(block['delete_btn'].width())
        return max(BLOCK_MIN_WIDTH if not compact else 0,
                   margins.left() + margins.right() + label_width + sum(controls)
                   + layout.spacing() * len(controls) + 4)

    def update_block_layout(self):
        """按当前宽度把各编码块摆到网格对应行列。

        列数或块数改变时必须先取出旧布局项再重排。QGridLayout 虽然允许用
        addWidget 把已有控件移到新单元格，但会保留旧列的尺寸缓存；纵排时曾经展开
        的多列会在切回横排后继续撑宽卡片，造成横向滚动和“没有自适应”。
        takeAt 只解除布局管理，不会删除控件或断开信号。列数与块数都没变时仍
        直接返回，保证 resizeEvent 高频触发时不会反复拆布局。
        宽度未知（卡片还没被布局过，例如往未显示的标签页复制框架）时用占位列数排一次
        但不写缓存，等 showEvent / 延时兜底再来一趟换成真实列数——否则占位的 2 列会
        被当成结果缓存住，页面显示出来后一直是每行两个。
        """
        per_row = self._codes_per_row()
        provisional = per_row <= 0
        per_row = FALLBACK_PER_ROW if provisional else per_row
        count = len(self.blocks)
        if not provisional and per_row == self._applied_per_row and count == self._applied_count:
            return
        self._applied_per_row = None if provisional else per_row
        self._applied_count = -1 if provisional else count
        # 先清掉旧的 QLayoutItem，让 QGridLayout 忘掉宽屏/纵排阶段产生的列尺寸。
        # 控件仍以 block_container 为父件，不会被销毁或丢失编辑状态。
        while self.block_layout.count():
            self.block_layout.takeAt(0)
        for i, block in enumerate(self.blocks):
            block['frame'].setMinimumWidth(self._block_width(block))
            self.block_layout.addWidget(block['frame'], i // per_row, i % per_row)
        self.block_layout.invalidate()
        self.block_container.updateGeometry()
        self.updateGeometry()

    def relayout(self):
        """宿主在批量增删或换页后强制重排一次。"""
        self._applied_per_row = None
        self._applied_count = -1
        self.update_block_layout()

    def resizeEvent(self, event):
        """卡片宽度变化后重新均分每行编码数量（自适应）"""
        super().resizeEvent(event)
        if getattr(self, "block_layout", None) is not None:
            self.update_block_layout()

    def showEvent(self, event):
        """首次显示时宽度才真正定下来，补一次重排（隐藏期间建的卡片靠这里自愈）。"""
        super().showEvent(event)
        if getattr(self, "block_layout", None) is not None:
            self.update_block_layout()

    # ---------- 折叠 ----------
    def handle_header_click(self, event):
        # 点在改名铅笔/删除叉上不算折叠
        if any(btn.geometry().contains(event.pos()) for btn in self.header_actions):
            return
        self.toggle_blocks()

    def toggle_blocks(self):
        visible = self.block_container.isHidden()
        self.block_container.setVisible(visible)
        self.add_code_btn.setVisible(visible and
                                     getattr(self.owner, "_display_mode", "standard") == "standard")

    def set_display_mode(self, mode):
        compact = mode == "compact"
        self._display_mode = 'compact' if compact else 'standard'
        self.add_code_icon_btn.setVisible(compact)
        self.add_code_btn.setVisible(not compact and not self.block_container.isHidden())
        self.layout().setSpacing(5 if compact else 8)
        for block in self.blocks:
            block['checkbox'].setVisible(not compact) if block['checkbox'] is not None else None
            block['delete_btn'].setVisible(not compact)
            self._update_block_appearance(block)
        self.relayout()

    def _update_block_appearance(self, block):
        frame = block['frame']
        frame.setProperty('selectedCode', self.selected_block is block)
        frame.style().unpolish(frame)
        frame.style().polish(frame)
        frame.update()

    def eventFilter(self, watched, event):
        if getattr(self, '_display_mode', 'standard') == 'compact':
            block = next((b for b in self.blocks if watched in (b['frame'], b['entry'])), None)
            if block is not None:
                if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                    if self.selectable:
                        self._select_block(block)
                    return watched is block['frame']
                if event.type() == QEvent.ContextMenu:
                    menu = theme.bind_menu_popup(QMenu(self))
                    action = menu.addAction('删除编码')
                    if menu.exec(event.globalPos()) is action:
                        answer = QMessageBox.question(self, '删除编码',
                            f'确定删除编码“{block["entry"].text()}”吗？',
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                        if answer == QMessageBox.Yes:
                            self._drop_block(block)
                    return True
        return super().eventFilter(watched, event)

    def _add_code_from_header(self):
        if self.block_container.isHidden():
            self.block_container.show()
        self.add_block()
        self.blocks[-1]['entry'].setFocus()

    def mouseDoubleClickEvent(self, event):
        """双击编辑维度名称"""
        self.edit_label()


class DimensionPanel(QWidget):
    """「维度编辑」整块：白底面板（标题 + 可滚动的维度卡片列表）+ 下方三个按钮。

    宿主差异靠两个注入口解决：
      sources_provider() -> [(显示名, [(维度名, [编码...])])]  可复制的框架来源
      clear_owns_dimensions=True 时「清空」由面板接管为清维度（聚类/滞后），
      留 False 则语义交给宿主（编码页清的是已记录数据，可用态也按数据量定）
    """

    dimension_added = Signal(object)
    dimension_removed = Signal(int)
    dimension_renamed = Signal()
    framework_replaced = Signal()

    def __init__(self, parent=None, title="维度编辑", selectable=True,
                 sources_provider=None, clear_tooltip="清空已记录的编码数据",
                 clear_owns_dimensions=False):
        super().__init__(parent)
        self.cards = []
        self._coding_density = "normal"
        self._display_mode = user_state.state.coding_dimension_mode()
        self.selectable = selectable
        self.sources_provider = sources_provider
        # 批量建卡（导入/复制/读档）期间压掉逐条 dimension_added，宿主只需收尾一次
        self._batch = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        # ---- 白底面板 ----
        self.panel_card = QWidget()
        self.panel_card.setProperty("dimPanel", True)
        panel_layout = QVBoxLayout(self.panel_card)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(8)
        self.title_label = QLabel(title)
        self.title_label.setProperty("section", True)
        panel_layout.addWidget(self.title_label)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout()
        self.cards_layout.setAlignment(Qt.AlignTop)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(12)
        self.cards_container.setLayout(self.cards_layout)
        self.scroll_area.setWidget(self.cards_container)
        # Windows 与 macOS 对 QScrollArea 的滚动条/视口几何提交时序不同。
        # 监听真正决定卡片可用宽度的视口，而不是猜测一次 timer 延迟是否足够。
        self._relayout_timer = QTimer(self)
        self._relayout_timer.setSingleShot(True)
        self._relayout_timer.timeout.connect(self.relayout_cards)
        self.scroll_area.viewport().installEventFilter(self)
        self.cards_container.installEventFilter(self)
        panel_layout.addWidget(self.scroll_area, 1)
        # 本卡片子树不含原生窗体，可以走图形效果投影
        theme.apply_card_shadow(self.panel_card)
        layout.addWidget(self.panel_card, 1)
        # ---- 按钮区域：导入框架 / 添加维度 / 清空 ----
        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(8)
        self.import_btn = QPushButton("导入框架")
        self.import_btn.setProperty("btn", "secondary")
        theme.bind_icon(self.import_btn, "import", 16)
        self.import_btn.clicked.connect(self.on_import_clicked)
        self.add_dimension_btn = QPushButton("添加维度")
        self.add_dimension_btn.setProperty("btn", "secondary")
        self.add_dimension_btn.setToolTip("新建一个维度，会先让你输入维度名")
        # clicked 带 checked 布尔值，不能直连 add_dimension（首参是维度名），隔一层处理
        self.add_dimension_btn.clicked.connect(self.on_add_dimension_clicked)
        self.clear_btn = QPushButton("清空")
        self.clear_btn.setProperty("btn", "secondary")
        self.clear_btn.setToolTip(clear_tooltip)
        self._clear_owns_dimensions = clear_owns_dimensions
        if clear_owns_dimensions:
            self.clear_btn.clicked.connect(self.clear_dimensions)
        for btn in (self.import_btn, self.add_dimension_btn, self.clear_btn):
            btn_layout.addWidget(btn, 1)
        layout.addWidget(btn_widget)
        self._sync_clear_state()

    def _sync_clear_state(self):
        """接管「清空」时：没有维度就没得清，按钮跟着置灰。"""
        if self._clear_owns_dimensions:
            self.clear_btn.setEnabled(bool(self.cards))

    def eventFilter(self, watched, event):
        """视口最终宽度改变后合并重排；覆盖 Windows 较晚确定滚动条宽度的情况。"""
        if event.type() == QEvent.Resize and (
                watched is self.scroll_area.viewport() or watched is self.cards_container):
            self.schedule_relayout_cards()
        return super().eventFilter(watched, event)

    def schedule_relayout_cards(self):
        """在当前一轮 Qt 几何更新结束后强制按实际宽度重排全部维度卡片。"""
        self._relayout_timer.start(0)

    def set_compact(self, size):
        self._coding_density = size
        self.cards_layout.setSpacing(theme.coding_density_value(size, 7, 12))
        for card in self.cards:
            self._compact_card(card, size)

    def set_display_mode(self, mode):
        self._display_mode = "compact" if mode == "compact" else "standard"
        for card in self.cards:
            card.set_display_mode(self._display_mode)
        self.schedule_relayout_cards()

    @staticmethod
    def _compact_card(card, size):
        card.layout().setSpacing(theme.coding_density_value(size, 5, 8))
        card.block_layout.setSpacing(theme.coding_density_value(size, 5, 8))
        card.label.setMinimumWidth(theme.coding_density_value(size, 76, 100))
        for block in card.blocks:
            block["frame"].layout().setContentsMargins(
                *(theme.coding_density_value(size, a, b)
                  for a, b in ((6, 10), (3, 5), (4, 6), (3, 5))))
            block["frame"].layout().setSpacing(theme.coding_density_value(size, 5, 8))
        card.relayout()

    # ---------- 读写框架 ----------
    def framework(self):
        """当前框架：[(维度名, [编码...])]，供其他模块复制。"""
        return [(card.name, card.code_values()) for card in self.cards]

    def has_framework(self):
        return bool(self.cards)

    def next_dimension_name(self):
        """新维度的建议名：维度N 顺延，跳过已被占用的名字。"""
        used = {card.name for card in self.cards}
        n = len(self.cards) + 1
        while f"维度{n}" in used:
            n += 1
        return f"维度{n}"

    def on_add_dimension_clicked(self):
        """「添加维度」按钮：先弹框让用户输入维度名，没给名字就不建卡。"""
        name = self.prompt_dimension_name()
        if name:
            self.add_dimension(name)

    def prompt_dimension_name(self):
        """弹框收集维度名（默认填好建议名）；取消或输入非法返回 None。

        不用 QInputDialog.getText 静态版：它拿不到按钮对象，确定/取消会停在
        英文 OK/Cancel（应用未加载 Qt 中文翻译），与整体中文界面对不上。
        """
        dlg = QInputDialog(self)
        dlg.setWindowTitle("添加维度")
        dlg.setLabelText("请输入维度名：")
        dlg.setInputMode(QInputDialog.TextInput)
        dlg.setTextValue(self.next_dimension_name())
        dlg.setOkButtonText("添加")
        dlg.setCancelButtonText("取消")
        dlg.setMinimumWidth(320)
        # 预选建议名：直接打字即覆盖，想保留建议名则按右方向键取消选区
        edit = dlg.findChild(QLineEdit)
        if edit is not None:
            edit.selectAll()
        ok = dlg.exec()
        name = dlg.textValue().strip()
        if not ok:
            return None
        if not name:
            QMessageBox.warning(self, "提示", "维度名不能为空。")
            return None
        return name

    def add_dimension(self, name=None, codes=None):
        """新增一个维度卡片；给了 codes 就按这份编码建（初始那行空编码会被换掉）。"""
        card = DimensionCard(self, name=name or self.next_dimension_name(),
                             selectable=self.selectable)
        card.set_display_mode(self._display_mode)
        self.cards.append(card)
        if self._coding_density != "normal":
            self._compact_card(card, self._coding_density)
        self.cards_layout.addWidget(card)
        card.removed.connect(lambda c=card: self.remove_dimension(c))
        card.renamed.connect(self.dimension_renamed.emit)
        if codes is not None:
            card.clear_blocks()
            for code in codes:
                card.add_block(code)
        if not self._batch:
            self.dimension_added.emit(card)
        self._sync_clear_state()
        return card

    def remove_dimension(self, card):
        """维度卡片上的叉按钮触发：先报索引，供宿主同步删掉对应表格列。"""
        if card not in self.cards:
            return
        index = self.cards.index(card)
        self.cards.remove(card)
        card.setParent(None)
        card.deleteLater()
        self.dimension_removed.emit(index)
        self._sync_clear_state()

    def clear_dimensions(self):
        """移除全部维度卡片（不发逐条 dimension_removed，整块替换由宿主收尾）。"""
        while self.cards:
            card = self.cards.pop()
            card.setParent(None)
            card.deleteLater()
        self._sync_clear_state()

    def apply_framework(self, items):
        """按 [(维度名, [编码...])] 覆盖式重建全部维度卡片。"""
        self.clear_dimensions()
        self._batch = True
        try:
            for name, codes in items:
                self.add_dimension(name, list(codes))
        finally:
            self._batch = False
        # 隐藏期间建的卡片建卡时宽度为 0，下一拍布局到位后再统一重排一次
        self.schedule_relayout_cards()
        self.framework_replaced.emit()

    def relayout_cards(self):
        for card in self.cards:
            card.relayout()

    # ---------- 导入框架 ----------
    def on_import_clicked(self):
        """先问要不要复制已有框架（取消即视为不复制），否则走 Excel 导入。"""
        sources = [(label, items) for label, items in (self._sources() or []) if items]
        if sources:
            labels = [label for label, _ in sources]
            item, ok = QInputDialog.getItem(
                self, "复制框架", "检测到已有编码框架，是否复制？", labels, 0, False)
            if ok and item:
                label, items = sources[labels.index(item)]
                self.apply_framework(items)
                QMessageBox.information(self, "成功", f"已从「{label}」复制框架！")
                return
            # 用户点了“取消”：不复制，继续按老路子弹文件框
        self.import_from_excel()

    def import_from_excel(self, file_path=None):
        """从 Excel 导框架；传了 path（「最近打开」重开）就不再弹文件框。"""
        if not file_path:
            file_path = user_state.open_file_name(
                self, user_state.BUCKET_FRAMEWORK, "选择编码框架文件",
                "Excel Files (*.xlsx);;All Files (*)")
        if not file_path:
            return
        try:
            items = read_framework_excel(file_path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导入失败：{str(e)}")
            return
        self.apply_framework(items)
        QMessageBox.information(self, "成功", "编码框架已成功导入！")

    def _sources(self):
        if self.sources_provider is None:
            return []
        try:
            return list(self.sources_provider() or [])
        except Exception:
            return []
