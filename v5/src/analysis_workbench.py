# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""个体模式 / 小组模式分析工作台：IND01–IND15、GRP01–GRP10 的统一外壳。

与「统计及可视化」同一套五卡观感（数据 → 方法 → 参数 → 质检 → 导出，右表下图），
差别只有三处，也都是这两个模式的固有要求：

1. **数据卡装的是「一批组」**：`GroupBundle` 列表 → `AnalysisContext`，双入口（当前
   编码会话 / 批量上传多个 Group Excel），上传后每组一行可覆写 group_id；
2. **参数卡顶部自动补「分析对象」行**：按 `MethodSpec.units` 生成学生/组/维度多选，
   方法专有参数排在其后（先选算谁、再选怎么算）；
3. **右侧多一张辅助卡**：R 交叉校验表 + 训练日志（深度学习类方法的 epoch 回调走
   `ctx.progress`，后台线程由 `ComputeWorker.progress` 信号接回主线程）。

训练不入 UI 线程：`engine == "optional"` 的方法（IND13–IND15 / GRP10）默认丢进 `QThread`，
参数卡上只在跑动期间浮现「取消计算」按钮，取消 = 给 `ctx.cancelled` 挂标志，
计算层在下一个 epoch / 折检查点自己收摊（见 `analysis_dl.cancelled`）。其余方法
依旧同步算完，`generate(wait=True)` 可强制同步（测试脚本用）。

不可用的方法**不隐藏**：下拉里照旧列出、文案后附原因，选中即阻断计算并在质检卡写明
理由 —— 用户永远能看到「为什么这个跑不了」，而不是发现功能凭空少了。

R 交叉校验的调用契约（`analysis_r_bridge`）：
    probe() -> {"available": bool, "version": str, "packages": {名: bool},
                "found": [名], "missing": [名], "rscript": 路径}
    reference_check(spec, result, ctx, params) -> (差异表 rows, 一句人话说明)
    ProbeWorker / ReferenceWorker / InstallWorker —— 子线程跑，done 信号回主线程
首次显示工作台只后台探测，不拿 `Rscript` 的启动开销去堵 UI；桥接层缺席（未装 R /
缺包 / 本方法没有参照）时本卡只给降级提示，不影响任何计算与导出。

对外契约（main_window / 冒烟脚本依赖）：
    ModeAnalysisWorkbench(mode=am.MODE_INDIVIDUAL)
    attach_session(summary_tab=…, group_encoder=…, pages_provider=…)
    select_file() / load_files(paths) / load_from_session() / set_method(id)
    generate(wait=False) / join_compute(timeout_ms) / cancel_current() / current_result()
    redraw_chart() / current_figure() / run_crosscheck(sync=False) / join_r_tasks()
    ctx / result / figures / result_tabs / file_combo
"""

from __future__ import annotations

import os
import time
import traceback

import pandas as pd
from PySide6.QtCore import QEvent, Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout,
                            QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                            QListWidget, QListWidgetItem, QMessageBox,
                            QPlainTextEdit, QPushButton, QScrollArea, QSplitter,
                            QTabWidget, QTableWidget, QTableWidgetItem,
                            QVBoxLayout, QWidget)
from matplotlib.figure import Figure

import analysis_core as ac
import analysis_export as ae
import analysis_methods as am
import analysis_params as ap
import analysis_plots as plots
import theme
import user_state
from language import translate_widget_tree
from statistic_visualization_data import QAItem, fmt_num, merge_qa, qa_summary

MISSING_TEXT = ap.MISSING_TEXT
#: 取数入口能读的文件类型：与「统计及可视化」同一套口径
DATA_FILTER = ("编码数据 (*.xlsx *.xls *.csv);;Excel (*.xlsx *.xls);;CSV (*.csv);;所有文件 (*)")
CROSS_COLUMNS = ("量", "Python", "R", "绝对差", "相对差", "判定")
LEVEL_ORDER = {"error": 0, "warn": 1, "info": 2, "note": 3}
GROUP_COLUMNS = ("来源文件", "组号", "成员数", "时间点", "事件表")
EDITABLE_GROUP_COLUMN = 1        # 只有「组号」能改：其它列都是现读现算的事实
TRAINING_ENGINES = (am.ENGINE_OPTIONAL,)   # 丢后台线程的引擎（能训几分钟的都在这）
# 左栏宽度：下限沿用五卡定式，上限按工作台宽度折算。五卡里最宽的一行不固定
# （学生清单、长条目下拉、带阻断说明的方法下拉），写不死一个数，只能现量；
# 但左栏终归只是入口，再宽就要吃掉右表下图的读数空间。
LEFT_PANEL_MIN_W = ap.LEFT_PANEL_W
LEFT_PANEL_RATIO = 0.42
#: 右侧「表 / 图 / 交叉校验」三卡之间的透明间隙：除要给卡片阴影（blur 28 / offset 6）
#: 留出落地的地方，还与工作台根布局的左右页边 16 同档，让右侧的纵向节奏与页边一致。
CARD_GAP = 16


class ComputeWorker(QThread):
    """把一次方法计算搬到子线程：IND13–IND15 / GRP10 的训练不能堵 UI。

    `progress` 转发 `ctx.progress`（逐 epoch 日志），`done` 回传 `(result, error)`；
    两条信号都回主线程交货 —— 计算线程绝不碰任何控件。
    """

    progress = Signal(str)
    done = Signal(object, object)

    def __init__(self, spec, ctx, params, parent=None):
        super().__init__(parent)
        self.spec, self.ctx, self.params = spec, ctx, params

    def run(self):
        try:
            result = self.spec.compute(self.ctx, self.params)
        except Exception as exc:                # noqa: BLE001  线程里抛不出去，只能回传
            self.done.emit(None, exc)
            return
        self.done.emit(result, None)


class ModeAnalysisWorkbench(QWidget):
    #: 子类可追加的组清单列（小组模式多一列「事件来源」）
    EXTRA_GROUP_COLUMNS = ()

    def __init__(self, mode=am.MODE_INDIVIDUAL, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.ctx = None
        self.result = None
        self.spec = None
        self.figures = {}
        self.sheet = None
        self._paths = []
        self._batches = {}
        self._params_cache = {}
        self._rows = []
        self._loading = False
        self._last_export = ""
        self._suspend_refit = False
        self._fitted_width = 0.0              # 上次量过的预览区宽（英寸）：防滚动条引起的重画振荡
        self._left_fit_timer = None               # initUI 前不得有人来要宽度（见 _request_left_fit）
        self._canvases = []
        self._r_status = None
        self._bridge_ref = None
        self._bridge_missing = False
        self._probe_task = None
        self._r_task = None
        self._install_task = None
        self._task = None                       # 正在后台跑的计算（ComputeWorker）
        self._busy = False
        self.summary_source = None
        self.group_source = None
        self.pages_provider = None
        self.session_group_id = None
        self._imported_codebook = None
        self.initUI()
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._populate_methods()
        self._empty_state("尚未装载数据：选择 Group Excel 文件，或从当前编码会话取数。")

    # ---------------------------------------------------------------- 布局
    def initUI(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 16)
        root.setSpacing(12)
        root.addWidget(self._build_left())
        root.addWidget(self._build_right(), 1)

    def _make_card(self, title=None):
        card = QWidget()
        card.setProperty("card", True)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        if title:
            label = QLabel(title)
            label.setProperty("section", True)
            lay.addWidget(label)
        theme.apply_card_shadow(card)
        return card, lay

    def _build_left(self):
        host = QWidget()
        host.setFixedWidth(LEFT_PANEL_MIN_W)
        outer = QVBoxLayout(host)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        # 横向一律不出滚动条：左栏是「读表前」的入口，横向滚一下就把生成结果推出
        # 视野。放不下时由 _fit_left_width 把左栏加宽，而不是把内容藏到滚动条后面。
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        column = QVBoxLayout(inner)
        column.setContentsMargins(0, 0, 8, 0)
        column.setSpacing(12)
        column.addWidget(self._build_data_card())
        column.addWidget(self._build_method_card())
        column.addWidget(self._build_params_card())
        column.addWidget(self._build_qa_card())
        column.addWidget(self._build_export_card())
        column.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll)
        self.left_host = host
        self.left_scroll = scroll
        self._left_fit_timer = QTimer(self)
        self._left_fit_timer.setSingleShot(True)
        self._left_fit_timer.timeout.connect(self._fit_left_width)
        self._left_fit_timer.start(0)
        return host

    def _request_left_fit(self):
        """把左栏宽度重算排到事件尾巴上：参数卡刚重建时子件尺寸还没更新，当场量只会量到旧值。"""
        if self._left_fit_timer is not None:
            self._left_fit_timer.start(0)

    def _fit_left_width(self):
        """左栏按五卡的最小需求加宽，最多加到工作台宽度的 ``LEFT_PANEL_RATIO``。"""
        inner = self.left_scroll.widget()
        bar_w = self.left_scroll.verticalScrollBar().sizeHint().width()
        need = inner.minimumSizeHint().width() + bar_w
        cap = max(LEFT_PANEL_MIN_W, int(self.width() * LEFT_PANEL_RATIO))
        self.left_host.setFixedWidth(int(min(max(need, LEFT_PANEL_MIN_W), cap)))

    def resizeEvent(self, event):
        """窗口变窄时上限跟着变：左栏得从加宽状态退回定宽。"""
        super().resizeEvent(event)
        self._request_left_fit()

    def _build_data_card(self):
        card, lay = self._make_card("数据")
        row = QHBoxLayout()
        row.setSpacing(8)
        self.file_btn = QPushButton("选择文件…")
        self.file_btn.setProperty("btn", "secondary")
        self.file_btn.setFixedHeight(32)
        theme.bind_icon(self.file_btn, "import", 16)
        self.file_btn.clicked.connect(self.select_file)
        row.addWidget(self.file_btn)
        self.session_btn = QPushButton("从当前会话取数")
        self.session_btn.setProperty("btn", "secondary")
        self.session_btn.setFixedHeight(32)
        theme.bind_icon(self.session_btn, "tab-data", 16)
        self.session_btn.clicked.connect(self.load_from_session)
        row.addWidget(self.session_btn)
        lay.addLayout(row)

        frame_row = QHBoxLayout()
        frame_row.setSpacing(8)
        self.framework_btn = QPushButton("导入编码框架 Excel")
        self.framework_btn.setProperty("btn", "ghost")
        self.framework_btn.setFixedHeight(28)
        theme.bind_icon(self.framework_btn, "cluster", 16)
        self.framework_btn.clicked.connect(self.import_framework)
        frame_row.addWidget(self.framework_btn)
        self.framework_label = QLabel("框架：自动解析")
        self.framework_label.setProperty("hint", True)
        frame_row.addWidget(self.framework_label, 1)
        lay.addLayout(frame_row)

        self.file_combo = QComboBox()
        self.file_combo.setProperty("i18nData", True)
        self.file_combo.setToolTip("最近装载过的批次")
        ap.shrink_to_fit(self.file_combo)     # 批次标签是一串组号，不能让它定左栏宽
        theme.bind_combo_popup(self.file_combo)
        self.file_combo.currentIndexChanged.connect(self._on_batch_changed)
        lay.addWidget(self.file_combo)

        self.data_hint = QLabel("尚未装载数据。")
        self.data_hint.setProperty("hint", True)
        self.data_hint.setWordWrap(True)
        lay.addWidget(self.data_hint)

        map_label = QLabel("组清单（改「组号」列可覆写 group_id，改完点应用）")
        map_label.setProperty("hint", True)
        map_label.setWordWrap(True)      # 不折行就是一整句不可断的长文本，会把数据卡顶宽
        lay.addWidget(map_label)
        columns = self._group_columns()
        self.group_table = QTableWidget(0, len(columns))
        self.group_table.setHorizontalHeaderLabels(list(columns))
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.setEditTriggers(QTableWidget.AllEditTriggers)
        self.group_table.setMaximumHeight(140)
        header = self.group_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, len(columns)):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
            self.group_table.setColumnWidth(column, 96 if column == EDITABLE_GROUP_COLUMN else 58)
        lay.addWidget(self.group_table)

        self.apply_btn = QPushButton("应用组号覆写")
        self.apply_btn.setProperty("btn", "ghost")
        self.apply_btn.setFixedHeight(28)
        self.apply_btn.clicked.connect(self.apply_group_overrides)
        lay.addWidget(self.apply_btn)
        return card

    def _build_method_card(self):
        card, lay = self._make_card("分析方法")
        self.method_combo = QComboBox()
        self.method_combo.setToolTip(self._method_tooltip())
        ap.shrink_to_fit(self.method_combo)   # 条目后拼的阻断理由再长也只截断显示
        theme.bind_combo_popup(self.method_combo)
        lay.addWidget(self.method_combo)
        self.method_hint = QLabel("")
        self.method_hint.setProperty("hint", True)
        self.method_hint.setWordWrap(True)
        lay.addWidget(self.method_hint)
        return card

    def _method_tooltip(self):
        prefix = "IND" if self.mode == am.MODE_INDIVIDUAL else "GRP"
        return "%s 系列方法，按 A 状态与轨迹 … G 组间比较分组；不可用的方法会写明原因" % prefix

    def _build_params_card(self):
        card, lay = self._make_card("参数设置")
        self.params_host = QWidget()
        self.params_grid = QGridLayout(self.params_host)
        self.params_grid.setContentsMargins(0, 0, 0, 0)
        self.params_grid.setHorizontalSpacing(8)
        self.params_grid.setVerticalSpacing(6)
        self.params_grid.setColumnStretch(1, 1)
        lay.addWidget(self.params_host)
        self.params_notice = QLabel("")
        self.params_notice.setProperty("notice", True)
        self.params_notice.setWordWrap(True)
        self.params_notice.setVisible(False)
        lay.addWidget(self.params_notice)
        self.generate_btn = QPushButton("生成结果")
        self.generate_btn.setProperty("btn", "primary")
        self.generate_btn.setFixedHeight(34)
        self.generate_btn.setEnabled(False)
        theme.bind_icon(self.generate_btn, "calculate", 16, "#FFFFFF")
        self.generate_btn.clicked.connect(self.generate_now)
        lay.addWidget(self.generate_btn)
        # 取消钮只在后台训练时浮现：它写的不是控件，是 ctx.cancelled
        self.cancel_btn = QPushButton("取消计算")
        self.cancel_btn.setProperty("btn", "secondary")
        self.cancel_btn.setFixedHeight(30)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setToolTip("深度学习类方法逐 epoch 检查取消标志，已完成的折仍会报出")
        theme.bind_icon(self.cancel_btn, "progress", 16)
        self.cancel_btn.clicked.connect(self.cancel_current)
        lay.addWidget(self.cancel_btn)
        return card

    def _build_qa_card(self):
        card, lay = self._make_card("自动质检")
        self.qa_notice = QLabel("装载数据后这里会列出阻断、警告与提示。")
        self.qa_notice.setProperty("notice", True)
        self.qa_notice.setWordWrap(True)
        lay.addWidget(self.qa_notice)
        self.qa_list = QListWidget()
        self.qa_list.setMaximumHeight(150)
        self.qa_list.setWordWrap(True)
        lay.addWidget(self.qa_list)
        return card

    def _build_export_card(self):
        card, lay = self._make_card("导出")
        self.tables_btn = QPushButton("导出统计表")
        self.tables_btn.setProperty("btn", "secondary")
        self.tables_btn.setFixedHeight(30)
        theme.bind_icon(self.tables_btn, "export", 16)
        self.tables_btn.clicked.connect(self.export_statistics)
        lay.addWidget(self.tables_btn)

        figure_row = QHBoxLayout()
        figure_row.setSpacing(6)
        self.figure_btn = QPushButton("导出图形")
        self.figure_btn.setProperty("btn", "secondary")
        self.figure_btn.setFixedHeight(30)
        theme.bind_icon(self.figure_btn, "chart-bar", 16)
        self.figure_btn.clicked.connect(self.export_chart)
        figure_row.addWidget(self.figure_btn, 1)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "SVG", "PDF"])
        self.format_combo.setMinimumWidth(ap.combo_width(self.format_combo))
        theme.bind_combo_popup(self.format_combo)
        figure_row.addWidget(self.format_combo)
        self.dpi_combo = QComboBox()
        self.dpi_combo.addItems(["300", "600"])
        self.dpi_combo.setMinimumWidth(ap.combo_width(self.dpi_combo))
        self.dpi_combo.setToolTip("仅对 PNG 生效")
        theme.bind_combo_popup(self.dpi_combo)
        figure_row.addWidget(self.dpi_combo)
        lay.addLayout(figure_row)

        self.meta_btn = QPushButton("导出配置 / 质检 / 模型卡")
        self.meta_btn.setProperty("btn", "secondary")
        self.meta_btn.setFixedHeight(30)
        theme.bind_icon(self.meta_btn, "save", 16)
        self.meta_btn.clicked.connect(self.export_metadata)
        lay.addWidget(self.meta_btn)

        self.all_btn = QPushButton("一键导出全部")
        self.all_btn.setProperty("btn", "dashed")
        self.all_btn.setFixedHeight(30)
        theme.bind_icon(self.all_btn, "folder", 16)
        self.all_btn.clicked.connect(self.export_everything)
        lay.addWidget(self.all_btn)

        self.export_hint = QLabel("统计表 / 图 / config / audit / 模型卡共用同一 analysis_id 前缀。")
        self.export_hint.setProperty("hint", True)
        self.export_hint.setWordWrap(True)
        lay.addWidget(self.export_hint)
        return card

    def _build_right(self):
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        # 同「统计及可视化」：卡间留出比阴影（blur 28 / offset 6）更宽的透明间隙，
        # 窄分割条贴在阴影边缘会被读成「底下压了一块灰色矩形」。
        splitter.setProperty("cardGap", True)
        splitter.setHandleWidth(CARD_GAP)

        table_card, table_lay = self._make_card("结果统计表")
        self.result_tabs = QTabWidget()
        self.result_tabs.setObjectName("innerTab")
        theme.install_underline_tab_bar(self.result_tabs, fit_text=True)
        table_lay.addWidget(self.result_tabs)

        figure_card, figure_lay = self._make_card("图形预览")
        head = QHBoxLayout()
        head.setSpacing(6)
        self.figure_combo = QComboBox()
        self.figure_combo.setToolTip("本方法输出的图不止一张时在此定位")
        self.figure_combo.setVisible(False)
        theme.bind_combo_popup(self.figure_combo)
        self.figure_combo.currentIndexChanged.connect(self._show_figure_at)
        head.addWidget(self.figure_combo, 1)
        self.figure_hint = QLabel("")
        self.figure_hint.setProperty("hint", True)
        self.figure_hint.setWordWrap(True)
        self.figure_hint.setVisible(False)
        head.addWidget(self.figure_hint)
        figure_lay.addLayout(head)
        self.figure_scroll = QScrollArea()
        self.figure_scroll.setFrameShape(QScrollArea.NoFrame)
        self.figure_scroll.setWidgetResizable(True)
        self.figure_scroll.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.figure_scroll.setToolTip("图按内容自动放大，超出预览区时可滚动查看完整内容")
        self.figure_host = QWidget()
        self.figure_box = QVBoxLayout(self.figure_host)
        self.figure_box.setContentsMargins(0, 0, 0, 0)
        self.figure_box.setSpacing(8)
        self.figure_box.setAlignment(Qt.AlignHCenter)
        self.figure = Figure(figsize=(9, 4.6))
        self.canvas = ap.PreviewCanvas(self.figure)
        self.figure_box.addWidget(self.canvas)
        self.figure_scroll.setWidget(self.figure_host)
        self._refit_timer = QTimer(self)
        self._refit_timer.setSingleShot(True)
        self._refit_timer.setInterval(120)
        self._refit_timer.timeout.connect(self._refit_figures)
        self.figure_scroll.viewport().installEventFilter(self)
        figure_lay.addWidget(self.figure_scroll)
        # 预览区右下角的浮动「全屏查看」：图挤在小视口里永远看不清细节
        self.fullscreen_btn = ap.attach_fullscreen_button(
            self.figure_scroll, self._preview_pixmaps, "结果图")
        self.canvas.draw()

        aux_card, aux_lay = self._make_card("交叉校验与训练日志")
        self.aux_tabs = QTabWidget()
        self.aux_tabs.setObjectName("innerTab")
        theme.install_underline_tab_bar(self.aux_tabs)
        self.aux_tabs.addTab(self._build_cross_tab(), "R 交叉校验")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(600)
        self.log_view.setPlaceholderText("训练日志：逐 epoch 输出会打在这里（深度学习类方法）。")
        self.aux_tabs.addTab(self.log_view, "训练日志")
        aux_lay.addWidget(self.aux_tabs)

        splitter.addWidget(table_card)
        splitter.addWidget(figure_card)
        splitter.addWidget(aux_card)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([240, 380, 150])
        return splitter

    def _build_cross_tab(self):
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.cross_btn = QPushButton("用 R 复算并对照")
        self.cross_btn.setProperty("btn", "secondary")
        self.cross_btn.setFixedHeight(28)
        theme.bind_icon(self.cross_btn, "calculate", 16)
        self.cross_btn.clicked.connect(self.run_crosscheck)
        row.addWidget(self.cross_btn)
        self.cross_hint = QLabel("R 参考层用于佐证 Python 实现，缺席不影响本工作台。")
        self.cross_hint.setProperty("hint", True)
        self.cross_hint.setWordWrap(True)
        row.addWidget(self.cross_hint, 1)
        lay.addLayout(row)
        setup = QHBoxLayout()
        setup.setSpacing(8)
        self.copy_cmd_btn = QPushButton("复制安装命令")
        self.copy_cmd_btn.setProperty("btn", "secondary")
        self.copy_cmd_btn.setFixedHeight(28)
        theme.bind_icon(self.copy_cmd_btn, "copy", 16)
        self.copy_cmd_btn.clicked.connect(self.copy_install_command)
        setup.addWidget(self.copy_cmd_btn)
        self.install_btn = QPushButton("安装 R 参考包")
        self.install_btn.setProperty("btn", "secondary")
        self.install_btn.setFixedHeight(28)
        theme.bind_icon(self.install_btn, "import", 16)
        self.install_btn.clicked.connect(self.install_r_packages)
        setup.addWidget(self.install_btn)
        self.repo_edit = QLineEdit()
        self.repo_edit.setPlaceholderText("CRAN 镜像（可留空），如 https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
        self.repo_edit.setFixedHeight(28)
        setup.addWidget(self.repo_edit, 1)
        lay.addLayout(setup)
        self.r_status_label = QLabel("R 参考层：尚未探测。")
        self.r_status_label.setProperty("hint", True)
        self.r_status_label.setWordWrap(True)
        lay.addWidget(self.r_status_label)
        self.cross_table = QTableWidget(0, len(CROSS_COLUMNS))
        self.cross_table.setHorizontalHeaderLabels(list(CROSS_COLUMNS))
        self.cross_table.verticalHeader().setVisible(False)
        self.cross_table.setProperty("gridTable", True)   # 与结果表同一圈外框
        self.cross_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cross_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        lay.addWidget(self.cross_table, 1)
        return host

    # ------------------------------------------------------ 预览图尺寸
    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Resize and getattr(self, "figure_scroll", None) is not None
                and obj is self.figure_scroll.viewport() and not self._suspend_refit):
            self._refit_timer.start()
        return super().eventFilter(obj, event)

    def _viewport_inches(self):
        """视口尺寸 → figure 英寸（乘 device_pixel_ratio，高分屏上字才不会虚）。"""
        ratio = getattr(self.canvas, "device_pixel_ratio", 1.0) or 1.0
        dpi = self.figure.dpi or 110.0
        size = self.figure_scroll.viewport().size()
        return (max(size.width(), 360) * ratio / dpi,
                max(size.height(), 220) * ratio / dpi)

    def _avail_inches(self, params=None):
        """预览区能给图多大（英寸）：再乘用户选的「图形尺寸」倍数。

        倍数乘在这里而不是塞进 plots：绘图层只认「画布现在多大、内容要多大」，
        不认 UI 词表。选“放大”档不等于允许标签叠：内容量尺照走，只是地板提高了。
        """
        width_in, height_in = self._viewport_inches()
        scale = am.figure_scale(self.current_params() if params is None else params)
        return width_in * scale, height_in * scale

    def _fit_canvas(self, canvas):
        width_in, height_in = self._avail_inches()
        plots.fit_figure(canvas.figure, width_in, height_in)
        canvas.sync_to_figure()

    def _refit_figures(self):
        """预览区变宽/变窄后重画图：版式（旋转角、抽稀、图例列数）与画布宽绑定。

        只改 figure 尺寸不重画的话，标签仍按旧宽排 —— 宽了也还是叠在一起。
        但滚动条自己占的那十几像素不算“用户改了宽度”，否则撑大图 → 出竖滚条
        → 视口变窄 → 重画会来回振荡，故设一个阀值并记住上次量过的宽。
        """
        if self.figure_scroll is None:
            return
        canvases = self._all_canvases()
        if not canvases:
            return
        width_in, _height_in = self._avail_inches()
        if abs(width_in - getattr(self, "_fitted_width", 0.0)) < 0.35:
            return
        self._suspend_refit = True
        try:
            self._fitted_width = width_in
            if self.result is None:
                # 占位图没有内容量尺，拉一下尺寸就够，不值得重画
                for canvas in canvases:
                    self._fit_canvas(canvas)
                    canvas.draw_idle()
            else:
                self.redraw_chart()
        finally:
            self._suspend_refit = False

    def _all_canvases(self):
        out = [canvas for canvas in self._canvases if canvas.isVisible()]
        if self.canvas is not None and self.canvas.isVisible():
            out.append(self.canvas)
        return out

    def _preview_pixmaps(self):
        """当前预览区里那几张图的高清位图：全屏窗口按它们显示。

        拿不到“可见”画布时退回登记清单，而不是直接拿占位画布：整个工作台
        不是当前页时（页面隐藏）所有画布都不“可见”，拿占位图充数会把
        「尚未生成结果」当成结果图端进全屏窗口。占位只在真的没图时才是主角。
        """
        canvases = self._all_canvases() or self._canvases
        if not canvases and self.canvas is not None and not self.canvas.isHidden():
            canvases = [self.canvas]
        return [ap.figure_pixmap(canvas.figure) for canvas in canvases]

    def current_figure(self):
        """当前显示的第一张图：主题冒烟与外部看图只看这张，没图时回占位 figure。"""
        return next(iter(self.figures.values()), self.figure)

    # ---------------------------------------------------------------- 会话接入
    def attach_session(self, summary_tab=None, group_encoder=None, pages_provider=None,
                       group_id=None):
        """接入「当前编码会话」的三个取数口，主窗口只给对象、不摊开控件。"""
        self.summary_source = summary_tab
        self.group_source = group_encoder
        self.pages_provider = pages_provider
        self.session_group_id = group_id

    def default_group_id(self):
        value = getattr(self.group_source, "group_id", None)
        return str(value) if value else "group_1"

    # ---------------------------------------------------------------- 数据装载
    def _remember_batch(self, paths, label):
        self._batches[label] = list(paths)
        self._loading = True
        index = self.file_combo.findText(label)
        if index < 0:
            self.file_combo.insertItem(0, label)
            index = 0
        self.file_combo.setCurrentIndex(index)
        self._paths = list(paths)
        self._loading = False

    def select_file(self):
        paths = user_state.open_file_names(self, self.data_bucket(),
                                          "选择 Group 编码数据文件（可多选，一文件一组）",
                                          DATA_FILTER)
        if not paths:
            return None                  # 用户按了取消：不是故障，不弹错误框
        return self.load_files(paths)

    def data_bucket(self):
        """取数入口的目录记忆按工作台分桶： IND 常去的目录与 GRP 常去的目录不串味。"""
        return (user_state.BUCKET_INDIVIDUAL if self.mode == am.MODE_INDIVIDUAL
                else user_state.BUCKET_GROUP)

    def load_files(self, paths, group_ids=None):
        try:
            bundles = ac.load_bundles(paths, group_ids=group_ids)
        except Exception as exc:
            traceback.print_exc()
            self._empty_state("文件读取失败：%s" % exc)
            return None
        names = [bundle.group_id for bundle in bundles]
        self._remember_batch(list(paths), "%d 个文件：%s" % (len(paths), "、".join(names)))
        self.sheet = "批量上传"
        return self.adopt_bundles(bundles)

    def plot_data(self, path):
        """兼容旧入口名：单文件装载。"""
        return self.load_files([path] if isinstance(path, str) else list(path))

    def load_from_session(self):
        if self.summary_source is None and self.group_source is None:
            QMessageBox.information(self, "编码会话", "尚未接入编码页。")
            return None
        group_id = self.session_group_id or self.default_group_id()
        pages = list(self.pages_provider()) if callable(self.pages_provider) else []
        try:
            bundle = ac.bundle_from_session(group_id=group_id, summary_tab=self.summary_source,
                                            group_encoder=self.group_source, encoder_pages=pages)
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "取数失败", str(exc))
            return None
        self._remember_batch(["<当前编码会话>"], "当前编码会话（%s）" % bundle.group_id)
        self.sheet = "当前编码会话"
        return self.adopt_bundles([bundle])

    def import_framework(self, path=None):
        """导入手写框架 Excel；传了 path（「最近打开」重开）就不再弹文件框。"""
        if not path:
            path = user_state.open_file_name(self, user_state.BUCKET_FRAMEWORK,
                                            "导入编码框架 Excel（两列：维度 / 类型）",
                                            "Excel (*.xlsx *.xls);;所有文件 (*)")
        if not path:
            return None
        try:
            self._imported_codebook = ac.read_framework_excel(path)
        except Exception as exc:
            QMessageBox.critical(self, "导入框架失败", str(exc))
            return None
        self.framework_label.setText("框架：%d 个维度（手工导入）" % len(self._imported_codebook))
        return self.adopt_bundles(list(self.ctx.bundles)) if self.ctx is not None else None

    def _on_batch_changed(self, index):
        if self._loading or index < 0:
            return
        label = self.file_combo.itemText(index)
        paths = self._batches.get(label) or []
        if not paths:
            return
        self._paths = list(paths)
        if paths[0].startswith("<"):
            self.load_from_session()
        else:
            self.load_files(paths)

    # ---------------------------------------------------------------- 组号覆写
    def _group_columns(self):
        return list(GROUP_COLUMNS) + list(self.EXTRA_GROUP_COLUMNS)

    def _extra_group_values(self, bundle):
        """子类追加的列值：基类没附加列，故返空。"""
        return ()

    def _refresh_group_table(self, bundles):
        self.group_table.setRowCount(len(bundles))
        for row, bundle in enumerate(bundles):
            source = QTableWidgetItem(os.path.basename(bundle.source_file)
                                      if bundle.source_file else bundle.group_id)
            source.setData(Qt.UserRole, bundle.source_file)
            source.setFlags(source.flags() & ~Qt.ItemIsEditable)
            self.group_table.setItem(row, 0, source)
            self.group_table.setItem(row, 1, QTableWidgetItem(bundle.group_id))
            values = [len(bundle.members), bundle.n_times,
                      "是" if bundle.has_events else "否"]
            values.extend(self._extra_group_values(bundle))
            for offset, value in enumerate(values):
                column = 2 + offset
                if column >= self.group_table.columnCount():
                    break
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.group_table.setItem(row, column, item)

    def apply_group_overrides(self):
        if not self._paths or self._paths[0].startswith("<"):
            QMessageBox.information(self, "组号覆写", "只有批量上传的文件可以改组号。")
            return None
        overrides = [self.group_table.item(row, 1).text().strip()
                     for row in range(self.group_table.rowCount())]
        return self.load_files(self._paths, group_ids=[item for item in overrides if item])

    # ---------------------------------------------------------------- 上下文
    def resolve_framework(self, bundles=None):
        """框架三档：手工导入的 Excel > 当前会话编码页框架 > bundle 自带 / 数据推断。"""
        if self._imported_codebook:
            return list(self._imported_codebook)
        pages = list(self.pages_provider()) if callable(self.pages_provider) else []
        return ac.framework_from_pages(pages) or None

    def adopt_bundles(self, bundles):
        self._yield_previous_task()
        head = bundles[0] if bundles else None
        codebook = ac.resolve_codebook(head, framework=self.resolve_framework())
        ctx = am.AnalysisContext(bundles, codebook=codebook, mode=self.mode)
        ctx.progress = self.log
        self.ctx = ctx
        self._refresh_group_table(bundles)
        self._refresh_data_hint(ctx)
        self._populate_methods()
        self._rebuild_params()
        self.generate_btn.setEnabled(bool(ctx.usable))
        if not ctx.usable:
            self.result = None
            self._show_qa(None)
            self._empty_state("未解析出可用的编码列：应含 Time 与 S1Encoding_1 形态的列。")
            return ctx
        self.generate()
        return ctx          # 装载入口统一回上下文：结果看 self.result，不回 Result

    def _refresh_data_hint(self, ctx):
        if not ctx.usable:
            self.data_hint.setText("未解析出可用的编码列，请检查列名是否为 Time + S1Encoding_1。")
            return
        mode_name = "个体模式" if self.mode == am.MODE_INDIVIDUAL else "小组模式"
        self.data_hint.setText(
            "%s ｜ 组 %d ｜ 独立单元（组__学生）%d ｜ 维度 %s ｜ 最长序列 %d 步"
            % (mode_name, ctx.n_groups, ctx.n_students, "、".join(ctx.dimensions) or "—",
               ctx.max_sequence_length))

    # ---------------------------------------------------------------- 方法与参数
    def _populate_methods(self):
        keep = self.method_combo.currentData() if self.method_combo.count() else None
        self._loading = True
        self.method_combo.clear()
        for name, specs in am.method_groups(self.mode):
            self.method_combo.addItem("%s（%d）" % (name, len(specs)))
            self.method_combo.model().item(self.method_combo.count() - 1).setEnabled(False)
            for spec in specs:
                ok, note = spec.availability(self.ctx)
                self.method_combo.addItem(spec.title() if ok
                                          else "%s — %s" % (spec.title(), note), spec.id)
        self._loading = False
        self._request_left_fit()      # 下拉里的阻断说明会改方法卡宽度，先重量一次
        index = self._first_selectable()
        if keep:
            for position in range(self.method_combo.count()):
                if self.method_combo.itemData(position) == keep:
                    index = position
                    break
        if index < 0:
            self.spec = None
            return
        self.method_combo.setCurrentIndex(index)
        self._select_spec(self.method_combo.itemData(index))

    def _first_selectable(self):
        """第一个真正可选的方法行：分组标题不可选，拿它当默认值会算不出东西。"""
        for position in range(self.method_combo.count()):
            if self.method_combo.itemData(position):
                return position
        return -1

    def _select_spec(self, method_id):
        """切方法但不由此触发计算：装载数据时要先建参数卡、再由 adopt_bundles 统一算。"""
        if not method_id:
            return
        if self.spec is not None:
            self._params_cache[self.spec.id] = self._collect_params()
        self.spec = am.method_by_id(method_id)
        self._rebuild_params()
        self._update_method_hint()

    def method_id(self):
        return self.method_combo.currentData()

    def set_method(self, method_id):
        """测试与外部调用入口：切换方法并重建参数卡。"""
        for index in range(self.method_combo.count()):
            if self.method_combo.itemData(index) == method_id:
                self.method_combo.setCurrentIndex(index)
                return self.method_id()
        return None

    def _on_method_changed(self, _index):
        method_id = self.method_id()
        if not method_id or self._loading:
            return
        self._select_spec(method_id)
        if self.ctx is not None and self.ctx.usable:
            self.generate()
        else:
            self._empty_state("请先装载数据")

    def _update_method_hint(self):
        if self.spec is None:
            return
        ok, note = self.spec.availability(self.ctx)
        text = self.spec.question
        if self.spec.formula_note:
            text = "%s\n口径：%s" % (text, self.spec.formula_note)
        if not ok and note:
            self.method_hint.setText("%s\n当前不可用：%s" % (text, note))
            self.params_notice.setText(note)
            self.params_notice.setVisible(True)
            return
        self.params_notice.setVisible(False)
        downgrade = ac.group_count_note(self.ctx.bundles) if self.ctx is not None else ""
        if downgrade and any(str(item).startswith("groups") for item in (self.spec.requires or ())):
            text = "%s\n%s" % (text, downgrade)
        if self.spec.r_reference:
            text = "%s\nR 参考：%s" % (text, self.spec.r_reference)
        self.method_hint.setText(text)

    def _rebuild_params(self):
        if self.spec is None:
            return
        ap.clear_grid(self.params_grid)
        self._rows = ap.build_param_rows(self.spec, self.params_grid, self.ctx,
                                         on_change=self._value_changed,
                                         extra=am.unit_specs(self.spec),
                                         tail=am.chart_specs())
        cached = self._params_cache.get(self.spec.id)
        if cached is not None:
            ap.apply_params(self._rows, cached)
        ap.apply_visibility(self._rows)
        self._request_left_fit()      # 参数卡是左栏最宽的一卡，重建完就得重量宽度

    def _value_changed(self, key):
        ap.apply_visibility(self._rows)
        self._request_left_fit()      # 显隐规则换了一批可见行，最宽的那一行也可能跟着换
        if key not in am.RENDER_ONLY_PARAMS or self.result is None or self._busy:
            return
        # 图例位置与图形尺寸只改看不改算：重算一次会白训一轮模型（DL 方法要几分钟），
        # 而且算出来的数一点都不会变
        params = self._collect_params()
        self._params_cache[self.spec.id] = params
        self._render(self.result, params)

    def _collect_params(self):
        return ap.collect_params(self._rows)

    def current_params(self):
        if self.spec is None:
            return {}
        return dict(self._params_cache.get(self.spec.id) or self._collect_params())

    # ---------------------------------------------------------------- 计算与呈现
    def generate_now(self):
        """按钮入口（不带 `clicked(checked)` 参数）。"""
        return self.generate()

    def current_result(self):
        """当前结果；后台计尚未交货时返回 None（测试脚本用 `join_compute` 等）。"""
        return None if self._busy else self.result

    def generate(self, wait=False):
        """算一次：重方法（可选后端）默认交后台线程，其余当场算完。

        `wait=True` 强制同步跑完并返回结果（单测与冒烟脚本要一个能直接拿返回值的入口）；
        异步路径下本方法先返 None，结果由 `_on_compute_done` 回填。
        """
        if self.spec is None:
            return None
        if self.ctx is None or not self.ctx.usable:
            self._empty_state("请先装载数据")
            return None
        self._yield_previous_task()
        if self._busy:
            self._log_line("上一次计算还没结束：%s 本次请求忽略" % self.spec.id)
            return self.result
        params = self._collect_params()
        self._params_cache[self.spec.id] = params
        ok, note = self.spec.availability(self.ctx)
        self._update_method_hint()
        self.ctx.cancelled = False               # 新一轮：上轮的取消标志不留存
        if not ok:
            return self._finish(self.spec.blocked_result(self.ctx, note), params)
        if wait or self.spec.engine not in TRAINING_ENGINES:
            self.ctx.progress = self.log
            try:
                return self._finish(self.spec.compute(self.ctx, params), params)
            except Exception as exc:             # noqa: BLE001
                traceback.print_exc()
                self._log_line("计算失败：%s" % exc)
                self._empty_state("计算失败：%s" % exc)
                return None
        # 训练类：整段计算（含建模型）搬进 QThread，UI 线程只收 epoch 日志与最终结果
        task = ComputeWorker(self.spec, self.ctx, params, parent=self)
        task.progress.connect(self._log_line)
        task.done.connect(self._on_compute_done)
        self.ctx.progress = task.progress.emit
        self._task = task
        self._set_busy(True, self.spec.id)
        task.start()
        return None

    def _finish(self, result, params):
        """结果落界面：表 / 质检 / 绘图 / 清旧对照表，同步与异步两条路共用这一份。"""
        self.result = result
        self._fill_tables(result)
        self._show_qa(result)
        self._render(result, params)
        self._reset_crosscheck()
        return result

    def _on_compute_done(self, result, error):
        """后台计算回主线程的交货口（信号槽，不跨线程摸控件）。"""
        task, self._task = self._task, None
        params = (task.params if task is not None
                  else self._params_cache.get(self.spec.id if self.spec else "", {}))
        if self.ctx is not None:
            self.ctx.progress = self.log
        self._set_busy(False)
        if error is not None:
            traceback.print_exception(type(error), error, error.__traceback__)
            self._log_line("计算失败：%s" % error)
            self._empty_state("计算失败：%s" % error)
            return None
        self._finish(result, params)
        if task is not None:
            task.wait(200)                      # run() 已返，只等它把线程收尾，别析构在跑上的 QThread
        return self.result

    def _set_busy(self, busy, method_id=""):
        """后台训练期间只改两个按钮与日志：其它控件保持可读。"""
        self._busy = bool(busy)
        self.cancel_btn.setVisible(self._busy)
        self.cancel_btn.setEnabled(self._busy)
        self.cancel_btn.setText("取消计算")
        self.generate_btn.setEnabled(not self._busy and bool(self.ctx and self.ctx.usable))
        if self._busy:
            self.aux_tabs.setCurrentIndex(1)      # 跳到「训练日志」：得让用户看到在动
            self._log_line("%s：计算已交后台线程，进度见本卡；可随时取消" % method_id)

    def cancel_current(self):
        """取消 = 给 `ctx.cancelled` 挂标志：计算层在下一个 epoch / 折检查点自己收摊。"""
        if self._task is None or not self._task.isRunning():
            return False
        if self.ctx is not None:
            self.ctx.cancelled = True
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("已请求取消…")
        self._log_line("已请求取消：当前 epoch / 折跑完就返回部分结果")
        return True

    def _yield_previous_task(self, timeout_ms=15000):
        """换方法 / 换数据前先让上一次计算收摊：置取消位，等它自己退出。"""
        if self._task is None:
            return True
        if self.ctx is not None:
            self.ctx.cancelled = True
        return self.join_compute(timeout_ms=timeout_ms)

    def join_compute(self, timeout_ms=600000):
        """等后台计算跑完（含信号回主线程）：只给测试脚本与退出清理用。"""
        task = self._task
        if task is None:
            return True
        deadline = time.monotonic() + max(float(timeout_ms), 0.0) / 1000.0
        while task.isRunning() and time.monotonic() < deadline:
            QApplication.processEvents()          # 信号在主线程投递，得回到事件循环
            time.sleep(0.02)
        if task.isRunning():
            self.cancel_current()
            task.wait(5000)
        QApplication.processEvents()
        return self._task is None

    def _render(self, result, params):
        self._clearCanvases()
        self.figures = {}
        self.figure_hint.setVisible(False)
        avail = self._avail_inches(params)
        # 本轮就是按这个宽度排的版：下次视口宽与它差过一个阀值才值得重画
        self._fitted_width = avail[0]
        try:
            # 两遍绘制在 plots.render 里做（量尺寸 → 定尺寸 → 重排），这里只给视口事实
            self.figures = plots.render(result.method_id, result, self.ctx, params,
                                        avail=avail)
        except Exception as exc:
            traceback.print_exc()
            self.figure_hint.setText("绘图失败：%s" % exc)
            self.figure_hint.setVisible(True)
        self.figure_combo.blockSignals(True)
        self.figure_combo.clear()
        self.figure_combo.blockSignals(False)
        if not self.figures:
            self._draw_placeholder(getattr(result, "blocked_reason", "")
                                   or "本方法没有可呈现的图形（结果只有表格）")
            self.figure_combo.setVisible(False)
            return
        names = list(self.figures)
        for title in names:
            canvas = ap.PreviewCanvas(self.figures[title])
            self._canvases.append(canvas)
            self.figure_box.addWidget(canvas, 0, Qt.AlignHCenter)
            canvas.show()
            self._fit_canvas(canvas)
            canvas.draw()
        if len(names) > 1:
            self.figure_combo.addItems(names)
            self.figure_combo.setVisible(True)

    def _show_figure_at(self, index):
        """多张图竖排滚动呈现，下拉只做定位：一次看全比只看一张有用。"""
        bar = self.figure_scroll.verticalScrollBar()
        canvases = self._canvases
        if 0 <= index < len(canvases):
            bar.setValue(max(int(canvases[index].geometry().top()), 0))

    def _clearCanvases(self):
        for canvas in self._canvases:
            self.figure_box.removeWidget(canvas)
            canvas.setParent(None)
            canvas.deleteLater()
        self._canvases = []
        self.figure_box.removeWidget(self.canvas)
        # removeWidget 不会 hide：占位画布只摘走不收起的话，它会以上一次的几何继续
        # 挂在 figure_host 上。于是真实结果画完之后，预览区下方多出一整块旧占位图
        # （「尚未生成结果」），全屏查看还把它当成第二张图一起端进去。
        # _draw_placeholder 里自带 show，这里收起来不会把占位态弄没。
        self.canvas.setVisible(False)
        self.figures = {}

    def _draw_placeholder(self, message):
        # 占位与结果图是互斥的两种状态，进来先清一轮：redraw_chart 在 result/spec
        # 为空时（方法不可选、批量数据换了口径、取消计算后）会直接走到这里，而画布
        # 可能还挂着上一轮的结果图 —— 那样下面那句 addWidget 会把占位画布**追加到
        # 结果图底下**，预览区就成了「图 + 一块写着尚未生成结果的白底」。
        self._clearCanvases()
        fig = self.figure
        fig.clear()
        theme.sync_figure(fig)
        width_in, height_in = self._avail_inches()
        plots.fit_figure(fig, width_in, height_in)
        axes = fig.add_subplot(111)
        plots.style(axes)
        axes.axis("off")
        axes.text(0.5, 0.5, str(message), ha="center", va="center",
                  fontsize=10, color=theme.tokens()["ink3"], wrap=True)
        self.canvas.figure = fig
        self.figure_box.addWidget(self.canvas, 0, Qt.AlignHCenter)
        self.canvas.show()
        self._fit_canvas(self.canvas)
        self.canvas.draw_idle()

    def redraw_chart(self):
        """主题切换后用新配色重画当前结果。"""
        if self.result is not None and self.spec is not None:
            self._render(self.result, self.current_params())
            return
        self._draw_placeholder("尚未生成结果")

    # ---------------------------------------------------------------- 结果表
    def _fill_tables(self, result):
        # clear() 只摘页不删控件：不手工 deleteLater 的话，每次切方法都会漏一整批表格
        while self.result_tabs.count():
            page = self.result_tabs.widget(0)
            self.result_tabs.removeTab(0)
            if page is not None:
                page.setParent(None)
                page.deleteLater()
        tables = {name: frame for name, frame in (result.tables or {}).items()
                  if name != "r_crosscheck"}      # 这张表归「R 交叉校验」卡专门展示
        if not tables:
            self.result_tabs.addTab(self._note_table(result.blocked_reason or "暂无结果"), "提示")
            return
        for name, frame in tables.items():
            self.result_tabs.addTab(self._make_table(frame), "%s (%d)" % (name, len(frame)))

    def _note_table(self, message):
        table = QTableWidget(1, 1)
        table.setItem(0, 0, QTableWidgetItem(str(message)))
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setVisible(False)
        return table

    def _make_table(self, frame):
        table = QTableWidget(len(frame), max(len(frame.columns), 1))
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setProperty("gridTable", True)   # 补一圈外框，见 theme 的表格段
        table.setHorizontalHeaderLabels([str(col) for col in frame.columns])
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        # 不开隔行色：本项目 QSS 写了 QTableWidget::item，Qt 会忽略 AlternateBase
        header.setStretchLastSection(True)
        for row, (_, record) in enumerate(frame.iterrows()):
            for column, value in enumerate(record.tolist()):
                table.setItem(row, column, QTableWidgetItem(self._cell_text(value)))
        translate_widget_tree(table)
        return table

    @staticmethod
    def _cell_text(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return MISSING_TEXT
        if isinstance(value, bool):
            return "是" if value else "否"
        if isinstance(value, (int, float)):
            return fmt_num(float(value))
        return str(value)

    # ---------------------------------------------------------------- 质检卡
    def _show_qa(self, result):
        self.qa_list.clear()
        combined = merge_qa(self.ctx.quality() if self.ctx is not None else None,
                            list(getattr(result, "qa", []) or []))
        combined.extend(self.r_probe_items())
        combined.extend(ap.Note(message) for message in (getattr(result, "notes", []) or []))
        combined.sort(key=lambda item: LEVEL_ORDER.get(getattr(item, "level", "info"), 2))
        for item in combined:
            row = QListWidgetItem("%s %s" % (ap.prefix_of(item), item.message))
            row.setData(Qt.UserRole, getattr(item, "code", ""))
            self.qa_list.addItem(row)
        errors, warns, infos = qa_summary(combined)
        head = "阻断 %d · 警告 %d · 提示 %d" % (errors, warns, infos)
        if getattr(result, "blocked", False):
            head = "结果已阻断：%s\n%s" % (result.blocked_reason, head)
        elif result is None:
            head = "未生成结果：" + head
        self.qa_notice.setText(head)

    def r_probe_items(self):
        """R 参考层状态进质检卡：只报「有没有、缺哪些包」，不影响任何计算。"""
        status = self._probe_r()
        if not status:
            return []
        if not status.get("available"):
            return [QAItem("info", "r.unavailable",
                           "未探测到 Rscript：R 交叉校验卡降级为提示，Python 实现照常可用", False)]
        packages = status.get("packages") or {}
        missing = [name for name, found in packages.items() if not found]
        note = "R %s 可用：%s" % (status.get("version", "?"),
                                  "、".join(status.get("found") or []) or "无参考包")
        if missing:
            note += "；缺包 %s（可跑 r_refs/setup_r_reference.R 一键安装）" % "、".join(missing)
        return [QAItem("info", "r.reference", note, False)]

    def _probe_r(self):
        """已探测到的 R 状态；未探到时返回空，等 `ProbeWorker` 回填（绝不在此阻塞 UI）。"""
        return self._r_status or {}

    def _bridge(self):
        """桥接层是可选件：拿不到就返回 None，R 校验卡退回提示态。"""
        if self._bridge_missing:
            return None
        if self._bridge_ref is None:
            try:
                import analysis_r_bridge as bridge
            except ImportError:
                self._bridge_missing = True
                return None
            self._bridge_ref = bridge
        return self._bridge_ref

    def showEvent(self, event):
        """首次显示时才后台探测 R：工作台建了却没打开时不该去起 Rscript。"""
        super().showEvent(event)
        self.ensure_r_probe()

    def ensure_r_probe(self):
        """后台探测 R 与参考包（幂等）：结果回写质检卡与本卡状态行。"""
        if self._r_status is not None or self._probe_task is not None:
            return self._r_status or {}
        bridge = self._bridge()
        if bridge is None:
            self._r_status = {}
        elif getattr(bridge, "ProbeWorker", None) is None:    # 无 Qt：只有纯脚本调用时才会这样
            self._r_status = dict(bridge.probe())
        else:
            self._probe_task = bridge.ProbeWorker(parent=None)
            self._probe_task.done.connect(self._on_r_probe)
            self._probe_task.start()
            return self._r_status or {}
        self._sync_r_status_label()
        return self._r_status or {}

    def _on_r_probe(self, status):
        self._probe_task = None
        self._r_status = dict(status or {})
        self._sync_r_status_label()
        if self.result is not None:
            self._show_qa(self.result)                   # 补上质检卡里那条 R 状态

    def _sync_r_status_label(self):
        items = self.r_probe_items()
        self.r_status_label.setText("\n".join(item.message for item in items)
                                    or "R 参考层：尚未探测。")

    def join_r_tasks(self, timeout_ms=3000):
        """退出前给后台 R 进程一个收尾窗口：`QThread` 还在跑就被销毁会崩。"""
        for task in (self._probe_task, self._r_task, self._install_task):
            if task is not None:
                try:
                    task.wait(int(timeout_ms))
                except Exception:                        # noqa: BLE001  收尾失败不挡退出
                    pass
        return True

    def _empty_state(self, message):
        self.result = None
        self.figures = {}
        self.result_tabs.clear()
        self.result_tabs.addTab(self._note_table(message), "提示")
        self.qa_notice.setText(message)
        self._clearCanvases()
        self._draw_placeholder(message)
        self.figure_combo.setVisible(False)
        self._reset_crosscheck()
        self._request_left_fit()

    # ---------------------------------------------------------------- R 交叉校验
    def _reset_crosscheck(self):
        """重算一次就清掉上回的差异表：数据或参数变了，旧对照不能再作准。"""
        self.cross_table.setRowCount(0)
        self._fill_crosscheck(None)

    def _fill_crosscheck(self, result, note=None):
        """填差异表与说明：`note` 是桥接层给的人话（R 版本 / 脚本 / 计数），有就用它。"""
        frame = ae.crosscheck_frame(result) if result is not None else pd.DataFrame()
        if not len(frame):
            self.cross_table.setRowCount(0)
            if note:
                self.cross_hint.setText(note)
                return
            reference = getattr(self.spec, "r_reference", "") if self.spec is not None else ""
            self.cross_hint.setText(
                "本方法没有 R 参考结果%s：点「用 R 复算并对照」现跑，或以 Python 为准。" % (
                    "（注册表声明 %s）" % reference if reference else "（注册表未声明 r_reference）"))
            return
        self.cross_table.setColumnCount(len(frame.columns))
        self.cross_table.setHorizontalHeaderLabels([str(col) for col in frame.columns])
        translate_widget_tree(self.cross_table)
        self.cross_table.setRowCount(len(frame))
        for row, (_, record) in enumerate(frame.iterrows()):
            for column, value in enumerate(record.tolist()):
                self.cross_table.setItem(row, column, QTableWidgetItem(self._cell_text(value)))
        if note:
            self.cross_hint.setText(note)
            return
        worst = None
        if "绝对差" in frame.columns:
            worst = pd.to_numeric(frame["绝对差"], errors="coerce").max()
        self.cross_hint.setText("R 对照 %d 格：最大绝对差 %s" % (
            len(frame), MISSING_TEXT if worst is None or pd.isna(worst) else worst))

    def _apply_crosscheck(self, rows, note):
        self._fill_crosscheck(_ResultLike(pd.DataFrame(list(rows or []))), note=note)

    def run_crosscheck(self, sync=False):
        """用 R 参考实现复算当前方法并填差异表。

        默认丢给 `ReferenceWorker` 后台跑（Rscript 最长等几十秒，界面不陪它卡）；
        `sync=True` 时直接在调用线程跑完并返回差异表 —— 离屏冒烟脚本要的是确定结果，
        不是信号时序。桥接层缺席时本卡只给降级提示。
        """
        if self.spec is None or self.result is None:
            if not sync:
                QMessageBox.information(self, "R 交叉校验", "请先生成结果。")
            return None
        bridge = self._bridge()
        if bridge is None:
            self.cross_hint.setText("未探测到 R 桥接层（analysis_r_bridge）：本卡保持提示态。")
            return None
        params = self.current_params()
        worker = getattr(bridge, "ReferenceWorker", None)
        if sync or worker is None:
            try:
                rows, note = bridge.reference_check(self.spec, self.result, self.ctx, params)
            except Exception as exc:                     # noqa: BLE001
                traceback.print_exc()
                self._apply_crosscheck([], "R 复算异常：%s" % exc)
                return None
            self._apply_crosscheck(rows, note)
            return rows
        if self._r_task is not None:                     # 上一次还在跑：不堆 Rscript 进程
            return None
        self._r_task = worker(self.spec, self.result, self.ctx, params, parent=None)
        self.cross_btn.setEnabled(False)
        self.cross_hint.setText("R 复算中……（最长 %d 秒，界面可继续操作）"
                                % int(getattr(bridge, "TIMEOUT_SECONDS", 120)))
        self._r_task.done.connect(self._on_crosscheck)
        self._r_task.start()
        return None

    def _on_crosscheck(self, rows, note):
        self._r_task = None
        self.cross_btn.setEnabled(True)
        self._apply_crosscheck(rows, note)

    # ---------------------------------------------------------------- R 参考包安装
    def copy_install_command(self):
        """把安装命令丢进剪贴板：离线机器 / 换台机器时最省事的一条路。"""
        bridge = self._bridge()
        if bridge is None:
            self.r_status_label.setText("未探测到 R 桥接层：无安装命令可复制。")
            return ""
        command = bridge.install_command(self.repo_edit.text().strip())
        try:
            from PySide6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(command)
        except Exception:                                # noqa: BLE001
            pass
        self._log_line("R 参考包安装命令：%s" % command)
        self.r_status_label.setText("安装命令已复制到剪贴板：%s" % command)
        return command

    def install_r_packages(self):
        """直接执行 `r_refs/setup_r_reference.R`：CRAN 下载编译可能十几分钟，必在子线程。"""
        bridge = self._bridge()
        if bridge is None:
            self.r_status_label.setText("未探测到 R 桥接层：无法安装。")
            return None
        if self._install_task is not None:
            return None
        reply = QMessageBox.question(
            self, "安装 R 参考包",
            "将调用 Rscript 把 TraMineR / seqHMM / dtwclust / rENA / jsonlite / readxl "
            "装到本应用目录下的 Rlibrary。\n联网下载可能持续十几分钟；"
            "装不上也不影响本工作台（Python 实现始终为准）。\n\n继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return None
        repo = self.repo_edit.text().strip()
        worker = getattr(bridge, "InstallWorker", None)
        if worker is None:
            report = bridge.install_references(repo=repo)
            self._on_install_done(report)
            return report
        self._install_task = worker(repo=repo, parent=None)
        self.install_btn.setEnabled(False)
        self._log_line("开始安装 R 参考包（镜像：%s）……" % (repo or "默认 CRAN"))
        self._install_task.done.connect(self._on_install_done)
        self._install_task.start()
        return None

    def _on_install_done(self, report):
        self._install_task = None
        self.install_btn.setEnabled(True)
        report = dict(report or {})
        for line in str(report.get("text") or "").splitlines():
            if line.strip():
                self._log_line(line)
        packages = report.get("packages") or {}
        found = "、".join(sorted(name for name, hit in packages.items() if hit)) or "无"
        self._r_status = None                            # 装完重探：包状态可能整批变了
        self.ensure_r_probe()
        self.r_status_label.setText("R 参考包安装%s（库目录 %s）：可用 %s。" % (
            "已执行" if report.get("ok") else "未成功（%s）" % report.get("status", "failed"),
            report.get("library") or "data/Rlibrary", found))

    # ---------------------------------------------------------------- 训练日志
    def log(self, *messages):
        """`ctx.progress` 的落点：DL 侧每 epoch 调一次，这里逐行显示。"""
        text = " ".join(str(item) for item in messages if str(item).strip())
        if text:
            self._log_line(text)

    def _log_line(self, text):
        self.log_view.appendPlainText(str(text))

    # ---------------------------------------------------------------- 导出
    def _exportable(self):
        """有无可导出的结果 —— 不弹窗的那一份判据，供 ``*_to`` 编程式调用用。"""
        return self.result is not None and not (self.result.blocked and not self.result.tables)

    def _require_result(self):
        if not self._exportable():
            QMessageBox.warning(self, "导出", "请先选择一个可用方法并生成结果。")
            return False
        return True

    def export_statistics_to(self, path):
        return [ae.export_tables(self.result, path)] if self._exportable() else []

    def export_chart_to(self, path):
        """一张图就直接写；多张图按 ``_1/_2`` 依次写，路径由调用方给基础名。"""
        if not self._exportable():
            return []
        figures = list(self.figures.values()) or [self.figure]
        stem, ext = os.path.splitext(path)
        written = []
        for index, fig in enumerate(figures, start=1):
            target = path if len(figures) == 1 else "%s_%d%s" % (stem, index, ext)
            written.append(ae.export_figure(fig, target, dpi=int(self.dpi_combo.currentText())))
        return written

    def export_metadata_to(self, path):
        if not self._exportable():
            return []
        directory = os.path.dirname(path) or "."
        config = os.path.join(directory, ae.file_name(ae.CONFIG_KIND, self.result, "json"))
        audit = os.path.join(directory, ae.file_name(ae.AUDIT_KIND, self.result, "json"))
        card = os.path.join(directory, ae.file_name(ae.MODEL_CARD_KIND, self.result, "json"))
        ae.export_config_ctx(self.result, self.ctx, config, spec=self.spec, sheet=self.sheet)
        ae.export_audit_ctx(self.result, self.ctx, audit)
        ae.export_model_card(self.result, self.ctx, card, spec=self.spec,
                             params=self.current_params())
        return [config, audit, card]

    def export_all_to(self, directory):
        if not self._exportable() or not directory:
            return []
        return ae.export_all(self.result, self.ctx, directory, spec=self.spec,
                             params=self.current_params(), figures=self.figures,
                             sheet=self.sheet, dpi=int(self.dpi_combo.currentText()))

    def export_statistics(self):
        if not self._require_result():
            return None
        path = user_state.save_file_name(
            self, user_state.BUCKET_EXPORT, "导出统计表",
            ae.file_name(ae.TABLE_KIND, self.result, "xlsx"),
            "Excel 工作簿 (*.xlsx);;CSV (*.csv)")
        return self._guarded(self.export_statistics_to, path)

    def export_chart(self):
        if not self._require_result():
            return None
        fmt = self.format_combo.currentText().lower()
        path = user_state.save_file_name(
            self, user_state.BUCKET_EXPORT, "导出图形",
            ae.file_name(ae.FIGURE_KIND, self.result, fmt),
            "%s (*.%s);;所有文件 (*)" % (fmt.upper(), fmt))
        return self._guarded(self.export_chart_to, path)

    def export_metadata(self):
        if not self._require_result():
            return None
        path = user_state.save_file_name(
            self, user_state.BUCKET_EXPORT, "导出运行配置",
            ae.file_name(ae.CONFIG_KIND, self.result, "json"), "JSON (*.json)")
        return self._guarded(self.export_metadata_to, path)

    def export_everything(self):
        if not self._require_result():
            return None
        directory = user_state.pick_directory(self, user_state.BUCKET_EXPORT, "一键导出到目录")
        return self._guarded(self.export_all_to, directory)

    def _guarded(self, action, argument):
        if not argument:
            return None
        try:
            written = [item for item in action(argument) if item]
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "导出失败", str(exc))
            return None
        names = [str(item) for item in written]
        self._last_export = names[0] if names else ""
        self.export_hint.setText("已写出 %d 个文件到 %s"
                                 % (len(names), os.path.dirname(names[0]) or "."))
        return names

    # ---------------------------------------------------------------- 存档
    def archive_state(self):
        """工作台自身不入存档（数据永远从编码页现取），只记方法/参数/最近批次观感。"""
        return {"mode": self.mode, "method": self.method_id(),
                "params": {key: value for key, value in self._params_cache.items()
                           if isinstance(value, dict)},
                "batch": self.file_combo.currentText()}

    def restore_state(self, state):
        state = state or {}
        for key, value in (state.get("params") or {}).items():
            self._params_cache[str(key)] = dict(value or {})
        method = state.get("method")
        if method:
            self.set_method(method)


class _ResultLike(object):
    """给 `_fill_crosscheck` 用的轻量壳：只带一张 r_crosscheck 表。"""

    def __init__(self, frame):
        self.tables = {"r_crosscheck": frame}
        self.meta = {}
