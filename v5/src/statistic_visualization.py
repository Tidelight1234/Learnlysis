# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""统计及可视化：M01–M16 方法工作台。

左列是「数据 → 方法 → 参数 → 质检 → 导出」五张卡，右列上表下图。
本文件只做 UI 编排：数据解析在 statistic_visualization_data，统计在
statistic_visualization_stats，绘图在 statistic_visualization_plots，方法与参数声明在
statistic_visualization_methods，落盘在 statistic_visualization_export。
命名统一用 statistic_visualization 而非 analysis，是为了和后续的推断性分析模块区分开。

对外契约（main_window / _smoke_theme 依赖）：
    StatisticVisualizationModule()  无参构造
    select_file()             「选择文件」按钮语义
    plot_data(path)           load_file(path) 的兼容别名
    redraw_chart()            主题切换后用新 rcParams 重绘当前方法
    file_combo / figure / canvas
"""

import os
import traceback

import pandas as pd
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                             QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QListWidget, QListWidgetItem,
                             QScrollArea, QSpinBox, QSplitter, QTabWidget,
                             QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget, QPushButton, QMessageBox)
from matplotlib.figure import Figure

import theme
from language import translate_widget_tree
import analysis_params as ap
import user_state
from statistic_visualization_data import (CodingDataset, fmt_num, load_dataframe, merge_qa,
                                          qa_summary)
from statistic_visualization_export import (AUDIT_KIND, CONFIG_KIND, FIGURE_KIND, TABLE_KIND,
                                            export_all, export_audit, export_config,
                                            export_figure, export_tables, file_name)
from statistic_visualization_methods import GROUP_ORDER, METHODS, method_spec
import statistic_visualization_plots as plots

# 小控件、常量与参数行工厂已迁到 analysis_params（个体/小组两排工作台共用同一份）；
# 旧名字在此保留，_verify_stats_ui.py 等调用点不改。
_CheckList = ap.CheckList
_Segmented = ap.Segmented
_PreviewCanvas = ap.PreviewCanvas
_Note = ap.Note
_prefix_of = ap.prefix_of
_combo_width = ap.combo_width
MISSING_TEXT = ap.MISSING_TEXT
LEVEL_PREFIX = ap.LEVEL_PREFIX
LEFT_PANEL_W = ap.LEFT_PANEL_W

#: 取数入口能读的文件类型：与 IND/GRP 两个工作台同一套口径
DATA_FILTER = ("编码数据 (*.xlsx *.xls *.csv);;Excel (*.xlsx *.xls);;CSV (*.csv);;所有文件 (*)")
#: 右侧「表 / 图」两卡之间的透明间隙：除要给卡片阴影（blur 28 / offset 6）
#: 留出落地的地方，还与 IND/GRP 两个工作台的 CARD_GAP 同档，三页观感一致。
CARD_GAP = 16


# ------------------------------------------------------------------ 主模块

class StatisticVisualizationModule(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.result = None
        self.spec = None
        self.summary_source = None
        self.sheet = None
        self._params_cache = {}
        self._rows = []                 # [(ParamSpec, 行容器, 控件)]
        self._loading = False
        self._last_export = ""
        self._placeholder = ""          # 预览区当前的占位文案（空 = 画布本来全空）
        self.figure_scroll = None
        self._refit_timer = None
        self._suspend_refit = False
        self.initUI()
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)
        self.set_method(METHODS[0].id)

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
        host.setFixedWidth(LEFT_PANEL_W)
        outer = QVBoxLayout(host)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        self.left_scroll = scroll
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
        return host

    def _build_data_card(self):
        card, lay = self._make_card("数据")

        row = QHBoxLayout()
        row.setSpacing(8)
        self.file_btn = QPushButton("选择文件")
        self.file_btn.setProperty("btn", "secondary")
        self.file_btn.setFixedHeight(32)
        theme.bind_icon(self.file_btn, "import", 16)
        self.file_btn.clicked.connect(self.select_file)
        row.addWidget(self.file_btn)
        self.summary_btn = QPushButton("从数据汇总取数")
        self.summary_btn.setProperty("btn", "secondary")
        self.summary_btn.setFixedHeight(32)
        theme.bind_icon(self.summary_btn, "tab-data", 16)
        self.summary_btn.clicked.connect(self.load_from_summary)
        row.addWidget(self.summary_btn)
        lay.addLayout(row)

        self.file_combo = QComboBox()
        self.file_combo.setProperty("i18nData", True)
        self.file_combo.setToolTip("最近使用过的文件")
        theme.bind_combo_popup(self.file_combo)
        self.file_combo.currentTextChanged.connect(self._on_file_changed)
        lay.addWidget(self.file_combo)

        self.data_hint = QLabel("尚未装载数据：选择 Excel / CSV，或直接从数据汇总页取数。")
        self.data_hint.setProperty("hint", True)
        self.data_hint.setWordWrap(True)
        lay.addWidget(self.data_hint)

        time_row = QHBoxLayout()
        time_row.setSpacing(8)
        time_label = QLabel("时间列")
        time_label.setProperty("hint", True)
        time_row.addWidget(time_label)
        self.time_combo = QComboBox()
        self.time_combo.setProperty("i18nData", True)
        self.time_combo.setEnabled(False)
        theme.bind_combo_popup(self.time_combo)
        time_row.addWidget(self.time_combo, 1)
        lay.addLayout(time_row)

        map_label = QLabel("列 → 学生 / 维度映射（可直接改单元格，留空即剔除该列）")
        map_label.setProperty("hint", True)
        lay.addWidget(map_label)
        self.map_table = QTableWidget(0, 3)
        self.map_table.setHorizontalHeaderLabels(["源列名", "学生 ID", "编码维度"])
        self.map_table.verticalHeader().setVisible(False)
        self.map_table.setEditTriggers(QTableWidget.AllEditTriggers)
        self.map_table.setMaximumHeight(132)
        map_header = self.map_table.horizontalHeader()
        map_header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2):                       # 学生/维度两列要能容下编辑框，固定给宽
            map_header.setSectionResizeMode(col, QHeaderView.Interactive)
            self.map_table.setColumnWidth(col, 92)
        lay.addWidget(self.map_table)

        self.apply_btn = QPushButton("应用时间列与列映射")
        self.apply_btn.setProperty("btn", "ghost")
        self.apply_btn.setFixedHeight(28)
        self.apply_btn.clicked.connect(self.apply_mapping)
        lay.addWidget(self.apply_btn)
        return card

    def _build_method_card(self):
        card, lay = self._make_card("分析方法")
        self.method_combo = QComboBox()
        self.method_combo.setToolTip("M01–M16，按 A 基础频次 … F 路径呈现分组")
        theme.bind_combo_popup(self.method_combo)
        for group in GROUP_ORDER:
            specs = [spec for spec in METHODS if spec.group == group]
            if not specs:
                continue
            self.method_combo.addItem("%s（%d）" % (group, len(specs)))
            self.method_combo.model().item(self.method_combo.count() - 1).setEnabled(False)
            for spec in specs:
                self.method_combo.addItem(spec.title(), spec.id)
        lay.addWidget(self.method_combo)
        self.method_hint = QLabel("")
        self.method_hint.setProperty("hint", True)
        self.method_hint.setWordWrap(True)
        lay.addWidget(self.method_hint)
        return card

    def _build_params_card(self):
        card, lay = self._make_card("参数设置")
        self.params_host = QWidget()
        self.params_grid = QGridLayout(self.params_host)
        self.params_grid.setContentsMargins(0, 0, 0, 0)
        self.params_grid.setHorizontalSpacing(8)
        self.params_grid.setVerticalSpacing(6)
        self.params_grid.setColumnStretch(1, 1)
        lay.addWidget(self.params_host)
        self.generate_btn = QPushButton("生成结果")
        self.generate_btn.setProperty("btn", "primary")
        self.generate_btn.setFixedHeight(34)
        self.generate_btn.setEnabled(False)
        theme.bind_icon(self.generate_btn, "calculate", 16, "#FFFFFF")
        self.generate_btn.clicked.connect(self.generate)
        lay.addWidget(self.generate_btn)
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
        self.format_combo.setMinimumWidth(_combo_width(self.format_combo))
        theme.bind_combo_popup(self.format_combo)
        figure_row.addWidget(self.format_combo)
        self.dpi_combo = QComboBox()
        self.dpi_combo.addItems(["300", "600"])
        self.dpi_combo.setMinimumWidth(_combo_width(self.dpi_combo))
        self.dpi_combo.setToolTip("仅对 PNG 生效")
        theme.bind_combo_popup(self.dpi_combo)
        figure_row.addWidget(self.dpi_combo)
        lay.addLayout(figure_row)

        self.meta_btn = QPushButton("导出配置与质检")
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

        self.export_hint = QLabel("文件名统一为 类型_方法_分析名_analysis_id，四类产物同前缀。")
        self.export_hint.setProperty("hint", True)
        self.export_hint.setWordWrap(True)
        lay.addWidget(self.export_hint)
        return card

    def _build_right(self):
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        # 上下两张卡之间要留出比阴影（blur 28 / offset 6）更宽的透明间隙，
        # 否则卡片的下阴影会贴在分割条上，看着像底下压了一块灰矩形。
        splitter.setProperty("cardGap", True)
        splitter.setHandleWidth(CARD_GAP)

        table_card, table_lay = self._make_card("结果统计表")
        self.result_tabs = QTabWidget()
        self.result_tabs.setObjectName("innerTab")
        # 与编码页标签栏同一观感：笔直下划线；fit_text 保证表名不被截断
        theme.install_underline_tab_bar(self.result_tabs, fit_text=True)
        table_lay.addWidget(self.result_tabs)

        figure_card, figure_lay = self._make_card("图形预览")
        self.figure = Figure(figsize=(9, 5.2))
        self.canvas = _PreviewCanvas(self.figure)
        self.figure_scroll = QScrollArea()
        self.figure_scroll.setFrameShape(QScrollArea.NoFrame)
        # 与 IND/GRP 工作台同一层结构：画布装进宿主件，宿主铺满视口。
        # 直接把画布当 scroll widget（不可拉伸）时，画布之外的那圈只剩
        # QScrollArea 的 viewport 裸面：样式表里它既不透明也没人画，切主题后
        # 就一直端着上一个主题的像素（深色下露白块、换回浅色露黑块）。
        self.figure_scroll.setWidgetResizable(True)
        self.figure_scroll.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.figure_scroll.setToolTip("图形按内容自动放大，超出预览区时可滚动查看完整内容")
        self.figure_host = QWidget()
        self.figure_box = QVBoxLayout(self.figure_host)
        self.figure_box.setContentsMargins(0, 0, 0, 0)
        self.figure_box.setSpacing(8)
        self.figure_box.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.figure_box.addWidget(self.canvas, 0, Qt.AlignTop | Qt.AlignHCenter)
        self.figure_scroll.setWidget(self.figure_host)
        self._refit_timer = QTimer(self)
        self._refit_timer.setSingleShot(True)
        self._refit_timer.setInterval(120)
        self._refit_timer.timeout.connect(self._refit_figure)
        self.figure_scroll.viewport().installEventFilter(self)
        figure_lay.addWidget(self.figure_scroll)
        # 预览区右下角的浮动「全屏查看」：与 IND/GRP 工作台同一个控件
        self.fullscreen_btn = ap.attach_fullscreen_button(
            self.figure_scroll, self._preview_pixmaps, "结果图")
        self.canvas.draw()

        splitter.addWidget(table_card)
        splitter.addWidget(figure_card)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([260, 460])
        return splitter

    # ------------------------------------------------------ 预览图尺寸

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Resize and self.figure_scroll is not None
                and obj is self.figure_scroll.viewport() and not self._suspend_refit):
            self._refit_timer.start()
        return super().eventFilter(obj, event)

    def _preview_pixmaps(self):
        """当前预览图的高清位图：全屏窗口按它显示（占位图也照走，不特殊处理）。"""
        return [ap.figure_pixmap(self.figure)]

    def _preview_inches(self):
        """视口尺寸 → figure 英寸（乘上 device_pixel_ratio，高分屏上字才不会虚）。"""
        ratio = getattr(self.canvas, "device_pixel_ratio", 1.0) or 1.0
        dpi = self.figure.dpi or 100.0
        size = self.figure_scroll.viewport().size()
        return (max(size.width(), 360) * ratio / dpi,
                max(size.height(), 240) * ratio / dpi)

    def _apply_figure_size(self):
        """宽度铺满预览区，高度按内容声明值与视口取大——宁可滚动也不压缩标签。"""
        width_in, height_in = self._preview_inches()
        self._suspend_refit = True
        try:
            size = plots.fit_figure(self.figure, width_in, height_in)
            self.canvas.sync_to_figure()
        finally:
            self._suspend_refit = False
        return float(size[0]), float(size[1])

    def _refit_figure(self):
        """预览区变宽/变窄后重设尺寸：版式跟画布宽度绑在一起，只改尺寸不重排会错位。"""
        if self.figure_scroll is None or self.canvas is None:
            return
        width_in, _ = self._preview_inches()
        if abs(width_in - self.figure.get_figwidth()) < 0.08:
            return
        self.redraw_chart()

    # ---------------------------------------------------------------- 数据装载
    def set_summary_source(self, tab):
        """接入数据汇总页：build_analysis_data() 与外部文件走完全相同的解析链路。"""
        self.summary_source = tab

    def select_file(self):
        path = user_state.open_file_name(self, user_state.BUCKET_STATVIZ,
                                        "选择编码数据文件", DATA_FILTER)
        if not path:
            return None                       # 用户按了取消：不是故障，不弹错误框
        return self.open_path(path)

    def open_path(self, path):
        """装载一个数据文件并记进文件下拉：菜单的「最近打开」也走这个入口。"""
        self._remember(path)
        self.load_file(path)
        return path

    def _remember(self, path):
        self._loading = True
        index = self.file_combo.findText(path)
        if index < 0:
            self.file_combo.insertItem(0, path)
            index = 0
        self.file_combo.setCurrentIndex(index)
        self._loading = False

    def _on_file_changed(self, text):
        if self._loading or not text or not os.path.exists(text):
            return
        self.load_file(text)

    def plot_data(self, path):
        """兼容旧入口名：等价于 load_file(path)。"""
        return self.load_file(path)

    def load_file(self, path):
        try:
            frame = load_dataframe(path)
        except Exception as exc:
            self._empty_state("文件读取失败：%s" % exc)
            return None
        self.sheet = None
        return self.adopt_dataset(CodingDataset.build(frame, source=str(path)))

    def load_from_summary(self):
        if self.summary_source is None:
            QMessageBox.information(self, "数据汇总", "尚未接入数据汇总页。")
            return None
        try:
            columns, rows = self.summary_source.build_analysis_data()
        except Exception as exc:
            QMessageBox.critical(self, "取数失败", str(exc))
            return None
        if not columns or not rows:
            QMessageBox.information(self, "数据汇总", "数据汇总页暂无可导出的编码数据。")
            return None
        frame = pd.DataFrame(rows, columns=columns)
        self.sheet = "数据汇总页"
        return self.adopt_dataset(CodingDataset.build(frame, source="数据汇总页"))

    def adopt_dataset(self, ds):
        self.dataset = ds
        self._refresh_profile(ds)
        self._refresh_time_combo(ds)
        self._refresh_mapping_table(ds)
        self._rebuild_params()
        self.generate_btn.setEnabled(bool(ds.usable))
        if ds.usable and not ds.blocking_qa:
            return self.generate()
        reason = ds.blocking_qa[0].message if ds.blocking_qa else "没有可解析的编码列"
        self.result = None
        self._show_qa(None)
        self._empty_state(reason)
        return ds

    def _refresh_profile(self, ds):
        if not ds.usable:
            self.data_hint.setText("未解析出可用的编码列，请在下方手工指定映射。")
            return
        self.data_hint.setText(
            "%s ｜ 学生 %s ｜ 维度 %s ｜ 长表 %d 行 ｜ 来源：%s"
            % (ds.time_info.describe(), "、".join(ds.students), "、".join(ds.dimensions),
               len(ds.long_df), ds.source or "内存"))

    def _refresh_time_combo(self, ds):
        self._loading = True
        self.time_combo.clear()
        columns = [str(col) for col in ds.raw_df.columns] if ds is not None else []
        self.time_combo.addItems(columns)
        if ds is not None and str(ds.time_col) in columns:
            self.time_combo.setCurrentText(str(ds.time_col))
        self.time_combo.setEnabled(bool(columns))
        self._loading = False

    def _refresh_mapping_table(self, ds):
        columns = [str(col) for col in ds.raw_df.columns if str(col) != str(ds.time_col)]
        self.map_table.setRowCount(len(columns))
        for row, col in enumerate(columns):
            student, dimension = ds.column_map.get(col, ("", ""))
            name_item = QTableWidgetItem(col)
            name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.map_table.setItem(row, 0, name_item)
            self.map_table.setItem(row, 1, QTableWidgetItem(student))
            self.map_table.setItem(row, 2, QTableWidgetItem(dimension))

    def apply_mapping(self):
        if self.dataset is None or self.dataset.raw_df.empty:
            return None
        mapping = {}
        for row in range(self.map_table.rowCount()):
            column = self.map_table.item(row, 0).text()
            student = self.map_table.item(row, 1).text().strip()
            dimension = self.map_table.item(row, 2).text().strip()
            if not student or not dimension:
                continue
            if dimension.isdigit():
                dimension = "D%d" % int(dimension)
            mapping[column] = (student, dimension)
        time_col = self.time_combo.currentText() or None
        return self.adopt_dataset(CodingDataset.build(
            self.dataset.raw_df, time_col=time_col, column_map=mapping or None,
            source=self.dataset.source))

    # ---------------------------------------------------------------- 方法与参数
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
        if self.spec is not None:
            self._params_cache[self.spec.id] = self._collect_params()
        self.spec = method_spec(method_id)
        self._rebuild_params()
        self._update_method_hint()
        if self.dataset is not None and self.dataset.usable:
            self.generate()
        else:
            self._empty_state("请先装载数据")

    def _update_method_hint(self):
        if self.spec is None:
            return
        ok, note = self.spec.availability(self.dataset)
        text = self.spec.question
        if not ok and note:
            text = "%s\n当前不可用：%s" % (text, note)
        elif note:
            text = "%s\n注意：%s" % (text, note)
        self.method_hint.setText(text)

    def _clear_grid(self):
        while self.params_grid.count():
            item = self.params_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _rebuild_params(self):
        if self.spec is None:
            return
        ap.clear_grid(self.params_grid)
        self._rows = ap.build_param_rows(self.spec, self.params_grid, self.dataset,
                                         on_change=self._value_changed)
        cached = self._params_cache.get(self.spec.id)
        if cached is not None:
            self._apply_params(cached)
        self._apply_visibility()

    def _make_param_widget(self, param):
        """控件工厂已下沉 analysis_params，保留入口给测试与旧调用点"""
        return ap.make_param_widget(param, self.dataset, self._value_changed)

    def _dataset_values(self, which):
        return ap.dataset_values(self.dataset, which)

    def _options_for(self, param):
        return ap.options_for(param, self.dataset)

    def _entry_for(self, key, visible_only=True):
        return ap.entry_for(self._rows, key, visible_only)

    def _read_widget(self, widget):
        return ap.read_widget(widget)

    def _collect_params(self):
        return ap.collect_params(self._rows)

    def _apply_params(self, params):
        ap.apply_params(self._rows, params)

    def _value_changed(self, key):
        if key == "bin_mode":
            self._sync_bin_value()
        self._apply_visibility()

    def _sync_bin_value(self):
        """切换窗口划分方式时，把共用的 bin_value 对齐到该方式的常规量级。"""
        entry = self._entry_for("bin_mode")
        if entry is None:
            return
        seconds = self._read_widget(entry[2]) == "seconds"
        kinds = (QDoubleSpinBox,) if seconds else (QSpinBox,)
        for param, _host, widget in self._rows:
            if param.key == "bin_value" and isinstance(widget, kinds):
                widget.setValue(300.0 if seconds else 6)

    def _apply_visibility(self):
        ap.apply_visibility(self._rows)

    # ---------------------------------------------------------------- 计算与呈现
    def generate(self):
        if self.spec is None or self.dataset is None or not self.dataset.usable:
            self._empty_state("请先装载数据")
            return None
        params = self._collect_params()
        self._params_cache[self.spec.id] = params
        try:
            result = self.spec.compute(self.dataset, params)
        except Exception as exc:
            traceback.print_exc()
            self._empty_state("计算失败：%s" % exc)
            return None
        self.result = result
        self._update_method_hint()
        self._fill_tables(result)
        self._show_qa(result)
        self._render(result, params)
        return result

    def current_params(self):
        return dict(self._params_cache.get(self.spec.id, {})) if self.spec else {}

    def _draw_once(self, result, params, quiet=False):
        """把当前方法的结果画到 self.figure 上（一次完整重绘）。"""
        self.figure.clear()
        theme.sync_figure(self.figure)
        plots.reset_size(self.figure)
        if result is None or result.blocked or not result.tables:
            self._draw_placeholder(
                getattr(result, "blocked_reason", "") or "当前条件不足，暂无图形")
            return
        if quiet:
            # 测量遍是在“错的”尺寸上画的，tight_layout 的抱怨没有参考价值
            ap.suppress_layout_warnings(self._safe_render, result, params)
            return
        self._safe_render(result, params)

    def _safe_render(self, result, params):
        try:
            self.spec.render(self.figure, result, self.dataset, params)
        except Exception as exc:
            traceback.print_exc()
            self.figure.clear()
            theme.sync_figure(self.figure)
            plots.reset_size(self.figure)
            self._draw_placeholder("绘图失败：%s" % exc)

    def _render(self, result, params):
        """两遍绘制：第一遍只为了量出内容需要多高，第二遍在最终画布上排版。

        tight_layout 与刻度间距都是按画图当时的画布尺寸算的，只画一次再撑高，
        版式仍停留在小画布上，标签照样重叠，所以尺寸变了必须重画。
        """
        drawn_w, drawn_h = float(self.figure.get_figwidth()), float(self.figure.get_figheight())
        self._draw_once(result, params, quiet=True)
        width_in, height_in = self._apply_figure_size()
        if abs(width_in - drawn_w) > 0.05 or abs(height_in - drawn_h) > 0.05:
            self._draw_once(result, params)
            self._apply_figure_size()
        self.canvas.draw_idle()

    def _draw_placeholder(self, message):
        """在画布中央写「为什么没有图」；文案记一笔，主题切换时要照原样重建。"""
        self._placeholder = str(message or "")
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, self._placeholder, ha="center", va="center",
                color=theme.tokens()["ink3"], wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _redraw_placeholder(self):
        """按当前主题重建占位图：清画布 → 建坐标轴 → 定尺寸 → 重绘。

        坐标区底色与文字颜色都是在创建那一刻从 rcParams / tokens 取的，主题切换后
        只改 figure 底色会让旧坐标区在图中央留一块残色（IND/GRP 工作台没这个毛病，
        因为它的 redraw_chart 每次都重建占位图），所以这里也走重建。
        """
        self.figure.clear()
        theme.sync_figure(self.figure)
        plots.reset_size(self.figure)
        if self._placeholder:
            self._draw_placeholder(self._placeholder)
        self._apply_figure_size()
        self.canvas.draw_idle()

    def redraw_chart(self):
        """主题切换后按新配色重绘当前方法的结果图。"""
        if self.result is not None and self.spec is not None:
            params = self._params_cache.get(self.spec.id) or self._collect_params()
            self._render(self.result, params)
            return
        self._redraw_placeholder()

    def _fill_tables(self, result):
        self.result_tabs.clear()
        if not result.tables:
            self.result_tabs.addTab(self._note_table(result.blocked_reason or "暂无结果"), "提示")
            return
        for name, frame in result.tables.items():
            self.result_tabs.addTab(self._make_table(frame), "%s (%d)" % (name, len(frame)))

    def _note_table(self, message):
        table = QTableWidget(1, 1)
        table.setItem(0, 0, QTableWidgetItem(message))
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
        # 窄表不把右半张卡片空出来：末列吃掉剩余宽度，列多时依旧横向滚动
        header.setStretchLastSection(True)
        # 不开隔行色：本项目的 QSS 写了 QTableWidget::item，Qt 在此情况下会
        # 忽略 palette 的 AlternateBase 而把偶数行画成纯黑，全应用其它表格也统一素底
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

    def _show_qa(self, result):
        self.qa_list.clear()
        combined = merge_qa(self.dataset.qa if self.dataset is not None else None,
                            getattr(result, "qa", []) or [])
        for note in getattr(result, "notes", []) or []:
            combined.append(_Note(note))
        for item in combined:
            row = QListWidgetItem("%s %s" % (_prefix_of(item), item.message))
            row.setData(Qt.UserRole, getattr(item, "code", ""))
            self.qa_list.addItem(row)
        errors, warns, infos = qa_summary(combined)
        head = "阻断 %d · 警告 %d · 提示 %d" % (errors, warns, infos)
        if getattr(result, "blocked", False):
            head = "结果已阻断：%s\n%s" % (result.blocked_reason, head)
        elif result is None:
            head = "未生成结果：" + head
        self.qa_notice.setText(head)

    def _empty_state(self, message):
        self.result = None
        self.result_tabs.clear()
        self.result_tabs.addTab(self._note_table(message), "提示")
        self.qa_notice.setText(message)
        self._placeholder = str(message or "")
        self._redraw_placeholder()

    # ---------------------------------------------------------------- 导出
    def _require_result(self):
        if self.result is None or (self.result.blocked and not self.result.tables):
            QMessageBox.warning(self, "导出", "请先选择一个可用方法并生成结果。")
            return False
        return True

    def export_statistics_to(self, path):
        return [export_tables(self.result, path)]

    def export_chart_to(self, path):
        return [export_figure(self.figure, path, dpi=int(self.dpi_combo.currentText()))]

    def export_metadata_to(self, path):
        audit = os.path.join(os.path.dirname(path), file_name(AUDIT_KIND, self.result, "json"))
        export_config(self.result, self.dataset, path, spec=self.spec, sheet=self.sheet)
        export_audit(self.result, self.dataset, audit)
        return [path, audit]

    def export_all_to(self, directory):
        return export_all(self.result, self.dataset, directory, spec=self.spec,
                          fig=self.figure, sheet=self.sheet,
                          dpi=int(self.dpi_combo.currentText()))

    def export_statistics(self):
        if not self._require_result():
            return None
        path = user_state.save_file_name(
            self, user_state.BUCKET_EXPORT, "导出统计表",
            file_name(TABLE_KIND, self.result, "xlsx"),
            "Excel 工作簿 (*.xlsx);;CSV (*.csv)")
        return self._guarded(self.export_statistics_to, path)

    def export_chart(self):
        if self.result is None:
            QMessageBox.warning(self, "导出", "请先生成结果。")
            return None
        fmt = self.format_combo.currentText().lower()
        path = user_state.save_file_name(
            self, user_state.BUCKET_EXPORT, "导出图形",
            file_name(FIGURE_KIND, self.result, fmt),
            "%s (*.%s);;所有文件 (*)" % (fmt.upper(), fmt))
        return self._guarded(self.export_chart_to, path)

    def export_metadata(self):
        if not self._require_result():
            return None
        path = user_state.save_file_name(
            self, user_state.BUCKET_EXPORT, "导出运行配置",
            file_name(CONFIG_KIND, self.result, "json"), "JSON (*.json)")
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
            written = action(argument)
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "导出失败", str(exc))
            return None
        names = [str(item) for item in written]
        self._last_export = names[0]
        self.export_hint.setText("已写出 %d 个文件到 %s"
                                 % (len(names), os.path.dirname(names[0]) or "."))
        return names
