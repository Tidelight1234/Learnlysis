# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""主窗口 MainWindow：自定义标题栏与菜单、标签页组装、自动存档与恢复。"""

import os
import sys
import json
import time
import logging

from PySide6.QtCore import Qt, QPoint, QEvent, QElapsedTimer, QTimer
from PySide6.QtGui import (QColor, QPainter, QRadialGradient, QCursor, QKeySequence,
                           QAction, QActionGroup)
from PySide6.QtWidgets import (QMainWindow, QTabWidget, QSplitter, QPushButton, QWidget,
                             QVBoxLayout, QHBoxLayout, QMessageBox, QMenu, QApplication,
                             QTabBar, QLineEdit, QComboBox, QAbstractSlider,
                             QAbstractItemView, QLabel, QInputDialog,
                             QSizePolicy, QLayout)

import icon_provider
import theme
import user_state
from language import LANGUAGES, set_language, source_text, tr, translate_widget_tree

from about_dialog import AboutDialog
from config import APP_NAME, APP_TAGLINE, ARCHIVE_FILE, ARCHIVE_TMP_FILE, data_dir
from summary_tab import SummaryTab
from video_player import VideoPlayerModule
from data_encoder import DataEncoderModule
from group_analysis import GroupAnalysisModule
from statistic_visualization import StatisticVisualizationModule
from individual_analysis import IndividualAnalysisModule
from group_encoder import GroupInteractionEncoder
from ai_assistant import AiAssistantModule

TITLE_HEIGHT = 40
RESIZE_EDGE = 6
DESIGN_WINDOW_SIZE = (1900, 800)
MIN_START_SIZE = (960, 640)

# 五个外层模块的横向可用性下限（逻辑像素）：数据编码 1236、统计 1096、
# IND/GRP 1244、AI 1164、顶部导航 900；取最大项并向上留整档余量。
ENCODER_REQUIRED_W = 1236
STATVIZ_REQUIRED_W = 1096
PATTERN_REQUIRED_W = 1244
AI_REQUIRED_W = 1164
NAV_REQUIRED_W = 900
WINDOW_MIN_WIDTH = ((max(ENCODER_REQUIRED_W, STATVIZ_REQUIRED_W,
                        PATTERN_REQUIRED_W, AI_REQUIRED_W, NAV_REQUIRED_W) + 39) // 40) * 40

# 自动保存：改动落定后防抖一拍再写盘（连续记录只算一次）；看门狗是兵来将挡——
# 万一某条改动路径没接到信号，最多 30 秒也会被补一次存档（没改动时直接返回，不写盘）。
AUTOSAVE_DEBOUNCE_MS = 1500
AUTOSAVE_WATCHDOG_MS = 30000

# 外层标签页序号：四个模块都被包在 Canvas 里（tab_widget.indexOf 拿不到），只能按装入顺序定
TAB_ENCODER, TAB_STATVIZ, TAB_INDIVIDUAL, TAB_GROUP = 0, 1, 2, 3
TAB_AI = 4

# 「最近打开」每一类用的图标（与文件菜单顶部分区同一套图标语言）
_RECENT_ICONS = {
    user_state.BUCKET_VIDEO: "video",
    user_state.BUCKET_STATVIZ: "chart-bar",
    user_state.BUCKET_INDIVIDUAL: "individual",
    user_state.BUCKET_GROUP: "network",
    user_state.BUCKET_FRAMEWORK: "import",
}

# WM_NCCALCSIZE 把客户区扩到整窗以隐藏原生标题栏（logo/名称随之消失），保留 Win11 圆角与 DWM 阴影。
# 拖拽与八向缩放不再依赖 WM_NCHITTEST：QVideoWidget 会令整棵子树变成原生窗口，子 HWND
# 拦截命中测试消息，顶层 nativeEvent 收不到，故改用应用级事件过滤器把系统手势转发给顶层窗口。
_HT_CLIENT = 1
_HT_CAPTION = 2
_HT_LEFT = 10
_HT_RIGHT = 11
_HT_TOP = 12
_HT_TOPLEFT = 13
_HT_TOPRIGHT = 14
_HT_BOTTOM = 15
_HT_BOTTOMLEFT = 16
_HT_BOTTOMRIGHT = 17

# 缩放边 HT 代码 → 悬停方向光标
_EDGE_CURSORS = {
    _HT_LEFT: Qt.SizeHorCursor,
    _HT_RIGHT: Qt.SizeHorCursor,
    _HT_TOP: Qt.SizeVerCursor,
    _HT_BOTTOM: Qt.SizeVerCursor,
    _HT_TOPLEFT: Qt.SizeFDiagCursor,
    _HT_BOTTOMRIGHT: Qt.SizeFDiagCursor,
    _HT_TOPRIGHT: Qt.SizeBDiagCursor,
    _HT_BOTTOMLEFT: Qt.SizeBDiagCursor,
}

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _WM_NCCALCSIZE = 0x0083
    _WM_NCHITTEST = 0x0084
    _WM_NCLBUTTONDOWN = 0x00A1
    _WM_NCLBUTTONDBLCLK = 0x00A3
    _SM_CXFRAME = 32
    _SM_CXPADDEDBORDER = 92
    _SM_CXDOUBLECLK = 36
    _SM_CYDOUBLECLK = 37
    # Windows 11 (build>=22000) 窗体圆角：DWMWA_WINDOW_CORNER_PREFERENCE / DWMWCP_ROUND
    _DWMWA_WINDOW_CORNER_PREFERENCE = 33
    _DWMWCP_ROUND = 2

    class _NCCALCSIZE_PARAMS(ctypes.Structure):
        _fields_ = [("rgrc", wintypes.RECT * 3), ("lppos", ctypes.c_void_p)]


class TitleBar(QWidget):
    """自定义顶栏：承载菜单按钮与窗口按钮，替代被移除的原生标题栏。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(TITLE_HEIGHT)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 5, 6, 5)
        lay.setSpacing(2)
        self.menu_buttons = {}
        for key, text in (("file", "文件"), ("view", "视图"), ("help", "说明")):
            btn = QPushButton(tr(text))
            btn.setProperty("btn", "ghost")
            # nav 命中的 QSS 规则把字号提到最大一档，同时收紧纵向内边距（按钮固定 30px 高）
            btn.setProperty("nav", True)
            btn.setFixedHeight(30)
            lay.addWidget(btn)
            self.menu_buttons[key] = btn
        lay.addStretch(1)
        self.min_btn = QPushButton()
        self.min_btn.setProperty("btn", "ghost")
        self.min_btn.setFixedSize(40, 30)
        self.min_btn.setToolTip(tr("最小化"))
        theme.bind_icon(self.min_btn, "minus", 15)
        lay.addWidget(self.min_btn)
        self.max_btn = QPushButton()
        self.max_btn.setProperty("btn", "ghost")
        self.max_btn.setFixedSize(40, 30)
        self.max_btn.setToolTip(tr("最大化"))
        theme.bind_icon(self.max_btn, "maximize", 15)
        lay.addWidget(self.max_btn)
        self.close_btn = QPushButton()
        self.close_btn.setProperty("btn", "titleClose")
        self.close_btn.setFixedSize(40, 30)
        self.close_btn.setToolTip(tr("关闭"))
        theme.bind_icon(self.close_btn, "close", 15)
        lay.addWidget(self.close_btn)

    def paintEvent(self, event):
        # 与下方 Canvas 共用同一径向渐变参数（换算到本控件坐标），保证顶栏与画布无缝衔接
        t = theme.tokens()
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(t["canvas"]))
        win = self.window()
        status_h = win.statusBar().height() if isinstance(win, QMainWindow) else 0
        cw = max(self.width() - 32, 1)
        ch = max(win.height() - status_h - TITLE_HEIGHT - 16, 1)
        g = QRadialGradient(16 + cw * 0.85, TITLE_HEIGHT - ch * 0.15, max(cw, 600) * 0.9)
        g.setColorAt(0.0, QColor(t["atmo_inner"]))
        g.setColorAt(1.0, QColor(t["atmo_outer"]))
        p.fillRect(self.rect(), g)
        p.end()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("%s - %s" % (APP_NAME, tr(APP_TAGLINE)))
        self.setMinimumWidth(WINDOW_MIN_WIDTH)
        self._set_initial_geometry()
        # 创建主布局容器
        main_container =QSplitter(Qt.Horizontal)
        tab_container=QSplitter(Qt.Horizontal)
        tab_module_container=QSplitter(Qt.Horizontal)
         # 创建标签页控件
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("outerTab")
        # 各页的舒展 sizeHint 不应成为整个窗口的刚性下限；标签栏本身可滚动。
        self.tab_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self.tab_widget.tabBar().setExpanding(False)
        self.tab_widget.tabBar().setElideMode(Qt.ElideNone)
        self.tab_widget.tabBar().setUsesScrollButtons(True)
        self.tabmodule_widget = QTabWidget()
        self.tabmodule_widget.setObjectName("innerTab")
        # 创建两个功能模块
        self.video_player = VideoPlayerModule()
        self.data_encoder = DataEncoderModule(tabmodule=self.tabmodule_widget)
        self.individual_analysis = IndividualAnalysisModule(tabmodule=self.tabmodule_widget)
        self.statistic_visualization = StatisticVisualizationModule()
        self.group_analysis = GroupAnalysisModule(tabmodule=self.tabmodule_widget)
        # 正确传入 tabmodule 参数
        self.summary_tab = SummaryTab(tabmodule=self.tabmodule_widget)
        # 小组交互页：成员与框架都从编码页现场取，不复制一份数据
        self.group_encoder = GroupInteractionEncoder(
            members_provider=self.group_member_sources,
            framework_provider=self.group_framework_source,
            summary_provider=self.group_state_frame)
        # 汇总页的「输出小组数据包」从小组页现场取数，不反向引用主窗口
        self.summary_tab.set_group_source(self.group_pack_info)
        # 统计及可视化可直接取数据汇总页拼好的 Time + S{i}Encoding_{j} 标准表
        self.statistic_visualization.set_summary_source(self.summary_tab)
        # 两个新工作台的「从当前会话取数」：只给三个取数口，不摊开内部控件
        for module in (self.individual_analysis, self.group_analysis):
            module.attach_session(summary_tab=self.summary_tab,
                                  group_encoder=self.group_encoder,
                                  pages_provider=self.encoder_pages)
        # 编码区“当前时间点”跟随视频当前播放片段的起点
        self.data_encoder.set_video_player(self.video_player)
        self.group_encoder.time_ctrl.set_video_player(self.video_player)
        # 将功能模块添加到标签页
        # 设置最小宽度替代固定宽度
        
        # 标签栏换成带笔直下划线的实现，并挂上“+ 新建编码页”按钮
        self.encoder_tab_bar = theme.install_underline_tab_bar(
            self.tabmodule_widget, with_add=True)
        self.encoder_tab_bar.addRequested.connect(self.add_new_tab)
        # 点击 / 右键标签页时提供“复制编码框架”入口，见 _copy_framework_menu
        self.encoder_tab_bar.tabClicked.connect(self._offer_copy_framework)
        self.encoder_tab_bar.setContextMenuPolicy(Qt.CustomContextMenu)
        self.encoder_tab_bar.customContextMenuRequested.connect(self._show_encoder_tab_menu)
        # 编码区横/纵布局切换：挂在标签栏最右端的固定位置（corner widget），
        # 不随标签增减移动；只作用于当前选中的编码页
        self.layout_toggle_btn = QPushButton()
        self.layout_toggle_btn.setProperty("btn", "roundSm")
        self.layout_toggle_btn.setProperty("layoutToggle", True)
        self.layout_toggle_btn.setFixedSize(theme.TAB_ADD_SIZE, theme.TAB_ADD_SIZE)
        self.layout_toggle_btn.setCheckable(True)
        self.layout_toggle_btn.setToolTip("切换编码区排列：左右 / 上下")
        theme.bind_icon(self.layout_toggle_btn, "layout-h", 15)
        self.layout_toggle_btn.clicked.connect(self._toggle_encoder_layout)
        self.tabmodule_widget.setCornerWidget(self.layout_toggle_btn, Qt.TopRightCorner)
        self._encoder_stacked = False   # 新建编码页沿用上次的排列选择
        self.tabmodule_widget.setMovable(False)
        self.tabmodule_widget.addTab(self.data_encoder, '编码个体1')
        self.tabmodule_widget.addTab(self.summary_tab, "数据汇总")
        self._ensure_group_tab()   # 只有一页时不会挂出「小组交互」，见该函数
        self._schedule_tab_bar_sync()  # 给各标签按规则挂上关闭叉，见 _sync_tab_bar
        self._sync_layout_toggle()
        tab_module_container = theme.Canvas()
        module_layout = QVBoxLayout()
        module_layout.setSizeConstraint(QLayout.SetNoConstraint)
        module_layout.addWidget(self.tabmodule_widget)
        module_layout.setContentsMargins(16, 12, 16, 16)
        module_layout.setSpacing(12)
        tab_module_container.setLayout(module_layout)
        # 构建外层标签页内容
        tab_container = theme.Canvas()
        splitter = QSplitter()
        self.data_splitter = splitter
        splitter.setHandleWidth(1)

        # 将两个控件添加到分割器中
        splitter.addWidget(self.video_player)
        splitter.addWidget(tab_module_container)
        # 子页面的 sizeHint 只描述舒展尺寸；窄屏时由明确的 minimumWidth 守住可用下限。
        self.video_player.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        tab_module_container.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        # 设置分割器的拉伸因子（可选）
        self.video_player.setMinimumWidth(420)
        self.video_player.setMaximumWidth(1200)
        self.encoder_container = tab_module_container
        tab_module_container.setMinimumWidth(480)
        tab_module_container.setMaximumWidth(1250)
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)

        main_layout.setSpacing(16)
        # 顶边距为 0：画布紧贴标题栏，氛围渐变才能连续
        main_layout.setContentsMargins(16, 0, 16, 16)
        tab_container.setLayout(main_layout)
        statviz_container = theme.Canvas()
        statviz_layout = QHBoxLayout()
        statviz_layout.addWidget(self.statistic_visualization)
        statviz_layout.setSpacing(16)
        statviz_layout.setContentsMargins(16, 0, 16, 16)
        statviz_container.setLayout(statviz_layout)
        self.tab_widget.addTab(tab_container, tr("数据编码"))
        self.tab_widget.addTab(statviz_container, tr("统计及可视化"))
        self.tab_widget.addTab(self.individual_analysis, tr("个体模式分析"))
        self.tab_widget.addTab(self.group_analysis, tr("小组模式分析"))
        # 第 5 个外层标签页：AI 助手（智能体档案 / 对话 / 模型设置三区）。
        # 只注入几个回调（取工作现场 / 框架目标与应用），不把内部控件结构摊开给 AI 模块。
        self.ai_assistant = AiAssistantModule()
        self.ai_assistant.set_context_provider(self._ai_context_snapshot)
        self.ai_assistant.set_framework_workflow(
            self._ai_framework_targets, self._ai_apply_framework)
        ai_container = theme.Canvas()
        ai_layout = QHBoxLayout()
        ai_layout.addWidget(self.ai_assistant)
        ai_layout.setSpacing(16)
        ai_layout.setContentsMargins(16, 0, 16, 16)
        ai_container.setLayout(ai_layout)
        self.tab_widget.addTab(ai_container, tr("AI 助手"))
        self.tab_widget.currentChanged.connect(self.on_outer_tab_change)
        # 外层模块标签也可右键复制编码框架（取代各模块里的“传递编码”按钮）
        self.outer_tab_bar = self.tab_widget.tabBar()
        self.outer_tab_bar.setContextMenuPolicy(Qt.CustomContextMenu)
        self.outer_tab_bar.customContextMenuRequested.connect(self._show_module_tab_menu)
        # 能维护编码框架的外层标签：只剩编码页（新工作台不持有维度面板，框架由数据契约提供）
        self._framework_tab_roles = {0: "encoder"}
        # 各模块面板的「导入框架」都从这份登记表取复制来源
        self._register_framework_panels()
        main_container.addWidget(self.tab_widget)
        main_container.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self._menu_icon_actions = []
        self._refresh_icons()
        # macOS 使用系统标题栏；Windows 保留自定义标题栏与命中测试。
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self.title_bar = None
        if sys.platform != "darwin":
            self.title_bar = TitleBar(central)
            central_layout.addWidget(self.title_bar)
        central_layout.addWidget(main_container, 1)
        self.setCentralWidget(central)
        translate_widget_tree(self)

        # 文件菜单：与各模块的取数 / 导出入口逐条对齐（见 _build_file_menu）
        self._build_file_menu()

        self.view_menu = self._popup_menu()
        self.dark_action = self.view_menu.addAction(tr("深色模式"))
        self.dark_action.setCheckable(True)
        self.dark_action.setChecked(theme.get_mode() == "dark")
        self.dark_action.toggled.connect(self._on_dark_toggled)
        self.view_menu.addSeparator()
        self.coding_menu = self._add_submenu(self.view_menu, 'info', tr('数据编码'))
        self.ai_coding_action = self.coding_menu.addAction(tr("显示 AI 编码工具"))
        self.ai_coding_action.setCheckable(True)
        self.ai_coding_action.setChecked(user_state.state.coding_ai_visible())
        self.ai_coding_action.toggled.connect(self._set_ai_coding_visible)
        self.coding_layout_menu = self._add_submenu(self.coding_menu, 'info', tr('编码区布局'))
        self.coding_layout_menu.addAction('恢复默认四格').triggered.connect(
            lambda: [page.reset_free_layout() for _title, page in self.encoder_pages()])
        for key, label in (('dimensions', '维度编辑'), ('controls', '时间、注释及按钮'),
                           ('table', '表格')):
            self.coding_layout_menu.addAction('隐藏 AI 时由%s占两格' % label).triggered.connect(
                lambda _checked=False, block=key:
                [page.set_span_block(block) for _title, page in self.encoder_pages()])
        self.coding_layout_menu.addSeparator()
        self.coding_span_side_group = QActionGroup(self)
        self.coding_span_side_group.setExclusive(True)
        for side, label in (('start', '隐藏 AI 时合并在上方／左侧'),
                            ('end', '隐藏 AI 时合并在下方／右侧')):
            action = self.coding_layout_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(user_state.state.coding_layout()['span_side'] == side)
            action.setData(side)
            self.coding_span_side_group.addAction(action)
            action.triggered.connect(lambda _checked=False, value=side:
                [page.set_span_side(value) for _title, page in self.encoder_pages()])
        self.coding_layout_menu.aboutToShow.connect(lambda: [
            action.setChecked(action.data() == user_state.state.coding_layout()['span_side'])
            for action in self.coding_span_side_group.actions()])
        self.coding_font_menu = self._add_submenu(self.coding_menu, "info", tr("字号"))
        self.coding_font_group = QActionGroup(self)
        self.coding_font_group.setExclusive(True)
        for value, label in (("small", "小"), ("medium", "中"), ("normal", "大（正常）")):
            action = self.coding_font_menu.addAction(tr(label))
            action.setCheckable(True)
            action.setChecked(user_state.state.coding_font_size() == value)
            self.coding_font_group.addAction(action)
            action.triggered.connect(lambda _checked=False, size=value: self._set_coding_font_size(size))
        self.custom_coding_font_action = self.coding_font_menu.addAction(tr("自定义字号…"))
        self.custom_coding_font_action.setCheckable(True)
        self.coding_font_group.addAction(self.custom_coding_font_action)
        self.custom_coding_font_action.setChecked(
            isinstance(user_state.state.coding_font_size(), dict))
        self.custom_coding_font_action.triggered.connect(self._choose_coding_font_size)
        self.dimension_mode_menu = self._add_submenu(self.coding_menu, 'info', tr('维度编辑显示模式'))
        self.dimension_mode_group = QActionGroup(self)
        self.dimension_mode_group.setExclusive(True)
        for mode, label in (('standard', '标准模式'), ('compact', '简洁模式')):
            action = self.dimension_mode_menu.addAction(tr(label))
            action.setCheckable(True)
            action.setChecked(user_state.state.coding_dimension_mode() == mode)
            self.dimension_mode_group.addAction(action)
            action.triggered.connect(lambda _checked=False, value=mode:
                                     self._set_dimension_display_mode(value))
        self.language_menu = self._add_submenu(self.view_menu, "info", tr("语言"))
        self.language_group = QActionGroup(self)
        self.language_group.setExclusive(True)
        self.language_actions = {}
        for code, label in LANGUAGES:
            action = self.language_menu.addAction(label)
            action.setCheckable(True)
            action.setData(code)
            action.setChecked(user_state.state.language() == code)
            self.language_group.addAction(action)
            action.triggered.connect(lambda _checked=False, value=code: self._change_language(value))
            self.language_actions[code] = action

        self.help_menu = self._popup_menu()
        about_action = self._add_icon_action(
            self.help_menu, "info", tr("关于 Learnlysis"), self._show_about)
        if sys.platform == "darwin":
            self.file_menu.setTitle(tr("文件"))
            self.view_menu.setTitle(tr("视图"))
            self.help_menu.setTitle(tr("说明"))
            # 按 macOS 标准把“关于 Learnlysis”迁移到应用菜单。
            about_action.setMenuRole(QAction.AboutRole)
            native_bar = self.menuBar()
            native_bar.setNativeMenuBar(True)
            native_bar.addMenu(self.file_menu)
            native_bar.addMenu(self.view_menu)
            native_bar.addMenu(self.help_menu)
        else:
            for key, menu in (("file", self.file_menu),
                              ("view", self.view_menu),
                              ("help", self.help_menu)):
                btn = self.title_bar.menu_buttons[key]
                btn.clicked.connect(lambda _checked=False, b=btn, m=menu: self._exec_menu(b, m))
            self.title_bar.min_btn.clicked.connect(self.showMinimized)
            self.title_bar.max_btn.clicked.connect(self._toggle_maximize)
            self.title_bar.close_btn.clicked.connect(self.close)

        self.statusBar().showMessage(tr("就绪"))

        # 现场改动 → 自动保存：先装好机械，再接线（接线本身会读控件，可能推信号）
        self._init_autosave()
        self._bind_content_signals(self.data_encoder)
        self._bind_group_page_signals()
        self._bind_summary_signals()
        self._bind_video_signals()

        splitter.splitterMoved.connect(self.on_outer_splitter_moved)
        self.tab_widget.currentChanged.connect(self._on_tab_fade)
        self.tabmodule_widget.currentChanged.connect(self._on_inner_tab_fade)
        self.tabmodule_widget.currentChanged.connect(self.on_inner_tab_changed)
        theme.themeChanged.connect(self._on_theme_changed)

        # —— 无边框窗口拖拽 / 八向缩放（应用级事件过滤器方案）——
        self._press_timer = QElapsedTimer()
        self._last_press_pos = QPoint()
        self._override_cursor_active = False
        self._override_cursor_shape = None
        # Qt 状态栏自带的缩放手柄会与系统缩放循环冲突，关掉
        self.statusBar().setSizeGripEnabled(False)
        # 悬停方向光标依赖无按键的 Move 事件；setMouseTracking 只向下传播给调用时已存在的子控件，
        # 必须在整棵子树组装完成后开启
        self._enable_mouse_tracking(self)
        QApplication.instance().installEventFilter(self)
        QTimer.singleShot(0, self._apply_resolution_layout)

    def _set_initial_geometry(self):
        """按当前显示器的逻辑工作区限制并居中初始窗口。"""
        app = QApplication.instance()
        screen = app.screenAt(QCursor.pos()) if app is not None else None
        if screen is None and app is not None:
            screen = app.primaryScreen()
        if screen is None:
            self.setGeometry(100, 100, *DESIGN_WINDOW_SIZE)
            return
        area = screen.availableGeometry()
        width = max(min(MIN_START_SIZE[0], area.width()),
                    min(DESIGN_WINDOW_SIZE[0], area.width()))
        height = max(min(MIN_START_SIZE[1], area.height()),
                     min(DESIGN_WINDOW_SIZE[1], area.height()))
        self.setGeometry(area.x() + max((area.width() - width) // 2, 0),
                         area.y() + max((area.height() - height) // 2, 0),
                         width, height)

    def _apply_resolution_layout(self):
        """按高 DPI 换算后的可用宽高自动调整编码页。"""
        splitter = getattr(self, "data_splitter", None)
        encoder = getattr(self, "encoder_container", None)
        video = getattr(self, "video_player", None)
        if splitter is None or encoder is None or video is None or splitter.width() <= 1:
            return
        # 用顶层窗口宽度判断档位，避免子 splitter 的旧 sizeHint 形成“越撑越宽”的反馈环。
        available = max(self.width() - 32, 1)
        if available < 1200:
            video_min, encoder_min, video_ratio = 360, 430, 0.43
        elif available < 1500:
            video_min, encoder_min, video_ratio = 420, 500, 0.45
        else:
            video_min, encoder_min, video_ratio = 520, 600, 0.48
        video.setMinimumWidth(video_min)
        encoder.setMinimumWidth(encoder_min)
        band = (video_min, encoder_min)
        if getattr(self, "_resolution_layout_band", None) != band:
            self._resolution_layout_band = band
            first = int(available * video_ratio)
            splitter.setSizes([first, max(available - first, 1)])
        video.set_compact_layout(splitter.height() < 720)
        translate_widget_tree(self)

    def _add_icon_action(self, menu, icon_name, text, slot, shortcut=None):
        action = menu.addAction(icon_provider.icon(icon_name, theme.tokens()["ink2"], 16), tr(text))
        action.triggered.connect(slot)
        self._menu_icon_actions.append((action, icon_name))
        if shortcut:
            # 菜单不是 QMainWindow 的 menubar，默认快捷键只在菜单弹出时生效；
            # 把 action 同时挂到窗口上，Ctrl+S 在任何页面都能存一把
            action.setShortcut(shortcut)
            action.setShortcutContext(Qt.WindowShortcut)
            self.addAction(action)
        return action

    def _change_language(self, code):
        if code == user_state.state.language():
            return
        set_language(code)

    def retranslate_ui(self):
        self.setWindowTitle("%s - %s" % (APP_NAME, tr(APP_TAGLINE)))

    def _popup_menu(self, title=""):
        """弹层菜单（含子菜单）都去阴影 + 圆角裁剪：子菜单也是独立顶层窗体。"""
        if sys.platform == "darwin":
            return QMenu(title, self)
        return theme.bind_menu_popup(QMenu(title, self))

    def _add_submenu(self, parent, icon_name, title):
        """在 parent 里开一个带图标的子菜单，返回子菜单本体（菜单标题即入口文字）。"""
        menu = self._popup_menu(tr(title))
        action = parent.addMenu(menu)
        action.setIcon(icon_provider.icon(icon_name, theme.tokens()["ink2"], 16))
        self._menu_icon_actions.append((action, icon_name))
        return menu

    def _build_file_menu(self):
        """文件菜单：把散在四个模块里的取数 / 导出入口收在一处。

        铁律：菜单只转发到模块自己的公开方法，不在这里多写一份实现；否则模块改了菜单
        不会跟着改（就是本次修掉的病）。子菜单按「取数 → 导出 → 存档」分区，
        顺序与外层标签页一致（数据编码 → 统计及可视化 → 个体 → 小组）。
        """
        self.file_menu = self._popup_menu("文件")
        self._add_icon_action(self.file_menu, "video", "导入视频 — 数据编码",
                              self._open_video_files)
        self.recent_menu = self._add_submenu(self.file_menu, "folder", "最近打开")
        self.recent_menu.aboutToShow.connect(self._rebuild_recent_menu)
        self.file_menu.addSeparator()

        coding = self._add_submenu(self.file_menu, "tab-data", "编码与汇总")
        self._encoder_actions = [
            self._add_icon_action(coding, "import", "导入编码框架 — 当前编码页",
                                  self._import_encoder_framework),
            self._add_icon_action(coding, "export", "输出编码表 — 当前编码页",
                                  self._export_encoder_page),
        ]
        # 小组页是条件页（编码页 ≥ 2 才挂出），菜单项不隐藏而是置灰，见 _sync_file_menu_state
        self._group_page_actions = [
            self._add_icon_action(coding, "export", "输出小组数据包 — 小组交互页",
                                  self._export_group_page),
        ]
        coding.addSeparator()
        self._add_icon_action(coding, "export", "输出分析格式文件 — 数据汇总",
                              self._export_analysis_format)
        self._add_icon_action(coding, "export", "输出分析格式文件（含注释）— 数据汇总",
                              self._export_analysis_format_notes)
        self._add_icon_action(coding, "export", "输出汇总表格 — 数据汇总",
                              self._export_summary_table)
        self._add_icon_action(coding, "export", "输出小组数据包 — 数据汇总",
                              self._export_group_pack)

        data_menu = self._add_submenu(self.file_menu, "chart-bar", "分析取数")
        self._add_icon_action(data_menu, "chart-bar", "选择文件 — 统计及可视化",
                              self._select_statviz_file)
        self._add_icon_action(data_menu, "tab-data", "从数据汇总取数 — 统计及可视化",
                              self._load_statviz_from_summary)
        data_menu.addSeparator()
        self._add_icon_action(data_menu, "individual", "选择文件 — 个体模式分析",
                              self._select_individual_files)
        self._add_icon_action(data_menu, "tab-data", "从当前会话取数 — 个体模式分析",
                              self._load_individual_session)
        self._add_icon_action(data_menu, "import", "导入编码框架 — 个体模式分析",
                              self._import_individual_framework)
        data_menu.addSeparator()
        self._add_icon_action(data_menu, "network", "选择文件 — 小组模式分析",
                              self._select_group_files)
        self._add_icon_action(data_menu, "tab-data", "从当前会话取数 — 小组模式分析",
                              self._load_group_session)
        self._add_icon_action(data_menu, "import", "导入编码框架 — 小组模式分析",
                              self._import_group_framework)

        export_menu = self._add_submenu(self.file_menu, "export", "结果导出")
        self._add_icon_action(export_menu, "export", "一键导出全部 — 统计及可视化",
                              self._export_statviz_all)
        self._add_icon_action(export_menu, "export", "一键导出全部 — 个体模式分析",
                              self._export_individual_all)
        self._add_icon_action(export_menu, "export", "一键导出全部 — 小组模式分析",
                              self._export_group_all)
        export_menu.addSeparator()
        self._add_icon_action(export_menu, "robot", "导出 AI 对话 / 分析规划 — AI 助手",
                              self._export_ai_conversation)

        self.file_menu.addSeparator()
        archive = self._add_submenu(self.file_menu, "save", "工作进度")
        self._add_icon_action(archive, "save", "立即保存存档", self._save_archive_now,
                              QKeySequence("Ctrl+S"))
        self.auto_save_action = self._add_icon_action(archive, "replay", "自动保存",
                                                      self._toggle_autosave)
        self.auto_save_action.setCheckable(True)
        self.auto_save_action.setToolTip("开启后任何改动都会在片刻后自动写入存档，退出时不再询问")
        self._add_icon_action(archive, "import", "重新载入存档", self._reload_archive)
        self._add_icon_action(archive, "close", "清除存档文件", self._clear_archive_file)
        archive.addSeparator()
        self._add_icon_action(archive, "folder", "打开存档所在文件夹", self._open_archive_folder)
        self._sync_file_menu_state()

    # ---------- 菜单项 → 各模块入口（只转发，不重写实现） ----------
    def _target_encoder_page(self):
        """菜单要作用的编码页：当前页是编码页就用它，否则退回第一个编码页。"""
        page = self._current_encoder()
        if page is not None:
            return page
        index = self._first_encoder_index()
        return self.tabmodule_widget.widget(index) if index >= 0 else None

    def _show_summary_page(self):
        self.tab_widget.setCurrentIndex(TAB_ENCODER)
        self.tabmodule_widget.setCurrentWidget(self.summary_tab)

    def _notice_no_encoder_page(self):
        self.statusBar().showMessage("还没有编码页：先在「数据编码」里新建一页", 6000)

    def _import_encoder_framework(self):
        page = self._target_encoder_page()
        if page is None:
            self._notice_no_encoder_page()
            return False
        self.tab_widget.setCurrentIndex(TAB_ENCODER)
        page.dim_panel.import_from_excel()
        return True

    def _export_encoder_page(self):
        page = self._target_encoder_page()
        if page is None:
            self._notice_no_encoder_page()
            return False
        self.tab_widget.setCurrentIndex(TAB_ENCODER)
        self.tabmodule_widget.setCurrentWidget(page)
        page.output()
        return True

    def _group_page_present(self):
        """小组交互页此刻在不在标签上：条件页的菜单入口只能置灰，不能报错。"""
        return self.tabmodule_widget.indexOf(self.group_encoder) >= 0

    def _export_group_page(self):
        """小组页自己那份数据包（不经过汇总页）：页不在时明说原因。"""
        self.tab_widget.setCurrentIndex(TAB_ENCODER)
        if not self._group_page_present():
            self.statusBar().showMessage("小组交互页未挂出：至少需要两个编码个体页", 6000)
            return False
        self.tabmodule_widget.setCurrentWidget(self.group_encoder)
        self.group_encoder.output()
        return True

    def _export_analysis_format(self):
        self._show_summary_page()
        self.summary_tab.output_analysis_data()

    def _export_analysis_format_notes(self):
        self._show_summary_page()
        self.summary_tab.output_analysis_data(with_comments=True)

    def _export_summary_table(self):
        self._show_summary_page()
        self.summary_tab.output_summary_data()

    def _export_group_pack(self):
        self._show_summary_page()
        self.summary_tab.output_group_pack()

    def _load_statviz_from_summary(self):
        self.tab_widget.setCurrentIndex(TAB_STATVIZ)
        self.statistic_visualization.load_from_summary()

    def _load_workbench_session(self, module, index):
        self.tab_widget.setCurrentIndex(index)
        module.load_from_session()

    def _load_individual_session(self):
        self._load_workbench_session(self.individual_analysis, TAB_INDIVIDUAL)

    def _load_group_session(self):
        self._load_workbench_session(self.group_analysis, TAB_GROUP)

    def _import_workbench_framework(self, module, index):
        self.tab_widget.setCurrentIndex(index)
        module.import_framework()

    def _import_individual_framework(self):
        self._import_workbench_framework(self.individual_analysis, TAB_INDIVIDUAL)

    def _import_group_framework(self):
        self._import_workbench_framework(self.group_analysis, TAB_GROUP)

    def _export_all_results(self, module, index):
        """一键导出：没结果时模块自己会弹说明，不在菜单里再判一遍。"""
        self.tab_widget.setCurrentIndex(index)
        module.export_everything()

    def _export_statviz_all(self):
        self._export_all_results(self.statistic_visualization, TAB_STATVIZ)

    def _export_individual_all(self):
        self._export_all_results(self.individual_analysis, TAB_INDIVIDUAL)

    def _export_group_all(self):
        self._export_all_results(self.group_analysis, TAB_GROUP)

    def _sync_file_menu_state(self):
        """条件页与空现场不让菜单说谎：没有编码页时依赖它的项置灰，小组页没挂出时它那项也置灰。"""
        if not getattr(self, "_encoder_actions", None):
            return
        available = self._target_encoder_page() is not None
        for action in self._encoder_actions:
            action.setEnabled(available)
        for action in getattr(self, "_group_page_actions", []):
            action.setEnabled(self._group_page_present())

    # ---------- 最近打开：菜单现列现建，路径失效就地剔除 ----------
    def _rebuild_recent_menu(self):
        menu = self.recent_menu
        menu.clear()
        records = user_state.state.recent()
        headed = set()
        for record in records:
            bucket = record["bucket"]
            if bucket not in headed:
                headed.add(bucket)
                if menu.actions():
                    menu.addSeparator()
                caption = menu.addAction(user_state.RECENT_LABELS.get(bucket, bucket))
                caption.setEnabled(False)
            path = record["path"]
            icon_name = _RECENT_ICONS.get(bucket, "file")
            action = menu.addAction(icon_provider.icon(icon_name, theme.tokens()["ink2"], 16),
                                    os.path.basename(path))
            action.setToolTip(path)
            action.setEnabled(os.path.exists(path))
            action.triggered.connect(lambda _=False, b=bucket, p=path: self.open_recent_file(b, p))
        if not records:
            empty = menu.addAction("暂无打开记录")
            empty.setEnabled(False)
        menu.addSeparator()
        clear = menu.addAction("清空最近打开记录")
        clear.setEnabled(bool(records))
        clear.triggered.connect(self._clear_recent_files)

    def _clear_recent_files(self):
        user_state.state.clear_recent()
        self.statusBar().showMessage("已清空最近打开记录（存档不受影响）", 5000)

    def open_recent_file(self, bucket, path):
        """从「最近打开」重开一个文件：走各模块自己的装载口，不在这里再写一份解析。"""
        if not os.path.exists(path):
            user_state.state.forget(path)
            self.statusBar().showMessage("文件已不在原位，已从最近记录中移除：%s" % path, 6000)
            return False
        if bucket == user_state.BUCKET_VIDEO:
            self.tab_widget.setCurrentIndex(TAB_ENCODER)
            self.video_player.reveal_in_playlist(path)
        elif bucket == user_state.BUCKET_STATVIZ:
            self.tab_widget.setCurrentIndex(TAB_STATVIZ)
            self.statistic_visualization.open_path(path)
        elif bucket == user_state.BUCKET_INDIVIDUAL:
            self.tab_widget.setCurrentIndex(TAB_INDIVIDUAL)
            self.individual_analysis.load_files([path])
        elif bucket == user_state.BUCKET_GROUP:
            self.tab_widget.setCurrentIndex(TAB_GROUP)
            self.group_analysis.load_files([path])
        elif bucket == user_state.BUCKET_FRAMEWORK:
            page = self._target_encoder_page()
            if page is None:
                self._notice_no_encoder_page()
                return False
            self.tab_widget.setCurrentIndex(TAB_ENCODER)
            page.dim_panel.import_from_excel(path)
        else:
            return False
        self.statusBar().showMessage("已打开：%s" % path, 5000)
        return True

    def _exec_menu(self, btn, menu):
        menu.exec(btn.mapToGlobal(QPoint(0, btn.height() + 4)))

    def _open_video_files(self):
        self.tab_widget.setCurrentIndex(TAB_ENCODER)
        self.video_player.open_file()

    def _select_statviz_file(self):
        self.tab_widget.setCurrentIndex(TAB_STATVIZ)
        self.statistic_visualization.select_file()

    def _select_individual_files(self):
        self.tab_widget.setCurrentIndex(TAB_INDIVIDUAL)
        self.individual_analysis.select_file()

    def _select_group_files(self):
        self.tab_widget.setCurrentIndex(TAB_GROUP)
        self.group_analysis.select_file()

    def _show_about(self):
        AboutDialog(self).exec()

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def changeEvent(self, event):
        if event.type() == QEvent.WindowStateChange:
            self._clear_override_cursor()
            maximized = self.isMaximized()
            if self.title_bar is not None:
                theme.set_icon(self.title_bar.max_btn, "restore" if maximized else "maximize", 15)
                self.title_bar.max_btn.setToolTip("还原" if maximized else "最大化")
            self._apply_window_corner_rounding()
        elif event.type() == QEvent.ActivationChange and not self.isActiveWindow():
            self._clear_override_cursor()
        super().changeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_window_corner_rounding()
        self._apply_resolution_layout()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_resolution_layout()

    def _apply_window_corner_rounding(self):
        """非最大化时给整窗四角加 Windows 11 原生圆角。

        WM_NCCALCSIZE 把客户区扩到整窗后，DWM 不再自动圆角，需显式设置
        DWMWA_WINDOW_CORNER_PREFERENCE=DWMWCP_ROUND。最大化时 DWM 自动不圆角。
        旧系统（build<22000）该属性不存在，调用失败即保持系统直角，优雅降级。
        """
        if sys.platform != "win32":
            return
        try:
            pref = ctypes.c_int(_DWMWCP_ROUND)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                ctypes.c_void_p(int(self.winId())),
                ctypes.c_uint(_DWMWA_WINDOW_CORNER_PREFERENCE),
                ctypes.byref(pref), ctypes.sizeof(pref))
        except Exception:
            pass

    def hideEvent(self, event):
        self._clear_override_cursor()
        super().hideEvent(event)

    def nativeEvent(self, eventType, message):
        if sys.platform == "win32" and eventType == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == _WM_NCCALCSIZE and msg.wParam:
                params = _NCCALCSIZE_PARAMS.from_address(msg.lParam)
                if self.isMaximized():
                    # 最大化时窗体四周仍有不可见缩放边，不让出就会裁掉内容
                    border = (ctypes.windll.user32.GetSystemMetrics(_SM_CXFRAME)
                              + ctypes.windll.user32.GetSystemMetrics(_SM_CXPADDEDBORDER))
                    rect = params.rgrc[0]
                    rect.left += border
                    rect.top += border
                    rect.right -= border
                    rect.bottom -= border
                return True, 0
            if msg.message == _WM_NCHITTEST:
                hit = self._hit_test(msg.lParam)
                if hit:
                    return True, hit
        return super().nativeEvent(eventType, message)

    def _hit_test(self, lparam):
        x = ctypes.c_short(lparam & 0xFFFF).value
        y = ctypes.c_short((lparam >> 16) & 0xFFFF).value
        local = self.mapFromGlobal(QPoint(x, y))
        if not self.isMaximized():
            left = local.x() < RESIZE_EDGE
            right = local.x() >= self.width() - RESIZE_EDGE
            top = local.y() < RESIZE_EDGE
            bottom = local.y() >= self.height() - RESIZE_EDGE
            if top and left:
                return 13  # HTTOPLEFT
            if top and right:
                return 14  # HTTOPRIGHT
            if bottom and left:
                return 16  # HTBOTTOMLEFT
            if bottom and right:
                return 17  # HTBOTTOMRIGHT
            if left:
                return 10  # HTLEFT
            if right:
                return 11  # HTRIGHT
            if top:
                return 12  # HTTOP
            if bottom:
                return 15  # HTBOTTOM
        pos = self.title_bar.mapFromGlobal(QPoint(x, y))
        if self.title_bar.rect().contains(pos):
            # 按钮等子控件自行处理点击，空白区域作为标题栏拖拽
            return _HT_CLIENT if self.title_bar.childAt(pos) is not None else _HT_CAPTION
        return 0

    # ------------------------------------------------------------------
    # 无边框窗口拖拽 / 缩放：应用级事件过滤器
    # QVideoWidget 令整棵子树成为原生窗口，子 HWND 拦截 WM_NCHITTEST，nativeEvent 收不到，
    # 因此在 QApplication 上装全局过滤器：按全局坐标自行命中测试得到 HT 代码，再
    # ReleaseCapture() + SendMessageW(顶层 HWND, WM_NCLBUTTONDOWN/DBLCLK, ht, lParam) 进入系统模态循环。
    # ------------------------------------------------------------------
    def _enable_mouse_tracking(self, widget):
        """递归开启鼠标追踪，使缩放边悬停时能收到无按键的 Move 事件。"""
        widget.setMouseTracking(True)
        for child in widget.children():
            if isinstance(child, QWidget):
                self._enable_mouse_tracking(child)

    def eventFilter(self, obj, event):
        et = event.type()
        if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            if self._begin_gesture(event.globalPos()):
                return True
        elif et == QEvent.MouseMove:
            self._update_edge_cursor(event.globalPos())
        elif et == QEvent.MouseButtonRelease:
            self._clear_override_cursor()
        return super().eventFilter(obj, event)

    def _begin_gesture(self, gpos):
        """命中拖拽/缩放区则转发给系统模态循环（含 Aero Snap），返回 True 吞掉事件。"""
        if sys.platform != "win32":
            return False
        if QApplication.activePopupWidget() is not None:
            # 菜单/下拉打开时不启动拖拽：事件交还 Qt 关闭弹层。
            # 否则窗体被移走而弹层是独立顶层窗体，会留在原位不跟随。
            return False
        ht = self._hit_test_point(gpos)
        if ht in (0, _HT_CLIENT):
            return False
        if ht == _HT_CAPTION:
            # 第一次按下即进入移动循环，Windows 不再上报 MouseButtonDblClick，须自行判定双击
            dbl = self._is_double_click(gpos)
            self._press_timer.restart()
            self._last_press_pos = QPoint(gpos)
            self._send_nc_button(_WM_NCLBUTTONDBLCLK if dbl else _WM_NCLBUTTONDOWN, ht, gpos)
            return True
        if self.isMaximized():
            return False
        self._clear_override_cursor()
        self._send_nc_button(_WM_NCLBUTTONDOWN, ht, gpos)
        return True

    def _is_double_click(self, gpos):
        if not self._press_timer.isValid():
            return False
        if self._press_timer.elapsed() > QApplication.doubleClickInterval():
            return False
        dcx = ctypes.windll.user32.GetSystemMetrics(_SM_CXDOUBLECLK)
        dcy = ctypes.windll.user32.GetSystemMetrics(_SM_CYDOUBLECLK)
        return (abs(gpos.x() - self._last_press_pos.x()) <= dcx
                and abs(gpos.y() - self._last_press_pos.y()) <= dcy)

    def _send_nc_button(self, msg, ht, gpos):
        hwnd = int(self.winId())
        ctypes.windll.user32.ReleaseCapture()
        wh = self.windowHandle()
        dpr = wh.devicePixelRatio() if wh is not None else 1.0
        # WM_NCLBUTTON* 的 lParam 是物理像素屏幕坐标，高 DPI 下须乘 devicePixelRatio 换算
        x = int(gpos.x() * dpr) & 0xFFFF
        y = int(gpos.y() * dpr) & 0xFFFF
        ctypes.windll.user32.SendMessageW(hwnd, msg, ht, (y << 16) | x)

    def _hit_test_point(self, gpos):
        """全局坐标命中测试，返回 HT 代码；0 表示放行给子控件处理。"""
        local = self.mapFromGlobal(gpos)
        # 窗外（含负坐标）不能误判为缩放边，先用真实客户区过滤
        if not self.rect().contains(local):
            return 0
        if not self.isMaximized():
            left = local.x() < RESIZE_EDGE
            right = local.x() >= self.width() - RESIZE_EDGE
            top = local.y() < RESIZE_EDGE
            bottom = local.y() >= self.height() - RESIZE_EDGE
            if (left or right or top or bottom) and not self._is_interactive(QApplication.widgetAt(gpos)):
                if top and left:
                    return _HT_TOPLEFT
                if top and right:
                    return _HT_TOPRIGHT
                if bottom and left:
                    return _HT_BOTTOMLEFT
                if bottom and right:
                    return _HT_BOTTOMRIGHT
                if left:
                    return _HT_LEFT
                if right:
                    return _HT_RIGHT
                if top:
                    return _HT_TOP
                return _HT_BOTTOM
        tb = self.title_bar.mapFromGlobal(gpos)
        if self.title_bar.rect().contains(tb):
            # 按钮等子控件自行处理点击，空白区域作为标题栏拖拽
            return _HT_CLIENT if self.title_bar.childAt(tb) is not None else _HT_CAPTION
        return 0

    @staticmethod
    def _is_interactive(widget):
        """缩放边带内命中交互控件时让位，避免吞掉页签/按钮/滑块等点击。"""
        return isinstance(widget, (QPushButton, QTabBar, QLineEdit, QComboBox,
                                   QAbstractSlider, QAbstractItemView))

    def _update_edge_cursor(self, gpos):
        if sys.platform != "win32" or self.isMaximized() or QApplication.mouseButtons() != Qt.NoButton:
            self._clear_override_cursor()
            return
        shape = _EDGE_CURSORS.get(self._hit_test_point(gpos))
        if shape is None:
            self._clear_override_cursor()
        elif shape != self._override_cursor_shape:
            # setOverrideCursor 会入栈，仅在方向变化时重设，避免栈无限增长
            self._clear_override_cursor()
            QApplication.setOverrideCursor(QCursor(shape))
            self._override_cursor_active = True
            self._override_cursor_shape = shape

    def _clear_override_cursor(self):
        if self._override_cursor_active:
            QApplication.restoreOverrideCursor()
            self._override_cursor_active = False
            self._override_cursor_shape = None

    def _on_dark_toggled(self, checked):
        from PySide6.QtWidgets import QApplication
        theme.apply_theme(QApplication.instance(), "dark" if checked else "light")
        self.statusBar().showMessage("已切换到深色模式" if checked else "已切换到浅色模式", 3000)

    def _set_ai_coding_visible(self, visible):
        user_state.state.set_coding_ai_visible(visible)
        for _title, page in self.encoder_pages():
            page.set_ai_coding_visible(visible)

    def _set_coding_font_size(self, size):
        user_state.state.set_coding_font_size(size)
        for _title, page in self.encoder_pages():
            page.set_coding_font_size(size)

    def _set_dimension_display_mode(self, mode):
        user_state.state.set_coding_dimension_mode(mode)
        for _title, page in self.encoder_pages():
            page.dim_panel.set_display_mode(mode)

    def _choose_coding_font_size(self, _checked=False):
        previous = user_state.state.coding_font_size()
        initial = theme.coding_font_pixels(previous)
        pixels, accepted = QInputDialog.getInt(
            self, tr("自定义字号"), tr("字号（像素，10–20）"), initial, 10, 20, 1)
        if accepted:
            self._set_coding_font_size({"custom": pixels})
        else:
            index = {"small": 0, "medium": 1, "normal": 2}.get(
                previous if isinstance(previous, str) else "", 3)
            self.coding_font_menu.actions()[index].setChecked(True)

    def _on_theme_changed(self, _mode):
        self._refresh_icons()
        self.individual_analysis.redraw_chart()
        self.group_analysis.redraw_chart()
        self.statistic_visualization.redraw_chart()

    def _refresh_icons(self):
        t = theme.tokens()
        self.tab_widget.setTabIcon(0, icon_provider.icon("tab-data", t["ink2"], 18))
        self.tab_widget.setTabIcon(1, icon_provider.icon("chart-bar", t["ink2"], 18))
        self.tab_widget.setTabIcon(2, icon_provider.icon("individual", t["ink2"], 18))
        self.tab_widget.setTabIcon(3, icon_provider.icon("network", t["ink2"], 18))
        self.tab_widget.setTabIcon(4, icon_provider.icon("robot", t["ink2"], 18))
        self.tabmodule_widget.setTabIcon(0, icon_provider.icon("tab-group", t["ink2"], 16))
        for action, name in self._menu_icon_actions:
            action.setIcon(icon_provider.icon(name, t["ink2"], 16))

    def _on_tab_fade(self, index):
        w = self.tab_widget.widget(index)
        if w is not None:
            theme.fade_in(w)

    def _on_inner_tab_fade(self, index):
        w = self.tabmodule_widget.widget(index)
        if w is not None:
            theme.fade_in(w)

    def on_inner_tab_changed(self, index):
        """
        内层标签页切换时，立即更新当前标签页的布局
        """
        widget = self.tabmodule_widget.widget(index)
        if isinstance(widget, DataEncoderModule):
            # 调用已有的布局更新逻辑
            widget.on_splitter_moved()
        self._sync_layout_toggle()
        self._sync_file_menu_state()
    def _current_encoder(self):
        """当前选中的编码页；非编码页（如数据汇总）返回 None。"""
        widget = self.tabmodule_widget.currentWidget()
        return widget if isinstance(widget, DataEncoderModule) else None
    def _toggle_encoder_layout(self):
        encoder = self._current_encoder()
        if encoder is None:
            return
        stacked = self.layout_toggle_btn.isChecked()
        encoder.set_stacked_layout(stacked)
        self._encoder_stacked = stacked
        self._sync_layout_toggle()
        self.statusBar().showMessage(
            "编码区已改为上下排列" if stacked else "编码区已改为左右排列", 3000)
    def _sync_layout_toggle(self):
        """按钮回显当前编码页的排列方向；非编码页置灰。"""
        if not hasattr(self, "layout_toggle_btn"):
            return
        encoder = self._current_encoder()
        self.layout_toggle_btn.setEnabled(encoder is not None)
        stacked = bool(encoder.is_stacked()) if encoder is not None else self._encoder_stacked
        # 先改图标再设勾选：setChecked 会触发样式重绘，顺序反过来会闪一下旧图标
        theme.set_icon(self.layout_toggle_btn, "layout-v" if stacked else "layout-h", 15)
        self.layout_toggle_btn.setToolTip(
            "改回左右排列" if stacked else "改为上下排列")
        self.layout_toggle_btn.setChecked(stacked)
    def on_outer_splitter_moved(self):
        """外层分割器移动后的处理"""
        # 更新内层标签页中的编码模块
        for i in range(self.tabmodule_widget.count()):
            widget = self.tabmodule_widget.widget(i)
            if isinstance(widget, DataEncoderModule):
                widget.on_splitter_moved()
    # ------------------------------------------------------------------ 自动保存
    def _init_autosave(self):
        """启动自动保存：状态栏常驻指示 + 防抖/看门狗两个定时器。

        offscreen（冒烟与渲染脚本）下默认不写盘：那类脚本会真的构造编码页并触发
        记录/删行回调，不挡一道就等于跑一次测试就把工程里的 archive.json 盖了。
        要在那种环境下验存档，置环境变量 LEARNLYSIS_AUTOSAVE=1。
        """
        app = QApplication.instance()
        headless = app is not None and app.platformName() == "offscreen"
        self._autosave_blocked = headless and os.environ.get("LEARNLYSIS_AUTOSAVE") != "1"
        self._autosave_want = user_state.state.autosave_enabled()
        self._autosave_on = self._autosave_want and not self._autosave_blocked
        self._dirty = False
        self._suppress_dirty = False
        self._last_saved_at = None
        self._last_payload = None
        self._save_error = ""
        self.save_status = None
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(AUTOSAVE_DEBOUNCE_MS)
        self._autosave_timer.timeout.connect(self._flush_autosave)
        self._watchdog_timer = QTimer(self)
        self._watchdog_timer.setInterval(AUTOSAVE_WATCHDOG_MS)
        self._watchdog_timer.timeout.connect(self._flush_autosave)
        self._watchdog_timer.start()
        self.save_status = QLabel("")
        self.save_status.setProperty("saveState", "idle")
        self.statusBar().addPermanentWidget(self.save_status)
        self.auto_save_action.setChecked(self._autosave_on)
        self._refresh_save_status()

    def _toggle_autosave(self):
        self.set_autosave_enabled(self.auto_save_action.isChecked())

    def set_autosave_enabled(self, want):
        """开关自动保存：关掉只是不再随手写盘，退出时仍会存一次（不丢现场）。"""
        self._autosave_want = bool(want)
        if not self._autosave_blocked:
            user_state.state.set_autosave_enabled(bool(want))   # 测试环境不写坏用户设置
        self._autosave_on = self._autosave_want and not self._autosave_blocked
        self.auto_save_action.setChecked(self._autosave_on)
        if not self._autosave_on:
            self._autosave_timer.stop()
        elif self._dirty:
            self._autosave_timer.start()
        self._refresh_save_status()

    def mark_dirty(self, *_args):
        """现场改了：所有改动信号都收进这一个口（各家参数不一律忽略）。"""
        if self._suppress_dirty:
            return                        # 读档期间的整表重画不算用户改动
        self._dirty = True
        self._save_error = ""
        if self._autosave_on:
            self._autosave_timer.start()
        self._refresh_save_status()

    def _flush_autosave(self):
        """定时器到点：没改动直接返回，有改动才写盘。"""
        if not self._dirty or not self._autosave_on:
            return False
        return self.save_archive()

    def _save_archive_now(self):
        """菜单/Ctrl+S：强制落一次盘，成不成都给一句明说。"""
        ok = self.save_archive(force=True)
        if ok:
            self.statusBar().showMessage("存档已保存", 3000)
        return ok

    def _reload_archive(self):
        """回到存档现场：丢掉最后一次落盘之后的改动（自动保存下通常只丢 1.5 秒）。"""
        if self._dirty:
            reply = QMessageBox.question(
                self, "重新载入存档",
                "有尚未写入存档的改动，重新载入会把它们丢掉。是否继续？",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return False
        self.restore_from_archive()
        self.statusBar().showMessage("已重新载入存档", 5000)
        return True

    def _clear_archive_file(self):
        """清掉存档文件：只动 data/archive.json，现场内容不帮用户拆。"""
        if os.path.exists(ARCHIVE_FILE):
            reply = QMessageBox.question(
                self, "清除存档文件",
                "将删除上次工作现场的存档文件，当前窗口里的内容不受影响。是否继续？\n%s"
                % ARCHIVE_FILE,
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return False
            try:
                os.remove(ARCHIVE_FILE)
            except OSError as exc:
                logging.error("存档清除失败：%s", exc)
                self.statusBar().showMessage("存档清除失败：%s" % exc, 6000)
                return False
        self._last_payload = None
        self._dirty = False
        self._refresh_save_status()
        self.statusBar().showMessage("存档文件已清除（下次改动会重新写一份）", 6000)
        return True

    def _open_archive_folder(self):
        """把用户带到存档文件跟前：自动保存后用户看不到“保存成功了没”，至少能查到。"""
        target = ARCHIVE_FILE if os.path.exists(ARCHIVE_FILE) else data_dir
        user_state.reveal_in_explorer(target)
        return True

    def _set_save_state(self, kind, text):
        """状态栏右端的保存指示：颜色由 QSS 按 saveState 属性给，业务侧不写样式。"""
        if self.save_status is None:
            return
        if self.save_status.property("saveState") != kind:
            self.save_status.setProperty("saveState", kind)
            theme.refresh_style(self.save_status)
        self.save_status.setText(("● " + text) if text else "")
        tip = "存档文件：%s" % ARCHIVE_FILE
        if self._save_error:
            tip += "\n最近一次失败：%s" % self._save_error
        self.save_status.setToolTip(tip)

    def _refresh_save_status(self):
        """保存情况的文案只用一处算，避免定时器与信号槽各说一套。"""
        if self._autosave_blocked:
            return self._set_save_state("off", "自动保存已停用（测试环境）")
        if self._save_error:
            return self._set_save_state("error", "保存失败，稍后重试")
        if not self._autosave_on:
            return self._set_save_state("off", "自动保存已关闭（退出时仍存档）")
        if self._dirty:
            return self._set_save_state("pending", "有改动，稍后自动保存")
        if self._last_saved_at:
            return self._set_save_state(
                "saved", "已自动保存 " + time.strftime("%H:%M:%S", self._last_saved_at))
        return self._set_save_state("idle", "自动保存已开启")

    # ---------- 改动信号接线：一个数据所有者一组接法 ----------
    def _bind_content_signals(self, page):
        """一页编码数据的改动源：现场（记录/删行/改格/撤销/清空）、框架、时间胶囊、注释。"""
        page.content_changed.connect(self.mark_dirty)
        panel = getattr(page, "dim_panel", None)
        if panel is not None:
            for signal in (panel.dimension_added, panel.dimension_removed,
                           panel.dimension_renamed, panel.framework_replaced):
                signal.connect(self.mark_dirty)
        controls = getattr(page, "time_ctrl", None)
        if controls is not None:
            controls.params_changed.connect(self.mark_dirty)
        comment = getattr(page, "comment_entry", None)
        if comment is not None:
            comment.textEdited.connect(self.mark_dirty)

    def _bind_group_page_signals(self):
        """小组页的现场改动页内自己发（组号/词表/对象字典不进撤销栈，接不得表模型）。"""
        self.group_encoder.content_changed.connect(self.mark_dirty)
        self.group_encoder.time_ctrl.params_changed.connect(self.mark_dirty)

    def _bind_summary_signals(self):
        """汇总表是从编码页推出来的，但它也进存档：整表重画同样记一次改动
        （写盘前会比内容，重推出同样的表不会真多写一次盘）。"""
        model = self.summary_tab.summary_table.model()
        if model is not None:
            for signal in (model.rowsInserted, model.rowsRemoved,
                           model.dataChanged, model.layoutChanged):
                signal.connect(self.mark_dirty)

    def _bind_video_signals(self):
        """存档里只记当前视频路径：列表增减与切项都要跟上。"""
        self.video_player.file_list.currentItemChanged.connect(self.mark_dirty)
        model = self.video_player.file_list.model()
        if model is not None:
            for signal in (model.rowsInserted, model.rowsRemoved):
                signal.connect(self.mark_dirty)

    # ------------------------------------------------------------------ 存档读写
    def get_archive_data(self):
        """构建存档数据"""
        archive_data = {
            "video": self.video_player.current_file_path,
            "tabs": [],
            "summary": [],
            "summary_columns": [],
            "group": {},
            "workbench": {},
        }

        # 保存编码个体标签页
        for i in range(self.tabmodule_widget.count()):
            widget = self.tabmodule_widget.widget(i)
            if isinstance(widget, DataEncoderModule) and widget is not self.summary_tab:
                tab_data = {
                    "name": source_text(self.tabmodule_widget.tabText(i)),
                    "data": widget.to_dict()
                }
                archive_data["tabs"].append(tab_data)

        # 保存汇总数据（列头一并存：表头带“标签页名·列名”前缀，光靠列序读不回来）
        summary_table = self.summary_tab.summary_table
        archive_data["summary_columns"] = [
            source_text(summary_table.horizontalHeaderItem(col).text())
            if summary_table.horizontalHeaderItem(col) else ""
            for col in range(summary_table.columnCount())]
        summary_data = []
        for row in range(summary_table.rowCount()):
            row_data = []
            for col in range(summary_table.columnCount()):
                item = summary_table.item(row, col)
                row_data.append(item.text() if item else "")
            summary_data.append(row_data)
        archive_data["summary"] = summary_data
        # 小组页不在时也要存档：标签页规则只是“暂不展示”，事件数据不能丢
        archive_data["group"] = self.group_encoder.to_dict()
        # 两个模式分析工作台的现场（选中方法/参数/取数批次）：以前只存编码页，
        # 重开软件要重新选一遍方法，现在跟着存档一起回来
        archive_data["workbench"] = {
            "individual": self.individual_analysis.archive_state(),
            "group": self.group_analysis.archive_state(),
        }
        return archive_data

    def save_archive(self, force=False):
        """写存档：原子替换，全程不弹模态。

        自动保存是随手一拍就写一次，弹框会砸在用户手上；失败一律走状态栏指示 +
        日志，具体原因放在指示的 tooltip 里。
        """
        self._set_save_state("saving", "正在保存…")
        try:
            payload = json.dumps(self.get_archive_data(), ensure_ascii=False, indent=2,
                                 default=str)
        except Exception as exc:                              # noqa: BLE001
            logging.exception("存档内容生成失败")
            self._save_error = "内容生成失败：%s" % exc
            self._refresh_save_status()
            return False
        if not force and payload == self._last_payload:
            # 内容没变（比如只是重画了表）：不重复写盘，直接报“已是最新”
            self._dirty = False
            self._refresh_save_status()
            return True
        try:
            os.makedirs(data_dir, exist_ok=True)
            if os.path.exists(ARCHIVE_FILE) and not os.access(ARCHIVE_FILE, os.W_OK):
                os.chmod(ARCHIVE_FILE, 0o666)             # 只读属性会挡住替换
            with open(ARCHIVE_TMP_FILE, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(ARCHIVE_TMP_FILE, ARCHIVE_FILE)    # 同一分区内替换，不会留半份
        except Exception as exc:                            # noqa: BLE001
            logging.error("存档写入失败：%s（文件路径：%s）", exc, ARCHIVE_FILE)
            self._save_error = str(exc)
            self._refresh_save_status()
            return False
        self._last_payload = payload
        self._dirty = False
        self._save_error = ""
        self._last_saved_at = time.localtime()
        self._refresh_save_status()
        return True

    def load_archive(self):
        """读存档。文件坏了不弹模态：记日志、把坏文件挪成 .bak 再当空档继续。

        自动保存下这个函数在启动时一定会跑到，用户没做错任何事，不该被一个警告框
        挡在门外；坏数据留着 .bak 也给了手工找回的余地。
        """
        if not os.path.exists(ARCHIVE_FILE):
            return None
        try:
            with open(ARCHIVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            logging.warning("存档加载失败：%s，已隔离为 .bak", exc)
            try:
                os.replace(ARCHIVE_FILE, ARCHIVE_FILE + ".bak")
            except OSError as be:
                logging.error("坏存档备份失败：%s", be)
            return None
        # 记下这份原文：下一次自动保存若内容没变就不必再写一遍
        self._last_payload = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        return data
    # ---------- 标签页管理栏：下划线 / “+” / 关闭叉 ----------
    def _schedule_tab_bar_sync(self, _index=-1):
        """标签增删后延后一帧同步：QTabBar 没有公开的 tabAdded 信号，
        故由各个增删入口显式调用；延后一帧等布局完成后再摆“+”与量宽度"""
        QTimer.singleShot(0, self._sync_tab_bar)

    def _first_encoder_index(self):
        """第一个编码页的序号（即默认的“编码个体1”）"""
        for i in range(self.tabmodule_widget.count()):
            if isinstance(self.tabmodule_widget.widget(i), DataEncoderModule):
                return i
        return -1

    # ---------- 小组交互页：存则常驻，成员/框架/状态表都现取 ----------
    def encoder_pages(self):
        """按标签顺序返回 (标题, 编码页)，汇总页与小组页不在内"""
        return [(source_text(self.tabmodule_widget.tabText(i)), self.tabmodule_widget.widget(i))
                for i in range(self.tabmodule_widget.count())
                if isinstance(self.tabmodule_widget.widget(i), DataEncoderModule)]

    # ---------- AI 助手：菜单转发 + 工作现场快照 ----------
    def _export_ai_conversation(self):
        """文件菜单「导出 AI 对话」：切到 AI 助手页，转发到模块自己的公开导出方法。"""
        self.tab_widget.setCurrentIndex(TAB_AI)
        self.ai_assistant.export_conversation()
        return True

    def _ai_context_snapshot(self):
        """给 AI 助手的「当前工作现场」快照：只交取数口，不摊开内部控件结构。

        数据汇总页给状态表、小组交互页给事件表、编码页给框架；可选附上当前正在看的
        某张结果表。任一取数口出错都由 ai_context 内部吞掉，绝不让 AI 页打断主窗口。
        """
        return {
            "summary_tab": self.summary_tab,
            "group_encoder": self.group_encoder,
            "encoder_pages": self.encoder_pages(),
            "group_id": "session",
            "result_text": self._ai_result_text(),
        }

    def _ai_result_text(self):
        """当前激活工作台的结果首表转纯文本（「结果」上下文用）；拿不到就返回空串。"""
        index = self.tab_widget.currentIndex()
        module = {TAB_INDIVIDUAL: self.individual_analysis,
                  TAB_GROUP: self.group_analysis}.get(index)
        result = getattr(module, "result", None) if module is not None else None
        if result is None:
            return ""
        try:
            parts = []
            name = getattr(result, "method_name", "") or getattr(result, "method_id", "")
            if name:
                parts.append("方法：%s" % name)
            frame = result.table() if hasattr(result, "table") else None
            if frame is not None and len(frame):
                parts.append(frame.head(40).to_string(index=False))
            notes = [str(note) for note in (getattr(result, "notes", []) or [])][:5]
            if notes:
                parts.append("说明：" + "；".join(notes))
            return "\n".join(parts)[:4000]
        except Exception:                       # noqa: BLE001 结果转文本失败不该打断对话
            return ""

    def _ai_framework_targets(self):
        """框架设计工作台可选择的编码页；只返回普通 dict，不暴露控件。"""
        return [{"id": title, "label": title,
                 "row_count": len(getattr(page, "selected_data", []) or []),
                 "framework": page.dim_panel.framework()}
                for title, page in self.encoder_pages()]

    def _ai_apply_framework(self, target_id, items):
        """研究者确认后的唯一应用口：默认新建页；替换旧页时再次检查数据影响。"""
        clean = [(str(name).strip(), [str(code).strip() for code in codes if str(code).strip()])
                 for name, codes in (items or []) if str(name).strip()]
        if not clean:
            return False, "框架中没有可应用的状态维度。"
        if target_id == "__new__":
            page = self.add_new_tab()
            page.dim_panel.apply_framework(clean)
            index = self.tabmodule_widget.indexOf(page)
            if index >= 0:
                self.tabmodule_widget.setCurrentIndex(index)
            self.mark_dirty()
            return True, "已新建编码页并应用状态框架；事件维度仍保留在框架设计项目中。"
        target = None
        for title, page in self.encoder_pages():
            if title == str(target_id):
                target = page
                break
        if target is None:
            return False, "目标编码页已不存在，请刷新后重试。"
        rows = len(getattr(target, "selected_data", []) or [])
        if rows:
            answer = QMessageBox.question(
                self, "确认替换编码框架",
                "目标页已有 %d 条记录。替换会保留注释，但清空这些记录中的编码值。是否继续？" % rows,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return False, "研究者取消了替换操作。"
        target.dim_panel.apply_framework(clean)
        self.mark_dirty()
        return True, "已应用到“%s”。" % target_id

    def group_member_sources(self):
        """成员候选 [(S{n}, 页标题)]：页多一页，小组页就多一个 actor。"""
        return [(page.student_id(), title) for title, page in self.encoder_pages()]

    def group_framework_source(self):
        """Codebook 行：取第一个已设框架的编码页（小组页不改框架，只回显）。"""
        for _title, page in self.encoder_pages():
            items = page.dim_panel.framework()
            if items:
                return items
        return []

    def group_state_frame(self):
        """Behavior_State：数据汇总页的 Time + S{i}Encoding_{j} 标准表。"""
        columns, rows = self.summary_tab.build_analysis_data()
        if not columns or not rows:
            return None
        import pandas as pd
        return pd.DataFrame(rows, columns=columns)

    def group_pack_info(self):
        """给汇总页导出用的一完整组现场：组号/组名/成员/词表/框架/事件表。"""
        page = self.group_encoder
        return {
            "group_id": page.group_id,
            "group_name": page.group_name,
            "members": list(page.members),
            "interaction_types": list(page.interaction_types),
            "codebook": self.group_framework_source(),
            "events": page.events_frame(),
        }

    def _ensure_group_tab(self):
        """编码个体页 ≥ 2 时挂出「小组交互」，< 2 时摘掉但保留实例与内存数据。

        摘页用 removeTab 而非 deleteLater：用户“先建两页编完再删一页”不应丢事件表，
        回到 ≥ 2 时原样挂回。
        """
        show = len(self.encoder_pages()) >= 2
        index = self.tabmodule_widget.indexOf(self.group_encoder)
        if show:
            self.group_encoder.refresh_members()
            if index < 0:
                summary_index = self.tabmodule_widget.indexOf(self.summary_tab)
                at = summary_index if summary_index >= 0 else self.tabmodule_widget.count()
                self.tabmodule_widget.insertTab(at, self.group_encoder, "小组交互")
                translate_widget_tree(self.tabmodule_widget)
            self._sync_file_menu_state()
            return True
        if index >= 0:
            current = self.tabmodule_widget.currentIndex()
            self.tabmodule_widget.removeTab(index)
            if current == index:
                self.tabmodule_widget.setCurrentIndex(max(index - 1, 0))
        self._sync_file_menu_state()
        return False

    def _tab_closable(self, index):
        """默认编码页、小组交互页与数据汇总页常驻，不提供关闭入口（至少留一个编码页）"""
        widget = self.tabmodule_widget.widget(index)
        if not isinstance(widget, DataEncoderModule):
            return False
        return index != self._first_encoder_index()

    def _sync_tab_bar(self):
        """按当前标签列表重建关闭叉：可关闭的页挂上叉按钮，常驻页不挂"""
        bar = self.encoder_tab_bar
        for i in range(bar.count()):
            btn = bar.tabButton(i, QTabBar.RightSide)
            if btn is not None and not btn.property("tabCloseBtn"):
                bar.setTabButton(i, QTabBar.RightSide, None)
                btn.deleteLater()
                btn = None
            if not self._tab_closable(i):
                if btn is not None:
                    bar.setTabButton(i, QTabBar.RightSide, None)
                    btn.deleteLater()
                continue
            if btn is None:
                bar.setTabButton(i, QTabBar.RightSide, self._make_close_button())
            # 有框架可复制的页顺带挂上提示，让人知道点/右键标签页有菜单
            target = self._copy_target_page(i)
            bar.setTabToolTip(i, "可复制其他模块的编码框架：点击或右键本标签页"
                              if target is not None and self.framework_sources(target) else "")
        bar.place_add_button()

    def _make_close_button(self):
        """标签页关闭叉：与维度/编码行的删除叉同一画法（ghost + close 图标）"""
        btn = QPushButton()
        btn.setFixedSize(theme.TAB_CLOSE_SIZE, theme.TAB_CLOSE_SIZE)
        btn.setProperty("btn", "tabClose")
        btn.setProperty("tabCloseBtn", True)
        btn.setToolTip("关闭该标签页")
        theme.bind_icon(btn, "close", 12)
        btn.clicked.connect(lambda _=False, b=btn: self.close_encoder_tab(b))
        return btn

    def close_encoder_tab(self, button):
        """点叉时按按钮反查标签号，避免闭包里的旧索引在增删后失效"""
        bar = self.encoder_tab_bar
        for i in range(bar.count()):
            if bar.tabButton(i, QTabBar.RightSide) is not button:
                continue
            if self._tab_closable(i):
                widget = self.tabmodule_widget.widget(i)
                self.tabmodule_widget.removeTab(i)
                if widget is not None:
                    widget.deleteLater()
                # 少于一页编码时小组页也要跟着摘（数据不丢）
                self._ensure_group_tab()
                self._schedule_tab_bar_sync()
                # 关掉的不一定是当前页，未必会发 currentChanged，这里补一次回显
                self._sync_layout_toggle()
                self._sync_file_menu_state()
                self.mark_dirty()
            return

    # ---------- 编码框架：跨编码页登记与复制（只有编码页持有通用面板） ----------
    def framework_hosts(self):
        """所有能维护编码框架的界面：只剩各编码标签页

        个体 / 小组两个工作台不再自带维度面板（框架由数据契约解析：会话 > 导入 Excel >
        数据推断），所以它们不在宿主名单里 —— 「右键标签页复制编码框架」在这两页取消。
        """
        hosts = []
        for i in range(self.tabmodule_widget.count()):
            widget = self.tabmodule_widget.widget(i)
            if isinstance(widget, DataEncoderModule):
                hosts.append((self.tabmodule_widget.tabText(i), widget))
        return hosts

    def framework_sources(self, target=None):
        """可复制来源 [(模块名, [(维度名, [编码...])])]：排除目标自身与还没框架的模块"""
        sources = []
        for label, widget in self.framework_hosts():
            if widget is target or not hasattr(widget, "dim_panel"):
                continue
            items = widget.dim_panel.framework()
            if items:
                sources.append((label, items))
        return sources

    def _register_framework_panels(self):
        """把登记表接到各面板的 sources_provider 上：「导入框架」据此问要不要复制"""
        for _label, widget in self.framework_hosts():
            widget.dim_panel.sources_provider = (
                lambda target=widget: self.framework_sources(target))

    def _framework_label(self, widget):
        """控件在界面上的名字：编码页取内层标签文字，分析模块取外层标签文字"""
        index = self.tabmodule_widget.indexOf(widget)
        if index >= 0:
            return self.tabmodule_widget.tabText(index)
        index = self.tab_widget.indexOf(widget)
        if index >= 0:
            return self.tab_widget.tabText(index)
        return "本模块"

    def _tab_framework_target(self, index):
        """外层标签对应的框架宓主；“统计及可视化”没有框架，返回 None"""
        role = self._framework_tab_roles.get(index)
        if role == "encoder":
            widget = self.tabmodule_widget.currentWidget()
            return widget if isinstance(widget, DataEncoderModule) else None
        return role

    def _copy_target_page(self, index):
        """该内层标签能否接受“复制编码框架”：必须是编码页（数据汇总页不是）"""
        if index < 0 or index >= self.tabmodule_widget.count():
            return None
        widget = self.tabmodule_widget.widget(index)
        return widget if isinstance(widget, DataEncoderModule) else None

    def _copy_framework_menu(self, target, owner_bar):
        """构造“复制编码框架”菜单：列出除目标自身外所有已设框架的模块"""
        sources = self.framework_sources(target)
        if not sources:
            return None
        # 菜单窗体需在原生窗体创建前去掉阴影方框 + 圆角裁剪，否则圆角外残留底层矩形
        menu = theme.bind_menu_popup(QMenu(owner_bar))
        for label, items in sources:
            action = menu.addAction(f"复制编码框架（来自「{label}」）")
            action.setToolTip("把该模块的维度与编码整份复制到本模块")
            action.triggered.connect(
                lambda _=False, t=target, l=label, i=items: self.copy_framework(t, i, l))
        return menu

    def copy_framework(self, target, items, source_label):
        """套用一个框架；编码页已有记录时先确认（复制会让旧记录的编码值作废）"""
        if isinstance(target, DataEncoderModule) and target.selected_data:
            reply = QMessageBox.question(
                self, "确认复制框架",
                f"「{self._framework_label(target)}」已有 {len(target.selected_data)} 条记录，"
                "复制框架会保留注释但清空这些记录的编码值，是否继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        target.dim_panel.apply_framework(items)
        self.statusBar().showMessage(
            f"已将「{source_label}」的编码框架复制到「{self._framework_label(target)}」", 5000)

    def _show_encoder_tab_menu(self, pos):
        """右键内层编码标签：可复制的编码页上弹出“复制编码框架”菜单"""
        bar = self.encoder_tab_bar
        target = self._copy_target_page(bar.tabAt(pos))
        if target is None:
            return
        menu = self._copy_framework_menu(target, bar)
        if menu is not None:
            menu.exec(bar.mapToGlobal(pos))

    def _show_module_tab_menu(self, pos):
        """右键外层模块标签：把其他模块已设好的编码框架复制到本模块"""
        bar = self.outer_tab_bar
        target = self._tab_framework_target(bar.tabAt(pos))
        if target is None:
            return
        menu = self._copy_framework_menu(target, bar)
        if menu is not None:
            menu.exec(bar.mapToGlobal(pos))

    def _offer_copy_menu(self, target, bar, index):
        """首次进入还没框架的模块时弹一次菜单：关掉即按空白起步，每个模块只提示一次"""
        if target is None or target.dim_panel.has_framework():
            return
        if getattr(target, "_copy_offered", False):
            return
        menu = self._copy_framework_menu(target, bar)
        if menu is None:
            return
        target._copy_offered = True
        global_pos = bar.mapToGlobal(bar.tabRect(index).bottomLeft())
        # 延后一帧：等标签选中状态与淡入微动效落地再弹菜单
        QTimer.singleShot(0, lambda m=menu, p=global_pos: m.exec(p))

    def _offer_copy_framework(self, index):
        """点进尚无框架的编码页时的提示入口"""
        self._offer_copy_menu(self._copy_target_page(index), self.encoder_tab_bar, index)

    def _offer_module_copy(self, index):
        """首次进入聚类/滞后时的提示入口（编码页已有内层标签的提示，不重复打扰）"""
        if self._framework_tab_roles.get(index) == "encoder":
            return
        self._offer_copy_menu(self._tab_framework_target(index), self.outer_tab_bar, index)

    def add_new_tab(self):
        used_indices = []
        for i in range(self.tabmodule_widget.count()):
            tab_text = source_text(self.tabmodule_widget.tabText(i))
            if tab_text == "数据汇总":
                continue
            if tab_text.startswith("编码个体"):
                try:
                    index = int(tab_text[4:])  # 提取数字部分
                    used_indices.append(index)
                except:
                    pass
        # 找到最小未使用的序号
        used_indices.sort()
        next_index = 1
        for idx in used_indices:
            if idx == next_index:
                next_index += 1
            elif idx > next_index:
                break
        # 创建新标签页并设置名称
        new_tab = DataEncoderModule(tabmodule=self.tabmodule_widget)
        self.tabmodule_widget.addTab(new_tab, f"编码个体{next_index}")
        translate_widget_tree(self.tabmodule_widget)
        # 新页也要把改动接进自动保存，否则新建页记了东西不存盘
        self._bind_content_signals(new_tab)
        # 新页的「导入框架」也要能看到其他模块的框架
        self._register_framework_panels()
        # 新页沿用当前选择的排列（上栏尺寸要等显示后才能算，故延后一拍）
        if self._encoder_stacked:
            new_tab.set_stacked_layout(True)
        self.move_summary_to_end()
        self._schedule_tab_bar_sync()
        self._sync_layout_toggle()
        self._sync_file_menu_state()
        self.mark_dirty()
        return new_tab
    def restore_from_archive(self):
        """从存档恢复数据。整表重画会一路推信号，但不算用户改动，全程压住脏标记。"""
        archive_data = self.load_archive()
        if not archive_data:
            return False
        prev_suppress = self._suppress_dirty
        self._suppress_dirty = True
        try:
            self._apply_archive(archive_data)
        finally:
            self._suppress_dirty = prev_suppress
            self._dirty = False
            self._refresh_save_status()
        return True

    def _apply_archive(self, archive_data):
        # 恢复视频（不在播放列表里就补上，再按老口径装载）
        video_path = archive_data.get("video")
        if video_path and os.path.exists(video_path):
            self.video_player.reveal_in_playlist(video_path)

        # 清空现有编码个体标签页（除了数据汇总）
        for i in reversed(range(self.tabmodule_widget.count())):
            widget = self.tabmodule_widget.widget(i)
            if isinstance(widget, DataEncoderModule) and widget is not self.summary_tab:
                self.tabmodule_widget.removeTab(i)

        # 恢复编码个体
        for tab_data in archive_data.get("tabs", []):
            tab_name = source_text(tab_data["name"])
            if tab_name == "数据汇总":
                continue

            new_tab = self.add_new_tab()
            self.tabmodule_widget.setTabText(self.tabmodule_widget.indexOf(new_tab), tab_name)
            translate_widget_tree(self.tabmodule_widget)
            new_tab.from_dict(tab_data["data"])

        # 恢复小组页（旧存档没这些键 → from_dict 直接返回，不猜数据）
        self.group_encoder.from_dict(archive_data.get("group"))
        self._ensure_group_tab()

        # 恢复汇总数据
        summary_data = archive_data.get("summary", [])
        if summary_data:
            self.summary_tab.update_table_with_data(
                summary_data, archive_data.get("summary_columns"))
            self.summary_tab.generate_summary_data()

        # 恢复两个模式分析工作台的选方法与参数（旧存档没这个键 → 什么都不做）
        workbench = archive_data.get("workbench") or {}
        self.individual_analysis.restore_state(workbench.get("individual"))
        self.group_analysis.restore_state(workbench.get("group"))

        # 如果没有存档数据，则添加一个新的“编码个体1”标签页
        if not archive_data.get("tabs"):
            self.add_new_tab()
        self._schedule_tab_bar_sync()
        self._sync_file_menu_state()

    def check_and_restore_archive(self):
        """启动时直接回到上次现场，不再弹窗问“是否恢复”。

        自动保存下存档恒存在，弹窗会变成每启动一次挡一次；而“否”会删档，
        一不当心就把别人的现场拆了。不想要旧现场时用「文件 → 工作进度 → 清除存档文件」。
        """
        if not os.path.exists(ARCHIVE_FILE):
            return False
        if self.restore_from_archive():
            self.statusBar().showMessage("已恢复上次的工作现场，后续改动会自动保存", 6000)
            return True
        return False

    def closeEvent(self, event):
        """关闭：先落一次盘再走，不弹“是否保存”。"""
        # 自动保存被关了也不能丢现场；但测试环境（offscreen）默认不写盘，别盖工程里的存档
        if not self._autosave_blocked and (self._dirty or self._autosave_on):
            self.save_archive()
        self._autosave_timer.stop()
        self._watchdog_timer.stop()

        # 后台 R 探测/复算、后台训练还在跑时先给个收尾窗口：QThread 被销毁前必须停
        for module in (self.individual_analysis, self.group_analysis):
            for name in ("join_compute", "join_r_tasks"):
                hook = getattr(module, name, None)
                if callable(hook):
                    try:
                        hook(1000)
                    except Exception:               # noqa: BLE001  收尾失败不挡退出
                        pass

        event.accept()
    def on_outer_tab_change(self, index):
        """外层标签页切换时的处理：首次进入无框架的分析模块时问一次是否复制"""
        self._offer_module_copy(index)
    def move_summary_to_end(self):
        """数据汇总永远最后一格；「小组交互」按编码页数量在位与否则紧随编码页。"""
        self._ensure_group_tab()
        # 逐个“拔下来再挂回去”：removeTab 不销毁控件，小组页拿回的仍是同一实例
        for widget, label in ((self.group_encoder, "小组交互"),
                              (self.summary_tab, "数据汇总")):
            index = self.tabmodule_widget.indexOf(widget)
            if index == -1:
                continue
            self.tabmodule_widget.removeTab(index)
            self.tabmodule_widget.addTab(widget, label)
        translate_widget_tree(self.tabmodule_widget)
        self._schedule_tab_bar_sync()
