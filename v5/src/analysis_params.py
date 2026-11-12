# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""分析类页面的共用参数控件：statviz / 个体模式 / 小组模式三处共用同一份实现。

「统计及可视化」先把五卡工作台做稳了，参数行的构造（checklist / combo / segmented /
spin / dspin / check / text 七种控件）、取值、回填、按控制器显隐这四件事都写在了
statviz 内部。个体模式与小组模式工作台要的是同一套观感与同一套语义——参数卡长得不
一样、或同一份 ParamSpec 在两处解析出不同值，比算法算错更难查。所以把这几件集中到
这里维护一次：

- ``CheckList`` / ``Segmented`` / ``PreviewCanvas`` / ``Note``：小控件与展示件。
- ``attach_fullscreen_button`` / ``FigureViewDialog``：预览区右下角的「全屏查看」。
- ``make_param_widget`` / ``read_widget`` / ``collect_params`` / ``apply_params`` /
  ``apply_visibility``：参数行的构造与读写，宿主只需提供 ``dataset`` 与回调。

行三元组 ``(param, host, widget)`` 的形态与 statviz 原有一致，测试与既有调用点不改。
"""

import io
import warnings

from PySide6.QtCore import QEvent, QObject, QSize, Qt
from PySide6.QtGui import QFontMetrics, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                             QHBoxLayout, QGridLayout, QLabel, QLineEdit,
                             QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout,
                             QWidget, QPushButton)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import theme

MISSING_TEXT = "—"
LEVEL_PREFIX = {"error": "【阻断】", "warn": "【警告】", "info": "【提示】"}
LEFT_PANEL_W = 452        # 左栏宽度下限：statviz 仍按它定宽，IND/GRP 工作台会按内容加宽
COMBO_ARROW_PAD = 52      # 下拉框里给箭头与左右内边距留的固定宽度
PARAM_LABEL_W = 96        # 参数行标题列宽：与 statviz 原有观感一致
CHECKLIST_COLUMNS = 3     # 多选清单每行最多几个候选（放不下会自动减列）
CHECKLIST_SPACING = 10    # 多选清单的列间距：减列时要按它算总宽
CHECKLIST_BUDGET_W = 300  # 参数行留给多选清单的宽度预算：左栏 452 - 卡片边距 - 标题列


def combo_width(combo, floor=80):
    """按最长条目算宽度：写死宽度会把 “PNG / 300” 截成 “PN / 30”。"""
    metrics = QFontMetrics(combo.font())
    widest = max((metrics.horizontalAdvance(combo.itemText(index))
                  for index in range(combo.count())), default=0)
    return max(floor, widest + COMBO_ARROW_PAD)


def shrink_to_fit(widget):
    """让控件只吃布局给它的宽度，不被自己的 sizeHint 顶开宿主。

    QComboBox / QLineEdit 默认是 Preferred 策略，布局会把 sizeHint（下拉按最长
    条目算）当成最小宽度 —— 一个长批次标签就能把左栏多撑出 100 px。改成
    Ignored 后控件就地截断，而下拉弹层仍按内容加宽（Qt 取 max(控件宽, 列表宽)），
    全文一个字都不丢。
    """
    widget.setSizePolicy(QSizePolicy.Ignored, widget.sizePolicy().verticalPolicy())
    return widget


class PreviewCanvas(FigureCanvas):
    """预览画布：尺寸由 figure 决定，不被视口反向压缩。

    matplotlib 的 Qt 后端会在 resizeEvent 里用控件尺寸改写 figure 尺寸，那样
    “按内容撑高 + 滚动查看”会立刻被打回原形，所以这里只接尺寸、不回写 figure，
    尺寸一律由宿主显式下达。
    """

    def sizeHint(self):
        # get_width_height 是 canvas 的方法（figure 上没有），返回逻辑像素：
        # 物理像素 / device_pixel_ratio，正好等于 matplotlib 原本会给控件的尺寸。
        # 该函数会被 Qt 在 C++ 侧回调，抛异常直接 abort 整个进程，所以兜底到底值。
        try:
            width, height = self.get_width_height()
        except Exception:
            width, height = 800, 480
        return QSize(max(int(width), 1), max(int(height), 1))

    def resizeEvent(self, event):
        event.accept()

    def sync_to_figure(self):
        """把控件钉成 figure 的尺寸，不够放时由外层滚动区出滚动条。"""
        self.setFixedSize(self.sizeHint())
        self.updateGeometry()


class Note:
    """把结果的说明文案（notes）并入质检卡展示。"""

    def __init__(self, message):
        self.message = message
        self.code = "note"
        self.level = "note"
        self.blocking = False


def prefix_of(item):
    if getattr(item, "level", "") == "note":
        return "【说明】"
    return LEVEL_PREFIX.get(getattr(item, "level", "info"), "【提示】")


def qa_rows(result):
    """质检卡条目：阻断 → 警告 → 提示 → 说明，级别内保持产生顺序。"""
    items = list(getattr(result, "qa", []) or [])
    items.extend(Note(message) for message in (getattr(result, "notes", []) or []))
    order = {"error": 0, "warn": 1, "info": 2, "note": 3}
    return sorted(items, key=lambda item: order.get(getattr(item, "level", "info"), 2))


# ------------------------------------------------------------------ 小控件

class CheckList(QWidget):
    """学生 / 维度的多选清单，带「全选」开关。

    三列是理想排布，但学生 uid（`{group_id}__{student}`）能吃到 110 px 一格，
    硬排三列会把整张参数卡撑到左栏之外 —— 左栏一出横向滚动条，「生成结果」就被
    推出视野。所以列数按预算现算：从 ``CHECKLIST_COLUMNS`` 往下减到放得下为止，
    宁可多一行也不横向溢出。
    """

    def __init__(self, options, on_change=None, parent=None, budget=CHECKLIST_BUDGET_W):
        super().__init__(parent)
        self._on_change = on_change
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(CHECKLIST_SPACING)
        lay.setVerticalSpacing(4)
        self.all_box = QCheckBox("全选")
        self.all_box.setChecked(True)
        self.all_box.toggled.connect(self._toggle_all)
        self._boxes = []
        for value in options:
            box = QCheckBox(str(value))
            box.setChecked(True)
            box.toggled.connect(self._changed)
            self._boxes.append((str(value), box))
        columns = self._columns_for(self._boxes, budget)
        lay.addWidget(self.all_box, 0, 0, 1, columns)
        for index, (_value, box) in enumerate(self._boxes):
            lay.addWidget(box, 1 + index // columns, index % columns)
        for column in range(columns):
            lay.setColumnStretch(column, 1)

    @staticmethod
    def _columns_for(boxes, budget):
        """按最宽的一格试列数：`列数 × 格宽 + 间距` 放得进预算才肯多开一列。"""
        widest = max((box.sizeHint().width() for _, box in boxes), default=0)
        columns = CHECKLIST_COLUMNS
        while columns > 1 and columns * widest + (columns - 1) * CHECKLIST_SPACING > budget:
            columns -= 1
        return columns

    def _toggle_all(self, state):
        for _, box in self._boxes:
            box.blockSignals(True)
            box.setChecked(bool(state))
            box.blockSignals(False)
        self._changed()

    def _changed(self):
        all_checked = bool(self._boxes) and all(box.isChecked() for _, box in self._boxes)
        self.all_box.blockSignals(True)
        self.all_box.setChecked(all_checked)
        self.all_box.blockSignals(False)
        if self._on_change is not None:
            self._on_change()

    def values(self):
        picked = [value for value, box in self._boxes if box.isChecked()]
        return picked or [value for value, _ in self._boxes]

    def set_values(self, values):
        wanted = {str(value) for value in (values or [])}
        if not wanted:
            for _, box in self._boxes:
                box.setChecked(True)
            return
        for value, box in self._boxes:
            box.setChecked(value in wanted)

    def options(self):
        return [value for value, _ in self._boxes]


class Segmented(QWidget):
    """互斥分段按钮：用于「必须先显式确认」的口径选择。"""

    def __init__(self, options, on_change=None, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self._buttons = {}
        self._value = None
        self._on_change = on_change
        group = QButtonGroup(self)
        group.setExclusive(True)
        for index, (value, label) in enumerate(options):
            button = QPushButton(str(label))
            button.setProperty("btn", "roundSm")
            button.setCheckable(True)
            button.setFixedHeight(28)
            group.addButton(button, index)
            self._buttons[str(value)] = button
            button.clicked.connect(lambda _=False, v=str(value): self.pick(v))
            lay.addWidget(button)
        lay.addStretch(1)

    def pick(self, value):
        self._value = value
        for key, button in self._buttons.items():
            button.setChecked(key == value)
        if self._on_change is not None:
            self._on_change()

    def setValue(self, value):
        self._value = None if value is None else str(value)
        for key, button in self._buttons.items():
            button.setChecked(key == self._value)

    def value(self):
        return self._value


# ------------------------------------------------------------------ 参数量身定制

def dataset_values(dataset, which):
    """checklist / combo 的候选来源：数据集上的 students / dimensions 等列表。"""
    if dataset is None:
        return []
    return list(getattr(dataset, which, []) or [])


# 多选清单可以按参数名直接向宿主取候选；名单外的键只认声明过的 options。
CHECKLIST_SOURCES = ("students", "groups", "dimensions", "features")


def checklist_options(param, dataset):
    """checklist 的候选值：声明了的优先，否则按参数名从数据推断。

    「特征族」这类固定枚举必须走 options —— 它们不在数据对象上，硬去 `getattr`
    只会拿到空列表，然后整行参数凭空消失，用户看到的是「方法少了个开关」。
    """
    if param.options:
        return [str(value) for value, _text in param.options]
    which = param.key if param.key in CHECKLIST_SOURCES else "dimensions"
    return [str(value) for value in dataset_values(dataset, which)]


def options_for(param, dataset):
    """combo 的候选：声明了的优先，否则按参数名从数据推断。"""
    if param.options:
        return list(param.options)
    if param.key in ("student", "student_a", "student_b"):
        return [(value, value) for value in dataset_values(dataset, "students")]
    if param.key in ("dimension", "dim_a", "dim_b"):
        return [(value, value) for value in dataset_values(dataset, "dimensions")]
    return []


def make_param_widget(param, dataset=None, on_change=None):
    """按 ``ParamSpec.kind`` 造控件；没有候选（如还没装载数据）时返回 None。

    ``on_change`` 只在“值变会牵动其他行”的控件上接（checklist / combo / segmented）：
    spin / check / text 的联动统一交给宿主的重算入口，不在此处发信号。
    """
    def changed():
        if on_change is not None:
            on_change(param.key)

    if param.kind == "checklist":
        options = checklist_options(param, dataset)
        return CheckList(options, changed) if options else None
    if param.kind == "combo":
        options = options_for(param, dataset)
        if not options:
            return None
        widget = QComboBox()
        theme.bind_combo_popup(widget)
        for value, text in options:
            widget.addItem(str(text), value)
        widget.currentIndexChanged.connect(lambda _=0: changed())
        return widget
    if param.kind == "segmented":
        return Segmented(param.options, changed)
    if param.kind == "spin":
        widget = QSpinBox()
        widget.setRange(int(param.minimum), int(param.maximum))
        widget.setValue(int(param.default if param.default is not None else param.minimum))
        return widget
    if param.kind == "dspin":
        # 窗长可以是亚秒级小数，整数 spinbox 表达不了
        widget = QDoubleSpinBox()
        widget.setDecimals(2)
        widget.setRange(float(param.minimum), float(param.maximum))
        widget.setValue(float(param.default if param.default is not None
                             else param.minimum))
        return widget
    if param.kind == "check":
        widget = QCheckBox()
        widget.setChecked(bool(param.default))
        return widget
    if param.kind == "text":
        widget = QLineEdit(str(param.default or ""))
        widget.setPlaceholderText("例如 0,600,1200,1880")
        return widget
    return None


def make_param_row(param, widget):
    """一行参数：标题 + 控件，返回承载两者的 host。

    下拉与文本框的横向策略走 ``shrink_to_fit``：它们的 sizeHint 由最长条目决定，而
    Preferred 策略会让布局把 sizeHint 当成最小宽度 —— 于是「自动（有 outcome 用
    outcome…）」这类长条目能把整行顶出左栏。截断只发生在收起态，弹层与 tooltip
    仍给全文，标题列也不会被挤。
    """
    host = QWidget()
    lay = QHBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(8)
    label = QLabel(param.label)
    label.setProperty("hint", True)
    label.setMinimumWidth(PARAM_LABEL_W)
    if param.tooltip:
        label.setToolTip(param.tooltip)
        widget.setToolTip(param.tooltip)
    if isinstance(widget, (QComboBox, QLineEdit)):
        shrink_to_fit(widget)
    lay.addWidget(label)
    lay.addWidget(widget, 1)
    return host


def build_param_rows(spec, grid, dataset=None, on_change=None, extra=(), tail=()):
    """按方法声明重建参数行，返回 ``(param, host, widget)`` 三元组列表。

    直接塞进宿主的 ``QGridLayout``：grid 归宿主所有，本函数只管往里排。
    ``extra`` 是宿主额外要的行（如分析对象选择），排在方法自带参数之前 ——
    「先选算谁、再选怎么算」的阅读顺序不能倒。
    ``tail`` 则是排在最后的通用行（图形呈现类参数：图例位置、图形尺寸）：
    它们不影响算数，只看图，摆在方法参数之后才不会被当成口径参数。
    """
    rows = []
    for param in tuple(extra) + tuple(spec.params) + tuple(tail):
        widget = make_param_widget(param, dataset, on_change)
        if widget is None:
            continue
        host = make_param_row(param, widget)
        grid.addWidget(host, len(rows), 0, 1, 2)
        rows.append((param, host, widget))
    return rows


def clear_grid(grid):
    """清空参数栅格：控件归宿主 deleteLater，不留半个残行。"""
    while grid.count():
        item = grid.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def entry_for(rows, key, visible_only=True):
    """按参数名找行；``visible_only=False`` 时也回被显隐规则藏起来的行。"""
    for entry in rows:
        if entry[0].key == key and entry[1].isVisible() == visible_only:
            return entry
    if visible_only:
        return None
    for entry in rows:
        if entry[0].key == key:
            return entry
    return None


def read_widget(widget):
    if isinstance(widget, CheckList):
        return widget.values()
    if isinstance(widget, QComboBox):
        return widget.currentData()
    if isinstance(widget, Segmented):
        return widget.value()
    if isinstance(widget, QSpinBox):
        return widget.value()
    if isinstance(widget, QDoubleSpinBox):
        return float(widget.value())
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QLineEdit):
        return widget.text().strip()
    return None


def collect_params(rows):
    """只收可见行的值：隐藏行是“当前口径下不存在”的参数，不得混进计算。"""
    params = {}
    for param, host, widget in rows:
        if not host.isVisible():
            continue
        value = read_widget(widget)
        if value is None or value == "":
            continue
        params[param.key] = value
    return params


def apply_params(rows, params):
    """回填一份参数快照（切回方法时恢复上次的选择）。"""
    for param, _host, widget in rows:
        if param.key not in params:
            continue
        value = params[param.key]
        if isinstance(widget, CheckList):
            widget.set_values(value)
        elif isinstance(widget, QComboBox):
            index = widget.findData(value)
            if index >= 0:
                widget.setCurrentIndex(index)
        elif isinstance(widget, Segmented):
            widget.setValue(value)
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))


def apply_visibility(rows):
    """``shows_for`` 条件显隐：控制器当前值等于指定值时才亮出这一行。"""
    for param, host, _widget in rows:
        visible = True
        if param.shows_for:
            controller = entry_for(rows, param.shows_for[0], visible_only=False)
            value = None if controller is None else read_widget(controller[2])
            visible = (str(value) == str(param.shows_for[1]))
        host.setVisible(visible)


def suppress_layout_warnings(func, *args, **kwargs):
    """测量遍里 tight_layout 的抱怨没有参考价值（画布尺寸是故意给错的）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return func(*args, **kwargs)


# ------------------------------------------------------------------ 图形全屏查看

FULLSCREEN_DPI = 200       # 全屏位图的采样密度（预览画布只按视口出图，直接放大会糊）
FULLSCREEN_BTN = 30        # 浮动按钮边长（btn="round" 的圆角正好是它的一半）
FULLSCREEN_MARGIN = 12     # 按钮与预览区右下角的内缩
#: 单张位图的像素预算（≈ 96 MB RGBA）。IND/GRP 的图最大可到 34×26 in，按 200 dpi
#: 采样就是 35 M 像素，全屏一次开几张就把内存吃干（`_smoke_modes` 里真踩过
#: RendererAgg 的 bad allocation）。超预算就按比例降 dpi —— 宁可少点细节也不炸。
FULLSCREEN_MAX_PIXELS = 24_000_000
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.2, 6.0, 1.25


def figure_pixmap(figure, dpi=FULLSCREEN_DPI):
    """figure → 高清位图：savefig 到内存再读回，不动画布也不落盘。

    `dpi` 只是「想要的采样密度」：图太大时按 ``FULLSCREEN_MAX_PIXELS`` 降一档。
    """
    size = figure.get_size_inches()
    pixels = max(float(size[0]), 0.1) * max(float(size[1]), 0.1) * dpi * dpi
    if pixels > FULLSCREEN_MAX_PIXELS:
        dpi = max(dpi * (FULLSCREEN_MAX_PIXELS / pixels) ** 0.5, 40.0)
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=dpi,
                       facecolor=theme.tokens()["card"])
    except Exception:                       # noqa: BLE001 看图失败不该弹错误框
        return QPixmap()
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue(), "PNG")
    # 一律按 1.0 算尺：窗口里还要按倍率缩放，带上 DPR 会让“实际大小”名不副实
    pixmap.setDevicePixelRatio(1.0)
    return pixmap


class FigureViewDialog(QDialog):
    """全屏查看：把当前预览图搬到独立窗口，Ctrl+滚轮缩放、Esc 关闭。

    预览区就那么点大，密一点的折线与长标签在里面永远看不清；这里按更高 dpi
    重画成位图再放大，不改预览区的任何状态。多张图照旧竖排，与预览区同序。
    """

    def __init__(self, pixmaps, title="", parent=None):
        super().__init__(parent)
        self._pixmaps = [pm for pm in (pixmaps or []) if pm and not pm.isNull()]
        self._ratio = 1.0
        self._fit = True
        self.setWindowTitle("图形预览 - 全屏" + ("：" + title if title else ""))
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.resize(1120, 780)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        self.zoom_out_btn = self._tool_button("minus", "缩小（-）", self._zoom_out)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setProperty("hint", True)
        self.zoom_in_btn = self._tool_button("plus", "放大（+）", self._zoom_in)
        self.fit_btn = self._text_button("适应宽度", "按窗口宽度缩放（0）", lambda: self._set_fit(True))
        self.actual_btn = self._text_button("1:1", "实际像素大小（1）", lambda: self._set_fit(False, 1.0))
        bar.addWidget(self.zoom_out_btn)
        bar.addWidget(self.zoom_label)
        bar.addWidget(self.zoom_in_btn)
        bar.addStretch(1)
        bar.addWidget(self.fit_btn)
        bar.addWidget(self.actual_btn)
        root.addLayout(bar)

        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QScrollArea.NoFrame)
        self.scroll.setWidgetResizable(True)
        host = QWidget()
        box = QVBoxLayout(host)
        box.setContentsMargins(0, 0, 8, 0)
        box.setSpacing(12)
        box.setAlignment(Qt.AlignHCenter)
        self._labels = []
        for pixmap in self._pixmaps:
            label = QLabel()
            label.setAlignment(Qt.AlignHCenter)
            box.addWidget(label, 0, Qt.AlignHCenter)
            self._labels.append(label)
        if not self._pixmaps:
            box.addWidget(QLabel("当前没有可显示的图形。"))
        box.addStretch(1)
        self.scroll.setWidget(host)
        root.addWidget(self.scroll, 1)

        self.scroll.viewport().installEventFilter(self)
        self._apply()

    @staticmethod
    def _tool_button(icon_name, tip, handler):
        btn = QPushButton()
        btn.setProperty("btn", "roundSm")
        btn.setFixedSize(theme.TAB_ADD_SIZE, theme.TAB_ADD_SIZE)
        btn.setToolTip(tip)
        theme.bind_icon(btn, icon_name, 14)
        btn.clicked.connect(lambda _=False: handler())
        return btn

    @staticmethod
    def _text_button(text, tip, handler):
        """“适应宽度 / 1:1”这类没有公认图标的动作宁可写字：图标词不达意比占地方更糟。"""
        btn = QPushButton(text)
        btn.setProperty("btn", "ghost")
        btn.setFixedHeight(theme.TAB_ADD_SIZE)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(tip)
        btn.clicked.connect(lambda _=False: handler())
        return btn

    # ---------- 缩放 ----------
    def _base_width(self):
        return max((pm.width() for pm in self._pixmaps), default=0)

    def _set_fit(self, fit, ratio=None):
        self._fit = fit
        if not fit and ratio is not None:
            self._ratio = ratio
        self._apply()

    def _step(self, direction):
        base = self._ratio if not self._fit else self._fit_ratio()
        if not base:
            return
        value = base * (ZOOM_STEP if direction > 0 else 1.0 / ZOOM_STEP)
        self._fit = False
        self._ratio = min(max(value, ZOOM_MIN), ZOOM_MAX)
        self._apply()

    def _zoom_in(self):
        self._step(1)

    def _zoom_out(self):
        self._step(-1)

    def _fit_ratio(self):
        """适应窗口：按最宽的一张图折算，窄图跟着同一倍率走（不致于忽大忽小）。"""
        base = self._base_width()
        if not base:
            return 1.0
        avail = max(self.scroll.viewport().width() - 24, 120)
        return min(max(avail / float(base), ZOOM_MIN), ZOOM_MAX)

    def _apply(self):
        ratio = self._fit_ratio() if self._fit else self._ratio
        for label, pixmap in zip(self._labels, self._pixmaps):
            width = max(int(round(pixmap.width() * ratio)), 1)
            label.setPixmap(pixmap.scaledToWidth(width, Qt.SmoothTransformation))
        self.zoom_label.setText("%d%%" % round(ratio * 100))

    # ---------- 交互 ----------
    def eventFilter(self, obj, event):
        if obj is self.scroll.viewport():
            if event.type() == QEvent.Wheel:
                # 只抢 Ctrl/Cmd + 滚轮：多张图竖排时普通滚轮得先拿去滚动看图，
                # 这是桌面看图器的通行口径（浏览器 / VS Code 预览皆如此）。
                if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
                    steps = event.angleDelta().y() // 120
                    if steps:
                        self._step(1 if steps > 0 else -1)
                    return True
            elif event.type() == QEvent.Resize and self._fit:
                self._apply()
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom_in()
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self._zoom_out()
        elif key == Qt.Key_0:
            self._set_fit(True)
        elif key == Qt.Key_1:
            self._set_fit(False, 1.0)
        else:
            super().keyPressEvent(event)      # Esc → reject → 关闭


_open_viewers = []      # 关掉即从表中移除：不留悬空引用，也不被提前 GC


def _forget_viewer(dialog):
    try:
        _open_viewers.remove(dialog)
    except ValueError:
        pass


def open_figure_view(pixmaps, title="", parent=None):
    dialog = FigureViewDialog(pixmaps, title, parent)
    _open_viewers.append(dialog)
    dialog.finished.connect(lambda _=0: _forget_viewer(dialog))
    dialog.showMaximized()
    return dialog


class _CornerAnchor(QObject):
    """把浮动按钮钉在宿主（滚动区视口）右下角：视口一改就跟着回位。"""

    def __init__(self, button, host):
        super().__init__(host)
        self._button = button
        self._host = host
        host.installEventFilter(self)
        self.place()

    def place(self):
        btn, host = self._button, self._host
        btn.move(max(host.width() - btn.width() - FULLSCREEN_MARGIN, 0),
                 max(host.height() - btn.height() - FULLSCREEN_MARGIN, 0))
        btn.raise_()

    def eventFilter(self, obj, event):
        if obj is self._host and event.type() in (QEvent.Resize, QEvent.Show):
            self.place()
        return False


def attach_fullscreen_button(scroll_area, pixmaps_provider, title=""):
    """在预览区视口右下角挂一个「全屏查看」按钮，返回按钮供宿主显隐。"""
    button = QPushButton(scroll_area.viewport())
    button.setProperty("btn", "round")
    button.setFixedSize(FULLSCREEN_BTN, FULLSCREEN_BTN)
    button.setToolTip("在新窗口中全屏查看图形（Ctrl+滚轮缩放，Esc 关闭）")
    button.setCursor(Qt.PointingHandCursor)
    theme.bind_icon(button, "maximize", 16)
    button.clicked.connect(
        lambda _=False: open_figure_view(pixmaps_provider(), title, button.window()))
    button._corner_anchor = _CornerAnchor(button, scroll_area.viewport())
    return button
