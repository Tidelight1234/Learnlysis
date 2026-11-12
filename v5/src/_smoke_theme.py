"""离屏冒烟测试：构造 MainWindow，来回切换深/浅色，验证图标、阴影、图表随主题重绘。"""
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from matplotlib import colors as mcolors
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QLabel, QWidget

import theme
import statistic_visualization_methods as _sv_methods

settings = theme.theme.settings_path()
backup = None
if os.path.exists(settings):
    with open(settings, encoding="utf-8") as f:
        backup = f.read()

failures = []


def check(label, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + label + ("  " + str(extra) if extra else ""))
    if not cond:
        failures.append(label)


app = QApplication(sys.argv)
family = theme.load_fonts()
theme.apply_theme(app, "light")
print("FONT_FAMILY =", family)

from main_window import MainWindow

w = MainWindow()
w.show()
app.processEvents()

check("outer tabs == 5", w.tab_widget.count() == 5, w.tab_widget.count())
check("inner tabs == 2", w.tabmodule_widget.count() == 2, w.tabmodule_widget.count())
# 编码框架已抽成通用面板：只剩编码页维护框架（IND/GRP 的框架由数据契约解析）
from data_encoder import DataEncoderModule
from dimension_panel import DimensionCard, DimensionPanel

_framework_hosts = [widget for _label, widget in w.framework_hosts()]
check("框架宿主只剩编码页",
      _framework_hosts and all(isinstance(m, DataEncoderModule) for m in _framework_hosts),
      [type(m).__name__ for m in _framework_hosts])
check("新工作台不再自带维度面板",
      not any(hasattr(m, "dim_panel") for m in (w.individual_analysis, w.group_analysis)))
check("dimension panel shared by encoder pages",
      all(isinstance(m.dim_panel, DimensionPanel) for m in _framework_hosts))
check("panel buttons shared",
      all(m.dim_panel.import_btn.text() == "导入框架"
          and m.dim_panel.add_dimension_btn.text() == "添加维度"
          and m.dim_panel.clear_btn.text() == "清空" for m in _framework_hosts))
# 有了通用面板的「导入框架」，各分析模块自己的“传递编码”按钮就该退场
check("传递编码 buttons removed",
      not any(hasattr(m, "copy_encoding_btn") for m in _framework_hosts))
check("两个工作台各自锁定模式",
      w.individual_analysis.findChild(type(w.individual_analysis.method_combo)) is not None
      and w.individual_analysis.method_combo is not w.group_analysis.method_combo)

qss_light = app.styleSheet()
icons_light = len(theme._icon_bindings)
shadows_light = len(theme._shadow_bindings)
check("icon bindings registered", icons_light > 20, icons_light)
check("shadow bindings registered", shadows_light > 4, shadows_light)

# 字号阶梯：顶栏菜单/页签 > 区块标题 > 正文 > 辅助文字，且均已比旧档位上调一档
_fs = theme.FONT_SIZES
check("font ladder keeps hierarchy",
      _fs["fs_nav"] >= _fs["fs_section"] > _fs["fs_base"] > _fs["fs_small"], _fs)
check("font ladder raised from previous scale",
      (_fs["fs_nav"], _fs["fs_section"], _fs["fs_base"], _fs["fs_small"])
      >= (16, 15, 14, 13), _fs)
for _key in _fs:
    check("QSS resolved %s" % _key, "{%s}" % _key not in qss_light)
check("tab bar uses nav size",
      "font-size: %dpx" % _fs["fs_nav"] in qss_light)

# 字重阶梯：MiSans 没有 Bold 字面，QSS 写 600 会被 Qt 换成“算法描粗”的假 Bold（小字号
# 下笔画糊成一团），且 500/600 会落在同一个 Semibold 上、层级消失。三档必须各自命中
# 一个真字面，故用墨量（中文笔画占像素）而不是字面名字来做断言。
from PySide6.QtCore import Qt as _Qt
from PySide6.QtGui import QFont, QImage, QPainter, QPen

_PROBE_TEXT = "协作学习编码分析工具"


def ink_of(css_weight):
    """按 QSS 数值渲染一行中文，返回墨量（1 - 平均亮度），越大越粗。"""
    img = QImage(520, 64, QImage.Format_RGB32)
    img.fill(_Qt.white)
    p = QPainter(img)
    p.setFont(QFont(family, 14, int(css_weight) // 8))   # QSS→QFont：约 /8，600 即 Bold
    p.setPen(QPen(_Qt.black))
    p.drawText(img.rect(), _Qt.AlignCenter, _PROBE_TEXT)
    p.end()
    a = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes()).reshape(
        img.height(), img.bytesPerLine())
    bgr = a[:, :img.width() * 4].reshape(img.height(), img.width(), 4)
    lum = bgr[:, :, 2] * 0.299 + bgr[:, :, 1] * 0.587 + bgr[:, :, 0] * 0.114
    return round(float(255.0 - lum.mean()) / 255.0, 5)


_W = theme.FONT_WEIGHTS
check("weight ladder resolved in QSS",
      all("{%s}" % k not in qss_light for k in _W), _W)
check("no synthetic-bold weight left in QSS",
      not any(("font-weight: %d" % n) in qss_light for n in (600, 700, 800, 900)))
_ink_r, _ink_m, _ink_s = (ink_of(_W[k]) for k in ("fw_regular", "fw_medium", "fw_semibold"))
check("three weights hit three distinct faces",
      _ink_m > _ink_r * 1.05 and _ink_s > _ink_m * 1.05 and _ink_s > _ink_r * 1.25,
      (_ink_r, _ink_m, _ink_s))
_ink_bold = ink_of(600)          # 旧写法：Qt 合成加粗
check("heaviest shipped weight is real, not synthetic",
      _ink_s < _ink_bold * 0.85, (_ink_s, _ink_bold))
check("app font weight == QSS regular weight after conversion",
      theme.APP_FONT_WEIGHT == _W["fw_regular"] // 8 and ink_of(_W["fw_regular"]) == _ink_r,
      (theme.APP_FONT_WEIGHT, _W["fw_regular"]))

# 含视频窗体的页面必须跳过淡入：QGraphicsOpacityEffect 会把整页画进位图，
# 原生 HWND 进不去，实测会让编码区空白两三百毫秒（看起像视频区在缩放）。
_enc_page = w.tab_widget.widget(0)
check("encoder page detected as holding a video surface",
      theme.has_video_surface(_enc_page))
check("other pages still eligible for fade",
      all(not theme.has_video_surface(w.tab_widget.widget(i))
          for i in range(1, w.tab_widget.count())))
theme.fade_in(_enc_page)
app.processEvents()
check("encoder page gets no opacity effect", _enc_page.graphicsEffect() is None,
      type(_enc_page.graphicsEffect()).__name__)

ind = w.individual_analysis
grp = w.group_analysis
sa = w.statistic_visualization


def hexof(rgba):
    return "#%02X%02X%02X" % tuple(round(c * 255) for c in rgba[:3])


def figure_face(fig):
    return hexof(fig.get_facecolor())


# 未做任何分析时切深色：空画布缓冲必须跟随主题，否则深色下露出白底
w.dark_action.setChecked(True)
app.processEvents()
check("empty individual canvas follows dark",
      figure_face(ind.figure) == theme.TOKENS["dark"]["card"].upper(), figure_face(ind.figure))
check("empty group canvas follows dark",
      figure_face(grp.figure) == theme.TOKENS["dark"]["card"].upper(), figure_face(grp.figure))
check("empty sequence canvas follows dark",
      figure_face(sa.figure) == theme.TOKENS["dark"]["card"].upper(), figure_face(sa.figure))
w.dark_action.setChecked(False)
app.processEvents()
check("empty canvases back to light",
      figure_face(ind.figure) == theme.TOKENS["light"]["card"].upper()
      and figure_face(grp.figure) == theme.TOKENS["light"]["card"].upper()
      and figure_face(sa.figure) == theme.TOKENS["light"]["card"].upper())


def alive(refs):
    return sum(1 for r in refs if r() is not None)


def sample_icon_color():
    img = None
    for ref, (_name, size, _color) in theme._icon_bindings:
        widget = ref()
        if widget is None:
            continue
        pm = widget.pixmap() if isinstance(widget, QLabel) else widget.icon().pixmap(size, size)
        if pm is not None and not pm.isNull():
            img = pm.toImage()
            break
    if img is None:
        return None
    best, best_a = None, 0
    for y in range(0, img.height(), 2):
        for x in range(0, img.width(), 2):
            c = img.pixelColor(x, y)
            if c.alpha() > best_a:
                best_a, best = c.alpha(), (c.red(), c.green(), c.blue())
    return best


color_light = sample_icon_color()
check("a bound icon has pixels", color_light is not None, color_light)

# 真数据上工作台：IND / GRP 各出一个真图，后面的主题往返全部看这两张
import glob as _glob

_TESTDATA = sorted(_glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "_testdata", "*.xlsx")))
check("合成数据就绪", bool(_TESTDATA), _TESTDATA[:1])
if _TESTDATA:
    check("individual 装载批量数据", ind.load_files(_TESTDATA) is not None)
    check("group 装载批量数据", grp.load_files(_TESTDATA) is not None)
    ind.set_method("IND01")
    ind.generate()
    grp.set_method("GRP01")
    grp.generate()
    check("individual 出图", bool(ind.figures) and bool(ind.current_figure().axes),
          list(ind.figures))
    check("group 出图", bool(grp.figures) and bool(grp.current_figure().axes),
          list(grp.figures))
    check("workbench figure uses light card bg",
          figure_face(ind.current_figure()) == theme.TOKENS["light"]["card"].upper()
          and figure_face(grp.current_figure()) == theme.TOKENS["light"]["card"].upper())
    _palette_used = {hexof(p.get_facecolor()) for _ax in ind.current_figure().axes
                     for p in _ax.patches if p.get_facecolor()[3] > 0}
    check("workbench colors come from CATEGORY_COLORS",
          not _palette_used or _palette_used <= ({c.upper() for c in theme.CATEGORY_COLORS}
                                                 | {theme.TOKENS["light"]["card"].upper(),
                                                    theme.TOKENS["light"]["canvas"].upper()}),
          sorted(_palette_used))

# 序列图：合成符合列约定的数据，_2 维度只有一种编码（覆盖原除零静默失败路径）
import pandas as pd
import tempfile

xlsx = os.path.join(tempfile.gettempdir(), "_smoke_seq.xlsx")
pd.DataFrame({
    "time": [0, 10, 20, 30, 40],
    "S1Encoding_1": ["提问", "讲解", None, "讨论", "提问"],
    "S2Encoding_1": ["讲解", "提问", "讨论", None, "讲解"],
    "S1Encoding_2": ["高", "高", "高", "高", "高"],
    "S2Encoding_2": ["高", "高", "高", "高", "高"],
}).to_excel(xlsx, index=False)

sa.file_combo.clear()
sa.file_combo.addItem(xlsx)
sa.load_file(xlsx)
check("方法下拉列出注册表全量方法",
      len([i for i in range(sa.method_combo.count())
           if sa.method_combo.itemData(i)]) == len(_sv_methods.METHODS),
      sa.method_combo.count())
sa.set_method("M07")
seq_axes = list(sa.figure.axes)
check("M07 按学生 × 维度分面", len(seq_axes) == 4, len(seq_axes))
# M07 的组标题用 loc="left"，get_title() 默认只读 center，三个位置都要扫
_titles = [ax.get_title(loc) for ax in seq_axes
           for loc in ("center", "left", "right") if ax.get_title(loc)]
check("M07 组标题写学生不写 channel",
      any("学生 S1" in text and "channel" not in text for text in _titles), _titles)

# 缺失时间点用独立斜纹块（line_strong），不属于编码色板，断言时先分流
hatched = [p for ax in seq_axes for p in ax.patches if p.get_hatch()]
spans = [p.get_facecolor() for ax in seq_axes for p in ax.patches
         if not p.get_hatch()]
check("sequence chart has colored spans", len(spans) > 4, len(spans))
check("缺失时间点画成斜纹块", len(hatched) >= 2, len(hatched))

palette = {c.upper() for c in theme.CATEGORY_COLORS}
allowed = palette | {hexof(mcolors.to_rgba(theme.tokens()["card"]))}
used = {hexof(c) for c in spans}
check("sequence colors come from CATEGORY_COLORS", used <= allowed, sorted(used))
sa.redraw_chart()
check("sequence redraw_chart ran (with file)", len(sa.figure.axes) == 4,
      len(sa.figure.axes))

# 图表文字跟着 UI 字号阶梯一起上调（matplotlib 默认 10pt → 主题里定为 11pt）
from matplotlib import pyplot as _plt
_tick_sizes = {round(_t.get_fontsize(), 1) for _ax in seq_axes
               for _t in _ax.get_xticklabels() + _ax.get_yticklabels()}
check("chart text follows theme font.size bump",
      bool(_tick_sizes) and min(_tick_sizes) >= _plt.rcParams["font.size"] > 10.0,
      (sorted(_tick_sizes), _plt.rcParams["font.size"]))

w.dark_action.setChecked(True)
# 视频占位背景靠 palette 直写（不走样式表的延迟 polish）：切换瞬间就必须是新色
check("video placeholder follows dark token immediately",
      w.video_player.video_widget.palette().color(QPalette.Window).name().lower()
      == theme.TOKENS["dark"]["video_bg"].lower(),
      w.video_player.video_widget.palette().color(QPalette.Window).name())
check("video placeholder no longer styled by QSS",
      "QVideoWidget {" not in app.styleSheet())
# QVideoWidget 是原生 HWND，Qt 的 repaint() 不会失效客户区（实测无效），
# 必须走 Win32 InvalidateRect；防止后续改动又退回 repaint。
import inspect as _inspect
import video_player as _vp
check("theme switch invalidates native video window",
      "_refresh_native_bg" in _inspect.getsource(_vp.VideoPlayerModule._on_theme_changed)
      and "repaint()" not in _inspect.getsource(_vp.VideoPlayerModule._on_theme_changed))
app.processEvents()
check("mode == dark after toggle", theme.get_mode() == "dark", theme.get_mode())
qss_dark = app.styleSheet()
check("QSS changed with theme", qss_dark != qss_light, (len(qss_light), len(qss_dark)))
check("dark QSS uses dark canvas", theme.TOKENS["dark"]["canvas"] in qss_dark)
color_dark = sample_icon_color()
check("icon repainted in dark", color_dark is not None and color_dark != color_light,
      (color_light, color_dark))
check("no icon binding died", len(theme._icon_bindings) == icons_light,
      (icons_light, len(theme._icon_bindings)))
check("shadow bindings rebuilt", len(theme._shadow_bindings) >= shadows_light,
      (shadows_light, len(theme._shadow_bindings)))
check("all shadow refs alive", alive(theme._shadow_bindings) == len(theme._shadow_bindings))
check("individual figure repainted dark",
      not _TESTDATA or figure_face(ind.current_figure()) == theme.TOKENS["dark"]["card"].upper(),
      figure_face(ind.current_figure()))
check("group figure repainted dark",
      not _TESTDATA or figure_face(grp.current_figure()) == theme.TOKENS["dark"]["card"].upper(),
      figure_face(grp.current_figure()))
check("sequence figure repainted dark",
      figure_face(sa.figure) == theme.TOKENS["dark"]["card"].upper(), figure_face(sa.figure))
with open(settings, encoding="utf-8") as f:
    check("ui_settings.json persisted dark", json.load(f).get("theme") == "dark")

grab_dark = w.tab_widget.grab().toImage()
check("window renders in dark", not grab_dark.isNull(), grab_dark.size())

w.dark_action.setChecked(False)
check("video placeholder follows light token",
      w.video_player.video_widget.palette().color(QPalette.Window).name().lower()
      == theme.TOKENS["light"]["video_bg"].lower())
app.processEvents()
check("mode back to light", theme.get_mode() == "light", theme.get_mode())
check("icon repainted in light", sample_icon_color() == color_light,
      (sample_icon_color(), color_light))
check("individual figure back to light",
      not _TESTDATA or figure_face(ind.current_figure()) == theme.TOKENS["light"]["card"].upper(),
      figure_face(ind.current_figure()))
check("group figure back to light",
      not _TESTDATA or figure_face(grp.current_figure()) == theme.TOKENS["light"]["card"].upper(),
      figure_face(grp.current_figure()))

# 持久化：新起一次 load_mode 应读回 light
theme.load_mode()
check("load_mode reads persisted light", theme.get_mode() == "light", theme.get_mode())

from matplotlib import font_manager, pyplot as plt
resolved = font_manager.findfont(plt.rcParams["font.sans-serif"][0])
check("matplotlib resolves MiSans (not DejaVu)", "MiSans" in os.path.basename(resolved),
      os.path.basename(resolved))

# M4 微动效：切换标签应触发 fade_in，动画结束后 effect 必须清空，否则控件一直半透明
import time

for i in range(w.tab_widget.count()):
    w.tab_widget.setCurrentIndex(i)
    app.processEvents()
faded = w.tab_widget.currentWidget()
check("fade animation attached on tab switch",
      faded.property("_fade_anim") is not None)
check("tab switched", w.tab_widget.currentIndex() == 4, w.tab_widget.currentIndex())

from PySide6.QtCore import QAbstractAnimation

STOPPED = QAbstractAnimation.Stopped
deadline = time.time() + 2
while time.time() < deadline:
    app.processEvents()
    a = faded.property("_fade_anim")
    if a is not None and a.state() == STOPPED:
        break
a = faded.property("_fade_anim")
check("fade animation finished", a is not None and a.state() == STOPPED,
      None if a is None else a.state())
check("opacity effect cleared after fade", faded.graphicsEffect() is None,
      type(faded.graphicsEffect()).__name__)

for i in range(w.tabmodule_widget.count()):
    w.tabmodule_widget.setCurrentIndex(i)
    app.processEvents()
check("inner tabs switchable", w.tabmodule_widget.currentIndex() == 1)

# ---------- 往未显示的模块复制框架：每行编码数量仍要自适应 ----------
# 隐藏期间建卡拿到的宽度是 0，只能先占位排一次；切过来后必须按真实宽度重排，
# 否则就是老毛病——复制过来的框架永远每行两个。
w.tab_widget.setCurrentIndex(0)
_inner_before = w.tabmodule_widget.count()
_deadline = time.time() + 0.4
while time.time() < _deadline:
    app.processEvents()
# 框架宿主只剩编码页：临时开一页当「隐藏期建卡」的靶子，用例结束再收回
w.add_new_tab()
app.processEvents()
_copied = w.encoder_pages()[-1][1]
w.tabmodule_widget.setCurrentWidget(w.data_encoder)     # 新页退回隐藏，下面在它上面建卡
app.processEvents()
_copied.dim_panel.apply_framework([(f"维度{i}", [f"c{j}" for j in range(6)]) for i in range(1, 3)])
w.tabmodule_widget.setCurrentWidget(_copied)
_deadline = time.time() + 0.6
while time.time() < _deadline:
    app.processEvents()
_c1 = _copied.dimension_cards[0]
check("copied framework lays out by real width",
      _c1.width() > 0 and _c1._applied_per_row == _c1._codes_per_row() > 0,
      (_c1.width(), _c1._applied_per_row))
check("copied cards keep codes", [c.get_codes() for c in _copied.dimension_cards]
      == [[f"c{j}" for j in range(6)]] * 2)
# 自适应的缓存契约：宽度未定时那次占位排布不能把 2 列写进缓存，
# 否则隐藏期建的卡片显出来后永远停在“每行两个”。
_probe_card = DimensionCard(None, name="探测", selectable=False)
for _i in range(6):
    _probe_card.add_block(f"c{_i}")
# 未挂窗的顶层卡片拿到的是 640 默认宽，先压回 0 才是“隐藏期建卡”的情形
_probe_card.resize(0, 0)
_probe_card.block_container.resize(0, 0)
_probe_card.relayout()
check("unlaid card caches no provisional column count",
      _probe_card._codes_per_row() == 0 and _probe_card._applied_per_row is None,
      (_probe_card.width(), _probe_card._applied_per_row))
_probe_card.resize(640, 200)
_probe_card.block_container.resize(640, 180)
_probe_card.update_block_layout()
check("wider card spreads codes over more columns",
      _probe_card._applied_per_row == 640 // 150, _probe_card._applied_per_row)
# 主窗口的登记表要能把框架从一个编码页复制到另一个；IND/GRP 已退出宿主名单
w.data_encoder.dim_panel.apply_framework([("互动类型", ["提问", "回应"])])
_hosts = w.framework_hosts()
_copied_label = w._framework_label(_copied)
_src_labels = [label for label, _ in w.framework_sources(_copied)]
check("framework registry offers other encoder pages as sources",
      bool(_src_labels) and w._framework_label(w.data_encoder) in _src_labels, _src_labels)
check("registry excludes the target itself",
      _copied_label not in _src_labels, (_copied_label, _src_labels))
check("两个工作台都不在框架宿主名单里",
      all(widget is not m for _l, widget in _hosts for m in (ind, grp))
      and not any(hasattr(m, "dim_panel") for m in (ind, grp)))
check("each host panel asks its own sources",
      bool(_hosts) and all(widget.dim_panel.sources_provider is not None
                           for _l, widget in _hosts))
# 编码页的「清空」连框架带数据一起清，两者有其一就能点
check("编码页清空按框架+数据置灰",
      _copied.dim_panel.clear_btn.isEnabled()
      == (bool(_copied.selected_data) or bool(_copied.dimension_cards)))
_copied.clear_btn.click()
check("清空 drops all dimensions", len(_copied.dimension_cards) == 0)
check("清空后按钮回到置灰", not _copied.clear_btn.isEnabled())
# 「添加维度」按钮现在会弹框问维度名，无头跑测不能等模态框，直接走建卡接口
_copied.dim_panel.add_dimension()
check("加一个维度后清空重新可用", _copied.clear_btn.isEnabled())
_copied.dim_panel.clear_dimensions()
# 收回临时页：后面的用例仍按「编码页 + 汇总页」两页寻址
_index = w.tabmodule_widget.indexOf(_copied)
if _index >= 0:
    w.tabmodule_widget.removeTab(_index)
_copied.setParent(None)
_copied.deleteLater()
w._ensure_group_tab()
w._schedule_tab_bar_sync()
w._sync_layout_toggle()
app.processEvents()
check("临时编码页已收回，内层标签数复位",
      w.tabmodule_widget.count() == _inner_before, w.tabmodule_widget.count())
w.tab_widget.setCurrentIndex(0)
app.processEvents()

# ---------- 视频区颜色、卡片投影与编码区布局切换 ----------
check("dark video stage matches the white placeholder",
      theme.TOKENS["dark"]["stage"].lower() == theme.TOKENS["dark"]["video_bg"].lower(),
      (theme.TOKENS["dark"]["stage"], theme.TOKENS["dark"]["video_bg"]))
check("light video stage stays dark",
      theme.TOKENS["light"]["stage"].lower() != theme.TOKENS["light"]["video_bg"].lower())


def _min_lum(widget):
    """控件截图的最小亮度：含墨迹/投影时明显低于白底。"""
    img = widget.grab().toImage().convertToFormat(QImage.Format_RGB32)
    a = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes()).reshape(
        img.height(), img.bytesPerLine())
    bgr = a[:, :img.width() * 4].reshape(img.height(), img.width(), 4)
    lum = bgr[:, :, 2] * 0.299 + bgr[:, :, 1] * 0.587 + bgr[:, :, 0] * 0.114
    return float(lum.min())


_enc = w.data_encoder
check("encoder white areas carry card shadows",
      _enc.dim_panel.panel_card.graphicsEffect() is not None
      and _enc.result_slot.graphicsEffect() is not None
      and bool(_enc.result_widget.property("card")))
def _shadowed_cards(module):
    """工作台卡片阴影：apply_card_shadow 直接挂在卡片自身（这些页没有原生窗体）。"""
    return [c for c in module.findChildren(QWidget)
            if c.property("card") and c.graphicsEffect() is not None]


check("workbench cards carry the same card shadow",
      len(_shadowed_cards(ind)) >= 5 and len(_shadowed_cards(grp)) >= 5,
      (len(_shadowed_cards(ind)), len(_shadowed_cards(grp))))
# 投影挂在卡片外的留白槽上，不挂在卡片上：QGraphicsDropShadowEffect 的输出会被裁进
# 控件自身矩形，挂在铺满整格的卡片上等于没阴影。槽的四边留白才是阴影能落笔的地方。
check("table card itself stays effect-free", _enc.result_widget.graphicsEffect() is None)
from PySide6.QtCore import QPoint as _P


def _divider_gap():
    """卡片到两栏分界线的空白距离（含 1px 分界线）：横排看左、纵排看上。"""
    top = _enc.result_widget.mapTo(_enc.splitter, _P(0, 0))
    first = _enc.splitter.widget(0)
    horiz = _enc.splitter.orientation() == _Qt.Horizontal
    return (top.x() - first.x() - first.width()) if horiz else \
        (top.y() - first.y() - first.height())


_gap_h = _divider_gap()
_enc.set_stacked_layout(True)
app.processEvents()
_gap_v = _divider_gap()
_enc.set_stacked_layout(False)
app.processEvents()
check("table card floats off the divider both ways", _gap_h > 8 and _gap_v > 8,
      (_gap_h, _gap_v))
_cv = theme.tokens()["canvas"].lstrip("#")
_cv_r = int(_cv[0:2], 16)
# 投影得从整窗截图看：只 grab 这个透明槽会把未绘制区域当成黑，量不出真实变暗。
# 上面把标签停在最后一个外层页与汇总页，先切回编码页并跑一会事件循环。
w.tab_widget.setCurrentIndex(0)
w.tabmodule_widget.setCurrentIndex(0)
_deadline = time.time() + 0.4
while time.time() < _deadline:
    app.processEvents()
_wimg = w.grab().toImage()
_org = _enc.result_slot.mapTo(w, _P(0, 0))
_mid = _org.y() + _enc.result_slot.height() // 2
_band = [_wimg.pixelColor(_org.x() + x, _mid).red() for x in (4, 8, 10)]
_inside = _wimg.pixelColor(_org.x() + 20, _mid).red()
check("table card shadow paints into the gutter",
      all(_cv_r - 15 <= v < _cv_r for v in _band) and _inside > 250,
      (_band, _inside, _cv_r))
# 视频卡片含原生窗体，只能手绘投影，绝不能被塞进 QGraphicsEffect
check("video card keeps no graphics effect", w.video_player.panel.graphicsEffect() is None)
_host_lum = _min_lum(w.video_player)
check("video card shadow actually painted", _host_lum < 235, _host_lum)

check("jump hour field present", w.video_player.jump_inpouthour.placeholderText() == "时")
check("jump hour counted into the target", w.video_player.format_time(3723000) == "1:02:03",
      w.video_player.format_time(3723000))
check("short time stays mm:ss", w.video_player.format_time(62000) == "01:02",
      w.video_player.format_time(62000))

# 上面把内层标签停在“数据汇总”，那里布局按钮是置灰的，先切回编码页
w.tabmodule_widget.setCurrentIndex(0)
app.processEvents()
check("layout toggle pinned at the tab bar corner",
      w.tabmodule_widget.cornerWidget() is w.layout_toggle_btn)
check("layout toggle enabled on encoder pages", w.layout_toggle_btn.isEnabled())
_orient0 = _enc.splitter.orientation()
w.layout_toggle_btn.click()
app.processEvents()
check("click stacks the encoder columns",
      _enc.splitter.orientation() != _orient0 and _enc.is_stacked(),
      _enc.splitter.orientation())
w.layout_toggle_btn.click()
app.processEvents()
check("click again restores the side-by-side layout",
      _enc.splitter.orientation() == _orient0 and not _enc.is_stacked(),
      _enc.splitter.orientation())

# 汇总页没有编码区，按钮应该置灰，否则点了没反应反而让人以为卡死
w.tabmodule_widget.setCurrentIndex(1)
app.processEvents()
check("layout toggle greyed out off encoder pages", not w.layout_toggle_btn.isEnabled())
w.tabmodule_widget.setCurrentIndex(0)
app.processEvents()
check("layout toggle re-enabled back on the encoder page",
      w.layout_toggle_btn.isEnabled() and not w.layout_toggle_btn.isChecked())

# 新建编码页沿用当前排列选择：先切到上下，再开一页
w.layout_toggle_btn.click()
app.processEvents()
w.add_new_tab()
app.processEvents()
_new = [w.tabmodule_widget.widget(i) for i in range(w.tabmodule_widget.count())
        if w.tabmodule_widget.widget(i) not in (_enc, w.summary_tab)
        and w.tabmodule_widget.widget(i).__class__.__name__ == "DataEncoderModule"]
check("a second encoder page was created", len(_new) == 1, w.tabmodule_widget.count())
check("new encoder page inherits the stacked choice", bool(_new) and _new[0].is_stacked())
w.tabmodule_widget.setCurrentWidget(_enc)
app.processEvents()
w.layout_toggle_btn.click()
app.processEvents()
check("toggling one page leaves the other's layout alone",
      not _enc.is_stacked() and bool(_new) and _new[0].is_stacked())

# 卡片投影与内层淡入动效共存：动效期间子树不能被画空，结束后 effect 必须释放
theme.fade_in(_enc)
_end = time.time() + 0.06
while time.time() < _end:
    app.processEvents()
_mid = _min_lum(_enc)
check("encoder page still paints during its fade", _mid < 200, _mid)
_deadline = time.time() + 2
while time.time() < _deadline:
    app.processEvents()
    _a = _enc.property("_fade_anim")
    if _a is not None and _a.state() == STOPPED:
        break
check("encoder fade released its effect", _enc.graphicsEffect() is None,
      type(_enc.graphicsEffect()).__name__)
check("encoder page paints normally after fade", _min_lum(_enc) < 120, _min_lum(_enc))

# 不 close()：closeEvent 里的模态 QMessageBox 在 offscreen 平台会段错误（Qt 平台插件限制）

if backup is not None:
    with open(settings, "w", encoding="utf-8") as f:
        f.write(backup)
elif os.path.exists(settings):
    os.remove(settings)

if os.path.exists(xlsx):
    os.remove(xlsx)

print()
if failures:
    print("SMOKE_FAILED:", failures)
    sys.exit(1)
print("SMOKE_OK  icons=%d shadows=%d qss=%d/%d" %
      (icons_light, shadows_light, len(qss_light), len(qss_dark)))
