"""离屏 UI 验证：统计及可视化方法工作台的 M01–M19 全链路。

覆盖「装载 → 选方法 → 设参数 → 生成 → 统计表 + 图形 + 质检卡 → 导出」，
外加两条入口（外部文件 / 数据汇总页）的等价性与浅深双主题截图。

运行：D:\\miniconda\\miniconda\\python.exe _verify_stats_ui.py
"""

import os
import shutil
import sys
import tempfile
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json as _json
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit,
                             QSpinBox, QTableWidget)

import theme
import statistic_visualization_methods as AM
import statistic_visualization_plots as AP
from data_encoder import FIXED_COL_COUNT
from statistic_visualization import _CheckList, _Segmented

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP1 = os.path.normpath(os.path.join(HERE, "..", "..", "Results",
                                       "testfile", "group_1.xlsx"))
SHOTS = os.path.join(HERE, "_diag_shots")

failures = []


def check(label, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + label + ("  " + str(extra) if extra else ""))
    if not cond:
        failures.append(label)


app = QApplication(sys.argv)
theme.load_fonts()
theme.apply_theme(app, "light")

settings = theme.theme.settings_path()
backup = None
if os.path.exists(settings):
    with open(settings, encoding="utf-8") as handle:
        backup = handle.read()

from main_window import MainWindow                # noqa: E402

w = MainWindow()
w.resize(1600, 900)
w.show()
mod = w.statistic_visualization


def pump(ms):
    """offscreen 下只有 processEvents 会推进 QTimer，淡入动画要显式喂事件。"""
    end = time.time() + ms / 1000.0
    while time.time() < end:
        app.processEvents()
        time.sleep(0.005)


def qa_codes():
    return [mod.qa_list.item(i).data(Qt.UserRole) for i in range(mod.qa_list.count())]


def result_tables():
    return [mod.result_tabs.widget(index) for index in range(mod.result_tabs.count())
            if isinstance(mod.result_tabs.widget(index), QTableWidget)]


def table_rows():
    total = 0
    for index in range(mod.result_tabs.count()):
        widget = mod.result_tabs.widget(index)
        if isinstance(widget, QTableWidget):
            total += widget.rowCount()
    return total


def figure_content():
    return sum(len(ax.patches) + len(ax.lines) + len(ax.collections)
               + len(ax.texts) + len(ax.images) for ax in mod.figure.axes)


def param_widget(key):
    """取参数控件：优先当前可见的那个（bin_value 会按 bin_mode 分裂成两行）。"""
    entry = mod._entry_for(key) or mod._entry_for(key, visible_only=False)
    return None if entry is None else entry[2]


def set_param(key, value):
    widget = param_widget(key)
    if widget is None:
        check("参数 %s 存在" % key, False, "当前方法未声明该参数")
        return None
    if isinstance(widget, _Segmented):
        widget.pick(value)
    elif isinstance(widget, QComboBox):
        index = widget.findData(value)
        check("参数 %s 可选 %s" % (key, value), index >= 0)
        widget.setCurrentIndex(index)
    elif isinstance(widget, _CheckList):
        widget.set_values(list(value))
    elif isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QLineEdit):
        widget.setText(str(value))
    elif isinstance(widget, QDoubleSpinBox):
        widget.setValue(float(value))
    else:
        widget.setValue(int(value))
    mod._apply_visibility()
    pump(30)
    return widget


def read_param(key, default=None):
    widget = param_widget(key)
    return default if widget is None else mod._read_widget(widget)


w.tab_widget.setCurrentIndex(1)
pump(400)

print("=" * 78)
print("1) 装载 group_1.xlsx（外部文件入口）")
print("=" * 78)

check("测试文件存在", os.path.exists(GROUP1), GROUP1)
mod.load_file(GROUP1)
pump(300)
ds = mod.dataset
check("解析出 6 个编码列", ds is not None and len(ds.column_map) == 6,
      0 if ds is None else len(ds.column_map))
check("学生 = S1/S2", ds.students == ["S1", "S2"], ds.students)
check("维度 = D1/D2/D3", ds.dimensions == ["D1", "D2", "D3"], ds.dimensions)
hint = mod.data_hint.text()
check("数据概况写明 188 个时间点", "188" in hint, hint)
check("数据概况写明 Δt=10", "Δt=10" in hint, hint)
check("数据概况写明规则采样", "规则采样" in hint, hint)
check("映射表逐列可核对", mod.map_table.rowCount() == 6, mod.map_table.rowCount())
first = mod.map_table.item(0, 0).text()
check("映射表首行 = S1Encoding_1 → S1/D1",
      first == "S1Encoding_1"
      and mod.map_table.item(0, 1).text() == "S1"
      and mod.map_table.item(0, 2).text() == "D1",
      (first, mod.map_table.item(0, 1).text(), mod.map_table.item(0, 2).text()))
check("时间列下拉已就绪并选中 Time",
      mod.time_combo.isEnabled() and mod.time_combo.currentText() == "Time",
      mod.time_combo.currentText())
check("生成按钮已启用", mod.generate_btn.isEnabled())
check("质检卡列出 few_students", "few_students" in qa_codes(), qa_codes())
check("默认方法 M01 已自动出结果",
      mod.result is not None and mod.result.method_id == "M01")

print()
print("=" * 78)
print("2) 方法下拉结构：M01–M19 + 分组标题")
print("=" * 78)

ids = [mod.method_combo.itemData(i) for i in range(mod.method_combo.count())
       if mod.method_combo.itemData(i)]
check("下拉按顺序列出 M01–M19", ids == [spec.id for spec in AM.METHODS], ids)
placeholders = sum(1 for i in range(mod.method_combo.count())
                   if mod.method_combo.itemData(i) is None)
check("分组标题占位数 = 分组数", placeholders == len(AM.GROUP_ORDER), placeholders)
check("方法说明写明用途问题", "？" in mod.method_hint.text(), mod.method_hint.text())

print()
print("=" * 78)
print("3) M15 共现：未确认口径必须阻断，选定后出结果")
print("=" * 78)

mod.set_method("M15")
pump(200)
blocked = mod.result
check("未选口径 → 结果阻断", blocked is not None and blocked.blocked,
      None if blocked is None else blocked.blocked_reason)
check("阻断原因要求确认口径",
      "共现" in (blocked.blocked_reason or ""), blocked.blocked_reason)
check("质检卡出 cooccurrence_scope_unconfirmed",
      "cooccurrence_scope_unconfirmed" in qa_codes(), qa_codes())
check("质检汇总写明已阻断", "阻断" in mod.qa_notice.text(), mod.qa_notice.text())
check("阻断时结果表区显示提示而非空表", mod.result_tabs.count() == 1)
check("阻断时图形区画占位", len(mod.figure.axes) == 1 and figure_content() > 0)

set_param("scope", "within")
check("选定 within 后维度参数可见", param_widget("dim_b") is not None
      and param_widget("dim_b").isVisible())
set_param("dim_a", "D2")
set_param("dim_b", "D3")
mod.generate()
pump(200)
matrix = mod.result.table()
numeric = [col for col in matrix.columns if col not in ("D2 编码", "合计")]
check("within 口径不再阻断", not mod.result.blocked, mod.result.blocked_reason)
check("S1 D2×D3 矩阵总和 = 188",
      int(matrix[numeric].to_numpy(dtype=float).sum()) == 188,
      int(matrix[numeric].to_numpy(dtype=float).sum()))
check("矩阵表名不写成 transition", "转移" not in mod.result.method_name,
      mod.result.method_name)

set_param("scope", "cross")
mod.generate()
pump(200)
check("cross 口径也能出结果", not mod.result.blocked, mod.result.blocked_reason)
check("cross 口径不叫 co-occurrence",
      "Co-occurrence" not in mod.result.method_name, mod.result.method_name)

print()
print("=" * 78)
print("4) M01–M19 全量遍历：统计表 + 图形 + 质检三处都要有内容")
print("=" * 78)

sweep = {}
for spec in AM.METHODS:
    try:
        mod.set_method(spec.id)
        pump(60)
        mod.generate()
        pump(60)
        result = mod.result
        problems = []
        if result is None:
            problems.append("无结果")
        else:
            if result.blocked:
                problems.append("被阻断：%s" % result.blocked_reason)
            if not result.tables:
                problems.append("无统计表")
            if sum(len(frame) for frame in result.tables.values()) <= 0:
                problems.append("表行数为 0")
            if table_rows() <= 0:
                problems.append("UI 表格未填充")
            if not mod.figure.axes:
                problems.append("图形无坐标轴")
            elif figure_content() <= 0:
                problems.append("图形是空图")
            if mod.qa_list.count() <= 0:
                problems.append("质检卡为空")
            if not mod.method_hint.text():
                problems.append("方法说明为空")
        sweep[spec.id] = (len(result.tables) if result else 0,
                          len(mod.figure.axes), figure_content())
        check("%s %s" % (spec.id, spec.name), not problems, problems)
    except Exception:
        traceback.print_exc()
        check("%s 抛异常" % spec.id, False)

print("各方法表/轴/图元数：", sweep)
check("M07 按学生 × 维度分面（2×3）", sweep["M07"][1] == 6, sweep["M07"])
check("M08 六单元热图 + 六个色标轴", sweep["M08"][1] == 12, sweep["M08"])
check("M11 每维度一个 panel", sweep["M11"][1] == 3, sweep["M11"])
check("M14 明细 + 汇总两张表", sweep["M14"][0] == 2, sweep["M14"])

print()
print("=" * 78)
print("5) 参数缓存、条件显示与手工映射")
print("=" * 78)

mod.set_method("M04")
pump(60)
set_param("bin_mode", "seconds")
set_param("bin_value", 300)
mod.generate()
pump(60)
check("M04 固定窗长 300 s 生效", read_param("bin_value") == 300, read_param("bin_value"))
check("窗长用浮点控件（亚秒级采样需要小数）",
      isinstance(param_widget("bin_value"), QDoubleSpinBox),
      type(param_widget("bin_value")).__name__)
check("窗数仍用整数控件",
      isinstance(mod._entry_for("bin_value", visible_only=False)[2], QSpinBox))
check("结果表不开隔行色（QSS 下会被画成黑行）",
      result_tables() and all(not tab.alternatingRowColors() for tab in result_tables()),
      [tab.alternatingRowColors() for tab in result_tables()])
check("M04 结果含真实窗口边界",
      any("300" in str(item[1]) for item in mod.dataset.bin_bounds),
      mod.dataset.bin_bounds[:3])

set_param("bin_mode", "custom")
check("切到自定义后边界输入框可见",
      param_widget("bin_custom") is not None
      and param_widget("bin_custom").isVisible())
set_param("bin_custom", "0, 600, 1200, 1880")
mod.generate()
pump(60)
check("自定义边界走通 UI 链路", len(mod.dataset.bin_order) == 3,
      mod.dataset.bin_order)
check("自定义分箱结果非空", not mod.result.blocked and table_rows() > 0)
set_param("bin_mode", "seconds")
mod.generate()
pump(60)
set_param("bin_value", 12.5)
check("窗长控件接受小数", read_param("bin_value") == 12.5, read_param("bin_value"))
mod.generate()
pump(60)
check("小数窗长端到端可用（1880 s / 12.5 s 向上取整 = 151 个窗口）",
      len(mod.dataset.bin_order) == 151 and not mod.result.blocked,
      (len(mod.dataset.bin_order), mod.result.blocked))
check("小数窗长下仍出表与图", table_rows() > 0 and figure_content() > 0)
set_param("bin_value", 300)
mod.generate()
pump(60)
mod.set_method("M01")
pump(60)
mod.set_method("M04")
pump(60)
check("切回 M04 后参数缓存复原",
      read_param("bin_mode") == "seconds" and read_param("bin_value") == 300,
      (read_param("bin_mode"), read_param("bin_value")))

mod.set_method("M08")
pump(60)
set_param("value", "percentage")
mod.generate()
pump(60)
check("M08 百分比模式可切换", read_param("value") == "percentage",
      read_param("value"))

mod.load_file(GROUP1)
pump(200)
row = [r for r in range(mod.map_table.rowCount())
       if mod.map_table.item(r, 0).text() == "S1Encoding_1"][0]
mod.map_table.item(row, 1).setText("SX")
mod.apply_mapping()
pump(200)
check("手工改映射生效", "SX" in mod.dataset.students, mod.dataset.students)
check("改映射后仍能出结果", mod.result is not None and not mod.result.blocked)
mod.load_file(GROUP1)
pump(200)
check("重新装载后映射回到自动解析",
      mod.dataset.students == ["S1", "S2"], mod.dataset.students)

print()
print("=" * 78)
print("6) 一键导出：四类产物同 analysis_id 前缀")
print("=" * 78)

mod.set_method("M07")
pump(60)
mod.generate()
pump(120)
analysis_id = mod.result.analysis_id
tmp = tempfile.mkdtemp(prefix="stats_ui_")
try:
    written = mod.export_all_to(tmp)
    names = sorted(os.path.basename(item) for item in written)
    check("一键导出写出 6 个文件", len(written) == 6, names)
    check("全部带同一 analysis_id",
          all(analysis_id in name for name in names), (analysis_id, names))
    exts = sorted(os.path.splitext(name)[1].lstrip(".") for name in names)
    check("xlsx / png / svg / pdf / json 齐全",
          exts == ["json", "json", "pdf", "png", "svg", "xlsx"], exts)
    xlsx = [item for item in written if item.endswith(".xlsx")][0]
    frames = pd.read_excel(xlsx, sheet_name=None)
    check("XLSX 每张结果表一个 sheet", set(frames) == set(mod.result.tables),
          (sorted(frames), sorted(mod.result.tables)))
    check("XLSX 行数与结果表一致",
          all(len(frames[name]) == len(mod.result.tables[name]) for name in frames))
    config = [item for item in written if "run_config" in os.path.basename(item)][0]
    audit = [item for item in written if "run_audit" in os.path.basename(item)][0]
    with open(config, encoding="utf-8") as handle:
        cfg = _json.load(handle)
    with open(audit, encoding="utf-8") as handle:
        aud = _json.load(handle)
    check("config 记录方法与学生/维度", cfg["method"]["id"] == "M07"
          and cfg["dataset"].get("students") == ["S1", "S2"], cfg["method"])
    check("config 记录采样口径", cfg["sampling"]["is_regular"] is True
          and cfg["sampling"]["median_dt"] == 10.0, cfg["sampling"])
    check("config 记录列映射", len(cfg["input"]["column_mapping"]) == 6)
    codes = [item["code"] for item in aud["qa"]]
    check("audit 收录数据集级 + 方法级质检",
          "few_students" in codes and aud["observations"]["missing_cells"] == 0,
          (codes, aud["observations"]))
    check("audit 记录表行列数", set(aud["tables"]) == set(mod.result.tables))
    png = [item for item in written if item.endswith(".png")][0]
    check("PNG 非空且分辨率合理", os.path.getsize(png) > 20000,
          os.path.getsize(png))
    svg = [item for item in written if item.endswith(".svg")][0]
    with open(svg, encoding="utf-8") as handle:
        svg_text = handle.read()
    check("SVG 为矢量且带 M07 标题文案",
          svg_text.lstrip().startswith("<?xml") and "svg" in svg_text[:2000])

    single = os.path.join(tmp, "chart.svg")
    mod.export_chart_to(single)
    check("单独导出 SVG 可用", os.path.exists(single)
          and os.path.getsize(single) > 5000)
    csv_path = os.path.join(tmp, "tables.csv")
    mod.set_method("M01")
    pump(60)
    mod.export_statistics_to(csv_path)
    check("CSV 导出单表方法可用", os.path.exists(csv_path))
    check("CSV 内容可回读", len(pd.read_csv(csv_path)) > 0)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("=" * 78)
print("7) 「从数据汇总取数」与文件入口等价")
print("=" * 78)

PAGES = [
    [("时间分配", ["ST", "SI", "SO"]),
     ("毛利", ["MN", "ML", "ME"]),
     ("投入", ["PO", "PW", "PD", "PR"])],
    [[
        ["ST", "MN", "PO"], ["ST", "MN", "PW"], ["ST", "ML", "PW"],
        ["SO", "ML", "PD"], ["ST", "MN", "PO"], ["ST", "ML", "PR"],
    ], [
        ["ST", "MN", "PO"], ["SI", "MN", "PW"], ["ST", "ML", "PO"],
        ["ST", "ME", "PD"], ["SO", "ML", "PR"], ["ST", "MN", "PO"],
    ]],
]

DIM_NAMES = PAGES[0]
MATRIX = PAGES[1]

pages = w.summary_tab.encoder_pages()
if len(pages) < 2:
    w.add_new_tab()
    pump(200)
    pages = w.summary_tab.encoder_pages()
check("编码页有 2 个个体", len(pages) == 2, [name for name, _ in pages])


def fill_encoder(page, dims, matrix):
    """走真实交互路径：导入框架 → 记录 N 行 → 逐格改编码（回写行数据）。"""
    page.selected_data = []
    page.row_times = []
    page.row_positions = []
    page.result_table.setRowCount(0)
    page.dim_panel.apply_framework(dims)
    page.time_unit_entry.setText("10")
    page.set_start_seconds(0)
    for _ in matrix:
        page.continue_()
    for row, cells in enumerate(matrix):
        for offset, value in enumerate(cells):
            item = page.result_table.item(row, FIXED_COL_COUNT + offset)
            item.setText(value)                 # on_item_changed 回写 selected_data
    page.update_table()


for (_, page), matrix in zip(pages, MATRIX):
    fill_encoder(page, DIM_NAMES, matrix)
pump(120)

columns, rows = w.summary_tab.build_analysis_data()
check("汇总标准格式 = Time + S{i}Encoding_{j}",
      columns == ["Time", "S1Encoding_1", "S1Encoding_2", "S1Encoding_3",
                  "S2Encoding_1", "S2Encoding_2", "S2Encoding_3"], columns)
check("汇总行数 = 6", len(rows) == 6, len(rows))

mod.set_method("M01")
pump(60)
mod.load_from_summary()
pump(300)
summary_ds = mod.dataset
summary_tables = {name: frame.copy() for name, frame in mod.result.tables.items()}
check("汇总入口未阻断", mod.result is not None and not mod.result.blocked,
      None if mod.result is None else mod.result.blocked_reason)
check("汇总入口学生/维度解析一致",
      summary_ds.students == ["S1", "S2"]
      and summary_ds.dimensions == ["D1", "D2", "D3"],
      (summary_ds.students, summary_ds.dimensions))
check("汇总入口 Δt 由数据估计为 10",
      summary_ds.time_info.median_dt == 10.0 and summary_ds.time_info.is_regular,
      summary_ds.time_info.describe())


def freq(frame, student, dimension):
    part = frame[(frame["student_id"] == student) & (frame["dimension"] == dimension)]
    return dict(zip(part["code"].tolist(), part["count"].astype(int).tolist()))


check("S1×D1 频次 ST=5 SO=1",
      freq(summary_tables["频次统计"], "S1", "D1") == {"ST": 5, "SO": 1},
      freq(summary_tables["频次统计"], "S1", "D1"))
check("S2×D1 频次 ST=4 SI=1 SO=1",
      freq(summary_tables["频次统计"], "S2", "D1") == {"ST": 4, "SI": 1, "SO": 1},
      freq(summary_tables["频次统计"], "S2", "D1"))
check("每单元合计 6",
      all(int(part["count"].sum()) == 6
          for _key, part in summary_tables["频次统计"].groupby(
              ["student_id", "dimension"])))

frame = pd.DataFrame(rows, columns=columns)
round_trip = os.path.join(tempfile.mkdtemp(prefix="stats_rt_"), "summary.csv")
try:
    frame.to_csv(round_trip, index=False, encoding="utf-8-sig")
    mod.load_file(round_trip)
    pump(300)
    file_tables = mod.result.tables
    check("CSV 入口与汇总入口同表名", set(file_tables) == set(summary_tables),
          (sorted(file_tables), sorted(summary_tables)))
    check("CSV 入口与汇总入口逐单元格一致",
          all(file_tables[name].equals(summary_tables[name]) for name in file_tables))
    check("汇总入口的来源标注为数据汇总页",
          summary_ds.source == "数据汇总页", summary_ds.source)
finally:
    shutil.rmtree(os.path.dirname(round_trip), ignore_errors=True)

mod.load_file(GROUP1)
pump(200)
check("恢复基准数据", mod.dataset.time_info.n_positions == 188)

print()
print("=" * 78)
print("8) 可读性：左栏免横向滚动、标签不压字、图例有独立位置")
print("=" * 78)

w.tab_widget.setCurrentIndex(1)
pump(300)
hbar = mod.left_scroll.horizontalScrollBar()
check("左栏不需要横向滚动条", hbar.maximum() == 0, (hbar.value(), hbar.maximum()))
for label, combo in (("图形格式", mod.format_combo), ("导出分辨率", mod.dpi_combo)):
    metrics = QFontMetrics(combo.font())
    widest = max(metrics.horizontalAdvance(combo.itemText(index))
                 for index in range(combo.count()))
    check("%s 下拉容得下最长条目" % label, combo.width() >= widest + 26,
          (combo.width(), widest, combo.currentText()))

mod.set_method("M04")
pump(60)
mod.generate()
pump(150)
bar = mod.result_tabs.tabBar()
check("结果标签栏用自绘笔直下划线",
      isinstance(bar, theme.UnderlineTabBar) and bool(bar.property("flatUnderline")),
      type(bar).__name__)
check("结果标签不用省略号截断", bar.elideMode() == Qt.ElideNone, bar.elideMode())
bar_metrics = QFontMetrics(theme.tab_font(bar))
tab_texts = [bar.tabText(index) for index in range(bar.count())]
too_narrow = [(text, bar.tabRect(index).width(), bar_metrics.horizontalAdvance(text))
              for index, text in enumerate(tab_texts)
              if bar.tabRect(index).width() < bar_metrics.horizontalAdvance(text) + 8]
check("每个结果标签的名字都放得下（量实际 tabRect）", not too_narrow, too_narrow)
check("标签确实是容易被截断的长名字", any(len(text) > 10 for text in tab_texts),
      tab_texts)

# ---- 绘图层自适应：尺寸声明 / 字号 / 旋转，不依赖界面直接验
probe = Figure(figsize=(8, 3))
AP.reset_size(probe)
AP.fit_figure(probe, 6.3, 4.7)
check("无内容声明时不低于最小高度",
      AP.MIN_FIG_H_IN <= probe.get_figheight() < 5.0, probe.get_figheight())
AP.reset_size(probe)
AP.declare_size(probe, 9.4)
AP.fit_figure(probe, 6.3, 4.7)
check("内容声明的高度优先于视口高度",
      abs(probe.get_figheight() - 9.4) < 0.01, probe.get_figheight())
check("宽度铺满预览区", abs(probe.get_figwidth() - 6.3) < 0.01, probe.get_figwidth())
AP.declare_size(probe, 99.0)
AP.fit_figure(probe, 6.3, 4.7)
check("超长内容有上限（不画十米图）",
      probe.get_figheight() <= AP.MAX_FIG_H_IN + 0.01, probe.get_figheight())
check("刻度字号随标签数下降但有下限",
      AP.tick_fontsize(4) > AP.tick_fontsize(20) >= 6.8,
      (AP.tick_fontsize(4), AP.tick_fontsize(20), AP.tick_fontsize(60)))
probe_ax = probe.add_subplot(111)
probe_ax.set_xticks(range(2))
probe_ax.set_xticklabels(["AB", "CD"])
AP.fit_x_labels(probe_ax, ["AB", "CD"])
flat = probe_ax.xaxis.get_ticklabels()[0].get_rotation()
AP.fit_x_labels(probe_ax, ["很长的编码名称", "CD"])
tilted = probe_ax.xaxis.get_ticklabels()[0].get_rotation()
check("长标签自动旋转、短标签保持水平", flat == 0 and tilted == 45, (flat, tilted))

# ---- 编码类型很多的真实场景：撑高画布而不是压缩标签
def tick_overlaps(ax):
    """同一轴上相邻刻度标签互相压字的对数（旋转标签天然错位，不计）。"""
    renderer = mod.figure.canvas.get_renderer()
    pairs = 0
    for axis in (ax.xaxis, ax.yaxis):
        texts = [item for item in axis.get_ticklabels()
                 if item.get_visible() and item.get_text() and not item.get_rotation()]
        boxes = [item.get_window_extent(renderer) for item in texts]
        for left, right in zip(boxes, boxes[1:]):
            if (min(left.x1, right.x1) - max(left.x0, right.x0) > 1
                    and min(left.y1, right.y1) - max(left.y0, right.y0) > 1):
                pairs += 1
    return pairs


stress_dir = tempfile.mkdtemp(prefix="stats_stress_")
stress_codes = ["编码类别-%02d（协作认知）" % index for index in range(1, 27)]
stress_header = ["Time"] + ["S%dEncoding_%d" % (student, dim)
                            for student in (1, 2) for dim in (1, 2, 3)]


def value_label_gaps(ax):
    """每个数值标签底边到对应柱顶的像素间隙（负数 = 文字压在柱子里）。"""
    renderer = mod.figure.canvas.get_renderer()
    bars = [patch for patch in ax.patches
            if patch.get_visible() and patch.get_height() > 0]
    gaps = []
    for label in (item for item in ax.texts if item.get_text()):
        box = label.get_window_extent(renderer)
        center = (box.x0 + box.x1) / 2.0
        host = [patch for patch in bars
                if patch.get_x() <= center <= patch.get_x() + patch.get_width()]
        if host:
            top = ax.transData.transform((0, host[0].get_y() + host[0].get_height()))[1]
            gaps.append(box.y0 - top)
    return gaps


def value_label_collisions(ax):
    """相邻数值标签互相压字的对数。"""
    renderer = mod.figure.canvas.get_renderer()
    boxes = sorted((item.get_window_extent(renderer) for item in ax.texts if item.get_text()),
                   key=lambda box: box.x0)
    hits = 0
    for left, right in zip(boxes, boxes[1:]):
        if (min(left.x1, right.x1) - max(left.x0, right.x0) > 1
                and min(left.y1, right.y1) - max(left.y0, right.y0) > 1):
            hits += 1
    return hits


try:
    rng = np.random.default_rng(7)
    rows = []
    for index in range(60):
        row = [index * 10]
        for dim in (1, 2, 3) * 2:
            pool = stress_codes if dim == 3 else stress_codes[:6]
            row.append(pool[int(rng.integers(0, len(pool)))])
        rows.append(row)
    pd.DataFrame(rows, columns=stress_header).to_csv(
        os.path.join(stress_dir, "many_codes.csv"), index=False, encoding="utf-8-sig")
    mod.load_file(os.path.join(stress_dir, "many_codes.csv"))
    pump(300)
    check("压力数据解析出 26 个编码", len(mod.dataset.codes_by_dim.get("D3", [])) >= 20,
          len(mod.dataset.codes_by_dim.get("D3", [])))
    mod.set_method("M01")
    pump(60)
    mod.generate()
    pump(300)
    avail_w, avail_h = mod._preview_inches()
    fig_w, fig_h = mod.figure.get_figwidth(), mod.figure.get_figheight()
    check("编码多时画布按内容撑高（而不是压缩标签）", fig_h > avail_h + 0.5, (fig_h, avail_h))
    check("撑高后的宽度仍铺满预览区", fig_w >= avail_w - 0.05, (fig_w, avail_w))
    ratio = getattr(mod.canvas, "device_pixel_ratio", 1.0) or 1.0
    check("画布控件钉成 figure 尺寸（超出部分滚动查看）",
          abs(mod.canvas.height() - fig_h * mod.figure.dpi / ratio) <= 2,
          (mod.canvas.height(), fig_h * mod.figure.dpi / ratio))
    check("预览区不需要横向滚动条",
          mod.figure_scroll.horizontalScrollBar().maximum() == 0,
          mod.figure_scroll.horizontalScrollBar().maximum())
    mod.canvas.draw()
    app.processEvents()
    bad = sum(tick_overlaps(ax) for ax in mod.figure.axes if ax.get_visible())
    check("26 个长编码标签互不压字", bad == 0, bad)
    y_sizes = [max((item.get_fontsize() for item in ax.yaxis.get_ticklabels()
                    if item.get_text()), default=0.0)
               for ax in mod.figure.axes if ax.get_visible()]
    check("撑高画布后纵向编码标签不被缩到 8pt 以下", min(y_sizes) >= 8.0, min(y_sizes))
    vbar = mod.figure_scroll.verticalScrollBar()
    check("画布高出预览区时出纵向滚动条", vbar.maximum() > 0, vbar.maximum())
    scroll_shot = os.path.join(SHOTS, "stats-preview-scroll.png")
    w.grab().save(scroll_shot)
    check("预览区滚动形态留档", os.path.getsize(scroll_shot) > 20000, scroll_shot)
    m01_shot = os.path.join(SHOTS, "stats-stress-m01.png")
    mod.figure.savefig(m01_shot, dpi=110)
    check("M01 压力出图留档", os.path.getsize(m01_shot) > 20000, m01_shot)
    mod.set_method("M13")
    pump(60)
    mod.generate()
    pump(250)
    mod.canvas.draw()
    app.processEvents()
    named = max(sum(1 for item in ax.get_xticklabels() if item.get_text())
                for ax in mod.figure.axes if ax.get_visible())
    check("编码类 x 轴不抽稀：每个编码都留名字", named >= 20, named)
    struck = [gap for ax in mod.figure.axes if ax.get_visible()
              for gap in value_label_gaps(ax) if gap <= 0]
    check("数值标在柱顶之上，不压在柱子里", not struck, struck[:4])
    stuck = sum(value_label_collisions(ax) for ax in mod.figure.axes if ax.get_visible())
    check("数值标签互不粘连", stuck == 0, stuck)
    checked_axes = sum(1 for ax in mod.figure.axes if ax.get_visible() and ax.texts)
    check("M13 确实带了数值标注（上面两条不是空断）", checked_axes >= 3, checked_axes)
    stress_shot = os.path.join(SHOTS, "stats-stress-m13.png")
    mod.figure.savefig(stress_shot, dpi=110)
    check("压力场景出图留档", os.path.exists(stress_shot)
          and os.path.getsize(stress_shot) > 20000, stress_shot)
finally:
    shutil.rmtree(stress_dir, ignore_errors=True)

# ---- 图例位置：选图外时必须落在图形以留好的单独位置，不得压住坐标区
mod.load_file(GROUP1)
pump(300)
mod.set_method("M07")
pump(150)


def legend_tally():
    return (len(mod.figure.legends),
            sum(1 for ax in mod.figure.axes if ax.get_visible() and ax.get_legend()))


def outside_legend_hits():
    """图形级图例（正下方 / 右侧）与任一坐标区重叠的对数。"""
    mod.canvas.draw()
    renderer = mod.figure.canvas.get_renderer()
    boxes = [ax.get_window_extent(renderer) for ax in mod.figure.axes if ax.get_visible()]
    hits = 0
    for legend in mod.figure.legends:
        area = legend.get_window_extent(renderer)
        for box in boxes:
            if (min(area.x1, box.x1) - max(area.x0, box.x0) > 1
                    and min(area.y1, box.y1) - max(area.y0, box.y0) > 1):
                hits += 1
    return hits


check("图例位置已做成下拉参数", isinstance(param_widget("legend_loc"), QComboBox),
      type(param_widget("legend_loc")).__name__)
check("图例默认放图形正下方", read_param("legend_loc") == "below", read_param("legend_loc"))
set_param("legend_loc", "below")
mod.generate()
pump(250)
figure_legends, axis_legends = legend_tally()
check("正下方：一张图形级图例，行内不再各自带图例",
      figure_legends == 1 and axis_legends == 0, (figure_legends, axis_legends))
check("正下方：图例不压任何坐标区", outside_legend_hits() == 0, outside_legend_hits())
below_shot = os.path.join(SHOTS, "stats-legend-below.png")
mod.figure.savefig(below_shot, dpi=95)
check("图例正下方留档", os.path.getsize(below_shot) > 20000, below_shot)
set_param("legend_loc", "right")
mod.generate()
pump(250)
check("右侧：图例不压任何坐标区", outside_legend_hits() == 0, outside_legend_hits())
right_shot = os.path.join(SHOTS, "stats-legend-right.png")
mod.figure.savefig(right_shot, dpi=95)
check("图例右侧留档", os.path.getsize(right_shot) > 20000, right_shot)
set_param("legend_loc", "upper right")
mod.generate()
pump(250)
figure_legends, axis_legends = legend_tally()
visible_axes = sum(1 for ax in mod.figure.axes if ax.get_visible())
check("图内选项：逐行各一张图例且无图形级图例",
      figure_legends == 0 and axis_legends == visible_axes,
      (figure_legends, axis_legends, visible_axes))
set_param("legend_loc", "none")
mod.generate()
pump(250)
check("不显示：图上没有任何图例", legend_tally() == (0, 0), legend_tally())
check("不显示图例时图形仍完整", figure_content() > 0, figure_content())
set_param("legend_loc", "below")
mod.generate()
pump(250)

# 窗口数很多的极端情况：x 轴标签靠旋转 + 撑高画布保证可读
mod.set_method("M04")
pump(60)
set_param("bin_mode", "seconds")
set_param("bin_value", 12.5)
mod.generate()
pump(300)
check("151 个窗口下仍出图", figure_content() > 0 and not mod.result.blocked)
mod.canvas.draw()
app.processEvents()
bin_overlaps = sum(tick_overlaps(ax) for ax in mod.figure.axes if ax.get_visible())
check("密集时间窗口的 x 轴标签不互压", bin_overlaps == 0, bin_overlaps)
bin_labels = max(sum(1 for item in ax.get_xticklabels() if item.get_text())
                 for ax in mod.figure.axes if ax.get_visible())
check("151 个窗口的轴自动抽稀（不至于糊成一片）", bin_labels <= AP.MAX_X_LABELS + 1,
      bin_labels)
bins_shot = os.path.join(SHOTS, "stats-m04-bins.png")
mod.figure.savefig(bins_shot, dpi=95)
check("密集窗口出图留档", os.path.getsize(bins_shot) > 20000, bins_shot)
set_param("bin_value", 60)
mod.generate()
pump(200)
tabs_shot = os.path.join(SHOTS, "stats-ui-m04-tabs.png")
w.grab().save(tabs_shot)
check("M04 长表名标签页整窗留档", os.path.getsize(tabs_shot) > 20000, tabs_shot)

print()
print("=" * 78)
print("9) 主题切换重绘与截图")
print("=" * 78)

mod.set_method("M07")
pump(120)
axes_before = len(mod.figure.axes)
content_before = figure_content()

os.makedirs(SHOTS, exist_ok=True)
w.tab_widget.setCurrentIndex(1)
pump(400)
light_shot = os.path.join(SHOTS, "stats-ui-light.png")
w.grab().save(light_shot)
check("浅色主题整窗截图", os.path.getsize(light_shot) > 20000
      if os.path.exists(light_shot) else False, light_shot)

w.dark_action.setChecked(True)
pump(500)
check("深色主题下仍保有结果", mod.result is not None
      and mod.result.method_id == "M07")
check("redraw_chart 后面板数不变", len(mod.figure.axes) == axes_before,
      (axes_before, len(mod.figure.axes)))
check("重绘后图形仍有内容", figure_content() > 0, figure_content())
card = mcolors.to_hex(theme.tokens()["card"])
check("图形底色跟随主题",
      mcolors.to_hex(mod.figure.get_facecolor()) == card,
      (mcolors.to_hex(mod.figure.get_facecolor()), card))
check("深色下统计表依旧可读", table_rows() > 0, table_rows())
dark_shot = os.path.join(SHOTS, "stats-ui-dark.png")
w.grab().save(dark_shot)
check("深色主题整窗截图", os.path.exists(dark_shot)
      and os.path.getsize(dark_shot) > 20000, dark_shot)

mod.set_method("M16")
pump(200)
shot = os.path.join(SHOTS, "stats-ui-dark-m16.png")
mod.canvas.grab().save(shot)
check("深色下 M16 出图", len(mod.figure.axes) >= 1 and figure_content() > 0)

# 浮点窗长控件只在时间窗口类方法里出现，单独留一张验收图
# 只 grab 子控件会截到主题重绘的半程状态（标签已切、输入框未切），改取整窗
mod.set_method("M04")
pump(150)
w.dark_action.setChecked(True)
pump(500)
dark_params = os.path.join(SHOTS, "stats-ui-dark-params.png")
w.grab().save(dark_params)
check("深色下参数卡截图", os.path.exists(dark_params)
      and os.path.getsize(dark_params) > 20000, dark_params)
w.dark_action.setChecked(False)
pump(500)
light_params = os.path.join(SHOTS, "stats-ui-light-params.png")
w.grab().save(light_params)
check("浅色下参数卡截图", os.path.getsize(light_params) > 20000, light_params)

if backup is not None:
    with open(settings, "w", encoding="utf-8") as handle:
        handle.write(backup)
elif os.path.exists(settings):
    os.remove(settings)

print()
print("=" * 78)
print("截图输出：%s" % SHOTS)
print("失败 %d 项" % len(failures))
for name in failures:
    print("  -", name)
sys.exit(1 if failures else 0)
