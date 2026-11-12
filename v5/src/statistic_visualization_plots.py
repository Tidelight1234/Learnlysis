# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""M01–M19 的 matplotlib 绘图。

所有函数签名统一为 ``plot(fig, result, ds, p)``：调用方负责 ``fig.clear()`` 与
``theme.sync_figure(fig)``，本模块只做坐标轴、色块与图例。

配色与 UI 共用 theme.CATEGORY_COLORS：同一维度内的编码在任何图里颜色一致。
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib.colors import LinearSegmentedColormap

import theme

MISSING_LABEL = "（缺失）"


# ------------------------------------------------------------------ 公共工具

def tokens():
    return theme.tokens()


def value_cmap():
    """热力图/流带用的紫色渐进色带，浅色与深色各一套。"""
    t = tokens()
    stops = [t["card"], t["tint"], t["purple"]] if t["card"] == "#FFFFFF" \
        else [t["card"], "#332C52", t["purple"]]
    return LinearSegmentedColormap.from_list("learnlysis", stops)


def color_map(ds, dimension):
    """同一维度的编码 → 固定颜色（按 codes_by_dim 顺序，保证可复现）。"""
    codes = ds.codes_by_dim.get(dimension, [])
    palette = theme.CATEGORY_COLORS
    mapping = {code: palette[idx % len(palette)] for idx, code in enumerate(codes)}
    return mapping


def missing_color():
    return tokens()["line_strong"]


def bin_short(label):
    return str(label).split(" ")[0] if label else ""


# ---------------------------------------------------- 画布尺寸（自适应，不靠压缩标签）

MIN_FIG_H_IN = 3.2      # 预览区再矮也给图留出的最小高度
MAX_FIG_H_IN = 26.0     # 上限：再长就该提示用户筛选项，而不是生成十米长图
BAND_H_IN = 0.62        # 一行时间轴色带（M06/M07/M10/M11 的单行）
FACET_H_IN = 1.85       # 一个坐标面板（柱状 / 热图 / 箱线）
BAR_H_IN = 0.26         # 水平条形图里的一条
COLUMN_W_IN = 0.34      # 竖向柱状图里的一根（含旋转标签与顶端数值）
CELL_H_IN = 0.22        # 热图里的一行


def reset_size(fig):
    """每次重绘前清掉上一次声明，否则图会只增不减。"""
    fig._sv_need = (0.0, 0.0)


def declare_size(fig, height_in, width_in=0.0):
    """绘图函数只负责声明“这些内容至少需要多高”，实际尺寸由预览区决定。"""
    need = getattr(fig, "_sv_need", (0.0, 0.0))
    fig._sv_need = (max(need[0], float(width_in or 0.0)),
                    max(need[1], float(height_in or 0.0)))


def fit_figure(fig, avail_w_in, avail_h_in):
    """宽度铺满预览区；高度不够就把画布撑大，由预览区滚动而不是压缩标签。"""
    need_w, need_h = getattr(fig, "_sv_need", (0.0, 0.0))
    width = max(float(avail_w_in), need_w, 4.0)
    height = min(max(float(avail_h_in), need_h, MIN_FIG_H_IN), MAX_FIG_H_IN)
    if abs(width - fig.get_figwidth()) > 0.02 or abs(height - fig.get_figheight()) > 0.02:
        fig.set_size_inches(width, height, forward=True)
    return fig.get_size_inches()


def tick_fontsize(count, base=9.5, floor=6.8, step=0.16):
    """标签越多字越小，但有下限：再挤就该撑高画布而不是继续缩字。"""
    return max(floor, base - step * max(0, int(count) - 8))


# 行距由 declare_size 保证的轴（M01–M03 / M08 / M11 / M12）：画布已经按条目数撑高了，
# 每行实际间距不随条目数变小，所以不该再因为「条数多」而缩字——只在顶到
# MAX_FIG_H_IN、行距真被压扁时才小幅下调。
PITCHED_BASE, PITCHED_FLOOR, PITCHED_STEP = 9.5, 7.6, 0.03


def pitched_fontsize(count):
    """间距有保证的纵向类目轴字号：26 项时给到 9.0pt，而不是旧的 6.8pt。"""
    return tick_fontsize(count, base=PITCHED_BASE, floor=PITCHED_FLOOR,
                         step=PITCHED_STEP)


MAX_X_LABELS = 14     # 时间窗口轴最多画几个标签：再多就抽稀，宁可省略也不叠字


def fit_x_labels(ax, labels, base=9.0, max_labels=None):
    """x 轴标签自适应：太长就旋转、字号有下限；给了 max_labels 才抽稀。

    时间窗口轴（M04/M05/M08）刻度可达上百个，抽稀只省掉部分窗口名；
    而编码类轴（M12/M13）每个柱子就是一个编码，抽稀等于藏掉编码，所以默认不抽。
    """
    count = len(labels)
    if not count:
        return 1
    longest = max(len(str(text)) for text in labels)
    stride = 1
    if max_labels and count > max_labels:
        stride = int(np.ceil(count / float(max_labels)))
        kept = list(range(0, count, stride))
        ax.set_xticks(kept)
        ax.set_xticklabels([str(labels[index]) for index in kept])
    rotation = 45 if longest > 5 or count > (max_labels or 12) else 0
    ax.tick_params(axis="x", labelsize=tick_fontsize(count // stride, base=base),
                   labelrotation=rotation)
    if rotation:
        for text in ax.get_xticklabels():
            text.set_ha("right")
    return stride


def place_value_labels(ax, xs, tops, texts, base=8.8, floor=6.4):
    """把数值标在柱子顶端的上方（而不是压在柱子头上）。

    tops 传的是「柱高 + 误差条上沿」，否则误差线会正好穿过数字。相邻柱子太密、
    横排数字会粘连时改竖排：数字竖着一列只占一个字宽，永远不会和邻居叠在一起。
    """
    count = len(list(xs))
    if not count:
        return True
    size = tick_fontsize(count, base=base, floor=floor)
    # 一列能占多宽：单轴面板约吃掉画布宽度的 78%，再按柱子数平分
    pitch_in = ax.get_figure().get_figwidth() * 0.78 / count
    widest = max((len(str(text)) for text in texts), default=1)
    horizontal = pitch_in >= widest * size * 0.58 / 72.0 + 0.06
    place = {} if horizontal else {"rotation": 90, "rotation_mode": "anchor"}
    for x, top, text in zip(xs, tops, texts):
        ax.annotate(str(text), (x, top), xytext=(0, 2.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=size, **place)
    return horizontal


# ------------------------------------------------------------------ 图例位置

LEGEND_OUTSIDE = ("below", "right")
LEGEND_FONTSIZE = 8.5


def _swatch(color, hatch=None):
    return Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none", hatch=hatch)


def _legend_inside(p, default="below"):
    """图例是否放在坐标轴内部（那样就要给面板多留行高）。"""
    return (p.get("legend_loc") or default) not in ("none",) + LEGEND_OUTSIDE


def _legend_pairs(items):
    """items 的每一项为 (label, color) 或 (label, color, hatch)。"""
    handles, labels = [], []
    for item in items:
        handles.append(_swatch(item[1], item[2] if len(item) > 2 else None))
        labels.append(str(item[0]))
    return handles, labels


def _legend_key(item):
    return (str(item[0]), item[1], item[2] if len(item) > 2 else None)


def add_legend(fig, panels, p, base_rect=(0.0, 0.0, 1.0, 1.0), default="below", ncol=8):
    """按用户选的「图例位置」放图例，返回 tight_layout 应当使用的 rect。

    panels 为 [(ax, items), ...]：图内位置逐面板放（带半透底色框）；图外位置
    合并成一张图形级图例，并预留对应边距，无论多少面板都不会压住数据。
    """
    loc = p.get("legend_loc") or default
    if loc == "none":
        return base_rect
    if loc not in LEGEND_OUTSIDE:
        for ax, items in panels:
            if not items:
                continue
            handles, labels = _legend_pairs(items)
            ax.legend(handles, labels, loc=loc, fontsize=LEGEND_FONTSIZE, frameon=True,
                      framealpha=0.92, ncol=1, borderpad=0.5, labelspacing=0.35,
                      handlelength=1.1, handletextpad=0.5)
        return base_rect
    labels, handles, seen = [], [], set()
    for _ax, items in panels:
        for item in items:
            key = _legend_key(item)
            if key in seen:
                continue
            seen.add(key)
            handles.append(_swatch(key[1], key[2]))
            labels.append(key[0])
    if not labels:
        return base_rect
    if loc == "right":
        width_in = 0.75 + 0.115 * min(max(len(text) for text in labels), 24)
        frac = min(0.46, max(0.14, width_in / max(fig.get_figwidth(), 1.0)))
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.93),
                   ncol=1, fontsize=LEGEND_FONTSIZE, frameon=False, labelspacing=0.42,
                   handlelength=1.1, handletextpad=0.45)
        return (base_rect[0], base_rect[1], base_rect[2] - frac, base_rect[3])
    cols = max(1, min(len(labels), ncol))
    rows = int(np.ceil(len(labels) / cols))
    frac = min(0.42, (0.16 + 0.20 * rows) / max(fig.get_figheight(), 1.0))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.004),
               ncol=cols, fontsize=LEGEND_FONTSIZE, frameon=False, columnspacing=1.2,
               labelspacing=0.5, handlelength=1.1, handletextpad=0.45)
    return (base_rect[0], base_rect[1] + frac, base_rect[2], base_rect[3])


def _grid(n):
    n = max(int(n), 1)
    ncols = 1 if n <= 3 else 2
    return int(np.ceil(n / ncols)), ncols


def _row_budgets(counts, ncols, per_item, pad, floor=FACET_H_IN):
    """分面网格每行的纵向预算。

    行高按「本行最多条目数」而不是「全表最多条目数」分配，否则 6 个编码的面板会被
    26 个的那一块撑成一米粗的巨柱。同一组预算同时交给 height_ratios 与 declare_size，
    两边口径一致，撑高后每行拿到的实际高度就是这里的值。
    """
    counts = list(counts) or [0]
    rows = [counts[start:start + ncols] for start in range(0, len(counts), ncols)]
    return [max(floor, max(row) * per_item + pad) for row in rows]


def _suptitle(fig, text):
    fig.suptitle(text, fontsize=12.5, x=0.5, ha="center")


def _unit_label(student, dimension):
    return "%s · %s" % (student, dimension)


def _annotate_bars(ax, bars, values, digits=0):
    for bar, value in zip(bars, values):
        ax.annotate(("%g" % value) if abs(value - round(value)) < 1e-9 else "%.2f" % value,
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize="small")


def _bar_facets(axes, frames, value_col, xlabel, title_fmt, p):
    """M01/M02/M03 共用：每个 student × dimension 一个水平条形面板。"""
    for ax, frame in zip(np.atleast_1d(axes), frames):
        if frame.empty:
            ax.set_visible(False)
            continue
        codes = frame["code"].astype(str).tolist()[::-1]
        values = frame[value_col].tolist()[::-1]
        bars = ax.barh(range(len(codes)), values, color=tokens()["purple"], height=0.62)
        ax.set_yticks(range(len(codes)))
        ax.set_yticklabels(codes)
        ax.tick_params(axis="y", labelsize=pitched_fontsize(len(codes)))
        ax.set_xlabel(xlabel)
        ax.set_title(title_fmt.format(student=frame["student_id"].iloc[0],
                                      dimension=frame["dimension"].iloc[0]),
                     fontsize=11.5)
        ax.margins(x=0.18)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        if p.get("show_values", True):
            _annotate_bars(ax, bars, values)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)


def _segments_from_sequence(ds, sequence):
    """编码序列 → [(start, span, code)]，连续相同状态合并成一个矩形。"""
    positions = ds.time_positions()
    durations = [ds.duration_by_time.get(t, 0.0) for t in positions]
    segments = []
    for code, start, length in _runs(sequence):
        begin = positions[start]
        span = sum(durations[start:start + length])
        if not span:
            span = float(length) * (ds.time_info.median_dt or 1.0)
        segments.append((begin, span, code))
    return segments


def _runs(values):
    runs = []
    for idx, value in enumerate(values):
        if runs and runs[-1][0] == value:
            runs[-1][2] += 1
        else:
            runs.append([value, idx, 1])
    return [(code, start, length) for code, start, length in runs]


def _time_axis(ax, ds, label="Time (s)"):
    ax.set_xlabel(label)
    ax.set_xlim(ds.time_info.t_min, ds.time_info.t_max + (ds.time_info.median_dt or 0))
    locator_step = max(ds.time_info.median_dt or 1.0, 1.0)
    ticks = np.arange(ds.time_info.t_min,
                      ds.time_info.t_max + locator_step, locator_step)
    if len(ticks) > 12:
        stride = int(np.ceil(len(ticks) / 12.0))
        ticks = ticks[::stride]
    ax.set_xticks(list(ticks))
    ax.tick_params(axis="x", rotation=0)


def _no_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ------------------------------------------------------------------ M01–M03

def plot_simple(fig, result, ds, p, value_col=None, xlabel=None):
    """M01 / M02 / M03 共用的分面条形图。"""
    frame = result.table()
    if frame.empty:
        return
    value_col = value_col or {"M01": "count", "M02": "percentage",
                              "M03": "duration_s"}[result.method_id]
    xlabel = xlabel or {"M01": "频次", "M02": "占比 (%)",
                        "M03": "累计时长 (s)"}[result.method_id]
    units = frame.groupby(["student_id", "dimension"], sort=False)
    groups = [part for _, part in units]
    nrows, ncols = _grid(len(groups))
    budgets = _row_budgets([len(part) for part in groups], ncols, BAR_H_IN, 0.6)
    axes = np.array(fig.subplots(nrows, ncols, squeeze=False,
                                 gridspec_kw={"height_ratios": budgets})).ravel()
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    _bar_facets(axes[:len(groups)], groups, value_col, xlabel, "{student} · {dimension}", p)
    declare_size(fig, sum(budgets) + 0.9)
    _suptitle(fig, "%s｜%s" % (result.method_name, ds.time_info.describe()))
    fig.tight_layout(rect=(0, 0, 1, 0.955))


def plot_frequency(fig, result, ds, p):
    plot_simple(fig, result, ds, p)


def plot_percentage(fig, result, ds, p):
    frame = result.table()
    if frame.empty:
        return
    units = frame.groupby(["student_id", "dimension"], sort=False)
    groups = [part for _, part in units]
    nrows, ncols = _grid(len(groups))
    axes = np.array(fig.subplots(nrows, ncols, squeeze=False)).ravel()
    for ax, part in zip(axes[:len(groups)], groups):
        codes = part["code"].astype(str).tolist()
        values = part["percentage"].tolist()
        left = 0.0
        for code, value in zip(codes, values):
            color = color_map(ds, part["dimension"].iloc[0]).get(
                code, tokens()["ink1"])
            ax.barh([0], [value], left=left, color=color, height=0.5)
            if p.get("show_values", True) and value >= 4:
                ax.text(left + value / 2, 0, "%g%%" % value, ha="center",
                        va="center", fontsize="small", color=tokens()["card"])
            left += value
        ax.set_yticks([0])
        ax.set_yticklabels([_unit_label(part["student_id"].iloc[0],
                                       part["dimension"].iloc[0])])
        ax.set_xlim(0, 100)
        ax.set_xlabel("占比 (%)")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    declare_size(fig, nrows * max(FACET_H_IN * 0.7, 0.9) + 0.9)
    _suptitle(fig, "Percentage｜每格为该 student × dimension 的 100% 构成")
    fig.tight_layout(rect=(0, 0, 1, 0.955))


def plot_duration(fig, result, ds, p):
    plot_simple(fig, result, ds, p, value_col="duration_s", xlabel="累计时长 (s)")


# ------------------------------------------------------------------ M04–M05

def plot_bin_frequency(fig, result, ds, p):
    """M04：x=时间窗口，色=编码的 grouped bar。"""
    frame = result.table()
    if frame.empty:
        return
    groups = [part for _, part in frame.groupby(["student_id", "dimension"], sort=False)]
    nrows, ncols = _grid(len(groups))
    axes = np.array(fig.subplots(nrows, ncols, squeeze=False)).ravel()
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    panels, widest = [], 0
    for ax, part in zip(axes, groups):
        dimension = part["dimension"].iloc[0]
        bins = list(dict.fromkeys(part["time_bin"].tolist()))
        codes = [c for c in ds.codes_by_dim.get(dimension, []) if c in set(part["code"])]
        mapping = color_map(ds, dimension)
        matrix = part.pivot_table(index="time_bin", columns="code", values="count",
                                 aggfunc="sum", fill_value=0)
        matrix = matrix.reindex(index=bins, columns=codes, fill_value=0)
        width = 0.82 / max(len(codes), 1)
        for idx, code in enumerate(codes):
            offsets = np.arange(len(bins)) + (idx - (len(codes) - 1) / 2.0) * width
            ax.bar(offsets, matrix[code].to_numpy(), width=width,
                   color=mapping[code], label=code)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([bin_short(b) for b in bins])
        fit_x_labels(ax, [bin_short(b) for b in bins], max_labels=MAX_X_LABELS)
        ax.set_ylabel("频次")
        ax.set_title(_unit_label(part["student_id"].iloc[0], dimension), fontsize=11.5)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        panels.append((ax, [(code, mapping[code]) for code in codes]))
        widest = max(widest, len(codes))
    rect = add_legend(fig, panels, p, base_rect=(0, 0, 1, 0.945))
    per_facet = FACET_H_IN + (0.17 * widest if _legend_inside(p) else 0.0)
    declare_size(fig, nrows * per_facet + 0.9)
    _suptitle(fig, "Time-bin Frequency｜窗口边界：%s" % _bin_note(ds))
    fig.tight_layout(rect=rect)


def plot_bin_percentage(fig, result, ds, p):
    """M05：100% stacked bar 或 stacked area。"""
    frame = result.table()
    if frame.empty:
        return
    groups = [part for _, part in frame.groupby(["student_id", "dimension"], sort=False)]
    nrows, ncols = _grid(len(groups))
    axes = np.array(fig.subplots(nrows, ncols, squeeze=False)).ravel()
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    area = p.get("chart_style") == "area"
    panels = []
    widest = 0
    for ax, part in zip(axes, groups):
        dimension = part["dimension"].iloc[0]
        bins = list(dict.fromkeys(part["time_bin"].tolist()))
        codes = [c for c in ds.codes_by_dim.get(dimension, []) if c in set(part["code"])]
        mapping = color_map(ds, dimension)
        matrix = part.pivot_table(index="time_bin", columns="code", values="percentage",
                                 aggfunc="sum", fill_value=0.0)
        matrix = matrix.reindex(index=bins, columns=codes, fill_value=0.0)
        if area and len(bins) > 2:
            x = np.arange(len(bins), dtype=float)
            ax.stackplot(x, *[matrix[c].to_numpy() for c in codes],
                         labels=codes,
                         colors=[mapping[c] for c in codes], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([bin_short(b) for b in bins])
            ax.set_xlim(0, len(bins) - 1)
            bin_labels = [bin_short(b) for b in bins]
        else:
            left = np.zeros(len(bins))
            for code in codes:
                values = matrix[code].to_numpy()
                ax.bar(np.arange(len(bins)), values, bottom=left, color=mapping[code],
                       label=code, width=0.82)
                if p.get("show_values", True):
                    for x, value, base in zip(range(len(bins)), values, left):
                        if value >= 7:
                            ax.text(x, base + value / 2, "%g%%" % round(value, 1),
                                    ha="center", va="center", fontsize="small",
                                    color=tokens()["card"])
                left += values
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels([bin_short(b) for b in bins])
            bin_labels = [bin_short(b) for b in bins]
            ax.set_ylim(0, 100)
        fit_x_labels(ax, bin_labels, max_labels=MAX_X_LABELS)
        ax.set_ylabel("窗口内占比 (%)")
        ax.set_title(_unit_label(part["student_id"].iloc[0], dimension), fontsize=11.5)
        panels.append((ax, [(code, mapping[code]) for code in codes]))
        widest = max(widest, len(codes))
    rect = add_legend(fig, panels, p, base_rect=(0, 0, 1, 0.945))
    per_facet = FACET_H_IN + (0.17 * widest if _legend_inside(p) else 0.0)
    declare_size(fig, nrows * per_facet + 0.9)
    _suptitle(fig, "Time-bin Percentage｜分母为该窗口内的有效观测数")
    fig.tight_layout(rect=rect)


def _bin_note(ds):
    if not ds.bin_bounds:
        return "未分箱"
    first = ds.bin_bounds[0]
    last = ds.bin_bounds[-1]
    return "%s – %s，共 %d 窗" % (first[0], last[0], len(ds.bin_bounds))


# ------------------------------------------------------------------ M06–M07

def plot_timeline(fig, result, ds, p):
    """M06：单学生单维度分类时间轴。"""
    obs = result.tables.get("时间轴观测")
    if obs is None or obs.empty:
        return
    student = obs["student_id"].iloc[0]
    dimension = obs["dimension"].iloc[0]
    mapping = color_map(ds, dimension)
    sequence = ds.sequence(student, dimension)
    ax = fig.add_subplot(111)
    segments = _segments_from_sequence(ds, sequence)
    for start, span, code in segments:
        ax.add_patch(Rectangle((start, -0.5), span, 1.0,
                               facecolor=missing_color() if code is None else mapping.get(
                                   code, tokens()["ink1"]),
                               edgecolor=tokens()["card"], linewidth=0.4,
                               hatch="///" if code is None else None))
    ax.set_yticks([0])
    ax.set_yticklabels([_unit_label(student, dimension)])
    ax.set_ylim(-0.9, 0.9)
    _time_axis(ax, ds)
    items = [(code, color) for code, color in mapping.items()]
    if any(code is None for _, _, code in segments) and p.get("show_missing", True):
        items.append((MISSING_LABEL, missing_color(), "///"))
    rect = add_legend(fig, [(ax, items)], p, base_rect=(0, 0, 1, 0.94))
    declare_size(fig, BAND_H_IN * 2.0 + 1.3)
    _suptitle(fig, "Timeline｜%s · %s（每个色块宽度 = 该时间点的 duration）"
              % (student, dimension))
    fig.tight_layout(rect=rect)


def plot_multidim_timeline(fig, result, ds, p):
    """M07：同一学生 D1/D2/D3 同步时间轴，本项目最核心的过程图。"""
    students = list(result.tables.keys())
    dimensions = _chosen_dimensions(ds, p)
    if not students or not dimensions:
        return
    axes = fig.subplots(len(students) * len(dimensions), 1, squeeze=False, sharex=True)
    axes = [ax for row in axes for ax in row]
    row = 0
    panels = []
    for table_name in students:
        student = table_name.replace("学生 ", "")
        block_start = row
        for dimension in dimensions:
            ax = axes[row]
            mapping = color_map(ds, dimension)
            sequence = ds.sequence(student, dimension)
            segments = _segments_from_sequence(ds, sequence)
            present = []
            has_missing = False
            for start, span, code in segments:
                if code is None:
                    has_missing = True
                    face, hatch = missing_color(), "///"
                else:
                    if code not in present:
                        present.append(code)
                    face, hatch = mapping.get(code, tokens()["ink1"]), None
                ax.add_patch(Rectangle((start, -0.5), span, 1.0, facecolor=face,
                                       edgecolor=tokens()["card"], linewidth=0.3,
                                       hatch=hatch))
            ax.set_yticks([0])
            ax.set_yticklabels([dimension])
            ax.set_ylim(-0.75, 0.75)
            ax.grid(True, axis="x", linestyle="--", alpha=0.3)
            if row < len(axes) - 1:
                ax.tick_params(labelbottom=False)
            items = [(code, mapping[code]) for code in present if code in mapping]
            if has_missing and p.get("show_missing", True):
                items.append((MISSING_LABEL, missing_color(), "///"))
            panels.append((ax, items))
            row += 1
        axes[block_start].set_title(
            "学生 %s：%s 共 %d 个编码维度" % (student, "/".join(dimensions), len(dimensions)),
            fontsize=11.5, loc="left", pad=8)
    _time_axis(axes[-1], ds)
    # 每行只有 0.6 in 高，图例塞进行内必定盖住色带：图内选项改成挂在行右侧，
    # 图外选项合并成一张图形级图例
    if _legend_inside(p):
        for ax, items in panels:
            _row_legend(ax, items)
        rect = (0.0, 0.0, 0.74, 0.94)
    else:
        rect = add_legend(fig, panels, p, base_rect=(0, 0, 0.99, 0.94))
    declare_size(fig, len(axes) * BAND_H_IN + len(students) * 0.5 + 1.1)
    _suptitle(fig, "Multi-dimensional Timeline｜每名学生一组，组内共享同一 Time 轴")
    fig.tight_layout(rect=rect)


def _row_legend(ax, items):
    """多维时间轴每行只图例本行出现的编码，放在该行坐标区右侧的空白里。"""
    if not items:
        return
    handles, labels = _legend_pairs(items)
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              ncol=1, fontsize=7.4, frameon=False, handlelength=1.0, handletextpad=0.4,
              labelspacing=0.22, borderaxespad=0.0)


def _chosen_dimensions(ds, p):
    chosen = p.get("dimensions") or []
    wanted = {str(c) for c in chosen}
    picked = [d for d in ds.dimensions if str(d) in wanted]
    return picked or list(ds.dimensions)


# ------------------------------------------------------------------ M08 / M15

def _heatmap_one(ax, matrix, title, p, note=""):
    codes = matrix.index.tolist()
    bins = matrix.columns.tolist()
    data = matrix.to_numpy(dtype=float)
    cmap = value_cmap()
    vmax = np.nanmax(data) if data.size else 0.0
    img = ax.imshow(np.ma.masked_invalid(
        data if vmax > 0 else np.full_like(data, np.nan)), aspect="auto", cmap=cmap,
        vmin=0, vmax=(vmax or 1))
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([str(c) for c in codes])
    ax.tick_params(axis="y", labelsize=pitched_fontsize(len(codes)))
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([bin_short(b) for b in bins])
    fit_x_labels(ax, [bin_short(b) for b in bins], base=8.6, max_labels=MAX_X_LABELS)
    ax.set_title(title, fontsize=11.5)
    if p.get("show_values", True) and len(codes) * len(bins) <= 260:
        for (row, col), value in np.ndenumerate(data):
            ax.text(col, row, ("%g" % value) if abs(value - round(value)) < 1e-9
                    else "%.1f" % value,
                    ha="center", va="center", fontsize="small",
                    color=tokens()["ink1"] if value < (vmax or 1) * 0.6 else tokens()["card"])
    bar = ax.figure.colorbar(img, ax=ax, pad=0.02)
    bar.ax.tick_params(labelsize="small")
    if note:
        ax.set_xlabel(note)
    return ax


def plot_heatmap(fig, result, ds, p):
    """M08：code × time_bin 热图，每个 student × dimension 一张。"""
    names = [k for k in result.tables if k != "长表"]
    if not names:
        return
    nrows, ncols = _grid(len(names))
    budgets = _row_budgets([len(result.tables[name]) for name in names], ncols,
                           CELL_H_IN, 1.1)
    axes = np.array(fig.subplots(nrows, ncols, squeeze=False,
                                 gridspec_kw={"height_ratios": budgets})).ravel()
    value = p.get("value", "count")
    for ax in axes[len(names):]:
        _no_axes(ax)
    for ax, name in zip(axes, names):
        table = result.tables[name]
        matrix = table.set_index(table.columns[0])
        _heatmap_one(ax, matrix, "%s（%s）" % (name, "计数" if value == "count" else "窗口内比例(%)"),
                     p, note="时间窗口")
    declare_size(fig, sum(budgets) + 0.9)
    _suptitle(fig, "Heatmap｜%s｜默认显示 count，颜色不代表概率" % _bin_note(ds))
    fig.tight_layout(rect=(0, 0, 0.97, 0.94))


# ------------------------------------------------------------------ M09–M10

def plot_state_distribution(fig, result, ds, p):
    """M09：各时间位置的状态构成（堆叠面积）。"""
    frame = result.tables.get("长表")
    if frame is None or frame.empty:
        return
    dimension = frame["dimension"].iloc[0]
    pivot = frame.pivot_table(index="time", columns="code", values="percentage",
                              aggfunc="sum", fill_value=0.0)
    codes = [c for c in ds.codes_by_dim.get(dimension, []) if c in pivot.columns]
    pivot = pivot.reindex(columns=codes, fill_value=0.0).sort_index()
    ax = fig.add_subplot(111)
    x = pivot.index.to_numpy(dtype=float)
    palette = color_map(ds, dimension)
    ax.stackplot(x, *[pivot[c].to_numpy() for c in codes], labels=codes,
                 colors=[palette[c] for c in codes], alpha=0.88)
    ax.set_ylim(0, 100)
    ax.set_ylabel("状态占比 (%)")
    _time_axis(ax, ds)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    rect = add_legend(fig, [(ax, [(c, palette[c]) for c in codes])], p,
                      base_rect=(0, 0, 1, 0.94))
    declare_size(fig, 2.7)
    n_students = frame["student_id"].iloc[0].count("、") + 1
    _suptitle(fig, "State Distribution｜%s｜跨 %s 个位置，N=%d 名学生"
              % (dimension, len(x), n_students))
    fig.tight_layout(rect=rect)


def plot_modal_state(fig, result, ds, p):
    """M10：主导状态条带，并列位置用斜纹标记。"""
    table = result.table()
    if table.empty:
        return
    dimension = (result.meta.get("filter", {}) or {}).get("dimensions") or [""]
    dimension = dimension[0]
    mapping = color_map(ds, dimension)
    ax = fig.add_subplot(111)
    step = (ds.time_info.median_dt or 1.0)
    ties = 0
    for _, row in table.iterrows():
        start = float(row["time"])
        code = row["modal_code"]
        tie = str(row["tie"]) == "是"
        ties += 1 if tie else 0
        ax.add_patch(Rectangle((start, -0.5), step, 1.0,
                               facecolor=missing_color() if code == MISSING_LABEL
                               else mapping.get(code, tokens()["ink1"]),
                               edgecolor=tokens()["card"], linewidth=0.3,
                               hatch="///" if tie else None))
    ax.set_yticks([0])
    ax.set_yticklabels(["主导状态"])
    ax.set_ylim(-0.8, 0.8)
    _time_axis(ax, ds)
    _suptitle(fig, "Modal State｜%s｜%s（argmax_c P_c,t，仅代表最常见状态）"
              % (dimension, "含 %d 个并列位置" % ties if ties else "无并列位置"))
    rect = add_legend(fig, [(ax, list(mapping.items()))], p, base_rect=(0, 0, 1, 0.95))
    declare_size(fig, BAND_H_IN * 2.0 + 1.3)
    fig.tight_layout(rect=rect)


# ------------------------------------------------------------------ M11

def plot_sequence_index(fig, result, ds, p):
    """M11：固定维度、每学生一行的序列索引图；每个维度一个 panel。"""
    names = [k for k in result.tables if k.startswith("维度 ")]
    if not names:
        return
    axes = fig.subplots(len(names), 1, squeeze=False)
    axes = [ax for row in axes for ax in row]
    panels, total_h = [], 0.0
    for ax, name in zip(axes, names):
        table = result.tables[name]
        dimension = name.replace("维度 ", "")
        mapping = color_map(ds, dimension)
        students = list(dict.fromkeys(table["student_id"].tolist()))
        for idx, student in enumerate(students):
            part = table[table["student_id"] == student]
            for _, seg in part.iterrows():
                start, end = float(seg["start_time"]), float(seg["end_time"])
                code = seg["code"]
                ax.add_patch(Rectangle((start, len(students) - 1 - idx - 0.34),
                                       max(end - start, 1e-9), 0.68,
                                       facecolor=missing_color() if seg["is_missing"] == "是"
                                       else mapping.get(code, tokens()["ink1"]),
                                       edgecolor=tokens()["card"], linewidth=0.3,
                                       hatch="///" if seg["is_missing"] == "是" else None))
        ax.set_yticks(range(len(students)))
        ax.set_yticklabels(list(reversed(students)))
        ax.tick_params(axis="y", labelsize=pitched_fontsize(len(students)))
        ax.set_ylim(-0.6, len(students) - 0.4)
        ax.set_title("维度 %s" % dimension, fontsize=11.5, loc="left")
        _time_axis(ax, ds)
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        panels.append((ax, list(mapping.items())))
        total_h += max(FACET_H_IN, len(students) * BAND_H_IN + 0.9)
    rect = add_legend(fig, panels, p, base_rect=(0, 0, 0.9, 0.95))
    declare_size(fig, total_h + 0.9)
    _suptitle(fig, "Sequence Index｜排序方式：%s" % _sort_name(p.get("sort_by", "student")))
    fig.tight_layout(rect=rect)


def _sort_name(key):
    return {"student": "按学生 ID", "first_code": "按起始状态",
            "code_frequency": "按主要状态累计时长"}.get(key, key)


# ------------------------------------------------------------------ M12–M13

def plot_sequence_frequency(fig, result, ds, p):
    """M12：Top-N 完整序列（或压缩后的相位序列）频次条形图。"""
    frame = None
    for name, table in result.tables.items():
        if name.startswith("Top-"):
            frame = table
            break
    if frame is None or frame.empty:
        return
    ax = fig.add_subplot(111)
    labels = []
    for text in frame["sequence"].tolist():
        text = str(text)
        labels.append(text if len(text) <= 46 else text[:22] + " … " + text[-20:])
    values = frame["count"].to_numpy(dtype=float)
    bars = ax.barh(range(len(labels))[::-1], values, color=tokens()["purple"], height=0.6)
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelsize=pitched_fontsize(len(labels)))
    for bar, value, text in zip(bars, values, frame["sequence"].tolist()):
        detail = "（%s）" % text if len(str(text)) <= 46 else ""
        ax.annotate("%g%s" % (value, detail), (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize="small")
    ax.set_xlabel("学生数")
    ax.margins(x=0.25)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    declare_size(fig, len(labels) * BAR_H_IN * 1.4 + 1.0)
    _suptitle(fig, "Sequence Frequency｜Top %d" % len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.95))


def plot_mean_time(fig, result, ds, p):
    """M13：每个状态的平均占用时长 + 误差条（SD 或 SE，纯描述性）。"""
    frame = result.table()
    if frame.empty:
        return
    dimensions = list(dict.fromkeys(frame["dimension"].tolist()))
    axes = np.array(fig.subplots(len(dimensions), 1, squeeze=False)).ravel()
    err_col = "se_s" if p.get("error_bar") == "se" else "sd_s"
    widest = 1
    for ax, dimension in zip(axes, dimensions):
        part = frame[frame["dimension"] == dimension]
        codes = part["code"].astype(str).tolist()
        means = part["mean_s"].to_numpy(dtype=float)
        errors = part[err_col].to_numpy(dtype=float)
        widest = max(widest, len(codes))
        mapping = color_map(ds, dimension)
        ax.bar(range(len(codes)), means, yerr=errors, capsize=4,
               color=[mapping.get(c, tokens()["purple"]) for c in codes],
               error_kw=dict(ecolor=tokens()["ink2"], lw=1.1))
        ax.set_xticks(range(len(codes)))
        ax.set_xticklabels(codes)
        fit_x_labels(ax, codes)
        ax.set_ylabel("平均占用时长 (s)")
        ax.set_title("维度 %s" % dimension, fontsize=11.5, loc="left")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        if p.get("show_values", True):
            # 数值放在误差条上沿之上：放在柱头会被误差线穿过，等于把数字划掉
            place_value_labels(ax, range(len(codes)),
                               [m + e for m, e in zip(means, errors)],
                               ["%g" % round(value, 1) for value in means])
        ax.margins(y=0.22)
    declare_size(fig, len(dimensions) * FACET_H_IN * 1.35 + 0.9,
                 width_in=widest * COLUMN_W_IN + 1.6)
    _suptitle(fig, "Mean Time｜误差条为 %s（描述性，不做显著性检验）"
              % ("SE" if err_col == "se_s" else "SD"))
    fig.tight_layout(rect=(0, 0, 1, 0.95))


# ------------------------------------------------------------------ M14

def plot_spells(fig, result, ds, p):
    """M14：spell 时长的 box / violin / dot 分布，按编码分组。"""
    detail = result.tables.get("Spell 明细")
    if detail is None or detail.empty:
        return
    units = list(detail.groupby(["student_id", "dimension"], sort=False))
    nrows, ncols = _grid(len(units))
    axes = np.array(fig.subplots(nrows, ncols, squeeze=False)).ravel()
    for ax in axes[len(units):]:
        ax.set_visible(False)
    style = p.get("plot_style", "box")
    for ax, (_, part) in zip(axes, units):
        dimension = part["dimension"].iloc[0]
        mapping = color_map(ds, dimension)
        codes = [c for c in ds.codes_by_dim.get(dimension, []) if c in set(part["code"])]
        data = []
        labels = []
        for code in codes:
            values = part[part["code"] == code]["duration_s"].to_numpy(dtype=float)
            if values.size:
                data.append(values)
                labels.append(code)
        positions = list(range(len(data)))
        if not data:
            ax.set_visible(False)
            continue
        colors = [mapping.get(c, tokens()["purple"]) for c in labels]
        if style == "violin" and len(data) > 1:
            parts = ax.violinplot(data, positions=positions, showmeans=True)
            for body, color in zip(parts["bodies"], colors):
                body.set_facecolor(color)
                body.set_alpha(0.55)
        elif style == "dot":
            for idx, (values, color) in enumerate(zip(data, colors)):
                jitter = np.random.RandomState(idx).uniform(-0.12, 0.12, values.shape)
                ax.scatter(np.full(values.shape, idx) + jitter, values, s=14,
                           color=color, alpha=0.75)
        else:
            box = ax.boxplot(data, positions=positions, patch_artist=True, showfliers=False)
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        fit_x_labels(ax, labels)
        ax.set_ylabel("spell 时长 (s)")
        ax.set_title(_unit_label(part["student_id"].iloc[0], dimension), fontsize=11.5)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    declare_size(fig, nrows * FACET_H_IN * 1.3 + 0.9)
    _suptitle(fig, "Spell / Run｜连续段时长分布（%s）" % style)
    fig.tight_layout(rect=(0, 0, 1, 0.95))


# ------------------------------------------------------------------ M15

def plot_cooccurrence(fig, result, ds, p):
    """M15：共现 / 同步矩阵热图。"""
    names = [k for k in result.tables if k != "长表"]
    if not names:
        return
    axes = np.array(fig.subplots(1, len(names), squeeze=False)).ravel()
    for ax, name in zip(axes, names):
        table = result.tables[name]
        if "合计" in table.columns:
            table = table.drop(columns=["合计"])
        matrix = table.set_index(table.columns[0])
        _heatmap_one(ax, matrix, name, p, note=str(table.columns.name or ""))
    tallest = max((len(result.tables[name]) for name in names), default=1)
    declare_size(fig, max(FACET_H_IN, tallest * CELL_H_IN + 1.3) + 0.9)
    _suptitle(fig, "%s｜矩阵总和 = 共同参与统计的有效时间点数" % result.method_name)
    fig.tight_layout(rect=(0, 0, 1, 0.93))


# ------------------------------------------------------------------ M16

def plot_alluvial(fig, result, ds, p):
    """M16：阶段（time-bin）之间的学生路径，matplotlib 自绘 Alluvial。"""
    flows = result.tables.get("阶段路径流量")
    nodes = result.tables.get("阶段节点")
    if flows is None or flows.empty or nodes is None or nodes.empty:
        return
    phases = list(dict.fromkeys(nodes["phase"].tolist()))
    codes = list(dict.fromkeys(nodes["code"].tolist()))
    dimension = (result.meta.get("filter", {}) or {}).get("dimensions") or [""]
    mapping = color_map(ds, dimension[0])
    counts = {(row["phase"], row["code"]): float(row["students"])
              for _, row in nodes.iterrows()}
    totals = {phase: sum(v for (p_name, _), v in counts.items() if p_name == phase)
              for phase in phases}
    max_total = max([v for v in totals.values() if v] or [1.0])
    unit = 1.0 / max_total
    gap = 0.035
    node_x = {phase: idx for idx, phase in enumerate(phases)}
    node_width = 0.11

    ax = fig.add_subplot(111)
    geometry = {}
    for phase in phases:
        y = 0.0
        for code in codes:
            value = counts.get((phase, code), 0.0)
            if value <= 0:
                continue
            geometry[(phase, code)] = (node_x[phase], y, y + value * unit)
            y += value * unit + gap

    link_cursor = {}
    t = tokens()
    for _, row in flows.sort_values(["from_phase", "to_code"]).iterrows():
        key = (row["from_phase"], row["from_code"])
        to_key = (row["to_phase"], row["to_code"])
        if key not in geometry or to_key not in geometry:
            continue
        value = float(row["students"]) * unit
        offset = link_cursor.get(key, 0.0)
        x0 = node_x[row["from_phase"]] + node_width / 2
        x1 = node_x[row["to_phase"]] - node_width / 2
        y0_top = geometry[key][2] - offset
        y1_top = geometry[to_key][2] - offset
        y0_bottom = y0_top - value
        y1_bottom = y1_top - value
        link_cursor[key] = offset + value
        ax.add_patch(_bezier_band(x0, x1, y0_bottom, y0_top, y1_bottom, y1_top,
                                  mapping.get(row["from_code"], t["purple"])))
    for (phase, code), (x, y0, y1) in geometry.items():
        ax.add_patch(Rectangle((x - node_width / 2, y0), node_width, y1 - y0,
                               facecolor=mapping.get(code, t["ink1"]),
                               edgecolor=t["card"], linewidth=0.6))
        ax.text(x - node_width / 2 - 0.02, (y0 + y1) / 2, "%s" % code,
                ha="right", va="center", fontsize="small")
        if y1 - y0 >= 0.045:
            ax.text(x + node_width / 2 + 0.02, (y0 + y1) / 2, "%d 人" % counts[(phase, code)],
                    ha="left", va="center", fontsize="x-small", color=t["ink3"])
    for phase in phases:
        short, _, bounds = phase.partition(" ")
        ax.text(node_x[phase], 1.02, short, ha="center", va="bottom", fontsize=11)
        ax.text(node_x[phase], -0.055, bounds or "", ha="center", va="top",
                fontsize="x-small", color=t["ink3"])
    ax.set_xlim(-0.38, len(phases) - 1 + 0.3)
    ax.set_ylim(-0.14, 1.14)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    rect = add_legend(fig, [(ax, [(c, mapping[c]) for c in codes if c in mapping])], p,
                      base_rect=(0.03, 0.02, 0.97, 0.92))
    declare_size(fig, 4.6 + 0.1 * len(codes))
    _suptitle(fig, "Alluvial｜阶段性路径展示（宽度 = 学生数），不构成转移推断")
    fig.tight_layout(rect=rect)


def _bezier_band(x0, x1, y0_bottom, y0_top, y1_bottom, y1_top, color):
    """左右两条三次贝塞尔曲线围出的流带。"""
    xi = np.linspace(0.0, 1.0, 40)
    cx0, cx1 = x0 + (x1 - x0) * 0.42, x0 + (x1 - x0) * 0.58
    def curve(ya, yb):
        mt = 1 - xi
        return (mt ** 3 * ya + 3 * mt ** 2 * xi * ya
                + 3 * mt * xi ** 2 * yb + xi ** 3 * yb)
    top = curve(y0_top, y1_top)
    bottom = curve(y0_bottom, y1_bottom)
    xs = np.concatenate([x0 + (x1 - x0) * xi, (x0 + (x1 - x0) * xi)[::-1]])
    ys = np.concatenate([top, bottom[::-1]])
    return Polygon(np.column_stack([xs, ys]), closed=True, facecolor=color,
                   edgecolor="none", alpha=0.38)


# ------------------------------------------------------------------ M17

def plot_state_entropy(fig, result, ds, p):
    """M17：逐单元堆叠条形，每段是一个状态的贡献 -p ln p，条长合计就是 H。"""
    frame = result.tables.get("各状态概率贡献")
    summary = result.tables.get("状态熵")
    if frame is None or frame.empty or summary is None or summary.empty:
        return
    norm = p.get("value", "entropy") == "H_norm"
    units = list(summary[["student_id", "dimension", "entropy", "H_norm",
                          "n_states", "N"]].to_dict("records"))
    nrows, ncols = _grid(len(units))
    axes = np.array(fig.subplots(nrows, ncols, squeeze=False)).ravel()
    for ax in axes[len(units):]:
        _no_axes(ax)
    for ax, unit in zip(axes, units):
        part = frame[(frame["student_id"] == unit["student_id"])
                     & (frame["dimension"] == unit["dimension"])]
        codes = part["code"].astype(str).tolist()
        mapping = color_map(ds, unit["dimension"])
        # 选归一熵时整条除以 ln K：各段同比缩小，合计长度正好变成 H_norm
        scale = 1.0 / np.log(unit["n_states"]) if (norm and unit["n_states"] > 1) else 1.0
        values = part["contribution"].to_numpy(dtype=float) * scale
        total = float(values.sum())
        left = 0.0
        for code, value in zip(codes, values):
            ax.barh([0], [value], left=left, color=mapping.get(code, tokens()["purple"]),
                    height=0.5, edgecolor=tokens()["card"], linewidth=0.6)
            # 段宽不够一个字就不写字：否则两个编码名会叠成乱码
            if p.get("show_values", True) and total and value > total * 0.09:
                ax.text(left + value / 2.0, 0, code, ha="center", va="center",
                        fontsize=tick_fontsize(len(codes), base=8.2), color=tokens()["card"])
            left += float(value)
        ax.set_xlim(0, max(left * 1.06, 1e-6))
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("归一熵 H/ln K（各段标注编码）" if norm else "熵 H（ln 口径，各段标注编码）")
        ax.set_title("%s\nH=%.3f｜H_norm=%.3f｜K=%d｜N=%d"
                     % (_unit_label(unit["student_id"], unit["dimension"]),
                        unit["entropy"], unit["H_norm"], unit["n_states"], unit["N"]),
                     fontsize=10.5)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    declare_size(fig, nrows * FACET_H_IN + 0.9)
    _suptitle(fig, "Entropy｜条长 = 各状态 -p ln p 之和（ln 口径，K 为实际出现状态数）")
    fig.tight_layout(rect=(0, 0, 1, 0.94))


# ------------------------------------------------------------------ M18

def plot_switch_retention(fig, result, ds, p):
    """M18：每个单元一行双条（切换率 / 保持率），两条共用分母所以必然互补为 1。"""
    frame = result.tables.get("切换与保持")
    if frame is None or frame.empty:
        return
    rows = frame.to_dict("records")
    ax = fig.add_subplot(111)
    positions = np.arange(len(rows), dtype=float)
    ax.barh(positions + 0.19, [r["switch_rate"] for r in rows], height=0.34,
            color=tokens()["purple"], label="switch_rate（切换）")
    ax.barh(positions - 0.19, [r["self_transition"] for r in rows], height=0.34,
            color=tokens()["success"], label="self_transition（保持）")
    if p.get("show_values", True):
        for row, y in zip(rows, positions):
            ax.annotate("%.3f" % row["switch_rate"], (row["switch_rate"], y + 0.19),
                        xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
            ax.annotate("%.3f" % row["self_transition"], (row["self_transition"], y - 0.19),
                        xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(positions)
    ax.set_yticklabels([_unit_label(r["student_id"], r["dimension"])
                        + "｜n=%d｜runs=%d｜均长%.1f"
                        % (r["n_transitions"], r["n_runs"], r["mean_run_length"])
                        for r in rows])
    ax.tick_params(axis="y", labelsize=pitched_fontsize(len(rows)))
    ax.invert_yaxis()
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("比率（分母 = 相邻有效观测对数 T-1）")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    declare_size(fig, max(len(rows), 1) * 0.62 + 1.5)
    _suptitle(fig, "Switch & Retention｜两条同分母互补为 1，mean_run_length 以时间格计")
    fig.tight_layout(rect=(0, 0, 1, 0.95))


# ------------------------------------------------------------------ M19

def _prob_matrix_one(ax, codes, matrix, title, p):
    """行内归一的转移条件分布热图：每行相加 = 1，未观察到的转移留空不涂色。"""
    data = np.asarray(matrix, dtype=float)
    img = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap=value_cmap(),
                    vmin=0, vmax=1)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([str(c) for c in codes])
    ax.tick_params(axis="y", labelsize=pitched_fontsize(len(codes)))
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels([str(c) for c in codes])
    fit_x_labels(ax, [str(c) for c in codes], base=8.6)
    ax.set_title(title, fontsize=10.5)
    if p.get("show_values", True) and len(codes) <= 12:
        for (row, col), value in np.ndenumerate(data):
            if value != value:
                continue
            ax.text(col, row, "%.2f" % value, ha="center", va="center", fontsize=7.6,
                    color=tokens()["ink1"] if value < 0.6 else tokens()["card"])
    ax.set_xlabel("下一状态")
    ax.set_ylabel("当前状态")
    return img


def _m19_compare(fig, rows, p):
    """M19 默认视图：逐单元 H(Y) 与 H(Y|X) 成对条，条上标 reduction。"""
    ax = fig.add_subplot(111)
    positions = np.arange(len(rows), dtype=float)
    ax.bar(positions - 0.2, [r["H_Y"] for r in rows], width=0.4,
           color=tokens()["purple"], label="H(Y)（无条件熵）")
    ax.bar(positions + 0.2, [r["H_Y_given_X"] for r in rows], width=0.4,
           color=tokens()["success"], label="H(Y|X)（条件熵）")
    labels = [_unit_label(r["student_id"], r["dimension"]) for r in rows]
    if p.get("show_values", True):
        place_value_labels(ax, list(positions - 0.2), [r["H_Y"] for r in rows],
                           ["%.3f" % r["H_Y"] for r in rows])
        place_value_labels(ax, list(positions + 0.2), [r["H_Y_given_X"] for r in rows],
                           ["Δ%.3f" % r["reduction"] for r in rows])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    fit_x_labels(ax, labels)
    ax.set_ylabel("熵（ln）")
    ax.set_ylim(0, max(max(r["H_Y"] for r in rows), max(r["H_max"] for r in rows)) * 1.24)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    declare_size(fig, len(rows) * COLUMN_W_IN + 2.6,
                 width_in=len(rows) * COLUMN_W_IN * 2 + 2.0)
    _suptitle(fig, "Transition Entropy｜Δ = reduction = H(Y) - H(Y|X)，越大越可预测")
    fig.tight_layout(rect=(0, 0, 1, 0.94))


def _m19_decompose(fig, ds, rows, breakdown, p):
    """M19 分解视图：逐前态的 p(x)·H(Y|X=x)，相加 = H(Y|X)。"""
    frames = []
    for row in rows:
        part = breakdown[(breakdown["student_id"] == row["student_id"])
                         & (breakdown["dimension"] == row["dimension"])]
        items = part.to_dict("records")
        items.sort(key=lambda item: -item["contribution"])
        frames.append((row, [item["from_code"] for item in items],
                       [item["contribution"] for item in items]))
    nrows, ncols = _grid(len(frames))
    budgets = _row_budgets([len(codes) for _, codes, _ in frames], ncols, 0.34, 1.2)
    axes = np.array(fig.subplots(nrows, ncols, squeeze=False,
                                 gridspec_kw={"height_ratios": budgets})).ravel()
    for ax in axes[len(frames):]:
        _no_axes(ax)
    for ax, (row, codes, values) in zip(axes, frames):
        mapping = color_map(ds, row["dimension"])
        ax.barh(range(len(codes)), values, color=[mapping.get(c, tokens()["purple"])
                                                  for c in codes], height=0.62)
        ax.set_yticks(range(len(codes)))
        ax.set_yticklabels(codes)
        ax.tick_params(axis="y", labelsize=pitched_fontsize(len(codes)))
        ax.invert_yaxis()
        ax.set_xlabel("贡献 p(x)·H(Y|X=x)")
        ax.set_title("%s\nH(Y|X)=%.3f｜H(Y)=%.3f｜K=%d"
                     % (_unit_label(row["student_id"], row["dimension"]),
                        row["H_Y_given_X"], row["H_Y"], row["n_states"]), fontsize=10.5)
        if p.get("show_values", True):
            _annotate_bars(ax, list(ax.patches), values, digits=3)
        ax.margins(x=0.2)
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    declare_size(fig, sum(budgets) + 0.9)
    _suptitle(fig, "Transition Entropy｜逐前态对 H(Y|X) 的贡献（相加即条件熵）")
    fig.tight_layout(rect=(0, 0, 1, 0.94))


def _m19_matrix(fig, rows, detail, p):
    """M19 热图视图：行归一的 P(下一状态 | 当前状态)。"""
    panels = []
    for row in rows:
        part = detail[(detail["student_id"] == row["student_id"])
                      & (detail["dimension"] == row["dimension"])]
        codes = sorted(set(part["from_code"]) | set(part["to_code"]), key=str)
        if len(codes) < 2:
            continue
        slot = {code: index for index, code in enumerate(codes)}
        # 没发生过的转移留 NaN：涂成 0 会让「不可能」和「概率很小」看起来一样
        matrix = np.full((len(codes), len(codes)), np.nan)
        for item in part.to_dict("records"):
            if item["count"]:
                matrix[slot[item["from_code"]], slot[item["to_code"]]] = item["p_to_given_from"]
        panels.append((row, codes, matrix))
    if not panels:
        return False
    nrows, ncols = _grid(len(panels))
    budgets = _row_budgets([len(codes) for _, codes, _ in panels], ncols,
                           CELL_H_IN * 1.6, 1.3)
    axes = np.array(fig.subplots(nrows, ncols, squeeze=False,
                                 gridspec_kw={"height_ratios": budgets})).ravel()
    for ax in axes[len(panels):]:
        _no_axes(ax)
    for ax, (row, codes, matrix) in zip(axes, panels):
        _prob_matrix_one(ax, codes, matrix,
                         "%s｜H(Y|X)=%.3f｜可预测性=%.3f"
                         % (_unit_label(row["student_id"], row["dimension"]),
                            row["H_Y_given_X"], row["predictability"]), p)
    declare_size(fig, sum(budgets) + 0.9, width_in=ncols * 3.6)
    _suptitle(fig, "Transition Entropy｜条件转移概率热图（每行相加 = 1，空白 = 未观察到）")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return True


def plot_transition_entropy(fig, result, ds, p):
    """M19 三种视图：compare（默认）/ decompose / matrix。后两种需明细表，缺失时退回 compare。"""
    frame = result.tables.get("条件熵")
    detail = result.tables.get("转移条件分布")
    breakdown = result.tables.get("前态熵分解")
    if frame is None or frame.empty:
        return
    rows = frame.to_dict("records")
    view = p.get("view", "compare")
    if (view == "matrix" and detail is not None and not detail.empty
            and _m19_matrix(fig, rows, detail, p)):
        return
    if view == "decompose" and breakdown is not None and not breakdown.empty:
        _m19_decompose(fig, ds, rows, breakdown, p)
        return
    _m19_compare(fig, rows, p)
