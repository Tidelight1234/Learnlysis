# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""IND / GRP 全部方法的绘图层。

统一签名 ``plot(fig_holder, result, ctx, params)`` 太宽泛，实际按方法各写一个函数、
再由 ``PLOTTERS`` 按 method_id 派发；工作台只调 `render(method_id, result, ctx, params)`
拿到 ``{图名: Figure}``，不关心画了什么。

四条硬规矩：
- 一律 `matplotlib.figure.Figure` 直接构造，**不用 pyplot**（工作台常驻画布 + 线程安全）；
- 配色只取 `theme.CATEGORY_COLORS`，文字/网格取 `theme.tokens()`，最后 `theme.sync_figure`；
- 每张图都必须能在数据缺失时给出"没有可画"的空白图，而不是抛异常；
- **放不下就把画布撑大，不许缩字藏标签**：尺寸口径与「统计及可视化」一致（见下），
  超出的部分交给预览区滚动；图例位置是用户参数，不许压住坐标区。

口径为什么和 statviz 分开写两份实现：两边的内容形状不同（这边大量是 n×n 矩阵与
网络环布局，那边是分面色带），但 `declare_size / claim_margin / planned_size` 三个
动作的语义必须逐字对齐，否则同一个「图形尺寸」参数在两个工作台上会长出两种行为。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import theme

MAX_PANELS = 6           # 一次最多画 6 个分面，再多就读不出来了
MAX_ITEMS = 15           # 条形图最多 15 项（只用于“取前 N”的排名图）
MAX_BARS = 48            # 一条 = 一个两两对的图不排名：能展就全展（高度靠撑画布）
# ---- 画布尺寸：按内容量尺，放不下就撑大画布去滚动，而不是压缩标签 ----
MIN_FIG_H_IN = 3.2       # 预览图再矮也留够读得下轴标签的高度
MAX_FIG_H_IN = 26.0      # 纵向上限：与 statviz 同档，再高就该让用户筛选项
MAX_FIG_W_IN = 34.0      # 横向上限：顶到上限还放不下才允许抽稀（见 fit_category_labels）
ROW_H_IN = 0.30          # 一条水平条（含它那格类目标签）的最小高度
BAND_H_IN = 0.62         # 一行堆叠时间带的最小高度
MIN_ROW_H_IN = 1.5       # 多行网格里一行的下限（带标题与刻度就得占这么多）
PANEL_DECOR_IN = 0.72    # 一个分面行自己占的标题 + 底部刻度带 + 行间隙（tight_layout 从坐标区里扣）
FACET_H_IN = 2.0         # 一个坐标面板（柱状 / 折线 / 散点）的最小高度
CELL_IN = 0.30           # 热图一格的最小边长：低于此格内数字必互相压字
COLUMN_W_IN = 0.46       # 竖向柱状图一根的最小宽度（含旋转后的类目标签）
LABEL_GAP_IN = 0.10      # 相邻类目标签之间的最小空隙：贴太紧就是一整条黑带
AXIS_W_FRACTION = 0.70   # 坐标区宽 ≈ 画布宽 × 它：量标签占用时按它反推画布要多宽
AXIS_H_FRACTION = 0.74   # 同上，纵向
LEGEND_FONTSIZE = 8.5    # 图例字号：与 statviz 同一档，不再随条目数缩
LEGEND_ROW_IN = 0.17     # 图例一行的实高（含 labelspacing）：算「竖着装得下几条」
LEGEND_OUTSIDE = ("below", "right")   # 这两个位置要单独给图例留一条边距
CBAR_LABEL_IN = 0.38     # 色条自己的刻度标签要吃掉的宽度
SUPTITLE_IN = 0.34       # 图名（suptitle）要吃掉的一条顶边
DEFAULT_AVAIL_IN = (9.0, 5.0)  # 宿主没给视口时的名义视口（测试脚本、非 UI 调用）


# ------------------------------------------------------------------ 公共小件

def colors():
    return list(theme.CATEGORY_COLORS)


def ink(key="ink2"):
    return theme.tokens().get(key, "#5C6070")


def color_map(universe):
    """编码 → 颜色：按出现顺序取色，超过 8 个色值时循环（保持同一编码同一色）。"""
    palette = colors()
    return {str(item): palette[index % len(palette)]
            for index, item in enumerate(universe)}


def new_figure(title="", width=7.2, height=4.0, panels=1, size=None):
    fig = Figure(figsize=(width, height), dpi=110)
    if size:
        # 两遍绘制的第二遍：先把画布调成上一遍量出来的尺寸，排版（旋转角、抽稀、
        # tight_layout）才能按真实宽度算，而不是先挤一遍再拉开
        fig.set_size_inches(float(size[0]), float(size[1]))
    axes = [fig.add_subplot(panels, 1, position + 1) for position in range(panels)]
    # 构造时给的那一档就是「一行面板的基准高」：存下来当量尺基线，否则两遍绘制
    # 会把上一遍量出的总高再乘一次行数，图一遍比一遍大
    fig._ly_row_h = float(height)
    reset_size(fig)
    if title:
        fig.suptitle(title, color=ink("ink1"), fontsize=11)
    for ax in axes:
        style(ax)
    theme.sync_figure(fig)
    return fig, (axes[0] if panels == 1 else axes)


def _grid_rows(fig):
    """子图最多几行：一行 4.2 in 够用，两行就得翻倍，不然标签必重。"""
    rows = 1
    for ax in fig.axes:
        spec = getattr(ax, "get_subplotspec", lambda: None)()
        geometry = getattr(spec, "get_geometry", lambda: None)()
        try:
            rows = max(rows, int(geometry[0]))
        except (TypeError, ValueError):
            continue
    return rows


# ------------------------------------------------------ 内容量尺：declare / claim / plan

def reset_size(fig):
    """每次重绘前清掉上一次声明，否则图会只增不减。"""
    fig._ly_need = (0.0, 0.0)
    fig._ly_margin = (0.0, 0.0)
    fig._ly_bands = {}
    fig._ly_row_need = {}


def declare_size(fig, height_in=0.0, width_in=0.0):
    """内容声明：这些坐标区至少要多高、画布至少要多宽（多面板取最大，不累加）。

    绘图函数只负责报「本图内容需要多少」，实际尺寸由预览区与上限算（见 planned_size）。
    同行并列的面板要把本行的总宽乘上列数再报，取最大才不会被当成「只需要一块」。
    多行分面要逐行报（见 claim_row_height）：行与行是叠起来的，只报最大值会少算其他行。
    """
    need = getattr(fig, "_ly_need", (0.0, 0.0))
    fig._ly_need = (max(need[0], float(width_in or 0.0)),
                    max(need[1], float(height_in or 0.0)))


def claim_margin(fig, height_in=0.0, width_in=0.0):
    """边距式需求（图例）：这一部分是内容之外另加的，必须累加而不是取最大。

    26 个编码的图例放正下方时要占三四行，拿 declare_size 报会被面板高度吃掉，
    结果图例又压回坐标区上 —— 那正是本次要修的病。
    """
    margin = getattr(fig, "_ly_margin", (0.0, 0.0))
    fig._ly_margin = (margin[0] + max(float(width_in or 0.0), 0.0),
                      margin[1] + max(float(height_in or 0.0), 0.0))


def claim_row_margin(fig, row, height_in=0.0):
    """按子图行归集标签带高度：同一行的几个面板共用一条底边，行与行才累加。

    逐面板累加会把一排 4 个面板的旋转标签算成四条底边，白撑高三倍。
    """
    bands = getattr(fig, "_ly_bands", {})
    key = max(int(row or 1), 1)
    bands[key] = max(bands.get(key, 0.0), float(height_in or 0.0))
    fig._ly_bands = bands


def claim_row_height(fig, row, height_in=0.0):
    """某一行子图自己要多高：同行取最大、行与行累加（`declare_size` 只能取最大）。

    多行分面里最高那一行的需求（例如 15 条两两 agreement 的条图）拿 `declare_size`
    报会被当成全图需求，行与行平分之后本行反而不够，标签必叠。
    """
    rows = getattr(fig, "_ly_row_need", {})
    key = max(int(row or 1), 1)
    rows[key] = max(rows.get(key, 0.0), float(height_in or 0.0))
    fig._ly_row_need = rows


def planned_size(fig, avail_w_in, avail_h_in=None):
    """纯算不落地：按内容声明与预览区得出目标尺寸（英寸）。"""
    need_w, need_h = getattr(fig, "_ly_need", (0.0, 0.0))
    margin_w, margin_h = getattr(fig, "_ly_margin", (0.0, 0.0))
    rows = _grid_rows(fig)
    # 逐行声明过的就按行累加（行与行是叠起来的，不是取最大）
    row_h = sum(getattr(fig, "_ly_row_need", {}).values())
    # 基线取构造时的单行高（拿当前总高乘行数会逐遍自我放大）
    base = getattr(fig, "_ly_row_h", float(fig.get_size_inches()[1]))
    if rows <= 1:
        floor = max(base, MIN_FIG_H_IN)
    else:
        # 多行不能拿「单面板基准高 × 行数」当地板：那一档是「一张图」的高，不是
        # 「一格分面」的高，六条色带会被撑成六块面板（GRP01 实测 8.2 → 25.2 in），
        # 只保一条「一行至少还读得下」的地板
        floor = max(rows * MIN_ROW_H_IN, need_h, row_h, MIN_FIG_H_IN)
    if avail_h_in:
        floor = max(floor, float(avail_h_in) * AXIS_H_FRACTION)
    band_h = sum(getattr(fig, "_ly_bands", {}).values())
    height = min(max(floor, need_h, row_h) + band_h + margin_h, MAX_FIG_H_IN)
    width = min(max(float(avail_w_in), need_w) + margin_w, MAX_FIG_W_IN)
    return width, height


def fit_figure(fig, avail_w_in, avail_h_in=None):
    """把目标尺寸落到画布上：宽不够就加宽、高不够就加高，预览区两条滚动条自会出现。

    子图位置存的是比例坐标，所以画完再改物理尺寸不会错位；但字号是绝对值（pt），
    拉宽只会更敞亮，缩小才会叠字 —— 因此这里只允许“变大”。
    """
    size = planned_size(fig, avail_w_in, avail_h_in)
    # 与当前画布取大：两遍绘制的第二遍上来就是按目标尺寸建的，若这一遍的内容刚好
    # 被这个宽度满足（不再 declare_size），照计划值写回会把它缩回去，标签又被裁掉
    width, height = max(size[0], float(fig.get_figwidth())), max(size[1], float(fig.get_figheight()))
    fig.set_size_inches(width, height, forward=True)
    return width, height


def style(ax):
    ax.tick_params(colors=ink(), labelcolor=ink(), which="both")
    ax.title.set_color(ink("ink1"))
    ax.xaxis.label.set_color(ink())
    ax.yaxis.label.set_color(ink())
    for spine in ax.spines.values():
        spine.set_color(ink("line"))
    ax.set_axisbelow(True)
    ax.grid(True, color=ink("line"), linewidth=0.6, alpha=0.7)


# -------------------------------------------- 类目标签：先撑画布，撑饱了才准抽稀

def text_width_in(label, fontsize):
    """近似量一段标签多宽：中日韩一个字吃满一个字号，ASCII 约 0.55 个。

    故意不依赖 renderer：测量遍是在故意给错的画布尺寸上跑的，量不到真值；
    估算只要偏保守（宁可多留）就够用。多行标签取最宽的那一行。
    """
    lines = str(label).split("\n") or [""]
    units = max(sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in line) for line in lines)
    return units * float(fontsize) / 72.0


def _row_of(ax):
    """面板在第几行（1 起，tight_layout 后同一行共用底边）；拿不到就当第一行。

    `get_geometry()` 给的是 (总行数, 总列数, 起始格, 结束格)，第三项是**扁平行号**（0 基）：
    直接拿总行数当行号会把「图有几行」当成「在第几行」，多行分面的底边全挤在同一键上。
    """
    spec = getattr(ax, "get_subplotspec", lambda: None)()
    geometry = getattr(spec, "get_geometry", lambda: None)()
    try:
        _rows, n_cols, start, _stop = (int(value) for value in geometry)
    except (TypeError, ValueError):
        return 1
    return start // max(n_cols, 1) + 1


def fit_category_labels(ax, labels, base=8.4, floor=6.8, columns=1, gap_in=LABEL_GAP_IN,
                        max_labels=0, positions=None):
    """横轴类目标签：平排 → 旋转 45° → 旋转 90° → 加宽画布，顶到上限才抽稀。

    每个刻度都是一个编码 / 一个分析单元，抽稀等于把数据藏起来，所以默认先把
    画布按需求加宽（`declare_size`），只有宽度已顶到 ``MAX_FIG_W_IN`` 仍放不下时
    才退而抽稀；时间窗口轴可给 ``max_labels`` 主动稀（那上面每个刻度不是对象而是窗口）。

    ``columns`` 是本行并列的面板数（量的时候要乘上去，否则一排 6 个热图会按单面板算宽）。
    ``positions`` 是刻度实际所在坐标（分组柱状图的柱心不是整数），不给就按 0..n-1。
    返回实际使用的步长（1 = 一个没省）。
    """
    labels = [str(item) for item in labels]
    count = len(labels)
    if not count:
        return 1
    fig = ax.figure
    columns = max(int(columns or 1), 1)
    ticks = list(range(count)) if positions is None else [float(v) for v in positions]
    size = max(floor, base - 0.10 * max(0, count - 16))
    widest = max(labels, key=len)
    flat_in = text_width_in(widest, size) + gap_in
    # 一个面板现在有多少横向空间给坐标区（不是画布全宽）
    per_ax_in = max(fig.get_figwidth() / columns * AXIS_W_FRACTION, 1.2)
    rotation, pitch = 0, flat_in
    if count * pitch > per_ax_in:
        rotation, pitch = 45, flat_in * 0.71 + gap_in
    if count * pitch > per_ax_in and len(widest) > 6:
        rotation, pitch = 90, text_width_in(widest[:1], size) + gap_in
    stride = 1
    if max_labels and count > max_labels:
        stride = int(np.ceil(count / float(max_labels)))
    else:
        # 先把画布撑到“一个都不省”需要的宽度，再看现在到底撑到了多少
        declare_size(fig, width_in=min(count * pitch * columns / AXIS_W_FRACTION,
                                       MAX_FIG_W_IN))
        room = max(fig.get_figwidth() / columns * AXIS_W_FRACTION, 1.2)
        if count * pitch > room * 1.02:
            stride = int(np.ceil(count * pitch / room))
    kept = list(range(0, count, stride))
    ax.set_xticks([ticks[index] for index in kept])
    ax.set_xticklabels([labels[index] for index in kept], rotation=rotation,
                       ha="right" if rotation else "center",
                       va="center" if rotation == 90 else "top", fontsize=size)
    if rotation:
        # 旋转后标签往下探出坐标区：给本行声明一条底边，撑高画布而不是让 tight_layout
        # 把坐标区压扁
        depth = text_width_in(max((labels[index] for index in kept), key=len), size)
        claim_row_margin(fig, _row_of(ax), (depth * 0.71 if rotation == 45 else depth) + 0.28)
    return stride


def fit_bar_rows(ax, count, row_h_in=ROW_H_IN, base=8.4, floor=6.6, labels=None, columns=1):
    """水平条图：一行一条，高度按条数撑；y 侧标签太长就把画布加宽。

    这里不缩字也不省条：条图的每一条就是一个模式 / 一个学生，省了就是丢数据。
    ``columns`` 是本行并列的面板数：标签列得按面板数分摊画布宽，不然一排两个面板
    会各自以为整张画布的左边距都是给自己的。
    高度需求记在本行头上（`claim_row_height`）：多行分面里拿整图最大报，行与行
    一平分就不够了，而这正是条图最容易被挤到的那张。
    """
    fig = ax.figure
    columns = max(int(columns or 1), 1)
    count = max(int(count), 1)
    claim_row_height(fig, _row_of(ax), count * row_h_in + 0.9)
    if labels:
        size = max(floor, base - 0.10 * max(0, count - 16))
        widest = max(labels, key=len)
        need_in = text_width_in(widest, size) + 0.30
        have_in = max(fig.get_figwidth() * (1.0 - AXIS_W_FRACTION) / columns, 0.4)
        if need_in > have_in:
            # 目标宽只跟「标签要多宽」有关（不乘当前画布宽）：写成跟当前宽相关的话，
            # 两遍绘制会一遍比一遍宽，永远收不敛
            declare_size(fig, width_in=min(need_in * columns / (1.0 - AXIS_W_FRACTION),
                                           MAX_FIG_W_IN))
        ax.set_yticklabels(labels, fontsize=size)
    return count


def matrix_room(fig, n_rows, n_cols, panels_wide=1, label_in=0.95, row=None):
    """n×m 矩阵热图：每格至少 ``CELL_IN``，横竖都按内容撑，标签列/行另算。

    ``panels_wide`` 是这个热图在同一行里并列几块；``row`` 是它在第几行（多行分面
    才传）：逐行报才不会被当成「整张图只要这一块」，也不会把几块矩阵摞成一柱。
    """
    side_w = n_cols * CELL_IN + label_in
    side_h = n_rows * CELL_IN + label_in
    if row:
        claim_row_height(fig, row, side_h)
    else:
        declare_size(fig, height_in=side_h)
    declare_size(fig, width_in=side_w * max(int(panels_wide or 1), 1))
    return side_w, side_h


def label_column_in(labels, fontsize):
    """一列类目标签要多宽：最长那条 + 留白（矩阵面板的行名、条图的 y 名都用）。"""
    return max((text_width_in(item, fontsize) for item in labels), default=0.0) + 0.30


def row_fontsize(count, base=8.2, floor=6.4):
    """行数越多字号越小，但有下限：小于这一档就不再缩，改成把画布撑高。"""
    return max(floor, base - 0.06 * max(0, int(count) - 12))


def color_bar(fig, image, ax, fraction=0.046, pad=0.04):
    """给热图配一条色条：自建面板并挂在坐标区上（排版完再落位，见 tighten）。

    直接 `fig.colorbar(img, ax=ax)` 的话，色条是按当时的坐标区位置算出来的，
    而 tight_layout 会把坐标区重排回网格 —— 色条就停在原地被主图压住。
    所以这里自己建 cax、记下父坐标区，等排版完再贴着最终位置放下。
    """
    cax = fig.add_axes([0.9, 0.12, 0.02, 0.76])
    bar = fig.colorbar(image, cax=cax)
    ax._ly_cbar = cax
    ax._ly_cbar_geometry = (float(fraction), float(pad))
    cax.tick_params(labelsize=6.8, colors=ink(), labelcolor=ink())
    return bar


def _place_colorbars(fig):
    """排版完之后把色条重新贴到各自主图右边（只认 `_ly_cbar`，不猜野面板）。"""
    for ax in fig.axes:
        cax = getattr(ax, "_ly_cbar", None)
        if cax is None:
            continue
        fraction, pad = getattr(ax, "_ly_cbar_geometry", (0.046, 0.04))
        pos = ax.get_position()
        cax.set_position([pos.x1 + pos.width * pad, pos.y0,
                          pos.width * fraction, pos.height])


# ------------------------------------------------------------ 图例：给独立位置

def _legend_pairs(legend):
    """从已画好的 legend 里取出 (handle, label)，供重排时复用。"""
    handles = getattr(legend, "legend_handles", None)
    if handles is None:                      # matplotlib < 3.6 的老字段名
        handles = getattr(legend, "legendHandles", None) or []
    labels = [str(text.get_text()) for text in legend.get_texts()]
    return list(zip(list(handles), labels))


def collect_legends(fig):
    """把各面板自己 `ax.legend()` 出来的东西回收成 [(ax, pairs)]。

    绘图函数继续照自己的意思 `ax.legend(...)`（不要求人人都写图例），
    位置则一律由用户参数定：两套口径不会在某个面板里长出不一样的图例。
    """
    panels = []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None or not ax.get_visible():
            continue
        pairs = _legend_pairs(legend)
        if pairs:
            panels.append((ax, pairs))
        ax.legend_ = None                     # 先拆掉，否则下一步会和旧图例叠在一起
    for legend in list(fig.legends):
        legend.remove()                        # 重画时不留上一轮的图形级图例
    return panels


def _pair_key(handle, label):
    try:
        face = str(handle.get_facecolor())
    except Exception:                        # noqa: BLE001 标量 handle 拿不到面颜色
        face = ""
    return (label, face)


def _legend_ncol(labels, width_in):
    """按画布宽决定图例几列：宁可多几行，也不要一行拉到顶不动。"""
    widest = max(text_width_in(label, LEGEND_FONTSIZE) for label in labels) + 0.62
    return max(1, min(len(labels), 10, int(max(width_in, 1.0) // max(widest, 0.5))))


def apply_legends(fig, params, default="below"):
    """按用户的「图例位置」重放图例，返回 tight_layout 该用的 rect。

    图外两档（正下方 / 右侧）把各面板的条目去重合并成一张图形级图例，并给图例
    单独声明一条边距（见 claim_margin）；图内档位逐面板放（带半透底框）；
    “不显示”直接不放。
    """
    loc = (params or {}).get("legend_loc") or default
    panels = collect_legends(fig)
    rect = (0.0, 0.0, 1.0, 1.0)
    if loc == "none" or not panels:
        return rect
    if loc not in LEGEND_OUTSIDE:
        widest = 0
        for ax, pairs in panels:
            made = ax.legend([handle for handle, _ in pairs], [label for _, label in pairs],
                             loc=loc, fontsize=LEGEND_FONTSIZE, frameon=True,
                             framealpha=0.92, ncol=1, borderpad=0.5, labelspacing=0.35,
                             handlelength=1.1, handletextpad=0.5)
            if made is not None:
                # 图内图例是锚在坐标区里的：让它参与量尺的话，tight_layout 会为了
                # 装下它把坐标区压到装不下自己（“cannot make Axes height small enough”）
                made.set_in_layout(False)
            widest = max(widest, len(pairs))
        # 坐标区还是得长高一点：行内没空白，图例就是把曲线盖了
        claim_margin(fig, height_in=0.17 * widest)
        return rect
    seen, handles, labels = set(), [], []
    for _ax, pairs in panels:
        for handle, label in pairs:
            key = _pair_key(handle, label)
            if key in seen:
                continue
            seen.add(key)
            handles.append(handle)
            labels.append(label)
    if not labels:
        return rect
    if loc == "right":
        block_in = max(text_width_in(label, LEGEND_FONTSIZE) for label in labels) + 0.75
        # 条目多到一列装不下就并排两列：图例高出画布等于被裁掉，比占宽更糟
        cols = 1 if len(labels) * LEGEND_ROW_IN <= fig.get_figheight() * 0.86 else 2
        band_in = block_in * cols
        frac = min(0.46, max(0.14, band_in / max(fig.get_figwidth(), 1.0)))
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.93),
                   ncol=cols, fontsize=LEGEND_FONTSIZE, frameon=False, labelspacing=0.42,
                   handlelength=1.1, handletextpad=0.45)
        # 图例吃掉的宽度用边距声明，保证坐标区仍拿得到自己那一份：合并后的图形级
        # 图例只占一竖列，跟子图行数无关（乘上去会把画布白白撑宽几英寸）
        claim_margin(fig, width_in=band_in)
        return (rect[0], rect[1], rect[2] - frac, rect[3])
    cols = _legend_ncol(labels, fig.get_figwidth())
    rows = int(np.ceil(len(labels) / cols))
    band_in = 0.24 * rows + 0.18
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.004),
               ncol=cols, fontsize=LEGEND_FONTSIZE, frameon=False, columnspacing=1.2,
               labelspacing=0.5, handlelength=1.1, handletextpad=0.45)
    claim_margin(fig, height_in=band_in)
    return (rect[0], rect[1] + min(0.42, band_in / max(fig.get_figheight(), 1.0)),
            rect[2], rect[3])


def tighten(fig, rect=(0.0, 0.0, 1.0, 1.0), pad=0.7, quiet=False):
    """统一排版：给旋转标签、标题、图例各自让位，且不许把色条压住。

    以前这些图从不走 tight_layout：子图留在默认边距里，旋转 45° 的编码名直接
    被画布底边砍掉。色条是挂在主图右侧的一条窄面板，tight_layout 不认识它，
    所以先按它的实宽扣除右边界（多列时还要拉开列间隙），排版完再贴着最终位置
    放回去（见 color_bar）。

    ``quiet=True`` 只用于两遍绘制里的量尺那一遍（见 render）：那张画布是故意
    给小的，装不下是预期行为，报出来的「Tight layout not applied」只会遮住真问题。
    """
    free = list(rect)
    titled = getattr(fig, "_suptitle", None) is not None and fig._suptitle.get_text()
    if titled:
        # tight_layout 只认子图自己的标题，不管 suptitle：不先把顶边让出来，
        # 图名就会压在第一行面板的标题上（两边字叠在一起）
        free[3] = max(0.45, rect[3] - SUPTITLE_IN / max(fig.get_figheight(), 1.0))
    cbar_cols, cbar_need_in = 1, 0.0
    for ax in fig.axes:
        if getattr(ax, "_ly_cbar", None) is None:
            continue
        spec = getattr(ax, "get_subplotspec", lambda: None)()
        geometry = getattr(spec, "get_geometry", lambda: None)()
        try:
            cols = max(int(geometry[1]), 1)
        except (TypeError, ValueError):
            cols = 1
        fraction, bar_pad = getattr(ax, "_ly_cbar_geometry", (0.046, 0.04))
        cbar_cols = max(cbar_cols, cols)
        # 色条实宽 = 主图宽 × fraction + 间隙 + 它自己的刻度标签（英寸，不随列数缩）
        ax_w = max(free[2] * fig.get_figwidth() / cols, 0.5)
        cbar_need_in = max(cbar_need_in, ax_w * (fraction + bar_pad) + CBAR_LABEL_IN)
    reserve = cbar_need_in / max(fig.get_figwidth(), 1.0) if cbar_need_in else 0.0
    if reserve:
        free[2] = max(0.35, free[2] - reserve)
    w_pad = 1.4
    if cbar_cols > 1 and cbar_need_in:
        # 列间隙也得留够：色条是贴着主图右边往外吃的，不留就压到下一列面板的 y 标签
        # 上（tight_layout 的 w_pad 以字号为单位，所以这里换算成字号倍数）
        w_pad = max(w_pad, cbar_need_in * 72.0
                    / max(float(rcParams.get("font.size") or 10.0), 6.0))

    def layout(target):
        """跑一次 tight_layout，返回“版式被放弃”那类抱怨（色条那句不算）。"""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fig.tight_layout(rect=tuple(target), pad=pad, w_pad=w_pad, h_pad=1.8)
            except Exception:                   # noqa: BLE001 版式算不出也不能拿掉整张图
                pass
        return [str(item.message) for item in caught
                if "not compatible with tight_layout" not in str(item.message)]

    complaints = layout(free)
    if complaints and titled and free[3] < rect[3]:
        # 顶边让给图名之后版式反而装不下了（量尺那一遍的画布本就是故意给小的）：
        # 宁可图名离第一行近一点，也不能放弃整张图的排版
        complaints = layout(free[:3] + [rect[3]])
    if not quiet:
        for message in complaints:
            # “Tight layout not applied”不吞 —— 那句是真有东西顶出画布，得回方法里修
            warnings.warn(message, UserWarning, stacklevel=3)
    _place_colorbars(fig)
    return tuple(free)


def finish(fig, params, quiet=False):
    """每张图画完的统一收尾：回收图例 → 按用户选项重放 → 排版。

    尺寸要到宿主那里才定（两遍绘制），所以这里只按当前画布宽决定图例列数与 rect。
    """
    rect = apply_legends(fig, params)
    tighten(fig, rect, quiet=quiet)
    theme.sync_figure(fig)


def numeric(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def text(column):
    return column.astype(str)


# ------------------------------------------------------------------ IND01 / IND05 通用：热图

def heatmap(ax, matrix, labels, title="", panels_wide=1, x_labels=None):
    """方阵热图通用画法（调整残差 / 转移概率 / 发射概率都走这里）。

    尺寸口径写在 `matrix_room`：每格至少一格的边长，格内数字才可能不互压；
    横轴类目标签交给 `fit_category_labels`（先撑宽、顶到上限才抽稀）。
    """
    data = np.asarray(matrix, dtype=float)
    if data.size == 0:
        ax.text(0.5, 0.5, "没有可画的矩阵", ha="center", va="center", color=ink())
        ax.axis("off")
        return None
    labels = [str(item) for item in labels]
    fig = ax.figure
    rows, cols = data.shape
    y_size = row_fontsize(rows)
    # 行名（可能是“prop_会话参与度_社交”这种长特征名）先量一遍：label_in 不写死，
    # 否则 tight_layout 只会把坐标区压窄，字照样挤到画布外
    matrix_room(fig, rows, cols, panels_wide=panels_wide,
                label_in=max(0.95, label_column_in(labels[:rows], y_size)))
    image = ax.imshow(data, cmap="RdBu_r", vmin=-max(np.nanmax(np.abs(data)), 1e-9),
                      vmax=max(np.nanmax(np.abs(data)), 1e-9), aspect="auto")
    fit_category_labels(ax, (x_labels if x_labels is not None else labels)[:cols],
                        columns=panels_wide, base=8.0, floor=6.4)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(labels[:rows], fontsize=y_size)
    # 格内数字按格子实宽给字号，装不下就省字：颜色 + 行列刻度仍读得出，
    # 但两个数字叠在一起什么都读不出
    box = ax.get_position()
    cell_w = max(fig.get_figwidth() * box.width / max(cols, 1), 1e-6)
    cell_h = max(fig.get_figheight() * box.height / max(rows, 1), 1e-6)
    text_size = max(5.2, min(8.0, 0.66 * min(cell_w, cell_h) * 72.0))
    span = max(np.nanmax(np.abs(data)), 1e-9)
    for i in range(rows):
        for j in range(cols):
            value = data[i, j]
            if not np.isfinite(value):
                continue
            body = "%g" % round(float(value), 2)
            if text_width_in(body, text_size) > cell_w - 0.03:
                continue
            ax.text(j, i, body, ha="center", va="center", fontsize=text_size,
                    color="#FFFFFF" if abs(value) > 0.55 * span else ink("ink1"))
    if title:
        ax.set_title(title, fontsize=9)
    return image


# ------------------------------------------------------------------ IND01 LSA

def plot_lsa(fig, result, ctx, params):
    frame = result.tables.get("lsa_cells")
    if frame is None or not len(frame):
        return
    ax = fig.axes[0]
    subset = frame[frame["significant"] == True].copy()          # noqa: E712
    if not len(subset):
        subset = frame.copy()
        subset["_hint"] = "无显著格"
    subset["Z"] = numeric(subset, "Z")
    subset = subset.reindex(subset["Z"].abs().sort_values(ascending=False).index)[:MAX_ITEMS]
    labels = ["%s→%s\n%s lag=%s" % (row["from"], row["to"], str(row["dimension"])[:10],
                                    row["lag"]) for _index, row in subset.iterrows()]
    palette = colors()
    bars = ax.barh(range(len(subset)), subset["Z"].tolist(),
                   color=[palette[0] if value < 0 else palette[1]
                          for value in subset["Z"].tolist()])
    ax.set_yticks(range(len(subset)))
    # 标签是两行（码对 + 维度/滞后），一行要给两行字的高度；宽度由最宽那行决定
    fit_bar_rows(ax, len(subset), row_h_in=0.46, labels=labels)
    ax.axvline(0, color=ink("line_strong"), linewidth=1)
    threshold = float(result.meta.get("threshold") or 1.96)
    ax.axvline(threshold, color=palette[3], linestyle="--", linewidth=1)
    ax.axvline(-threshold, color=palette[3], linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.set_title("调整残差 Z（虚线 = |Z| 阈值，灰条 = 未达显著）", fontsize=9)
    if "_hint" in subset.columns:
        ax.text(0.02, 0.9, "本次没有 |Z| 超过阈值的格", transform=ax.transAxes,
                fontsize=8.5, color=ink())
    return bars


# ------------------------------------------------------------------ IND02 Markov

def plot_markov(fig, result, ctx, params):
    tables = [name for name in result.tables if name.startswith("probabilities_")]
    panels = min(len(tables), MAX_PANELS) or 1
    fig.clear()
    for index in range(panels):
        ax = fig.add_subplot(1, panels, index + 1)
        style(ax)
        name = tables[index]
        frame = result.tables[name]
        states = [str(value) for value in frame.iloc[:, 0].tolist()]
        matrix = frame.iloc[:, 1:].to_numpy(dtype=float)
        heatmap(ax, matrix, states, title=name.replace("probabilities_", ""),
                panels_wide=panels)
        ax.set_xlabel("to", fontsize=8)
        ax.set_ylabel("from", fontsize=8)
    theme.sync_figure(fig)


# ------------------------------------------------------------------ IND03 选阶

def plot_order(fig, result, ctx, params):
    frame = result.tables.get("summary")
    if frame is None or not len(frame):
        return
    ax = fig.axes[0]
    palette = colors()
    dimensions = list(dict.fromkeys(text(frame["dimension"])))
    for position, dimension in enumerate(dimensions[:MAX_PANELS]):
        part = frame[frame["dimension"] == dimension]
        ax.plot(part["order"].tolist(), numeric(part, "BIC").tolist(), marker="o",
                color=palette[position % len(palette)], label=str(dimension)[:14])
    ax.set_xlabel("阶数", fontsize=8)
    ax.set_ylabel("BIC（越小越好）", fontsize=8)
    ax.set_title("选阶曲线", fontsize=9)
    declare_size(fig, height_in=FACET_H_IN)
    star = frame[frame.get("选中阶", "").astype(str) == "★"] if "选中阶" in frame else frame
    if len(star):
        for _index, row in star.iterrows():
            ax.scatter([row["order"]], [row["BIC"]], s=90, facecolors="none",
                       edgecolors=palette[3], linewidths=1.6, zorder=5)
    if len(dimensions) > 1:
        ax.legend(fontsize=7.5, framealpha=0.9)


# ------------------------------------------------------------------ IND04 频繁模式

def plot_patterns(fig, result, ctx, params):
    frame = result.tables.get("ngrams")
    if frame is None or not len(frame):
        frame = result.tables.get("subsequences")
    if frame is None or not len(frame):
        return
    ax = fig.axes[0]
    subset = frame.head(MAX_ITEMS)
    values = numeric(subset, "support").tolist()
    ax.barh(range(len(subset)), values, color=colors()[0])
    ax.set_yticks(range(len(subset)))
    # 模式串（"A→B→C…"）可以很长：不折行也不截断，直接按标签宽撑画布
    fit_bar_rows(ax, len(subset), row_h_in=0.34,
                 labels=[str(item) for item in
                         (subset["gram"] if "gram" in subset else subset["子序列"]).tolist()])
    ax.invert_yaxis()
    ax.set_xlabel("support（分母 = 序列条数）", fontsize=8)
    ax.set_title("最常见的 %d 个模式" % len(subset), fontsize=9)
    for position, value in enumerate(values):
        ax.text(value, position, " %.3f" % value, va="center", fontsize=7.5, color=ink())


# ------------------------------------------------------------------ IND05 距离矩阵

def plot_distance(fig, result, ctx, params):
    frame = result.tables.get("distance_matrix")
    if frame is None or not len(frame):
        return
    ax = fig.axes[0]
    labels = [str(item) for item in frame.iloc[:, 0].tolist()]
    matrix = frame.iloc[:, 1:].to_numpy(dtype=float)
    data = np.asarray(matrix, dtype=float)
    matrix_room(fig, data.shape[0], data.shape[1], panels_wide=1,
                label_in=max(0.95, label_column_in(labels, row_fontsize(len(labels)))))
    image = ax.imshow(data, cmap="viridis", aspect="auto")
    # 距离阵的每个刻度就是一个分析单元：不抽稀，宁可把画布撑宽去滚动
    fit_category_labels(ax, labels[:data.shape[1]], base=8.0, floor=6.4)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(labels[:data.shape[0]], fontsize=row_fontsize(len(labels)))
    ax.set_title("%d × %d 距离矩阵" % (data.shape[0], data.shape[1]), fontsize=9)
    color_bar(fig, image, ax)


# ------------------------------------------------------------------ IND07 聚类

def plot_clusters(fig, result, ctx, params):
    dendrogram = result.tables.get("dendrogram")
    clusters = result.tables.get("clusters")
    fig.clear()
    if dendrogram is not None and len(dendrogram):
        ax = fig.add_subplot(1, 2, 1)
        style(ax)
        _draw_dendrogram(ax, dendrogram)
        ax.set_title("层次聚类树状图", fontsize=9)
    if clusters is not None and len(clusters):
        ax = fig.add_subplot(1, 2, 2) if dendrogram is not None else fig.add_subplot(111)
        style(ax)
        _draw_cluster_bars(ax, clusters)
    declare_size(fig, height_in=FACET_H_IN * 1.2)
    theme.sync_figure(fig)


def _draw_dendrogram(ax, frame):
    """自己画树状图：列号 → 叶子序号，合并高度 → 竖线顶。不依赖 scipy 的绘图函数。"""
    size = len(frame) + 1
    leaves = {index: (float(index), 0.0) for index in range(size)}
    positions = dict(leaves)
    for _index, row in frame.iterrows():
        left, right, height = int(row["合并a"]) - 1, int(row["合并b"]) - 1, float(row["高度"])
        x_left, y_left = positions.get(left, (float(left), 0.0))
        x_right, y_right = positions.get(right, (float(right), 0.0))
        ax.plot([x_left, x_left], [y_left, height], color=ink(), linewidth=0.9)
        ax.plot([x_right, x_right], [y_right, height], color=ink(), linewidth=0.9)
        ax.plot([x_left, x_right], [height, height], color=colors()[0], linewidth=1.1)
        positions[size + int(_index)] = ((x_left + x_right) / 2.0, height)
    # 叶子是一个一个序列单元：窄画布下按叶号抽稀（号仍在轴上，只是不逐个写），
    # 但先按内容把画布撑宽，所以只有顶到上限才会真的省
    fit_category_labels(ax, [str(index + 1) for index in range(size)],
                        columns=2, base=8.0, floor=6.6)
    ax.set_ylabel("距离", fontsize=8)


def _draw_cluster_bars(ax, frame):
    counts = frame["簇"].value_counts().sort_index()
    palette = colors()
    ax.bar([str(item) for item in counts.index], counts.tolist(),
           color=[palette[int(index) % len(palette)] for index in counts.index])
    fit_category_labels(ax, [str(item) for item in counts.index], columns=2, base=8.4)
    if "silhouette" in frame:
        means = frame.groupby("簇")["silhouette"].mean()
        for position, cluster in enumerate(counts.index):
            ax.text(position, counts[cluster] / 2.0, "%.2f" % means.get(cluster, 0.0),
                    ha="center", va="center", fontsize=7.5, color="#FFFFFF")
        ax.set_ylabel("簇规模（条上数字 = 平均 silhouette）", fontsize=8)
    ax.set_xlabel("簇", fontsize=8)
    ax.set_title("簇规模", fontsize=9)


# ------------------------------------------------------------------ GRP01 组状态分布

def plot_group_state(fig, result, ctx, params):
    frame = result.tables.get("trajectory")
    if frame is None or not len(frame):
        return
    states = sorted(set(text(frame["state"])))
    cmap = color_map(states)
    groups = list(dict.fromkeys(text(frame["group_id"])))
    fig.clear()
    panels = min(len(groups), MAX_PANELS) or 1
    for index, group_id in enumerate(groups[:panels]):
        ax = fig.add_subplot(panels, 1, index + 1)
        style(ax)
        claim_row_height(fig, index + 1, BAND_H_IN * 2.2)
        part = frame[frame["group_id"] == group_id]
        pivot = part.pivot_table(index="time", columns="state", values="share", aggfunc="mean")
        pivot = pivot.reindex(columns=states).fillna(0.0)
        ax.stackplot(pivot.index.tolist(),
                     *[pivot[state].tolist() for state in states],
                     labels=[str(state) for state in states],
                     colors=[cmap[state] for state in map(str, states)], alpha=0.85)
        ax.set_xlim(min(pivot.index), max(pivot.index))
        ax.set_ylim(0, 1)
        ax.set_title("%s / %s" % (group_id, str(part["dimension"].iloc[0]) if len(part) else ""),
                     fontsize=9)
        # 每一行都自带一条标题 + 时间刻度：不把这条装饰算进去，tight_layout 会把它
        # 从坐标区里扣走（声明 1.36 in 一行，实际只剩 0.8 in，色带读不出层次）
        claim_row_margin(fig, index + 1, PANEL_DECOR_IN)
        if index == 0:
            # stackplot 已带 labels，图例内容照旧；位置交给 finish 按用户选项重放
            ax.legend(fontsize=LEGEND_FONTSIZE)
    theme.sync_figure(fig)


# ------------------------------------------------------------------ GRP02 组熵

def plot_group_entropy(fig, result, ctx, params):
    frame = result.tables.get("trajectory")
    if frame is None or not len(frame):
        return
    ax = fig.axes[0]
    palette = colors()
    keys = list(dict.fromkeys(zip(text(frame["group_id"]), text(frame["dimension"]))))
    for position, (group_id, dimension) in enumerate(keys[:MAX_PANELS]):
        part = frame[(frame["group_id"] == group_id) & (frame["dimension"] == dimension)]
        ax.plot(part["time"].tolist(), numeric(part, "H").tolist(),
                color=palette[position % len(palette)], linewidth=1.3,
                label="%s %s" % (group_id, dimension[:10]))
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("组内熵 H_t (ln)", fontsize=8)
    ax.set_title("熵轨迹（越高越分散）", fontsize=9)
    declare_size(fig, height_in=FACET_H_IN)
    if len(keys) > 1:
        ax.legend(fontsize=LEGEND_FONTSIZE, ncol=2)


# ------------------------------------------------------------------ GRP03 同步度

def _pair_tag(row, keys):
    """两两对的后缀：成员名只在组内唯一（都叫 S1/S2），不带组与维度就分不开。"""
    parts = [str(row[key]) for key in keys if key in row.index]
    return "（%s）" % "·".join(parts) if parts else ""


def _sync_blocks(matrix):
    """把 `sync_matrix` 切回一块块真方阵。

    这张表是「每组每维度一个方阵」纵向摞起来的长表（一行 = 一个组×维度×成员），
    整块 `imshow` 会画出一个「45 行 × 3 列」的假矩阵：既不是方阵，也把不同组的
    成员拌在同一个坐标系里。按 (group_id, dimension) 切回一块块再分面画。
    """
    keys = [column for column in ("group_id", "dimension") if column in matrix.columns]
    blocks = []
    if keys:
        grouped = matrix.groupby(keys, sort=False)
    else:
        grouped = [(None, matrix)]
    for key, part in grouped:
        body = part.drop(columns=keys, errors="ignore")
        if body.shape[1] < 2:
            continue
        labels = [str(item) for item in body.iloc[:, 0].tolist()]
        rest = body.iloc[:, 1:]
        # 成员数不同的组 concat 后会出现全空列（别的组的成员名），按实际非空列取
        columns = [name for name in rest.columns if rest[name].notna().any()]
        data = np.nan_to_num(rest[columns].to_numpy(dtype=float))
        title = " / ".join(str(item) for item in (key if isinstance(key, tuple) else
                                                  (key,) if key is not None else ()))
        blocks.append((title, labels, [str(name) for name in columns], data))
    return blocks


def plot_synchrony(fig, result, ctx, params):
    matrix = result.tables.get("sync_matrix")
    pairs = result.tables.get("pairs")
    fig.clear()
    blocks = _sync_blocks(matrix) if matrix is not None and len(matrix) else []
    shown = blocks[:MAX_PANELS]
    has_pairs = pairs is not None and len(pairs)
    panels = len(shown) + (1 if has_pairs else 0)
    if not panels:
        return
    columns_wide = min(panels, 3)
    rows_wide = int(np.ceil(panels / float(columns_wide)))
    # 分面顶到上限时把“省了多少”写在图上：不让用户以为只有这六块
    cut = ("（共 %d 个组×维度，仅展前 %d）" % (len(blocks), len(shown))
           if len(blocks) > len(shown) else "")
    for slot, (title, labels, x_labels, data) in enumerate(shown):
        row = slot // columns_wide + 1
        ax = fig.add_subplot(rows_wide, columns_wide, slot + 1)
        style(ax)
        image = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        matrix_room(fig, data.shape[0], data.shape[1], panels_wide=columns_wide, row=row,
                    label_in=max(0.95, label_column_in(labels, row_fontsize(len(labels)))))
        # 成员名（group_1__S2 形态）比编码长得多：旋转 + 撑宽，不省人
        fit_category_labels(ax, x_labels[:data.shape[1]], columns=columns_wide,
                            base=8.0, floor=6.4)
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(labels[:data.shape[0]], fontsize=row_fontsize(len(labels)))
        ax.set_title((title or "同步度矩阵") + (cut if not slot else ""), fontsize=9)
        color_bar(fig, image, ax)
    if has_pairs:
        ax = fig.add_subplot(rows_wide, columns_wide, panels)
        style(ax)
        # 这一条只是分面里的一格：行高被同列的矩阵限住，45 条挤进去必压字，
        # 所以只取前若干条（省了多少写进标题，不静默丢）
        subset = pairs.head(MAX_ITEMS)
        values = numeric(subset, "agreement")
        ax.barh(range(len(subset)), values.tolist(), color=colors()[4])
        pair_labels = ["%s·%s%s" % (row["成员A"], row["成员B"],
                                    _pair_tag(row, ("group_id", "dimension")))
                       for _index, row in subset.iterrows()]
        ax.set_yticks(range(len(subset)))
        fit_bar_rows(ax, len(subset), row_h_in=ROW_H_IN, labels=pair_labels,
                     columns=columns_wide)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        extra = "（共 %d 对，仅展前 %d 对）" % (len(pairs), len(subset)) if len(pairs) > len(subset) else ""
        ax.set_title("两两 agreement%s" % extra, fontsize=9)
    theme.sync_figure(fig)


# ------------------------------------------------------------------ GRP04 领先-滞后

def plot_lead_lag(fig, result, ctx, params):
    frame = result.tables.get("pairs")
    if frame is None or not len(frame):
        return
    ax = fig.axes[0]
    palette = colors()
    # 一行就是一个 (组, 维度, A, B)：按三维分组会把 D2/D3 挤掉（只留 iloc[0] 那一行），
    # 等于图上看不到的那几维被静默丢了
    subset = frame.head(MAX_BARS)
    values = numeric(subset, "最优lag").tolist()
    labels = ["%s–%s%s" % (row["成员A"], row["成员B"],
                           _pair_tag(row, ("group_id", "dimension")))
              for _index, row in subset.iterrows()]
    ax.barh(range(len(values)), values,
            color=[palette[1] if value >= 0 else palette[0] for value in values])
    ax.set_yticks(range(len(labels)))
    fit_bar_rows(ax, len(labels), row_h_in=ROW_H_IN, labels=labels)
    ax.invert_yaxis()
    ax.axvline(0, color=ink("line_strong"), linewidth=1)
    ax.set_xlabel("最优 lag（正 = 前者领先）", fontsize=8)
    cut = "（共 %d 对，仅展前 %d 对）" % (len(frame), len(subset)) if len(frame) > len(subset) else ""
    ax.set_title("领先-滞后方向%s" % cut, fontsize=9)


# ------------------------------------------------------------------ GRP05 组级残差

def plot_group_transitions(fig, result, ctx, params):
    frame = result.tables.get("residual_matrix")
    if frame is None or not len(frame):
        return
    keys = list(dict.fromkeys(zip(text(frame["group_id"]), text(frame["口径"]))))
    fig.clear()
    panels = min(len(keys), MAX_PANELS) or 1
    for index, (group_id, flavor) in enumerate(keys[:panels]):
        ax = fig.add_subplot(1, panels, index + 1)
        style(ax)
        part = frame[(frame["group_id"] == group_id) & (frame["口径"] == flavor)]
        body = part.drop(columns=[column for column in ("group_id", "口径", "状态")
                                  if column in part], errors="ignore")
        states = [str(item) for item in part["状态"].tolist()] if "状态" in part else []
        heatmap(ax, body.to_numpy(dtype=float), states,
                title="%s · %s" % (group_id, flavor), panels_wide=panels)
        ax.set_xlabel("to", fontsize=8)
        ax.set_ylabel("from", fontsize=8)
    theme.sync_figure(fig)


# ------------------------------------------------------------------ GRP06 异质性

def plot_heterogeneity(fig, result, ctx, params):
    frame = result.tables.get("heterogeneity")
    if frame is None or not len(frame) or "CV" not in frame:
        return
    ax = fig.axes[0]
    pivot = frame.pivot_table(index="特征", columns="group_id", values="CV", aggfunc="max")
    ranked = pivot.mean(axis=1).sort_values(ascending=False)
    chosen = list(ranked.index[:4]) or list(pivot.index[:4])
    pivot = pivot.reindex(chosen)
    groups = [str(column) for column in pivot.columns]
    palette = colors()
    width = 0.8 / max(len(groups), 1)
    for position, group_id in enumerate(groups[:MAX_PANELS]):
        series = pd.to_numeric(pivot[group_id], errors="coerce").fillna(0.0)
        ax.bar(np.arange(len(chosen)) + position * width, series.tolist(), width=width,
               color=palette[position % len(palette)], label=group_id)
    ax.set_xticks(np.arange(len(chosen)) + width * (len(groups) - 1) / 2.0)
    # 特征名是 prop_run_x / switch_rate_1 这种长串：交给统一的类目轴处理
    # （柱心不在整数上，刻度位置必须一并传进去，不然标签会跑到邻居柱上去）
    fit_category_labels(ax, [str(item) for item in chosen], base=8.2,
                        positions=np.arange(len(chosen)) + width * (len(groups) - 1) / 2.0)
    ax.set_ylabel("CV（组内成员差异）", fontsize=8)
    ax.set_title("异质性最高的特征", fontsize=9)
    declare_size(fig, height_in=FACET_H_IN)
    if len(groups) > 1:
        ax.legend(fontsize=LEGEND_FONTSIZE, ncol=min(len(groups), 6))


# ------------------------------------------------------------------ GRP09 网络

def ring_positions(nodes):
    """节点环形布局：网络图不跑力导向（不可复现且慢），固定角度保证同数据同图。"""
    nodes = list(nodes)
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: (0.0, 0.0)}
    return {node: (float(np.cos(2 * np.pi * index / len(nodes))),
                   float(np.sin(2 * np.pi * index / len(nodes))))
            for index, node in enumerate(nodes)}


def draw_weighted_network(ax, frame, title="", directed=True, arc=0.12):
    """加权网络通用画法（列约定：source / target / weight）：线宽 = 边权，环布局。"""
    nodes = sorted(set(text(frame["source"])) | set(text(frame["target"])))
    if not nodes:
        ax.text(0.5, 0.5, "没有可画的边", ha="center", va="center", color=ink())
        ax.set_axis_off()
        return
    cmap = color_map(nodes)
    positions = ring_positions(nodes)
    weights = numeric(frame, "weight")
    scale = float(weights.max()) if len(weights) and weights.max() > 0 else 1.0
    for index, row in frame.iterrows():                       # noqa: PER001
        x1, y1 = positions[str(row["source"])]
        x2, y2 = positions[str(row["target"])]
        ratio = min(max(float(weights.iloc[index]) / scale, 0.0), 1.0)
        if directed:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", color=ink("ink3"),
                                        alpha=0.25 + 0.6 * ratio,
                                        linewidth=0.4 + 2.2 * ratio,
                                        shrinkA=9, shrinkB=9,
                                        connectionstyle="arc3,rad=%g" % arc))
        else:
            ax.plot([x1, x2], [y1, y2], color=ink("ink3"), alpha=0.25 + 0.6 * ratio,
                    linewidth=0.4 + 2.2 * ratio)
    for node in nodes:
        x, y = positions[node]
        ax.scatter([x], [y], s=180, color=cmap[node], zorder=5, edgecolors=ink("card"),
                   linewidths=1.2)
        ax.text(x, y, node, ha="center", va="center", fontsize=7, color="#FFFFFF", zorder=6)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.25)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
    # 环布局上节点越多越要更大的面：周长按点数摊宽，不然名字必叠在一起
    side = max(4.6, len(nodes) * 0.42)
    declare_size(ax.figure, height_in=side * 0.86, width_in=side * 0.94)


def plot_network(fig, result, ctx, params):
    """GRP09：优先画 ONA 有向时序网，没转移边时退到 ENA 无向共现网。"""
    frame = result.tables.get("ona_edges")
    directed = frame is not None and len(frame) > 0
    if not directed:
        frame = result.tables.get("ena_edges")
        if frame is None or not len(frame):
            return
        frame = frame.rename(columns={"节点A": "source", "节点B": "target",
                                      "共现次数": "weight"})
    else:
        frame = frame.rename(columns={"权重": "weight"})     # ONA 边表的权列叫「权重」
    frame = frame.copy()
    frame["weight"] = numeric(frame, "weight")
    ax = fig.axes[0]
    draw_weighted_network(ax, frame, directed=directed,
                          title="%s网络（线宽 = 边权，%d 节点 / %d 边）"
                          % ("有向" if directed else "无向",
                             int(frame["source"].nunique() + frame["target"].nunique()) // 2,
                             len(frame)))


# ------------------------------------------------------------------ IND06 DTW

def _matrix_panel(fig, panels, slot, frame, title, cmap="viridis"):
    """距离阵热图通用面板：每个刻度都是一个分析单元，不省只撑宽。"""
    ax = fig.add_subplot(1, panels, slot + 1)
    style(ax)
    labels = [str(item) for item in frame.iloc[:, 0].tolist()]
    data = np.asarray(frame.iloc[:, 1:].to_numpy(dtype=float), dtype=float)
    image = ax.imshow(data, cmap=cmap, aspect="auto")
    matrix_room(fig, data.shape[0], data.shape[1], panels_wide=panels,
                label_in=max(0.95, label_column_in(labels, row_fontsize(len(labels)))))
    fit_category_labels(ax, labels[:data.shape[1]], columns=panels, base=8.0, floor=6.4)
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(labels[:data.shape[0]], fontsize=row_fontsize(len(labels)))
    ax.set_title(title, fontsize=9)
    color_bar(fig, image, ax)
    return ax


def plot_dtw(fig, result, ctx, params):
    """IND06：左 = DTW 距离热图，右 = DBA 重心（堆叠面积，逐位置的码质量分布）。"""
    distance = result.tables.get("distance_matrix")
    center = result.tables.get("barycenter")
    soft = result.tables.get("soft_distance_matrix")
    blocks = [item for item in (("距离矩阵", distance), ("soft-DTW 距离", soft),
                               ("重心", center)) if item[1] is not None and len(item[1])]
    if not blocks:
        return
    fig.clear()
    panels = len(blocks)
    for slot, (name, frame) in enumerate(blocks):
        if name == "重心":
            ax = fig.add_subplot(1, panels, slot + 1)
            style(ax)
            body = frame.drop(columns=[column for column in ("重心位置", "主导码")
                                       if column in frame], errors="ignore")
            positions = numeric(frame, "重心位置").to_numpy(dtype=float)
            bottom = np.zeros(len(frame))
            palette = colors()
            for number, column in enumerate(body.columns):
                values = numeric(body, column).to_numpy(dtype=float)
                ax.fill_between(positions, bottom, bottom + values, color=palette[
                    number % len(palette)], alpha=0.75, label=str(column))
                bottom = bottom + values
            ax.set_xlabel("重心位置", fontsize=8)
            ax.set_ylabel("码质量", fontsize=8)
            ax.set_title("DBA 重心（各位置上的码占比）", fontsize=9)
            # 图例位置现在由用户定，不再因为「码多于 8 个」就干脆不标
            ax.legend(fontsize=LEGEND_FONTSIZE, ncol=2)
            declare_size(fig, height_in=FACET_H_IN)
            continue
        _matrix_panel(fig, panels, slot, frame, name)
    theme.sync_figure(fig)


# ------------------------------------------------------------------ IND08 HMM

def plot_hmm(fig, result, ctx, params):
    """IND08：发射概率热图（逐维/通道）+ AIC/BIC 选状态数曲线。"""
    emission = result.tables.get("emission")
    selection = result.tables.get("model_selection")
    if (emission is None or not len(emission)) and (selection is None or not len(selection)):
        return
    panels = int(bool(emission is not None and len(emission))) \
        + int(bool(selection is not None and len(selection))) or 1
    fig.clear()
    slot = 0
    if emission is not None and len(emission):
        dimensions = list(dict.fromkeys(text(emission["维度"])))
        channels = list(dict.fromkeys(text(emission["通道"])))
        part = emission[(emission["维度"] == dimensions[0])
                        & (emission["通道"] == channels[0])]
        pivot = part.pivot_table(index="state", columns="symbol", values="p", aggfunc="max")
        ax = fig.add_subplot(1, panels, slot + 1)
        style(ax)
        heatmap(ax, pivot.to_numpy(dtype=float), [str(item) for item in pivot.index],
                title="发射概率 B（%s / %s）" % (dimensions[0], channels[0]),
                x_labels=[str(item) for item in pivot.columns], panels_wide=panels)
        ax.set_xlabel("观测码", fontsize=8)
        ax.set_ylabel("隐状态", fontsize=8)
        slot += 1
    if selection is not None and len(selection):
        ax = fig.add_subplot(1, panels, slot + 1)
        style(ax)
        palette = colors()
        for number, column in enumerate(("AIC", "BIC")):
            if column not in selection:
                continue
            for dimension, part in list(selection.groupby("维度", sort=False))[:4]:
                ax.plot(numeric(part, "状态数"), numeric(part, column), marker="o",
                        linewidth=1.2, alpha=0.85,
                        color=palette[(number * 4 + list(selection["维度"]).index(dimension))
                                      % len(palette)],
                        label="%s %s" % (dimension, column) if len(selection["维度"].unique()) > 1
                        else column)
        ax.set_xlabel("隐状态数 k", fontsize=8)
        ax.set_ylabel("越小越好", fontsize=8)
        ax.set_title("选状态数（%s）" % (" / ".join(
            sorted(set(text(selection["维度"]))))[:2]), fontsize=9)
        declare_size(fig, height_in=FACET_H_IN)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=LEGEND_FONTSIZE, ncol=2)
    theme.sync_figure(fig)


# ------------------------------------------------------------------ IND09 变点 / 分段

def plot_segmentation(fig, result, ctx, params):
    """IND09：Hellinger 滑窗轨迹 + 变点竖线；第二面 = 每条序列的段数。"""
    trace = result.tables.get("hellinger")
    segments = result.tables.get("segments")
    changes = result.tables.get("changepoints")
    if (trace is None or not len(trace)) and (segments is None or not len(segments)):
        return
    panels = int(bool(trace is not None and len(trace))) + int(bool(segments is not None
                                                                   and len(segments))) or 1
    fig.clear()
    slot = 0
    if trace is not None and len(trace):
        ax = fig.add_subplot(1, panels, slot + 1)
        style(ax)
        palette = colors()
        labels = list(dict.fromkeys(text(trace["label"])))[:4]
        marks = []
        if changes is not None and len(changes):
            marks = [numeric(changes[changes["label"] == item], "位置").tolist()
                     for item in labels]
        for number, label in enumerate(labels):
            part = trace[trace["label"] == label]
            ax.plot(numeric(part, "位置"), numeric(part, "Hellinger"), linewidth=1.2,
                    color=palette[number % len(palette)], label=label)
            for position in (marks[number] if number < len(marks) else []):
                ax.axvline(float(position), color=palette[number % len(palette)],
                           linestyle="--", linewidth=0.9, alpha=0.8)
        ax.set_xlabel("时点", fontsize=8)
        ax.set_ylabel("Hellinger 距离", fontsize=8)
        ax.set_title("滑窗分布距离（虚线 = 判出的变点）", fontsize=9)
        declare_size(fig, height_in=FACET_H_IN)
        if len(labels) > 1:
            ax.legend(fontsize=LEGEND_FONTSIZE)
        slot += 1
    if segments is not None and len(segments):
        ax = fig.add_subplot(1, panels, slot + 1)
        style(ax)
        counts = segments.groupby("label")["段号"].max().sort_values(ascending=False)
        chosen = counts.head(MAX_ITEMS)
        ax.barh(range(len(chosen)), chosen.to_numpy(dtype=float), color=colors()[1])
        ax.set_yticks(range(len(chosen)))
        fit_bar_rows(ax, len(chosen), row_h_in=ROW_H_IN,
                     labels=[str(item) for item in chosen.index])
        ax.invert_yaxis()
        ax.set_xlabel("段数", fontsize=8)
        ax.set_title("每条序列被切成的段数", fontsize=9)
    theme.sync_figure(fig)


# ------------------------------------------------------------------ IND10 ENA / TCA

def plot_ena(fig, result, ctx, params):
    """IND10：ENA 双标图 —— 学生坐标散点（按组着色）+ 节点载荷箭头。"""
    coords = result.tables.get("coordinates")
    loadings = result.tables.get("loadings")
    if coords is None or not len(coords):
        return
    axis_columns = [column for column in coords.columns if str(column).startswith("坐标")]
    if len(axis_columns) < 2:
        return
    ax = fig.axes[0]
    xs = numeric(coords, axis_columns[0]).to_numpy(dtype=float)
    ys = numeric(coords, axis_columns[1]).to_numpy(dtype=float)
    palette = colors()
    groups = list(dict.fromkeys(text(coords["group_id"])))
    for index, (x, y) in enumerate(zip(xs, ys)):
        color = palette[groups.index(text(coords["group_id"]).iloc[index]) % len(palette)]
        ax.scatter([x], [y], s=42, color=color, zorder=5, edgecolors=ink("card"),
                   linewidths=0.8)
        ax.text(x, y, str(coords["uid"].iloc[index]).split("__")[-1], fontsize=6.5,
                ha="center", va="center", color="#FFFFFF", zorder=6)
    if loadings is not None and len(loadings):
        loading_axes = [column for column in loadings.columns
                        if str(column).startswith("坐标")][:2]
        body = loadings.copy()
        body["长度"] = np.hypot(numeric(body, loading_axes[0]), numeric(body, loading_axes[1]))
        for _index, row in body.sort_values("长度", ascending=False).head(10).iterrows():
            dx, dy = float(row[loading_axes[0]]), float(row[loading_axes[1]])
            ax.annotate("", xy=(dx * 1.6, dy * 1.6), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=ink("ink3"), alpha=0.75,
                                        linewidth=1.0))
            ax.text(dx * 1.72, dy * 1.72, str(row["node"]), fontsize=6.5, color=ink(),
                    ha="center", va="center")
        # 载荷与坐标不同量纲：不把它们算进显示范围的话，箭头与节点名会伸出坐标区，
        # 既压住轴标签，也让 tight_layout 量到一个顶出画布几倍的包围盒（整张图版式直接放弃）
        reach = max(float(np.nanmax(np.abs(xs))), float(np.nanmax(np.abs(ys))),
                    1.72 * float(np.nanmax(np.abs(numeric(body, loading_axes[0])))),
                    1.72 * float(np.nanmax(np.abs(numeric(body, loading_axes[1])))))
        ax.set_xlim(-reach * 1.18, reach * 1.18)
        ax.set_ylim(-reach * 1.18, reach * 1.18)
    ax.axhline(0, color=ink("line"), linewidth=0.8)
    ax.axvline(0, color=ink("line"), linewidth=0.8)
    if len(groups) > 1:
        # 散点是一个个画的（为了逐点写 uid），图例另给一组代理 handle：
        # 不然按组着色只有颜色、没有对照，用户只能猜哪个色是哪组
        ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=6.5,
                                  markerfacecolor=palette[index % len(palette)],
                                  markeredgecolor=ink("card"), label=name)
                           for index, name in enumerate(groups)],
                  fontsize=LEGEND_FONTSIZE)
    ax.set_xlabel(axis_columns[0], fontsize=8)
    ax.set_ylabel(axis_columns[1], fontsize=8)
    tca = result.tables.get("tca")
    explained = ""
    if tca is not None and len(tca):
        explained = "（%s: %.0f%% / %s: %.0f%%）" % (
            axis_columns[0], float(numeric(tca, "解释方差%").iloc[0]),
            axis_columns[1], float(numeric(tca, "解释方差%").iloc[1]))
    ax.set_title("网络坐标 %s%s" % ("、".join(groups[:3]) if len(groups) > 1 else "全体",
                                    explained), fontsize=9)
    # 双标图是方形布局：点越多越要把画布撑大，否则圆点与旁边的字糊成一团
    spread = max(4.8, len(coords) * 0.30)
    declare_size(fig, height_in=spread, width_in=spread * 1.10)


# ------------------------------------------------------------------ IND11 ONA

def plot_ona(fig, result, ctx, params):
    """IND11：合并有向网（Top 边）+ 逐单元密度条。"""
    pooled = result.tables.get("transitions")
    metrics = result.tables.get("metrics")
    if pooled is None or not len(pooled):
        return
    top = pooled.sort_values("weight", ascending=False).head(60)
    panels = 2 if metrics is not None and len(metrics) else 1
    fig.clear()
    ax = fig.add_subplot(1, panels, 1)
    style(ax)
    draw_weighted_network(ax, top, directed=True,
                          title="合并有向时序网（前 %d 条边）" % len(top))
    if panels > 1:
        ax = fig.add_subplot(1, panels, 2)
        style(ax)
        part = metrics
        keys = [column for column in ("uid", "dimension") if column in part]
        label = part.apply(lambda row: " / ".join(str(row[item]) for item in keys), axis=1)
        values = numeric(part, "密度").to_numpy(dtype=float)
        order = np.argsort(values)[::-1][:MAX_ITEMS]
        ax.barh(range(len(order)), [values[item] for item in order], color=colors()[2])
        ax.set_yticks(range(len(order)))
        fit_bar_rows(ax, len(order), row_h_in=ROW_H_IN,
                     labels=[str(label.iloc[item]) for item in order])
        ax.invert_yaxis()
        ax.set_xlabel("密度 = E / N(N−1)", fontsize=8)
        ax.set_title("各分析单元的网密度", fontsize=9)
    theme.sync_figure(fig)


# ------------------------------------------------------------------ GRP07 角色发现

def plot_roles(fig, result, ctx, params):
    """GRP07：特征×簇 的 σ 偏离热图 + 各簇规模。"""
    profile = result.tables.get("profile")
    roles = result.tables.get("roles")
    if profile is None or not len(profile):
        return
    pivot = profile.pivot_table(index="特征", columns="簇", values="偏离(σ)", aggfunc="max")
    cluster_names = ["簇 %s" % item for item in pivot.columns]
    fig.clear()
    ax = fig.add_subplot(1, 2, 1)
    style(ax)
    heatmap(ax, pivot.to_numpy(dtype=float), [str(item) for item in pivot.index],
            title="角色画像（簇均值偏离全体几个 σ）",
            x_labels=cluster_names, panels_wide=2)
    ax.set_xlabel("", fontsize=8)
    ax.set_ylabel("", fontsize=8)
    ax = fig.add_subplot(1, 2, 2)
    style(ax)
    if roles is not None and len(roles):
        summary = roles.groupby(["簇", "角色"]).size().reset_index(name="规模")
        palette = colors()
        for index, row in summary.iterrows():
            ax.bar(index, float(row["规模"]), color=palette[index % len(palette)])
            ax.text(index, float(row["规模"]), "\n%s" % str(row["角色"])[:12],
                    ha="center", va="bottom", fontsize=6.5, color=ink())
        ax.set_xticks(range(len(summary)))
        fit_category_labels(ax, ["簇 %s" % item for item in summary["簇"]], columns=2)
        ax.set_ylim(0, float(summary["规模"].max()) * 1.6)
    declare_size(fig, height_in=FACET_H_IN)
    ax.set_ylabel("成员数", fontsize=8)
    ax.set_title("簇规模与角色名", fontsize=9)
    theme.sync_figure(fig)


# ------------------------------------------------------------------ GRP08 交互网络

def plot_interaction(fig, result, ctx, params):
    """GRP08：PERSON→PERSON 有向网 + 逐节点出/入强度。"""
    edges = result.tables.get("edges")
    nodes = result.tables.get("node_metrics")
    if edges is None or not len(edges):
        ax = fig.axes[0]
        summary = result.tables.get("summary")
        if summary is not None and len(summary):
            ax.barh(range(len(summary)), numeric(summary, "事件数"), color=colors()[0])
            ax.set_yticks(range(len(summary)))
            fit_bar_rows(ax, len(summary), row_h_in=ROW_H_IN,
                         labels=[str(item) for item in summary["group_id"]])
            ax.invert_yaxis()
            ax.set_xlabel("事件数", fontsize=8)
            ax.set_title("没有 PERSON→PERSON 边，只报事件计数", fontsize=9)
        return
    panels = 2 if nodes is not None and len(nodes) else 1
    fig.clear()
    ax = fig.add_subplot(1, panels, 1)
    style(ax)
    draw_weighted_network(ax, edges, directed=True,
                          title="交互网：%d 人 / %d 条边" % (
                              len(set(text(edges["source"])) | set(text(edges["target"]))),
                              len(edges)))
    if panels > 1:
        ax = fig.add_subplot(1, panels, 2)
        style(ax)
        part = nodes.sort_values("出强度", ascending=False).head(MAX_ITEMS)
        positions = np.arange(len(part))
        width = 0.38
        palette = colors()
        ax.bar(positions - width / 2, numeric(part, "出强度"), width=width,
               color=palette[0], label="出（发起）")
        ax.bar(positions + width / 2, numeric(part, "入强度"), width=width,
               color=palette[1], label="入（接收）")
        ax.set_xticks(positions)
        fit_category_labels(ax, [str(item) for item in part["节点"]], columns=2)
        declare_size(fig, height_in=FACET_H_IN)
        ax.set_ylabel("事件数", fontsize=8)
        ax.set_title("谁在说、谁被说", fontsize=9)
        ax.legend(fontsize=LEGEND_FONTSIZE)
    theme.sync_figure(fig)


# ------------------------------------------------------------------ IND12 特征型 ML

def plot_ml(fig, result, ctx, params):
    """IND12：ΔMacro-F1 重要度 + 逐折 Macro-F1（含多数类基线）。"""
    importance = result.tables.get("importance")
    metrics = result.tables.get("metrics")
    fig.clear()
    ax = fig.add_subplot(1, 2, 1)
    style(ax)
    if importance is not None and len(importance):
        part = importance.groupby("特征")["重要度(ΔMacro-F1)"].mean().sort_values()
        part = part.tail(MAX_ITEMS)
        palette = colors()
        ax.barh(range(len(part)), part.to_numpy(dtype=float),
                color=[palette[index % len(palette)] for index in range(len(part))])
        ax.set_yticks(range(len(part)))
        fit_bar_rows(ax, len(part), row_h_in=ROW_H_IN,
                     labels=[str(item) for item in part.index])
        ax.set_xlabel("重排后 ΔMacro-F1 均值", fontsize=8)
        ax.set_title("哪些特征真的在起作用", fontsize=9)
    else:
        selection = result.tables.get("model_selection")
        if selection is not None and len(selection):
            ax.barh(range(len(selection)), numeric(selection, "Macro-F1 均值"),
                    color=colors()[0])
            ax.set_yticks(range(len(selection)))
            fit_bar_rows(ax, len(selection), row_h_in=ROW_H_IN,
                         labels=[str(item) for item in selection["模型"]])
            ax.invert_yaxis()
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("Macro-F1 均值", fontsize=8)
        ax.set_title("重排次数为 0，未计算重要度", fontsize=9)
    ax = fig.add_subplot(1, 2, 2)
    style(ax)
    if metrics is not None and len(metrics):
        models = list(dict.fromkeys([str(item) for item in metrics["模型"]]))
        palette = colors()
        positions = np.arange(len(models))
        for position, model_text in enumerate(models):
            values = numeric(metrics[metrics["模型"] == model_text], "Macro-F1")
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            ax.scatter(np.full(len(values), position) +
                       np.random.default_rng(position).normal(0, 0.05, len(values)),
                       values.to_numpy(dtype=float), s=22, alpha=0.8,
                       color=palette[position % len(palette)])
            ax.plot([position - 0.28, position + 0.28], [float(values.mean())] * 2,
                    color=ink("ink1"), linewidth=1.4)
        if bool(result.meta.get("label_source", "").startswith("伪标签")):
            ax.set_title("逐折 Macro-F1（伪标签：只说明能复现簇）", fontsize=9)
        else:
            ax.set_title("逐折 Macro-F1（横线 = 折均值）", fontsize=9)
        ax.set_xticks(positions)
        fit_category_labels(ax, [str(item) for item in models], columns=2)
        declare_size(fig, height_in=FACET_H_IN)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Macro-F1", fontsize=8)
    theme.sync_figure(fig)


# ------------------------------------------------------------------ IND13–IND15 / GRP10 深度学习

def blank(ax, note="没有可画的内容"):
    """空图也要占位：不能抛异常，也不能留一张白到看不出什么的画布。"""
    ax.axis("off")
    ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=9, color=ink())


def as_float(value, default=None):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def fold_macro_scatter(ax, metrics, title="", floor=None):
    """逐折 Macro-F1：每模型一列散点 + 折均值横线，虚线 = 地板。"""
    style(ax)
    if metrics is None or not len(metrics):
        blank(ax, "没有可画的折指标")
        return
    models = list(dict.fromkeys([str(item) for item in metrics["模型"]]))
    palette = colors()
    for position, model_text in enumerate(models):
        part = metrics[metrics["模型"] == model_text]
        values = numeric(part, "Macro-F1")
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        ax.scatter(np.full(len(values), position)
                   + np.random.default_rng(position).normal(0, 0.05, len(values)),
                   values.to_numpy(dtype=float), s=22, alpha=0.85,
                   color=palette[position % len(palette)])
        ax.plot([position - 0.28, position + 0.28], [float(values.mean())] * 2,
                color=ink("ink1"), linewidth=1.4)
    score = as_float(floor)
    if score is not None:
        ax.axhline(score, color=ink("ink2"), linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(models)))
    fit_category_labels(ax, models, columns=2)
    declare_size(ax.figure, height_in=FACET_H_IN)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Macro-F1", fontsize=8)
    ax.set_title(title, fontsize=9)


def plot_dl_next_state(fig, result, ctx, params):
    """IND13：逐折 Macro-F1 散点（含地板虚线）+ 模型排名条形。"""
    metrics = result.tables.get("metrics")
    selection = result.tables.get("model_selection")
    fig.clear()
    ax = fig.add_subplot(1, 2, 1)
    fold_macro_scatter(ax, metrics, "逐折 Macro-F1（虚线 = 地板最高分）",
                       floor=(result.meta or {}).get("floor_macro_f1"))
    ax = fig.add_subplot(1, 2, 2)
    style(ax)
    if selection is None or not len(selection):
        blank(ax, "没有可汇总的模型指标")
    else:
        part = selection.tail(MAX_ITEMS)
        means = numeric(part, "Macro-F1 均值").to_numpy(dtype=float)
        errors = numeric(part, "Macro-F1 标准差").to_numpy(dtype=float)
        palette = colors()
        ax.barh(range(len(part)), means, xerr=np.clip(errors, 0.0, None),
                color=[palette[index % len(palette)] for index in range(len(part))],
                error_kw=dict(ecolor=ink("ink2"), linewidth=0.8))
        ax.set_yticks(range(len(part)))
        fit_bar_rows(ax, len(part), row_h_in=ROW_H_IN,
                     labels=[str(item) for item in part["模型"]], columns=2)
        ax.set_xlim(0.0, 1.05)
        ax.set_xlabel("Macro-F1 均值（误差棒 = 折间标准差）", fontsize=8)
        fields = (result.meta or {}).get("receptive_field") or {}
        extra = "、".join("%s 感受野 %s" % (key, value) for key, value in sorted(fields.items()))
        ax.set_title(("模型排名（%s）" % extra if extra else "模型排名"), fontsize=9)
    theme.sync_figure(fig)


def plot_dl_embedding(fig, result, ctx, params):
    """IND14：折外 embedding 的 PC1/PC2 散点 + 重建 NLL 最高的序列。"""
    embeddings = result.tables.get("embeddings")
    palette = colors()
    fig.clear()
    ax = fig.add_subplot(1, 2, 1)
    style(ax)
    if embeddings is None or not len(embeddings):
        blank(ax, "没有可用的折外 embedding")
    else:
        x = numeric(embeddings, "PC1").to_numpy(dtype=float)
        y = numeric(embeddings, "PC2").to_numpy(dtype=float)
        labels = [str(value) for value in embeddings["标签"]] if "标签" in embeddings else []
        if any(labels):
            mapping = color_map(sorted(set(labels)))
            for index, label in enumerate(labels):
                ax.scatter(x[index], y[index], s=26, alpha=0.85,
                           color=mapping.get(label, palette[0]),
                           marker="o" if label else "+")
            explained = (result.meta or {}).get("pca_explained") or []
            note = "、".join("%s %.0f%%" % ("PC%d" % (index + 1), 100.0 * float(value))
                         for index, value in enumerate(explained[:2]))
            ax.set_title("折外 embedding（按标签着色，%s）" % note if note
                         else "折外 embedding（按标签着色）", fontsize=9)
        else:
            hits = numeric(embeddings, "异常").to_numpy(dtype=float) \
                if "异常" in embeddings else np.zeros(len(embeddings))
            for index in range(len(embeddings)):
                ax.scatter(x[index], y[index], s=26, alpha=0.85,
                           color=palette[1] if hits[index] > 0.5 else palette[0])
            ax.set_title("没有可用标签：按是否标为异常着色", fontsize=9)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
    ax = fig.add_subplot(1, 2, 2)
    style(ax)
    if embeddings is None or not len(embeddings) or "重建NLL" not in embeddings:
        blank(ax, "没有可排的重建误差")
    else:
        part = embeddings.sort_values("重建NLL", ascending=False).tail(MAX_ITEMS)
        values = numeric(part, "重建NLL").to_numpy(dtype=float)
        marks = numeric(part, "异常").to_numpy(dtype=float)
        ax.barh(range(len(part)), values,
                color=[palette[1] if mark > 0.5 else palette[0] for mark in marks])
        ax.set_yticks(range(len(part)))
        fit_bar_rows(ax, len(part), row_h_in=ROW_H_IN, columns=2,
                     labels=["%s·%s" % (str(uid), str(number)) for uid, number in
                             zip(part["单元"], part["序列号"])])
        ax.set_xlabel("重建/遮码 NLL（红 = 异常）", fontsize=8)
        ax.set_title("谁最不象大家的轨迹", fontsize=9)
    theme.sync_figure(fig)


def plot_dl_gat(fig, result, ctx, params):
    """IND15：注意力最强的几条边 + 每组折外的“真实类概率”。"""
    attention = result.tables.get("attention")
    predictions = result.tables.get("predictions")
    palette = colors()
    fig.clear()
    ax = fig.add_subplot(1, 2, 1)
    style(ax)
    if attention is None or not len(attention):
        blank(ax, "没有可画的注意力权重")
    else:
        part = attention.sort_values("平均注意力").tail(MAX_ITEMS)
        values = numeric(part, "平均注意力").to_numpy(dtype=float)
        ax.barh(range(len(part)), values, color=palette[0])
        ax.set_yticks(range(len(part)))
        fit_bar_rows(ax, len(part), row_h_in=ROW_H_IN, columns=2,
                     labels=["%s→%s · 桶%s" % (str(source), str(target), bucket)
                             for source, target, bucket in
                             zip(part["source"], part["target"], part["桶"])])
        ax.set_xlabel("跨折平均注意力", fontsize=8)
        counts = numeric(part, "次数") if "次数" in part else None
        ax.set_title("模型在看哪几条边（条末数字 = 事件次数）" if counts is not None
                     else "模型在看哪几条边", fontsize=9)
        if counts is not None:
            for index in range(len(part)):
                ax.text(values[index], index, "  ×%d" % int(counts.iloc[index]),
                        va="center", fontsize=6.5, color=ink())
            ax.set_xlim(0.0, float(max(values.max(), 1e-9)) * 1.35)
    ax = fig.add_subplot(1, 2, 2)
    style(ax)
    if predictions is None or not len(predictions):
        blank(ax, "没有折外预测可画")
    else:
        part = predictions.sort_values("真实概率")
        values = numeric(part, "真实概率").to_numpy(dtype=float)
        hits = numeric(part, "预测正确").to_numpy(dtype=float)
        ax.barh(range(len(part)), values,
                color=[palette[2] if hit > 0.5 else palette[0] for hit in hits])
        ax.axvline(1.0 / max(len(dict.fromkeys([str(value) for value in part["真实"]])), 1),
                   color=ink("ink2"), linestyle="--", linewidth=1.0)
        ax.set_yticks(range(len(part)))
        fit_bar_rows(ax, len(part), row_h_in=ROW_H_IN, columns=2,
                     labels=["%s/%s→%s" % (str(group), truth, pred)
                             for group, truth, pred in
                             zip(part["组"], part["真实"], part["预测"])])
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("真实类的预测概率（绿 = 猜对，虚线 = 均匀分布）", fontsize=8)
        ax.set_title("LOYO 折外：每组恰好预测一次", fontsize=9)
    theme.sync_figure(fig)


def plot_group_prediction(fig, result, ctx, params):
    """GRP10：组级特征热图（按簇排序）+ 各模型 LOYO 折外 Macro-F1。"""
    frame = result.tables.get("group_features")
    clusters = result.tables.get("clusters")
    metrics = result.tables.get("metrics")
    fig.clear()
    ax = fig.add_subplot(1, 2, 1)
    style(ax)
    if frame is None or not len(frame):
        blank(ax, "组级特征表为空")
    else:
        columns = [column for column in frame.columns
                   if column not in ("group_id", "label")]
        shown = columns[:MAX_ITEMS]
        if len(columns) > MAX_ITEMS:
            # 截了就要说：只删不写等于把“图上少了什么”藏起来
            suffix = "（共 %d 个特征，仅展前 %d 个）" % (len(columns), MAX_ITEMS)
        else:
            suffix = ""
        order = list(frame["group_id"].astype(str))
        if clusters is not None and len(clusters):
            ranking = dict(zip(clusters["组"].astype(str), numeric(clusters, "簇")))
            order = sorted(order, key=lambda item: (int(ranking.get(item, 99)), item))
        matrix = frame.set_index(frame["group_id"].astype(str)).loc[order, shown]
        values = matrix.apply(lambda column: pd.to_numeric(column, errors="coerce")) \
            .fillna(0.0).to_numpy(dtype=float)
        span = values.std(axis=0, ddof=0)
        safe = np.where(span > 1e-12, span, 1.0)
        scaled = np.clip((values - values.mean(axis=0)) / safe, -3.0, 3.0)
        image = heatmap(ax, scaled, [str(item) for item in order],
                        title="组级特征 z 分数（行按簇排序）%s" % suffix,
                        x_labels=[str(item) for item in shown], panels_wide=2)
        ax.set_xlabel("", fontsize=8)
        ax.set_ylabel("", fontsize=8)
        if image is not None:
            color_bar(fig, image, ax)
    ax = fig.add_subplot(1, 2, 2)
    style(ax)
    if metrics is None or not len(metrics):
        blank(ax, "没有可比的模型指标（缺 outcome？）")
    else:
        means = metrics.groupby("模型")["Macro-F1"].mean().sort_values()
        palette = colors()
        ax.barh(range(len(means)), means.to_numpy(dtype=float),
                color=[palette[index % len(palette)] for index in range(len(means))])
        ax.axvline(float(means.min()) if len(means) else 0.0, color=ink("line"), linewidth=0.6)
        ax.set_yticks(range(len(means)))
        fit_bar_rows(ax, len(means), row_h_in=ROW_H_IN, columns=2,
                     labels=[str(item) for item in means.index])
        ax.set_xlim(0.0, 1.05)
        ax.set_xlabel("LOYO 折外 Macro-F1 均值", fontsize=8)
        ax.set_title("地板与网络同列对比", fontsize=9)
    theme.sync_figure(fig)


# ------------------------------------------------------------------ 派发

PLOTTERS = {
    "IND01": plot_lsa,
    "IND02": plot_markov,
    "IND03": plot_order,
    "IND04": plot_patterns,
    "IND05": plot_distance,
    "IND06": plot_dtw,
    "IND07": plot_clusters,
    "IND08": plot_hmm,
    "IND09": plot_segmentation,
    "IND10": plot_ena,
    "IND11": plot_ona,
    "IND12": plot_ml,
    "IND13": plot_dl_next_state,
    "IND14": plot_dl_embedding,
    "IND15": plot_dl_gat,
    "GRP01": plot_group_state,
    "GRP02": plot_group_entropy,
    "GRP03": plot_synchrony,
    "GRP04": plot_lead_lag,
    "GRP05": plot_group_transitions,
    "GRP06": plot_heterogeneity,
    "GRP07": plot_roles,
    "GRP08": plot_interaction,
    "GRP09": plot_network,
    "GRP10": plot_group_prediction,
}

TITLES = {
    "IND01": "LSA 调整残差", "IND02": "转移概率矩阵", "IND03": "选阶曲线",
    "IND04": "频繁模式", "IND05": "序列距离", "IND06": "DTW 距离与重心",
    "IND07": "聚类结果", "IND08": "HMM 发射与选模", "IND09": "变点与分段",
    "IND10": "ENA 网络坐标", "IND11": "ONA 有向网", "IND12": "特征重要度与逐折指标",
    "IND13": "next-state 模型对比", "IND14": "自监督表征与异常",
    "IND15": "GAT 注意力与折外预测",
    "GRP01": "组状态构成", "GRP02": "组内熵轨迹", "GRP03": "同步度",
    "GRP04": "领先-滞后", "GRP05": "组级残差", "GRP06": "异质性",
    "GRP07": "角色画像", "GRP08": "交互网络", "GRP09": "组级网络",
    "GRP10": "组比较与组预测",
}


def _figure_for(method_id, result, ctx, params, painter, strict=False, size=None):
    title = "%s · %s" % (method_id, TITLES.get(method_id, result.method_name))
    fig, _ax = new_figure("", height=4.2, size=size)
    fig.paint_error = None
    try:
        painter(fig, result, ctx, params)
    except Exception as error:                       # noqa: BLE001 图可以空，不能崩
        fig.paint_error = "%s: %s" % (type(error).__name__, error)
        if strict:
            raise
        fig.clear()
        ax = fig.add_subplot(111)
        style(ax)
        ax.axis("off")
        ax.text(0.5, 0.5, "这张图暂时画不出来：%s" % error, ha="center", va="center",
                fontsize=9, color=ink())
    # 图名在画完之后才贴：一半的 plotter 会 `fig.clear()` 重建面板，先贴就被清掉，
    # 导出的 PNG 上就看不出这是哪个方法的结果
    if title:
        fig.suptitle(title, color=ink("ink1"), fontsize=11)
        claim_margin(fig, height_in=SUPTITLE_IN)
    # 收尾不分方法：图例放哪、怎么排版都只这一处，新增方法不会再长出第三种图例行为；
    # `size is None` 就是量尺那一遍（画布故意只有 7.2×4.2），那一遍的装不下不算账
    finish(fig, params, quiet=size is None)
    return fig


def render(method_id, result, ctx, params, strict=False, avail=None):
    """返回 ``{图名: Figure}``；没有注册绘图函数的方法给空字典（工作台显示占位卡）。

    工作台走非严格模式（画坏了也不能卡住分析），冒烟测试走 strict 把异常顶出来。

    ``avail=(宽, 高)`` 是预览区能给多大（英寸），没给就用 ``DEFAULT_AVAIL_IN``。
    无论哪种都走两遍：第一遍在默认画布上量“内容到底需要多大”，第二遍在最终尺寸
    上重排。字号是绝对量（pt），只撑大画布不重画的话版式仍停在窄画布上，
    旋转角与抽稀都按错的宽度算过，标签照样叠 —— 与 statviz 同一个理由。
    """
    painter = PLOTTERS.get(method_id)
    if painter is None or result is None or getattr(result, "blocked", False):
        return {}
    name = TITLES.get(method_id, method_id)

    def build(size=None):
        return {name: _figure_for(method_id, result, ctx, params, painter, strict, size=size)}

    figures = build()
    if not figures:
        return figures
    width_in, height_in = (float(avail[0]) if avail else DEFAULT_AVAIL_IN[0],
                           float(avail[1]) if avail else DEFAULT_AVAIL_IN[1])
    drawn = {title: (fig.get_figwidth(), fig.get_figheight())
             for title, fig in figures.items()}
    targets = {title: fit_figure(fig, width_in, height_in) for title, fig in figures.items()}
    if any(abs(targets[title][0] - drawn[title][0]) > 0.05
           or abs(targets[title][1] - drawn[title][1]) > 0.05 for title in targets):
        figures = build(size=targets[name])
        fit_figure(figures[name], width_in, height_in)
    return figures
