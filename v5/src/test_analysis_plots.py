"""IND / GRP 绘图层（`analysis_plots`）离线单测。

断的是这次改版立下的四条规矩，全部不依赖真实数据、不依赖 R / torch：

1. 图例按用户选的位置放，图外两档不许压住坐标区，图内档位不许长出第二套图例；
2. 类目标签放不下就把画布撑大，撑到上限才抽稀 —— 不许缩字，也不许叠字；
3. 矩阵面板按**实际列数**出刻度（长表摞起来的非方阵不得当成方阵画）；
4. 图名（suptitle）在 `fig.clear()` 之后仍然在，且不与子图标题重叠；
5. 多行分面的高度按行累加：不被「单面板基准高 × 行数」撑爆，也不被行与行平分挤扁。

运行方式：
    cd version/v5
    pytest test_analysis_plots.py -v
"""

import itertools
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
import pytest                                                   # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg     # noqa: E402
from matplotlib.figure import Figure                            # noqa: E402

_V5_DIR = os.path.dirname(os.path.abspath(__file__))
if _V5_DIR not in sys.path:
    sys.path.insert(0, _V5_DIR)

import analysis_methods as am                                   # noqa: E402
import analysis_plots as plots                                  # noqa: E402


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 量字的小工具：判据是渲染后的窗口矩形，不是“看起来应该没问题”
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def renderer_of(fig):
    """裸 ``Figure`` 挂在 FigureCanvasBase 上（没有 renderer），量之前先补一个。"""
    if not isinstance(fig.canvas, FigureCanvasAgg):
        FigureCanvasAgg(fig)
    return fig.canvas.get_renderer()


def intersects(box, other, tol=1.0):
    """两个窗口矩形真重叠（两个轴都相交才算，贴边 1 px 以内不算）。"""
    return (min(box.x1, other.x1) - max(box.x0, other.x0) > tol
            and min(box.y1, other.y1) - max(box.y0, other.y0) > tol)


def overlapping(texts):
    """同一根轴上互相压字的标签对数（1 px 以内的贴边不算）。"""
    live = [item for item in texts if item.get_text() and item.get_visible()]
    if len(live) < 2:
        return 0
    renderer = renderer_of(live[0].get_figure())
    boxes = [item.get_window_extent(renderer) for item in live]
    return sum(1 for left, right in itertools.combinations(boxes, 2) if intersects(left, right))


def probe_overlaps():
    """自检：故意把长标签挤在窄画布上，探针必须报重叠，否则下面的断言不算数。"""
    fig = Figure(figsize=(3, 2))
    ax = fig.add_subplot(111)
    ax.set_xticks(range(30))
    ax.set_xticklabels(["编码名称%02d" % index for index in range(30)],
                       rotation=90, fontsize=8)
    return overlapping(ax.get_xticklabels())


def twice(painter, avail=(9.0, 5.0), params=None):
    """照 `render` 的两遍口径画一次：量尺 → 定尺寸 → 重排。"""
    settings = {"legend_loc": "below"}
    settings.update(params or {})
    first = figure()
    painter(first)
    plots.finish(first, settings)
    size = plots.fit_figure(first, avail[0], avail[1])
    second = plots.new_figure("", height=4.2, size=size)[0]
    painter(second)
    plots.finish(second, settings)
    plots.fit_figure(second, avail[0], avail[1])
    return second


def figure(title=""):
    """`new_figure` 返回 (fig, axes)，这些测试只要画布。"""
    return plots.new_figure(title, height=4.2)[0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 夹具：与真实结果表同形状的小数据
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _sync_tables(members=("S1", "S2", "S3"), groups=("group_1", "group_2"),
                 dimensions=("D1", "D2")):
    """`sync_matrix` 是「每组每维度一个方阵」摞起来的长表（非方阵才是常态）。"""
    rows, pairs = [], []
    for group in groups:
        for dimension in dimensions:
            size = len(members)
            matrix = np.array([[0.0 if i == j else round(0.25 + 0.2 * i + 0.1 * j, 3)
                                for j in range(size)] for i in range(size)])
            for index, member in enumerate(members):
                record = {"成员": member}
                record.update({name: matrix[index, column]
                               for column, name in enumerate(members)})
                record.update({"group_id": group, "dimension": dimension})
                rows.append(record)
            for left in range(size):
                for right in range(left + 1, size):
                    pairs.append({"group_id": group, "dimension": dimension,
                                  "成员A": members[left], "成员B": members[right],
                                  "lag": 0, "agreement": matrix[left, right]})
    return (pd.DataFrame(rows, columns=["成员"] + list(members)
                         + ["group_id", "dimension"]),
            pd.DataFrame(pairs))


def _lead_lag_table(groups=("group_1",), dimensions=("D1", "D2"),
                    members=("S1", "S2", "S3")):
    rows = []
    for group in groups:
        for dimension in dimensions:
            for left in range(len(members)):
                for right in range(left + 1, len(members)):
                    rows.append({"group_id": group, "dimension": dimension,
                                 "成员A": members[left], "成员B": members[right],
                                 "最优lag": right - left, "领先者": members[left],
                                 "跟随者": members[right]})
    return pd.DataFrame(rows)


def _trajectory_table(groups=("group_1", "group_2", "group_3", "group_4",
                              "group_5", "group_6"),
                      states=("MA", "MB", "MC", "MD"), steps=24):
    """GRP01 的 `trajectory`：一行 = 一个 （组, 时间, 状态） 的份额。"""
    rng = np.random.default_rng(11)
    rows = []
    for group in groups:
        for time in range(0, steps * 10, 10):
            shares = rng.random(len(states))
            shares = shares / shares.sum()
            rows.extend({"group_id": group, "dimension": "D1", "time": time,
                         "state": state, "成员数": 3, "有码成员数": 3, "share": value}
                        for state, value in zip(states, shares))
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1 图例：位置由用户定，且不许压图
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _two_lines(fig):
    ax = fig.axes[0]
    ax.plot([0, 1], [0, 1], label="快速笔记")
    ax.plot([0, 1], [1, 0], label="口头解释")
    ax.legend()
    return ax


@pytest.mark.parametrize("loc", ["below", "right"])
def test_outside_legend_does_not_cover_axes(loc):
    fig = figure("统计图")
    ax = _two_lines(fig)
    plots.finish(fig, {"legend_loc": loc})
    renderer = renderer_of(fig)
    box = ax.get_window_extent(renderer)
    canvas = fig.bbox
    for legend in fig.legends:
        spot = legend.get_window_extent(renderer)
        assert not intersects(box, spot), "图例（%s）压在坐标区上" % loc
        # 图例在坐标区之外，纵向（below）或横向（right）必然与坐标区共轴，
        # 所以只能判整块相交；另外它必须整个落在画布里。
        assert spot.x0 >= -1 and spot.y0 >= -1 \
            and spot.x1 <= canvas.x1 + 1 and spot.y1 <= canvas.y1 + 1, \
            "图例（%s）跑出画布了" % loc


def test_inside_legend_stays_per_axes():
    fig = figure("统计图")
    ax = _two_lines(fig)
    plots.finish(fig, {"legend_loc": "best"})
    assert not fig.legends, "图内档位不该长出图形级图例"
    assert ax.get_legend() is not None


def test_legend_none_removes_everything():
    fig = figure("统计图")
    _two_lines(fig)
    plots.finish(fig, {"legend_loc": "none"})
    assert not fig.legends
    assert fig.axes[0].get_legend() is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2 标签：撑画布而不是缩字
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_probe_detects_squeezed_labels():
    assert probe_overlaps() > 0, "重叠探针失效，后面的断言都不算数"


def test_many_category_labels_do_not_overlap():
    labels = ["编码类别%02d" % index for index in range(24)]

    def paint(fig):
        ax = fig.axes[0]
        ax.bar(range(len(labels)), range(len(labels)))
        plots.fit_category_labels(ax, labels)

    fig = twice(paint)
    assert overlapping(fig.axes[0].get_xticklabels()) == 0
    assert fig.get_figwidth() > 9.0, "放不下就该把画布撑宽，而不是停在预览区宽度"


def test_heatmap_cell_labels_do_not_overlap():
    codes = ["状态%02d" % index for index in range(18)]

    def paint(fig):
        ax = fig.axes[0]
        matrix = np.random.default_rng(7).normal(size=(len(codes), len(codes)))
        plots.heatmap(ax, matrix, codes, title="调整残差")

    fig = twice(paint)
    ax = fig.axes[0]
    assert overlapping(ax.get_xticklabels()) == 0
    assert overlapping(ax.get_yticklabels()) == 0


def test_fit_figure_only_grows():
    fig = figure()
    plots.declare_size(fig, height_in=3.0, width_in=4.0)
    small = plots.fit_figure(fig, 4.0, 3.0)
    large = plots.fit_figure(fig, 12.0, 8.0)
    bigger = plots.fit_figure(fig, 4.0, 3.0)
    assert large[0] > small[0]
    assert bigger[0] >= large[0], "预览区变窄不该把已经撑开的画布缩回去"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3 矩阵：按实际列数出刻度；色条留在画布内
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_synchrony_panels_are_real_squares():
    matrix, pairs = _sync_tables()
    result = am.result("GRP03", "组内同步度", {"sync_matrix": matrix, "pairs": pairs})
    figures = plots.render("GRP03", result, None, {"legend_loc": "below"}, strict=True)
    fig = list(figures.values())[0]
    panels = [ax for ax in fig.axes if ax.get_images()]
    assert len(panels) == 4, "两维 × 两组应各出一块方阵"
    for ax in panels:
        assert len(ax.get_yticklabels()) == 3
        assert len(ax.get_xticklabels()) == 3, "非方阵不得按行数画列刻度"
    for ax in panels:
        bar = getattr(ax, "_ly_cbar", None)
        assert bar is not None and bar.get_position().x1 <= 1.0, "色条被推出了画布"


def test_lead_lag_keeps_every_dimension():
    frame = _lead_lag_table()
    result = am.result("GRP04", "领先-滞后", {"pairs": frame})
    figures = plots.render("GRP04", result, None, {"legend_loc": "none"}, strict=True)
    fig = list(figures.values())[0]
    labels = [item.get_text() for item in fig.axes[0].get_yticklabels() if item.get_text()]
    assert len(labels) == len(frame), "每个 (组, 维度, 对) 都该有一条"
    assert sum("D2" in text for text in labels) == 3, "另一维被静默丢了"


def test_suptitle_survives_fig_clear():
    matrix, pairs = _sync_tables()
    result = am.result("GRP03", "组内同步度", {"sync_matrix": matrix, "pairs": pairs})
    fig = list(plots.render("GRP03", result, None, {"legend_loc": "below"},
                            strict=True).values())[0]
    assert fig._suptitle is not None and "GRP03" in fig._suptitle.get_text()
    heads = [fig._suptitle] + [ax.title for ax in fig.axes
                               if ax.get_subplotspec() and ax.title.get_text()]
    assert overlapping(heads) == 0, "图名压住了子图标题"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4 多行分面：行与行累加，既不被基准高撑爆也不被平分挤扁
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_row_of_is_the_actual_grid_row():
    """`get_geometry()` 只给扁平行号：不换算成「第几行」，多行的底边会全挤在同一键上。"""
    fig = Figure(figsize=(8, 6))
    cells = [fig.add_subplot(2, 3, slot + 1) for slot in range(6)]
    assert [plots._row_of(ax) for ax in cells] == [1, 1, 1, 2, 2, 2]


def test_claim_row_height_sums_rows_and_merges_columns():
    fig = figure()
    plots.claim_row_height(fig, 1, 2.0)
    plots.claim_row_height(fig, 1, 3.5)        # 同行取最大
    plots.claim_row_height(fig, 2, 1.25)       # 行与行累加
    assert sum(fig._ly_row_need.values()) == pytest.approx(4.75)
    height = plots.planned_size(fig, 9.0, 5.0)[1]
    assert height >= 4.75, "逐行声明的总高不该在量尺里丢掉"


def test_multirow_bands_are_not_inflated_nor_squeezed():
    """GRP01：六条色带不该被「单面板基准高 × 行数」撑到顶天，也不该被压成细条。"""
    result = am.result("GRP01", "组状态构成", {"trajectory": _trajectory_table()})
    fig = list(plots.render("GRP01", result, None, {"legend_loc": "below"},
                            strict=True, avail=(9.0, 5.0)).values())[0]
    panels = [ax for ax in fig.axes if ax.get_subplotspec()]
    assert len(panels) == 6
    tall = min(ax.get_position().height for ax in panels) * fig.get_figheight()
    assert tall >= 1.2, "每行色带至少得读得出层次（实测 %.2f in）" % tall
    assert fig.get_figheight() < 18.0, "六条色带不该顶到画布高上限"
    assert overlapping(panels[0].get_xticklabels()) == 0


def test_bar_panel_inside_a_grid_keeps_its_labels():
    """多行分面里的条图：本行需求被当成全图需求时，行一平分标签就叠。"""
    matrix, pairs = _sync_tables(groups=tuple("group_%d" % n for n in range(1, 5)),
                                 dimensions=("D1", "D2", "D3"))
    result = am.result("GRP03", "组内同步度", {"sync_matrix": matrix, "pairs": pairs})
    fig = list(plots.render("GRP03", result, None, {"legend_loc": "below"},
                            strict=True, avail=(9.0, 5.0)).values())[0]
    bars = [ax for ax in fig.axes if not ax.get_images() and ax.get_yticklabels()
            and ax.get_subplotspec()]
    assert bars, "没找到两两 agreement 的条图面板"
    for ax in bars:
        assert overlapping(ax.get_yticklabels()) == 0, str(ax.get_title())[:20]


def test_right_legend_width_claim_ignores_row_count():
    """合并后的图形级图例只占一竖列：宽需求乘子图行数会白白撑宽几英寸。"""
    result = am.result("GRP01", "组状态构成", {"trajectory": _trajectory_table()})
    figures = plots.render("GRP01", result, None, {"legend_loc": "right"},
                           strict=True, avail=(9.0, 5.0))
    fig = list(figures.values())[0]
    width = fig.get_figwidth()
    assert width < 11.0, "六行图选「右侧」把画布撑到了 %.1f in" % width
    renderer = renderer_of(fig)
    for legend in fig.legends:
        spot = legend.get_window_extent(renderer)
        assert spot.x1 <= fig.bbox.x1 + 1 and spot.y0 >= -1, "右侧图例跑出画布"
