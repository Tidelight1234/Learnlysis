"""排版冒烟：逐方法把图画出来，再量标签与图例的窗口矩形有没有互相压字。

判据不是「看起来挤」，而是拿 renderer 量每个 Text 的包围盒，两轴同时相交 >1 px 才算压字；
旋转 90° 的长标签只要留白不够就会互相切边，正是用户报的那个病。图外图例（below/right）
既不许压坐标区，也不许跑出画布——这两条在单测里只有小夹具，这里用真方法真数据全量扫一遍。

跑法（PowerShell）：

    cd version\v5; ..\..\..\.conda\python.exe -X utf8 _smoke_layout.py

依赖 `_testdata/`（先跑 `_gen_testdata.py`）；有 `_testdata_real/` 时再扫一趟真实数据——
GRP01 那个「六条色带被撑成 26 in 画布」的病只在真实数据（7 组 × 18 状态 × 188 时点）上暴露。
"""
import glob
import itertools
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg   # noqa: E402

import _smoke_modes as sm                          # noqa: E402  (导入即配好图表字体)
import analysis_core as ac                         # noqa: E402
import analysis_methods as am                      # noqa: E402
import analysis_plots as plots                     # noqa: E402

#: 两档预览区：小窗（用户默认）与大窗（选了「超大图」或最大化窗口）都得零压字
AVAILS = ((9.0, 5.0), (21.6, 12.0))
#: 图外档要额外验「不压坐标区」，图内档只验「不跑出画布」
OUTSIDE_LOCS = ("below", "right")
LEGS = ("below", "right", "best")
TOL_PX = 1.0


def renderer_of(fig):
    """裸 Figure 默认挂在 FigureCanvasBase 上（没有 renderer），量字之前先补一个。"""
    if not isinstance(fig.canvas, FigureCanvasAgg):
        FigureCanvasAgg(fig)
    return fig.canvas.get_renderer()


def intersects(box, other, tol=TOL_PX):
    """两个窗口矩形真重叠（两个轴都相交才算，贴边 1 px 以内不算）。"""
    return (min(box.x1, other.x1) - max(box.x0, other.x0) > tol
            and min(box.y1, other.y1) - max(box.y0, other.y0) > tol)


def label_clash(ax, which):
    """同一根轴上互相压字的刻度标签对数（以及参与计量的标签条数）。"""
    labels = ax.get_xticklabels() if which == "x" else ax.get_yticklabels()
    live = [item for item in labels if item.get_text() and item.get_visible()]
    if len(live) < 2:
        return 0, 0
    renderer = renderer_of(ax.figure)
    boxes = [item.get_window_extent(renderer) for item in live]
    bad = sum(1 for left, right in itertools.combinations(boxes, 2) if intersects(left, right))
    return bad, len(live)


def title_clash(fig):
    """图名与子图标题/轴标题互相压字的条数。"""
    renderer = renderer_of(fig)
    boxes = []
    head = getattr(fig, "_suptitle", None)
    if head is not None and head.get_text():
        boxes.append(("图名", head))
    for ax in fig.axes:
        if not ax.get_subplotspec():
            continue
        for item in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if item.get_text():
                boxes.append((item.get_text()[:16], item))
    bad = []
    for (left_name, left), (right_name, right) in itertools.combinations(boxes, 2):
        if intersects(left.get_window_extent(renderer), right.get_window_extent(renderer)):
            bad.append("%s <-> %s" % (left_name, right_name))
    return bad


def legend_check(fig, loc):
    """图例必须整块在画布内；图外档还不得压任何坐标区。"""
    renderer = renderer_of(fig)
    canvas = fig.bbox
    problems = []
    for legend in fig.legends:
        box = legend.get_window_extent(renderer)
        if (box.x0 < canvas.x0 - TOL_PX or box.y0 < canvas.y0 - TOL_PX
                or box.x1 > canvas.x1 + TOL_PX or box.y1 > canvas.y1 + TOL_PX):
            problems.append("图例跑出画布")
        if loc not in OUTSIDE_LOCS:
            continue
        for ax in fig.axes:
            if ax.get_visible() and ax.get_subplotspec() and intersects(box, ax.get_window_extent(renderer)):
                problems.append("图例压在坐标区上（%s）" % (ax.get_title()[:16] or "无标题"))
    return problems


def sweep(tag, bundles, hits):
    for avail in AVAILS:
        for mode in (am.MODE_INDIVIDUAL, am.MODE_GROUP):
            ctx = am.AnalysisContext(bundles, mode=mode)
            for spec in am.methods_for(mode):
                params = sm.default_params(spec, ctx)
                try:
                    out = spec.compute(ctx, params)
                except Exception as exc:                                # noqa: BLE001
                    hits.append("%s COMPUTE-FAIL %s %s" % (tag, spec.id, exc))
                    continue
                if out.blocked:
                    continue
                for loc in LEGS:
                    trial = dict(params)
                    trial["legend_loc"] = loc
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        figures = plots.render(spec.id, out, ctx, trial, strict=True, avail=avail)
                    for warning in caught:
                        if "layout" in str(warning.message).lower():
                            hits.append("%s WARN %.1f %s/%s -> %s"
                                        % (tag, avail[0], spec.id, loc, str(warning.message)[:70]))
                    for name, fig in figures.items():
                        size = "%.1fx%.1f" % (fig.get_figwidth(), fig.get_figheight())
                        for ax in fig.axes:
                            if not ax.get_visible() or not ax.get_subplotspec():
                                continue
                            xbad, _x = label_clash(ax, "x")
                            ybad, _y = label_clash(ax, "y")
                            if xbad or ybad:
                                hits.append("%s LABEL-OVERLAP %.1f %s/%s %s x=%d y=%d %s"
                                            % (tag, avail[0], spec.id, loc, name, xbad, ybad, size))
                        for text in title_clash(fig):
                            hits.append("%s TITLE-OVERLAP %.1f %s/%s %s %s"
                                        % (tag, avail[0], spec.id, loc, name, text))
                        for problem in legend_check(fig, loc):
                            hits.append("%s LEGEND %.1f %s/%s %s：%s"
                                        % (tag, avail[0], spec.id, loc, name, problem))
                        print("   %-6s %-5s %-11s avail=%.1f %s" % (tag, spec.id, loc, avail[0], size))


def main():
    paths = sorted(glob.glob(sm.DATA_GLOB))
    if not paths:
        print("缺 _testdata/，先跑 _gen_testdata.py"); return 2
    # 自检：故意把 30 个长标签挤在 3in 宽的面板上，探针必须报重叠，否则「零重叠」的结论不算数
    from matplotlib.figure import Figure                                # noqa: E402
    probe = Figure(figsize=(3, 2))
    ax = probe.add_subplot(111)
    ax.set_xticks(range(30))
    ax.set_xticklabels(["编码名称%02d" % index for index in range(30)], rotation=90, fontsize=8)
    bad, total = label_clash(ax, "x")
    print("PROBE %d/%d %s" % (bad, total, "OK" if bad else "DEAD-PROBE"))
    if not bad:
        return 2

    hits = []
    sweep("synth", ac.load_bundles(paths), hits)
    real_glob = sm.REAL_GLOB if os.path.isdir(os.path.dirname(sm.REAL_GLOB)) else sm.REAL_GLOB_RAW
    real = sorted(glob.glob(real_glob))[:sm.REAL_FILES]
    if real:
        sweep("real", ac.load_bundles(real), hits)
    else:
        print("（无真实数据副本，跳过 real 趟：跑 _normalize_realtest.py 生成）")

    for hit in hits:
        print(hit)
    print("LAYOUT_HITS=%d" % len(hits))
    print("LAYOUT_SMOKE_OK" if not hits else "LAYOUT_SMOKE_FAIL")
    return 0 if not hits else 1


if __name__ == "__main__":
    sys.exit(main())
