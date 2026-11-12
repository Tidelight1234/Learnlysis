"""离屏逐方法冒烟：IND / GRP 每个方法都要能出结果、图不能抛、质检要有预期条目。

跑三趟：合成数据（`_testdata/`）、真实数据（`_testdata_real/group_*.xlsx` 前若干个，
含逐方法落盘导出）、真工作台 UI 通道（选→生成→表/图/质检→一键导出）。

跑法（PowerShell）：
    cd version\\v5; $env:QT_QPA_PLATFORM="offscreen"; ..\\.conda\\python.exe -X utf8 _smoke_modes.py

torch / R 缺席时对应方法只断言"降级说明已给出"，不算失败；在场时 DL 只跑微型配置。
这是开发期脚本，不是产品代码（下划线前缀），也不进 requirements 链路。
"""

from __future__ import annotations

import glob
import os
import sys
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")

import analysis_core as ac                           # noqa: E402
import analysis_methods as am                        # noqa: E402


def configure_chart_fonts():
    """把图表字体设成和真应用一致（只动 rcParams，不依赖 Qt）。

    计算趟不建 QApplication，也就走不到 `theme.apply_theme` 里的 `_apply_matplotlib`；
    不补这一步 matplotlib 会退回 DejaVu Sans，中文标题全画成方框、日志全刷 glyph
    warning，导出图上看不出问题、冒烟输出也埋了。（产品链路里应用启动就调了，不重复。）
    """
    from matplotlib import font_manager

    import theme
    for fname in ("MiSans-Regular.ttf", "MiSans-Medium.ttf", "MiSans-Semibold.ttf"):
        path = os.path.join(theme.FONTS_DIR, fname)
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:                     # noqa: BLE001
                pass
    known = {entry.name for entry in font_manager.fontManager.ttflist}
    chosen = [family for family in ("MiSans", theme.FONT_FALLBACK, "Microsoft YaHei", "SimHei")
              if family in known]
    if not chosen:
        return
    matplotlib.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False


configure_chart_fonts()

DATA_GLOB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_testdata", "*.xlsx")

#: 真实数据（汇总页导出的标准分析格式）：完成定义要求 IND/GRP 每个方法在它上面也能出结果。
#: 读的是 `_testdata_real/` —— `Results/testfile/` 的副本，只把老导出的 `S1Eoding_1`
#: 拼写错误补成 `Encoding`（原目录受保护，只读不写），副本由 `_normalize_realtest.py` 生成。
REAL_GLOB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_testdata_real",
                         "group_*.xlsx")
#: 副本没生成时的退路：直接读原目录，只有两个文件能解析出成员（等于跑不了跨组方法）
REAL_GLOB_RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "Results", "testfile", "group_*.xlsx")
#: 只取前 N 个可用组：规范化后 24 个文件里有 23 个可解析（55 个独立学生），全跑一遍
#: 会把 OM/DTW 的距离矩阵推到五十多条序列；8 组已足够让 students>=8 与 groups>=3 成立
REAL_FILES = 8


def default_params(spec, ctx):
    params = {}
    for item in spec.params:
        params[item.key] = item.default
    params.setdefault("students", list(ctx.students))
    params.setdefault("dimensions", list(ctx.dimensions))
    if spec.engine == am.ENGINE_OPTIONAL:
        # 可选后端（torch）在场时只跑微型配置：冒烟验的是接线，不是收敛质量
        params.update({"epochs": 2, "folds": 2, "max_samples": 400, "max_features": 12,
                       "layers": 1, "heads": 2, "embed_dim": 8, "hidden_size": 8,
                       "stages": 2, "kernel_size": 2, "window": 4, "max_len": 32,
                       "buckets": 2, "width": 8, "out_dim": 4, "k": 2})
    return params


def check_result(spec, out, failures, ctx=None):
    tag = spec.id
    if out is None:
        failures.append("%s：返回 None" % tag)
        return False
    ok, note = spec.availability(ctx) if ctx is not None else (True, "")
    if out.blocked:
        # requires 没满足时阻断是正确行为，但必须把原因写清楚（方法不隐藏，只降级）
        if not ok:
            if not out.blocked_reason:
                failures.append("%s：条件不满足却没写降级原因" % tag)
            return False
        failures.append("%s：意外阻断（%s）" % (tag, out.blocked_reason))
        return False
    if not out.tables:
        failures.append("%s：结果表为空" % tag)
        return False
    for name, frame in out.tables.items():
        if frame is None or not len(getattr(frame, "index", [])):
            failures.append("%s：表 %s 是空表" % (tag, name))
    return True


def plot_result(spec, out, ctx, params, failures):
    """跑一遍绘图层，返回图集（导出通道要拿它往盘上写真文件）。"""
    try:
        import analysis_plots as plots
    except ImportError:
        return {}                           # 绘图层还没落地时不判失败
    try:
        figures = plots.render(spec.id, out, ctx, params, strict=True)
    except Exception:                     # noqa: BLE001
        failures.append("%s：绘图抛错\n%s" % (spec.id, traceback.format_exc()))
        return {}
    if spec.id not in plots.PLOTTERS:
        failures.append("%s：没有注册绘图函数" % spec.id)
    elif not figures:
        failures.append("%s：render 返回空图集" % spec.id)
    for name, fig in figures.items():
        if not fig.axes:
            failures.append("%s：图「%s」没有 axes" % (spec.id, name))
    return figures or {}


def export_channel(spec, out, ctx, params, figures, tag, failures):
    """把一次分析真写到盘上：完成定义里的「可导出」不是「内存里有表」。

    只在真实数据趟里跑：合成数据那一套已在 ui_channel 逐方法导过，而真实数据
    才有「时间列是纯数字、没有 group_id 列、各组人数不一」这类不常见的输入。
    """
    import shutil
    import tempfile

    import analysis_export as ae

    root = tempfile.mkdtemp(prefix="_smoke_modes_export_")
    try:
        written = ae.export_all(out, ctx, os.path.join(root, spec.id), spec=spec,
                                params=params, figures=figures, dpi=60,
                                formats=("png",)) or []
        names = [os.path.basename(path) for path in written]
        for prefix in ("tables_", "figure_", "run_config_", "run_audit_", "model_card_"):
            if not any(name.startswith(prefix) for name in names):
                failures.append("%s/%s：真实数据导出缺 %s（%s）"
                                % (tag, spec.id, prefix, sorted(names)[:4]))
        for path in written:
            if os.path.getsize(path) == 0:
                failures.append("%s/%s：导出文件为空 %s"
                                % (tag, spec.id, os.path.basename(path)))
    except Exception:                     # noqa: BLE001
        failures.append("%s/%s：导出抛错\n%s" % (tag, spec.id, traceback.format_exc()))
    finally:
        shutil.rmtree(root, ignore_errors=True)


#: R 对照要扫的参数变体（默认参数之外再推几个分支）
CROSSCHECK_VARIANTS = {
    "IND05": [{"method": "OM"}, {"method": "HAM"}, {"method": "DHD"},
              {"method": "MCO"}, {"method": "MCM"}],
}


def crosscheck_channel(spec, out, ctx, params, failures):
    """R 参考层通道：R 在场逐格断数值一致，不在场（或参照未启用）只断「有诚实说明」。"""
    try:
        import analysis_r_bridge as rb
    except ImportError:
        failures.append("R 对照：analysis_r_bridge 导入失败")
        return
    plans = getattr(rb, "CHECK_PLANS", {})
    if spec.id not in plans:
        declared = getattr(spec, "r_reference", "") or ""
        if declared and declared not in ("—", "-"):
            failures.append("%s：注册表声明 R 参照 %s，桥接层却没有对照计划"
                            % (spec.id, declared))
        return
    available = bool(rb.probe().get("available"))
    for override in CROSSCHECK_VARIANTS.get(spec.id, [{}]):
        trial = dict(params)
        trial.update(override)
        res = out
        if override:
            try:
                res = spec.compute(ctx, trial)
            except Exception:                     # noqa: BLE001
                failures.append("%s：R 对照变体 %s 计算抛错\n%s"
                                % (spec.id, override, traceback.format_exc()))
                continue
            if res.blocked:
                continue
        rows, note = rb.reference_check(spec, res, ctx, trial)
        tag = "%s%s" % (spec.id, "/" + ",".join("%s=%s" % kv for kv in sorted(override.items()))
                        if override else "")
        if not str(note or "").strip():
            failures.append("%s：R 通道没给任何说明（静默降级不合规）" % tag)
            continue
        if not available or not rows:
            continue                              # R 缺席 / 参照未启用：有说明就够
        full = (res.meta or {}).get("r_crosscheck") or rows
        if not (res.meta or {}).get("r_crosscheck"):
            failures.append("%s：差异表没写回 meta，导出会缺 r_crosscheck.csv" % tag)
        _agreed, failed, missing, worst = rb.summarize_rows(full)
        if failed or missing or (worst if worst is not None else 0.0) > 1e-6:
            failures.append("%s：R 对照超差 %d、缺失 %d、最大绝对差 %s" % (tag, failed, missing, worst))
            for row in rows[:3]:
                failures.append("    ! %s" % row)


#: 训练类参数的微型取值：参数卡上默认是“真分析”量级，冒烟只验接线、不求收敛
MICRO_TRAINING = {"epochs": 2, "folds": 2, "max_samples": 400, "max_features": 12,
                  "layers": 1, "heads": 2, "embed_dim": 8, "hidden_size": 8,
                  "stages": 2, "kernel_size": 2, "window": 4, "max_len": 32,
                  "buckets": 2, "width": 8, "out_dim": 4, "k": 2, "lr": "0.01"}


def shrink_training_widgets(module, spec, failures):
    """把可选后端方法的训练旋钮拧到最小（控件层，不只改 params 字典）。"""
    import analysis_params as ap

    rows = getattr(module, "_rows", []) or []
    keys = {param.key for param, _host, _widget in rows}
    wanted = {key: value for key, value in MICRO_TRAINING.items() if key in keys}
    if "epochs" not in wanted:
        failures.append("%s：可选后端方法的参数卡里没有 epochs 旋钮，微型配置无从生效"
                        % spec.id)
    ap.apply_params(rows, wanted)


def ui_channel(paths, failures):
    """真工作台通道：逐方法在 UI 上选→生成→看表/图/质检→导出。

    计算层单跑不碰控件，测不出「参数卡读回来的值」「质检卡有没有话」「导出文件名」
    这些环节 —— 这些只有在真控件上过一遍才算数。
    """
    import shutil
    import tempfile
    from PySide6.QtWidgets import QApplication

    from group_analysis import GroupAnalysisModule
    from individual_analysis import IndividualAnalysisModule

    app = QApplication.instance() or QApplication(sys.argv)
    try:
        import analysis_r_bridge as rb
        r_plans = set(rb.CHECK_PLANS)
        r_available = bool(rb.probe().get("available"))
    except ImportError:
        r_plans, r_available = set(), False
    r_checked = {}
    out_root = os.path.join(tempfile.mkdtemp(prefix="_smoke_modes_ui_"))
    try:
        for mode, module in ((am.MODE_INDIVIDUAL, IndividualAnalysisModule()),
                             (am.MODE_GROUP, GroupAnalysisModule())):
            ctx = module.load_files(paths)
            if ctx is None or not ctx.usable:
                failures.append("%s 工作台装载失败：%s" % (mode, module.data_hint.text()))
                continue
            specs = am.methods_for(mode)
            in_combo = {module.method_combo.itemData(i)
                        for i in range(module.method_combo.count())
                        if module.method_combo.itemData(i) is not None}
            missing = {spec.id for spec in specs} - in_combo
            if missing:
                failures.append("%s 工作台方法下拉缺失：%s" % (mode, sorted(missing)))
            for spec in specs:
                if module.set_method(spec.id) != spec.id:
                    failures.append("%s：下拉里选不到 %s" % (mode, spec.id))
                    continue
                ok, _note = spec.availability(ctx)
                if spec.engine == am.ENGINE_OPTIONAL:
                    shrink_training_widgets(module, spec, failures)
                result = module.generate()
                app.processEvents()
                if result is None and spec.engine == am.ENGINE_OPTIONAL and ok:
                    # 训练类方法必须已交 QThread（不得在 UI 线程算）：本方法验完就等它交货
                    if module.current_result() is not None:
                        failures.append("%s：可选后端方法在 UI 线程同步算完了" % spec.id)
                    if not module.join_compute(timeout_ms=1800000):
                        failures.append("%s：后台计算没在时限内结束" % spec.id)
                        continue
                    result = module.result
                if result is None:
                    failures.append("%s：UI generate() 返回 None" % spec.id)
                    continue
                if not ok:
                    if not (result.blocked and result.blocked_reason):
                        failures.append("%s：条件不满足时应阻断并写明原因" % spec.id)
                    if not module.method_hint.text() and not module.params_notice.text():
                        failures.append("%s：不可用方法没在界面上给说明" % spec.id)
                    continue
                if not check_result(spec, result, failures):
                    continue
                if result.blocked:
                    failures.append("%s：UI 里意外阻断（%s）" % (spec.id, result.blocked_reason))
                    continue
                if module.result_tabs.count() == 0:
                    failures.append("%s：结果标签页没填" % spec.id)
                if not module.figures:
                    failures.append("%s：预览无图" % spec.id)
                for name, fig in module.figures.items():
                    if not fig.axes:
                        failures.append("%s：预览图「%s」没有 axes" % (spec.id, name))
                if module.qa_list.count() == 0:
                    failures.append("%s：质检卡为空" % spec.id)
                written = module.export_all_to(os.path.join(out_root, spec.id))
                kinds = [os.path.basename(path) for path in (written or [])]
                for prefix in ("tables_", "figure_", "run_config_", "run_audit_",
                               "model_card_"):
                    if not any(name.startswith(prefix) for name in kinds):
                        failures.append("%s：一键导出缺 %s（%s）"
                                        % (spec.id, prefix, sorted(kinds)[:4]))
                if spec.id not in " ".join(kinds):
                    failures.append("%s：导出文件名没带方法号（%s）" % (spec.id, kinds[:3]))
                if spec.id in r_plans and not r_checked.get(mode):
                    # UI 通道只跑一个方法：验的是「按钮背后那条线」接上了没，不是重跑算法
                    r_checked[mode] = True
                    module.run_crosscheck(sync=True)
                    if not module.cross_hint.text().strip():
                        failures.append("%s：UI 里 R 校验卡没给说明" % spec.id)
                    if r_available and module.cross_table.rowCount() == 0:
                        failures.append("%s：UI 里 R 对照表为空" % spec.id)
                    again = module.export_all_to(os.path.join(out_root, spec.id + "_r")) or []
                    names = [os.path.basename(path) for path in again]
                    if r_available and not any(name.startswith("r_crosscheck") for name in names):
                        failures.append("%s：R 对照没进一键导出（缺 r_crosscheck.csv：%s）"
                                        % (spec.id, sorted(names)[:5]))
            state = module.archive_state()
            from analysis_workbench import ModeAnalysisWorkbench
            fresh = ModeAnalysisWorkbench(mode=mode)
            fresh.restore_state(state)
            if fresh.method_id() != state["method"]:
                failures.append("%s：存档往返没回到方法 %s（得到 %s）"
                                % (mode, state["method"], fresh.method_id()))
            fresh.setParent(None)
            fresh.deleteLater()
            module.setParent(None)
            module.deleteLater()
        # 导出目录必须真的落在盘上：只在内存里攒结果不算「可导出」
        names = [entry for _root, _dirs, files in os.walk(out_root) for entry in files]
        if len(names) < 10:
            failures.append("导出产物过少（%d 个文件）" % len(names))
        for path in ae_fig_paths(out_root):
            if os.path.getsize(path) == 0:
                failures.append("导出文件为空：%s" % os.path.basename(path))
    finally:
        shutil.rmtree(out_root, ignore_errors=True)


def ae_fig_paths(root):
    out = []
    for _dir, _sub, files in os.walk(root):
        out.extend(os.path.join(_dir, name) for name in files)
    return out


def sweep(bundles, tag, failures, with_crosscheck=True, with_export=False):
    """逐方法跑一遍计算层 + 绘图层（+ 可选 R 对照与落盘导出），返回本趟的合法阻断清单。

    `check_result` 已经区分了两种阻断：`requires` 未满足时阻断是正确行为（只要有原因），
    条件都满足却阻断则算失败。同一套口径既能跑合成数据，也能跑只有一组、没事件表的真实数据。
    但「不算失败」不等于「真的算了」：调用方拿得到 [(方法, 原因)] 才能分清这两件事，
    不会把降级报成通过。
    """
    blocked = []
    for mode in (am.MODE_INDIVIDUAL, am.MODE_GROUP):
        ctx = am.AnalysisContext(bundles, mode=mode)
        for spec in am.methods_for(mode):
            params = default_params(spec, ctx)
            try:
                out = spec.compute(ctx, params)
            except Exception:                       # noqa: BLE001
                failures.append("%s/%s：计算抛错\n%s" % (tag, spec.id, traceback.format_exc()))
                continue
            check_result(spec, out, failures, ctx)
            if not out.blocked:
                figures = plot_result(spec, out, ctx, params, failures)
                if with_crosscheck:
                    crosscheck_channel(spec, out, ctx, params, failures)
                if with_export:
                    export_channel(spec, out, ctx, params, figures, tag, failures)
            else:
                blocked.append((spec.id, out.blocked_reason or "（没写原因）"))
            if not spec.formula_note:
                failures.append("%s：没有口径说明 formula_note" % spec.id)
            if not spec.question:
                failures.append("%s：没有「回答什么问题」" % spec.id)
    return blocked


def report_blocked(tag, blocked):
    """把合法阻断打印出来：没打印的降级容易被当成「这个数据上一切方法都跑通了」。"""
    print("%s趟合法阻断 %d 项：%s" % (tag, len(blocked),
                                    "、".join("%s（%s）" % item for item in blocked) or "无"))


def main():
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print("缺少 _testdata，请先跑 _gen_testdata.py")
        return 2
    failures = []
    report_blocked("合成", sweep(ac.load_bundles(paths), "合成", failures))
    real = sorted(glob.glob(REAL_GLOB))
    if not real:
        # 副本没生成就退回原目录：那里 21 个文件是 `Eoding` 旧拼写，只剩 2 组可用
        real = sorted(glob.glob(REAL_GLOB_RAW))
        print("真实数据通道：缺 _testdata_real，改用原目录（只跑一次 "
              "_normalize_realtest.py 即可拿全 23 组）")
    if not real:
        failures.append("真实数据通道：找不到 %s" % REAL_GLOB)
    else:
        # 完成定义要求真实导出文件也能跑：时间列是纯数字、空表头变 int、各组人数不一、
        # 没事件表——这些“不干净”输入全部在这一趟里暴露。表头形状根本对不上的文件
        # （老版中文表头导出）装不出成员，跳过并把数目打印出来，不算失败也不算假装通过。
        bundles = ac.load_bundles(real)
        parsed = [bundle for bundle in bundles if bundle.members]
        print("真实数据通道：%d/%d 个文件可按列名契约解析，共 %d 个独立学生"
              % (len(parsed), len(bundles), sum(len(b.members) for b in parsed)))
        if len(parsed) < 2:
            failures.append("真实数据通道：可解析的组不足 2 个（%d），跑不了跨组方法" % len(parsed))
        before = len(failures)
        report_blocked("真实", sweep(parsed[:REAL_FILES], "真实", failures,
                                   with_crosscheck=False, with_export=True))
        if len(failures) == before:
            # 一个可以单独 grep 的哨兵：真实数据上逐方法算完且真的落盘了
            print("REAL_SWEEP_EXPORT_OK %d 组×逐方法导出全部落盘" % min(len(parsed), REAL_FILES))
    ui_channel(paths, failures)
    if failures:
        print("\n".join(failures))
        print("SMOKE_MODES_FAILED %d 项" % len(failures))
        return 1
    print("SMOKE_MODES_OK individual=%d group=%d"
          % (len(am.methods_for(am.MODE_INDIVIDUAL)), len(am.methods_for(am.MODE_GROUP))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
