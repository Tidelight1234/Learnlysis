"""Phase 3 本地训练脚手架离屏冒烟：降级 / 特征 / 数据集 / 训练 / 预测 / 存读 / Worker / 对话框闭环。

本机通常无 mediapipe/faster-whisper，正好验证「如实降级但仍可用序列特征训练」；训练跑真
sklearn（核心依赖），Worker 走真 QThread（轮询等完成），写回打到真实编码页验不变式（只填空、
不覆盖人工码、标注本地建议、可撤销）。save/load 用临时目录，全程不写 data/、不联网。
"""
import os
import sys
import time
import tempfile
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

import theme

app = QApplication(sys.argv)
theme.load_fonts()
theme.apply_theme(app, "light")

import auto_coding_train as act                   # noqa: E402
import auto_coding as ac                          # noqa: E402

fails = []


def check(label, fn):
    try:
        fn()
        app.processEvents()
        print("PASS  " + label)
    except Exception:
        print("FAIL  " + label)
        traceback.print_exc(file=sys.stdout)
        fails.append(label)


FW = [("互动类型", ["ST", "SI", "SQ"]), ("认知投入", ["高", "低"])]


def _pattern_rows(n=12):
    """可学习的确定性模式：dim0 交替 ST/SI（从 lag1 完全可判），dim1 每两行翻转。"""
    rows = []
    for i in range(n):
        rows.append(["ST" if i % 2 == 0 else "SI",
                     "高" if (i // 2) % 2 == 0 else "低", ""])
    return rows


def _solid_image(width, height, gray):
    """合成一张纯灰度 QImage（代替抽帧，不依赖 opencv/ffmpeg）。"""
    image = QImage(width, height, QImage.Format_RGB888)
    gray = max(0, min(255, int(gray)))
    image.fill(0xFF000000 | (gray << 16) | (gray << 8) | gray)
    return image


# ── 可选后端与降级 ──────────────────────────────────────────────────────
def _backends_and_degrade():
    backends = act.optional_backends()
    assert set(backends) == {"faster_whisper", "mediapipe", "opencv", "ffmpeg", "sklearn"}, backends
    assert backends["sklearn"] is True, "sklearn 是核心依赖，必须在场"
    present, table = act.av_backends_present()
    assert present == any(table.get(k) for k in ("mediapipe", "faster_whisper")), (present, table)
    hint = act.install_hint()
    assert "pip install mediapipe" in hint and "faster-whisper" in hint, hint
    # train_available：sklearn 在场 → 只卡样本数 / 类别数
    assert act.train_available(10, 3) == (True, ""), act.train_available(10, 3)
    ok, reason = act.train_available(2, 2)
    assert not ok and "太少" in reason, (ok, reason)
    ok2, reason2 = act.train_available(10, 1)
    assert not ok2 and "单一类别" in reason2, (ok2, reason2)


check("optional_backends / av 降级 / install_hint / train_available", _backends_and_degrade)


# ── 特征工程一致性 ──────────────────────────────────────────────────────
def _feature_consistency():
    names = act.sequence_feature_names(FW)
    # dim0: ST/SI/SQ + <空> = 4；dim1: 高/低 + <空> = 3 → 共 7（无时间特征）
    assert len(names) == 7, names
    assert "t" not in names, "已移除绝对时间特征，names 不该含 t"
    rows = _pattern_rows()
    for index in range(len(rows) + 1):           # 含 index==len(rows)（预测下一行）
        vec = act.sequence_feature_vector(rows, FW, index)
        assert len(vec) == len(names), (index, len(vec), len(names))
    # 第 0 行无 prev → 两维度都落 <空>
    v0 = act.sequence_feature_vector(rows, FW, 0)
    assert v0 == [0, 0, 0, 1, 0, 0, 1], v0
    # 第 1 行 prev=rows[0]=[ST,高] → dim0=ST(idx0), dim1=高(idx4)
    v1 = act.sequence_feature_vector(rows, FW, 1)
    assert v1[0] == 1.0 and v1[4] == 1.0 and sum(v1) == 2.0, v1


check("sequence_feature_names/vector 一致（含预测下一行）", _feature_consistency)


def _time_block_groups():
    g = act.time_block_groups(11, 5)
    assert len(g) == 11, g
    # 连续块：同块下标相邻，不交错
    for i in range(1, len(g)):
        assert g[i] == g[i - 1] or g[i] != g[i - 1], g
    seen = []
    for key in g:
        if not seen or seen[-1] != key:
            seen.append(key)
    assert len(seen) == len(set(seen)), ("块必须连续不回头", seen)
    assert act.time_block_groups(0, 5) == [], "空行数 → 空组"
    assert len(set(act.time_block_groups(3, 5))) <= 3, "块数不超过行数"


check("time_block_groups 连续分块（block CV 防泄漏）", _time_block_groups)


def _visual_features():
    # 空图 → []
    assert act.visual_features(None) == [] and act.visual_features(QImage()) == []
    img = QImage(32, 24, QImage.Format_RGB888)
    img.fill(0xFF808080)
    feats = act.visual_features(img, bins=4)
    # 2（均值/标准差）+ 4（直方图）= 6
    assert len(feats) == 6, feats
    assert abs(feats[0] - 128.0) < 2.0, ("灰度均值应≈128", feats[0])
    assert abs(sum(feats[2:]) - 1.0) < 1e-6, ("直方图应归一", feats[2:])


check("visual_features（合成 QImage / 空图降级）", _visual_features)


# ── 数据集构建 ─────────────────────────────────────────────────────────
def _build_dataset():
    rows = _pattern_rows(12)
    ds = act.build_dataset(rows, FW, 0, groups=act.time_block_groups(12, 4))
    assert ds["n"] == 12, ds["n"]                 # 全部在词表内
    assert ds["X"].shape == (12, 7), ds["X"].shape
    assert len(ds["feature_names"]) == 7 and ds["X"].shape[1] == len(ds["feature_names"])
    assert set(ds["y"]) == {"ST", "SI"}, ds["y"]
    assert len(set(ds["groups"])) == 4, ds["groups"]
    # 只收词表内非空标注：塞入越界码 + 空码，应被跳过
    rows2 = [["ST", "高", ""], ["乱码", "低", ""], ["", "高", ""], ["SI", "低", ""]]
    ds2 = act.build_dataset(rows2, FW, 0)
    assert ds2["n"] == 2 and ds2["y"] == ["ST", "SI"], ds2
    # 维度号越界 → n=0 + reason
    ds3 = act.build_dataset(rows, FW, 9)
    assert ds3["n"] == 0 and ds3.get("reason"), ds3


check("build_dataset（收词表内非空 / 跳过越界空 / 维度守卫）", _build_dataset)


def _build_dataset_with_visual():
    rows = _pattern_rows(6)
    groups = act.time_block_groups(6, 3)
    # 每个标注行都给一张合成帧 → 视听特征全有一致，并入
    frames = {i: _solid_image(16, 12, 40 + i * 20) for i in range(len(rows))}
    ds = act.build_dataset(rows, FW, 0, groups=groups, frames=frames)
    assert ds["visual"] is True, ds
    # 7 序列 + 8 视觉（2 统计 + 6 直方图 bins）= 15
    assert ds["X"].shape == (6, 15), ds["X"].shape
    assert len(ds["feature_names"]) == 15 and ds["feature_names"][-1] == "vis7", ds["feature_names"]
    # 缺帧（只给部分行）→ 诚实退回纯序列，不造参差矩阵
    ds2 = act.build_dataset(rows, FW, 0, groups=groups, frames={0: frames[0], 1: frames[1]})
    assert ds2["visual"] is False and ds2["X"].shape == (6, 7), (ds2["visual"], ds2["X"].shape)
    # 空图帧（isNull）→ visual_features 返回 [] → 退回纯序列
    ds3 = act.build_dataset(rows, FW, 0, groups=groups, frames={i: QImage() for i in range(6)})
    assert ds3["visual"] is False and ds3["X"].shape == (6, 7), ds3["visual"]
    # 并入视听特征后仍能训练（维度一致，不报错）
    result = act.train_coding_model(ds, "rf", folds=3)
    assert result["ok"], result


check("build_dataset 视听特征并入（全有或全无降级）", _build_dataset_with_visual)


# ── 训练 + 预测（纯逻辑） ───────────────────────────────────────────────
def _train_and_predict():
    rows = _pattern_rows(12)
    ds = act.build_dataset(rows, FW, 0, groups=act.time_block_groups(12, 4))
    result = act.train_coding_model(ds, "rf", folds=4)
    assert result["ok"], result
    assert result["engine"] and result["model"] is not None, result
    assert set(result["classes"]) == {"ST", "SI"}, result["classes"]
    assert result["per_fold"] and len(result["per_fold"]) >= 2, result["per_fold"]
    summary = result["summary"]
    assert 0.0 <= summary["balanced_accuracy"] <= 1.0, summary
    assert summary["folds"] == len(result["per_fold"]), summary
    # 交替模式完全可学：跨块泛化应显著高于随机（0.5）
    assert summary["balanced_accuracy"] >= 0.75, ("交替模式应学得动", summary)
    # 预测下一行（feat_index==len(rows)）：特征长度仍与训练一致，不报错
    code, proba = act.predict_dim_code(result, act.sequence_feature_vector(rows, FW, 12))
    assert code in ("ST", "SI", "SQ"), code
    assert proba is None or 0.0 <= proba <= 1.0, proba
    # 无模型 → 空码
    assert act.predict_dim_code({}, [0] * 7) == ("", None)
    assert act.predict_dim_code(None, [0] * 7) == ("", None)


check("train_coding_model 分组 CV + predict_dim_code", _train_and_predict)


def _train_guards():
    # 样本太少
    tiny = act.build_dataset(_pattern_rows(2), FW, 0)
    r = act.train_coding_model(tiny, "rf", folds=3)
    assert not r["ok"] and "太少" in r["reason"], r
    # 单一类别
    mono = act.build_dataset([["ST", "高", ""]] * 6, FW, 0)
    r2 = act.train_coding_model(mono, "rf", folds=3)
    assert not r2["ok"] and "单一类别" in r2["reason"], r2
    # 已取消的 ctx → ok=False + cancelled
    ds = act.build_dataset(_pattern_rows(12), FW, 0, groups=act.time_block_groups(12, 4))
    ctx = act.TrainContext(cancelled=lambda: True)
    r3 = act.train_coding_model(ds, "rf", folds=4, ctx=ctx)
    assert not r3["ok"] and r3.get("cancelled"), r3
    # TrainContext.report 回调可被驱动
    calls = []
    ctx2 = act.TrainContext(progress=lambda s, t, m: calls.append((s, t, m)))
    ctx2.report(1, 3, "x")
    assert calls == [(1, 3, "x")], calls


check("train 守卫（太少 / 单类 / 取消 / 进度回调）", _train_guards)


def _save_load_roundtrip():
    rows = _pattern_rows(12)
    ds = act.build_dataset(rows, FW, 0, groups=act.time_block_groups(12, 4))
    bundle = act.train_coding_model(ds, "rf", folds=4)
    assert bundle["ok"], bundle
    with tempfile.TemporaryDirectory() as folder:      # 临时目录，绝不写 data/
        path = os.path.join(folder, "coding_model.pkl")
        assert act.save_bundle(bundle, path), "保存应成功"
        assert os.path.exists(path) and os.path.getsize(path) > 0
        loaded = act.load_bundle(path)
        assert loaded is not None and loaded.get("model") is not None, loaded
        assert loaded["classes"] == bundle["classes"], loaded["classes"]
        code, _proba = act.predict_dim_code(loaded, act.sequence_feature_vector(rows, FW, 12))
        assert code in ("ST", "SI", "SQ"), code
    # 读不存在的文件 → None（不抛）
    assert act.load_bundle(os.path.join(tempfile.gettempdir(), "__no_such_model__.pkl")) is None


check("save_bundle / load_bundle 往返（临时目录）", _save_load_roundtrip)


# ── CodingTrainWorker（真 QThread） ────────────────────────────────────
def _worker_roundtrip():
    rows = _pattern_rows(12)
    ds = act.build_dataset(rows, FW, 0, groups=act.time_block_groups(12, 4))
    got = {}
    worker = act.CodingTrainWorker(ds, model_name="rf", folds=4)
    worker.progress.connect(lambda s, t, m: got.setdefault("progress", []).append((s, t, m)))
    worker.done.connect(lambda r: got.__setitem__("done", r))
    worker.error.connect(lambda m: got.__setitem__("error", m))
    worker.start()
    deadline = time.time() + 30.0
    while time.time() < deadline and "done" not in got and "error" not in got:
        app.processEvents()
        time.sleep(0.01)
    worker.wait(5000)
    assert "error" not in got, got.get("error")
    assert "done" in got and got["done"]["ok"], got
    assert got.get("progress"), "分组 K 折应至少回调一次进度"


check("CodingTrainWorker 真线程训练（progress + done）", _worker_roundtrip)


# ── LocalTrainDialog 闭环（打到真实编码页） ─────────────────────────────
from main_window import MainWindow                # noqa: E402

w = MainWindow()
w.show()
app.processEvents()
de = w.data_encoder


def _prep_train_encoder():
    de.selected_data = []
    de.row_times = []
    de.row_positions = []
    de.history = []
    de.result_table.setRowCount(0)
    de.dim_panel.apply_framework([("互动类型", ["ST", "SI", "SQ"]), ("认知投入", ["高", "低"])])
    de.time_unit_entry.setText("10")
    de.set_start_seconds(0)
    de.update_table_columns()
    de.update_action_state()
    for _ in range(12):
        de.continue_()
    for i, row in enumerate(de.selected_data):
        while len(row) < 3:
            row.append("")
        row[0] = "ST" if i % 2 == 0 else "SI"
        row[1] = "高" if (i // 2) % 2 == 0 else "低"
        row[2] = ""
    de.update_table()
    assert len(de.selected_data) == 12, len(de.selected_data)


def _wait_worker(dlg, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if dlg._worker is None:
            return True
        time.sleep(0.01)
    return False


def _dialog_construct_and_degrade():
    _prep_train_encoder()
    dlg = act.LocalTrainDialog(de)
    # 无视听后端时如实标注 + 安装命令（不静默假装具备视听识别）
    assert "视听后端" in dlg.backend_notice.text(), dlg.backend_notice.text()
    if not act.av_backends_present()[0]:
        assert "pip install" in dlg.backend_notice.text(), dlg.backend_notice.text()
    assert dlg.dim_combo.count() == 2, dlg.dim_combo.count()
    assert dlg._dim_index() == 0
    assert "12" in dlg.sample_label.text(), dlg.sample_label.text()
    assert dlg.train_btn.isEnabled(), "样本充足应可训练"
    assert not dlg.predict_btn.isEnabled() and not dlg.save_btn.isEnabled(), "训练前禁用预测/保存"


check("LocalTrainDialog 构造 + 降级标注 + 维度/样本刷新", _dialog_construct_and_degrade)


def _dialog_train_and_writeback():
    _prep_train_encoder()
    dlg = act.LocalTrainDialog(de)
    dlg.folds_spin.setValue(4)
    dlg._on_train()
    assert _wait_worker(dlg), "训练未在超时内完成"
    assert dlg._bundle is not None and dlg._bundle["ok"], dlg._bundle
    assert dlg.predict_btn.isEnabled() and dlg.save_btn.isEnabled()
    text = dlg.metrics.toPlainText()
    assert "balanced-acc" in text and "macro-F1" in text, text
    assert "人在环" in text, text

    # 预测写回：把当前时刻指到最后一行（t=110），清空其 dim0，预测应「只填空」写入 + 本地标记
    de.timepoint_edit.setText("00:01:50")
    de._on_timepoint_text_edited("00:01:50")
    times = de.computed_row_times()
    assert 110.0 in [float(t) for t in times], times
    row_index = times.index(110.0) if 110.0 in times else len(times) - 1
    de.selected_data[row_index][0] = ""           # 清空目标维度，制造可填的空位
    de.update_table()
    hist_before = len(de.history)
    dirty = []
    handler = lambda: dirty.append(1)
    de.content_changed.connect(handler)
    try:
        dlg._on_predict()
    finally:
        de.content_changed.disconnect(handler)
    filled = de.selected_data[row_index][0]
    assert filled in ("ST", "SI", "SQ"), ("应填入词表内建议码", filled)
    assert act.SUGGEST_MARKER_LOCAL in de.selected_data[row_index][-1], de.selected_data[row_index]
    assert len(de.history) == hist_before + 1, "写回应可撤销（压栈一次）"
    assert dirty, "写回应发 content_changed 脏标记"
    de.undo()
    assert de.selected_data[row_index][0] == "", ("撤销应还原为空", de.selected_data[row_index])


check("LocalTrainDialog 训练→预测→写回（只填空/标记/可撤销/脏标记）", _dialog_train_and_writeback)


def _dialog_no_overwrite_existing():
    """已有人工码的单元格：写回绝不覆盖，且全被占满时不误打「待确认」标记 / 不发脏 / 不压栈。"""
    _prep_train_encoder()
    # 直接验写回口径（不依赖模型预测的具体码、不过 worker）：row0 两维度都是人工码
    de.selected_data[0] = ["ST", "高", "人工原注"]
    de.update_table()
    assert de.selected_data[0][0] == "ST", de.selected_data[0]
    times = de.computed_row_times()
    dirty = []
    handler = lambda: dirty.append(1)
    de.content_changed.connect(handler)
    hist_before = len(de.history)
    try:
        out = ac.apply_suggestions(de, float(times[0]), [(0, "SI"), (1, "低")],
                                   marker=act.SUGGEST_MARKER_LOCAL)
    finally:
        de.content_changed.disconnect(handler)
    assert out["written"] == 0 and out["skipped"] == 2, out
    assert de.selected_data[0][0] == "ST" and de.selected_data[0][1] == "高", de.selected_data[0]
    # 未写入任何建议 → 不该把该行标成「待确认」（避免误导），也不发脏 / 不压栈
    assert act.SUGGEST_MARKER_LOCAL not in de.selected_data[0][-1], de.selected_data[0]
    assert "人工原注" in de.selected_data[0][-1], de.selected_data[0]
    assert not dirty, "全跳过不该发 content_changed"
    assert len(de.history) == hist_before, "全跳过不该压撤销栈"


check("写回：人工码占满时不覆盖/不误标/不发脏", _dialog_no_overwrite_existing)


def _dialog_predict_keeps_human_code():
    """对话框预测写回：当前时刻已有人工码时，绝不覆盖（无论模型预测什么）。"""
    _prep_train_encoder()
    dlg = act.LocalTrainDialog(de)
    dlg.folds_spin.setValue(3)
    dlg._on_train()
    assert _wait_worker(dlg), "训练未在超时内完成"
    de.timepoint_edit.setText("00:00:00")
    de._on_timepoint_text_edited("00:00:00")
    assert float(de.time_ctrl.parse_timepoint()) == 0.0, de.time_ctrl.parse_timepoint()
    de.selected_data[0] = ["ST", "高", "keep"]     # row0 两维度都是人工码
    de.update_table()
    assert de.selected_data[0][0] == "ST", ("predict 前人工码就丢了", de.selected_data[0])
    dlg._on_predict()
    assert de.selected_data[0][0] == "ST" and de.selected_data[0][1] == "高", \
        ("人工码被覆盖", de.selected_data[0])
    assert "keep" in de.selected_data[0][-1], de.selected_data[0]


check("LocalTrainDialog 预测不覆盖已有人工码", _dialog_predict_keeps_human_code)


def _dialog_no_framework_guard():
    de.clear()
    dlg = act.LocalTrainDialog(de)
    assert dlg.dim_combo.count() == 0, dlg.dim_combo.count()
    assert not dlg.train_btn.isEnabled(), "无框架应禁用训练"
    # 训练按钮守卫：直接调 _on_train 也不该崩（样本 0 → reason）
    dlg._on_train()
    app.processEvents()


check("LocalTrainDialog 无框架守卫（禁用 + 不崩）", _dialog_no_framework_guard)


def _encoder_button_guard():
    assert hasattr(de, "local_train_btn") and de.local_train_btn.text(), "缺 Phase 3 本地训练入口"
    de.clear()
    assert not de.dimension_cards
    de._open_local_train()          # 无维度 → 只提示后返回，不会 exec_ 阻塞
    app.processEvents()


check("encoder 本地训练入口与守卫", _encoder_button_guard)

print()
if fails:
    print("TRAIN_FAILED: %s" % fails)
    sys.exit(1)
print("TRAIN_OK")
