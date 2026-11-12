"""Phase 2 自动编码离屏冒烟：video_probe 降级 + auto_coding 解析/校验/写回 + 对话框闭环。

本机通常无 opencv/ffmpeg，正好验证「如实降级」；抽帧走桩（synthetic QImage），LLM 走桩
transport（不连真网），写回打到真实编码页验不变式（不覆盖人工码 / 标注待确认 / 可撤销 /
发脏标记）。全程不写 data/、不动直播播放器。
"""
import os
import sys
import json
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

import theme

app = QApplication(sys.argv)
theme.load_fonts()
theme.apply_theme(app, "light")

import video_probe as vp                          # noqa: E402
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


CODEBOOK = [("互动类型", ["ST", "SI", "SQ"]), ("认知投入", ["高", "低"])]


# ── video_probe：后端探测与降级 ──────────────────────────────────────────
def _vp_degrade():
    backends = vp.optional_backends()
    assert isinstance(backends, dict) and set(backends) == {"opencv", "ffmpeg"}, backends
    avail, table = vp.frames_available()
    assert avail == any(table.values()), (avail, table)
    if not avail:
        assert "pip install opencv-python" in vp.install_hint(), vp.install_hint()
    # 不存在的文件 → (None, 人话)，绝不抛
    img, note = vp.extract_frame("Z:/__definitely_missing__.mp4", 1.0)
    assert img is None and "不存在" in note, note
    # 存在但不是视频（本脚本自身）：无后端时给安装提示，有后端时给读不到帧的原因
    img2, note2 = vp.extract_frame(os.path.abspath(__file__), 0.0)
    assert img2 is None and note2, note2


check("video_probe 后端探测与降级", _vp_degrade)


def _vp_data_uri():
    img = QImage(64, 48, QImage.Format_RGB888)
    img.fill(0xFF2266AA)
    uri = vp.qimage_to_data_uri(img)
    assert uri.startswith("data:image/jpeg;base64,") and len(uri) > 60, uri[:48]
    assert vp.qimage_to_data_uri(img, fmt="PNG").startswith("data:image/png;base64,")
    # 超大图按最长边缩放（体积受控）：仍能编出合法 data URI
    big = QImage(4000, 2000, QImage.Format_RGB888)
    big.fill(0xFF112233)
    assert vp.qimage_to_data_uri(big, max_side=800).startswith("data:image/jpeg;base64,")
    # 空图 → 空串
    assert vp.qimage_to_data_uri(None) == "" and vp.qimage_to_data_uri(QImage()) == ""


check("video_probe QImage→data URI（含缩放/空图）", _vp_data_uri)


# ── auto_coding：prompt / 多模态消息 ─────────────────────────────────────
def _ac_prompt():
    system_text, user_text = ac.build_prompt(CODEBOOK, "我们先分工", 12.5)
    assert "ST、SI、SQ" in system_text and "认知投入" in system_text, system_text
    assert "JSON" in system_text, system_text
    assert "t=12.5" in user_text and "我们先分工" in user_text, user_text
    msgs = ac.coding_messages(CODEBOOK, ["data:image/jpeg;base64,AA"], "转写", 3.0)
    assert [m["role"] for m in msgs] == ["system", "user"], msgs
    assert isinstance(msgs[1]["content"], list), "带图应是多模态内容块"
    assert any(part.get("type") == "image_url" for part in msgs[1]["content"])
    # 无图 → 退化为纯文本 user 消息
    assert isinstance(ac.coding_messages(CODEBOOK, [], "x", 3.0)[1]["content"], str)


check("auto_coding prompt/多模态消息", _ac_prompt)


def _ac_parse():
    r = ac.parse_suggestions('{"互动类型":"ST","认知投入":"超高"}', CODEBOOK)
    assert r["ok"], r
    d = {x["dimension"]: x for x in r["dims"]}
    assert d["互动类型"]["code"] == "ST" and d["互动类型"]["in_vocab"], d
    assert d["认知投入"]["code"] == "超高" and not d["认知投入"]["in_vocab"], "越界必须标注"
    # 围栏 + 包一层 codes + 对象值带 reason
    r2 = ac.parse_suggestions('```json\n{"codes":{"互动类型":{"code":"SI","reason":"提问"}}}\n```',
                              CODEBOOK)
    d2 = {x["dimension"]: x for x in r2["dims"]}
    assert d2["互动类型"]["code"] == "SI" and d2["互动类型"]["in_vocab"], d2
    assert d2["互动类型"]["reason"] == "提问", d2
    assert d2["认知投入"]["code"] == "" and not d2["认知投入"]["in_vocab"], "模型没答→空"
    # 前后夹散文也能抠出 JSON
    r3 = ac.parse_suggestions('好的：{"互动类型":"SQ"} 供参考', CODEBOOK)
    assert r3["ok"] and {x["dimension"]: x["code"] for x in r3["dims"]}["互动类型"] == "SQ", r3
    # 完全不是 JSON → ok=False 且带错误说明、dims 空
    r4 = ac.parse_suggestions("我无法完成", CODEBOOK)
    assert not r4["ok"] and r4["error"] and r4["dims"] == [], r4


check("auto_coding JSON 解析与越界标注", _ac_parse)


# ── 码本释义导入：Excel 解析 / prompt 注入 / 设置往返 / 对话框入口 ────────
import tempfile                                       # noqa: E402
import pandas                                         # noqa: E402
import ai_settings                                    # noqa: E402


def _write_codebook_xls(dirpath, name, header, rows):
    path = os.path.join(dirpath, name)
    pandas.DataFrame([dict(zip(header, r)) for r in rows]).to_excel(path, index=False)
    return path


def _cb_read_variants():
    """两种列名写法都能宽松解析；无释义列也不报错，只提维度/编码。"""
    with tempfile.TemporaryDirectory() as d:
        p1 = _write_codebook_xls(
            d, "cn.xlsx", ["维度", "编码", "释义", "示例"],
            [["互动类型", "ST", "学生主动提问", "“老师能再讲讲吗？”"],
             ["互动类型", "SI", "学生间相互回应", "“我同意你的看法”"],
             ["认知投入", "高", "专注推进任务", ""]])
        notes = ac.read_codebook_excel(p1)
        assert set(notes) == {"互动类型", "认知投入"}, notes
        assert notes["互动类型"]["ST"]["desc"] == "学生主动提问", notes
        assert notes["互动类型"]["ST"]["example"] == "“老师能再讲讲吗？”", notes
        assert notes["认知投入"]["高"]["example"] == "", notes
        # 英文列头 + 「类型」代替「编码」
        p2 = _write_codebook_xls(
            d, "en.xlsx", ["dimension", "type", "description"],
            [["Interaction", "ASK", "asking a question"]])
        notes2 = ac.read_codebook_excel(p2)
        assert notes2["Interaction"]["ASK"]["desc"] == "asking a question", notes2


def _cb_read_rejects():
    """缺维度/编码列 → ValueError（人话），不静默返回空。"""
    with tempfile.TemporaryDirectory() as d:
        p = _write_codebook_xls(d, "bad.xlsx", ["名字", "备注"], [["x", "y"]])
        try:
            ac.read_codebook_excel(p)
            assert False, "应抛 ValueError"
        except ValueError as exc:
            assert "列" in str(exc), exc


def _cb_prompt_injection():
    """notes 在场时：逐码附注 + 多一条判据规则；不在场时保持原样。"""
    notes = {"互动类型": {"ST": {"desc": "主动提问", "example": "能不能再看看"}}}
    sys_t, _ = ac.build_prompt(CODEBOOK, "x", 1.0, code_notes=notes)
    assert "ST（释：主动提问" in sys_t, sys_t
    assert "据语义判断归属" in sys_t, sys_t
    plain, _ = ac.build_prompt(CODEBOOK, "x", 1.0)
    assert "ST（" not in plain and "据语义判断归属" not in plain, plain


def _cb_settings_roundtrip():
    """AiSettings 内存往返（offscreen 不写盘）：set → get 保真，脏行被洗掉。"""
    s = ai_settings.AiSettings()
    s.set_codebook_notes({"互动类型": {"ST": {"desc": "提问", "example": ""},
                                     "XX": {"desc": "", "example": ""}},  # 空释义→应被洗掉
                         "空维度": {}})
    got = s.codebook_notes()
    assert "XX" not in got.get("互动类型", {}), got
    assert got["互动类型"]["ST"]["desc"] == "提问", got
    assert "空维度" not in got, got
    assert s.codebook_notes_counts() == (1, 1), s.codebook_notes_counts()
    s.set_codebook_notes({})
    assert s.codebook_notes() == {}


def _cb_dialog_import():
    """真对话框：导入直传路径不弹框 → 状态标更新 → _notes() 取到 → 清除回空。"""
    _prep_encoder()
    with tempfile.TemporaryDirectory() as d:
        p = _write_codebook_xls(
            d, "cb.xlsx", ["维度", "编码", "描述", "示例"],
            [["互动类型", "ST", "主动提问", "能再看看吗"],
             ["认知投入", "高", "专注推进", ""]])
        st = ai_settings.AiSettings()
        dlg = ac.AutoCodingDialog(de, transport=_stub_transport, settings=st)
        assert "未导入码本释义" in dlg.codebook_label.text(), dlg.codebook_label.text()
        assert dlg._on_import_codebook(p) is True
        assert "已载入码本释义" in dlg.codebook_label.text(), dlg.codebook_label.text()
        assert dlg._notes()["互动类型"]["ST"]["desc"] == "主动提问", dlg._notes()
        # 坏文件 → 导入失败但不抛，状态栏写明原因
        bad = os.path.join(d, "bad.xlsx")
        pandas.DataFrame([{"名字": "x"}]).to_excel(bad, index=False)
        assert dlg._on_import_codebook(bad) is False
        assert "导入失败" in dlg.codebook_label.text(), dlg.codebook_label.text()
        dlg._on_clear_codebook()
        assert dlg._notes() == {}


check("码本释义 Excel 宽松解析（中/英文列头）", _cb_read_variants)
check("码本释义 缺列拒绝", _cb_read_rejects)
check("码本释义 prompt 注入", _cb_prompt_injection)
check("码本释义 AiSettings 往返", _cb_settings_roundtrip)


# ── auto_coding：写回编码页的不变式 ───────────────────────────────────
from main_window import MainWindow                # noqa: E402

w = MainWindow()
w.show()
app.processEvents()
de = w.data_encoder


def _prep_encoder():
    de.selected_data = []
    de.row_times = []
    de.row_positions = []
    de.history = []
    de.result_table.setRowCount(0)
    de.dim_panel.apply_framework([("互动类型", ["ST", "SI"]), ("认知投入", ["高", "低"])])
    de.time_unit_entry.setText("10")
    de.set_start_seconds(0)
    de.update_table_columns()
    de.update_action_state()


def _ac_apply_no_overwrite():
    """已有行：非空维度绝不覆盖，空的才填；注释追加标记不吞原文；可撤销；发脏标记。"""
    _prep_encoder()
    de.selected_data = [["ST", "", "已有注释"]]
    de.row_times = [None]
    de.row_positions = [0]
    de.update_table()
    dirty = []
    handler = lambda: dirty.append(1)
    de.content_changed.connect(handler)
    try:
        out = ac.apply_suggestions(de, 0.0, [(0, "SI"), (1, "高")])
    finally:
        de.content_changed.disconnect(handler)
    assert out["written"] == 1 and out["skipped"] == 1, out
    row = de.selected_data[0]
    assert row[0] == "ST", ("人工码被覆盖了", row)
    assert row[1] == "高", row
    assert ac.SUGGEST_MARKER in row[-1] and "已有注释" in row[-1], row
    assert dirty, "写回该发 content_changed"
    assert len(de.history) == 1, de.history
    de.undo()
    assert de.selected_data[0] == ["ST", "", "已有注释"], de.selected_data


check("auto_coding 写回：不覆盖/标注/可撤销/脏标记", _ac_apply_no_overwrite)


def _ac_apply_new_row_and_guards():
    _prep_encoder()
    de.continue_()
    de.continue_()
    assert de.computed_row_times() == [0, 10], de.computed_row_times()
    out = ac.apply_suggestions(de, 20.0, [(0, "SI"), (1, "低")])
    assert out["written"] == 2 and out["row_index"] is not None, out
    times = de.computed_row_times()
    assert 20.0 in times, times
    row = de.selected_data[times.index(20.0)]
    assert row[:2] == ["SI", "低"] and row[-1] == ac.SUGGEST_MARKER, row
    # 空建议 / 越界码（调用方已滤，这里再兜一层）→ 拒绝写回，给说明
    out2 = ac.apply_suggestions(de, 30.0, [])
    assert out2["written"] == 0 and out2["note"], out2
    # 无维度框架 → 拒绝写回
    de.clear()
    out3 = ac.apply_suggestions(de, 0.0, [(0, "ST")])
    assert out3["written"] == 0 and out3["note"], out3


check("auto_coding 写回：新建行 + 空/无框架守卫", _ac_apply_new_row_and_guards)


def _ac_find_row():
    _prep_encoder()
    de.continue_()
    de.continue_()
    assert ac.find_row_at_time(de, 0.0) == 0
    assert ac.find_row_at_time(de, 10.0) == 1
    assert ac.find_row_at_time(de, 999.0) is None


check("auto_coding find_row_at_time", _ac_find_row)


# ── 对话框闭环：桩 transport → 复核表 → 应用写回 ─────────────────────────
class _FakeSettings:
    def provider(self):
        return {"base_url": "https://stub/v1", "api_key": "k", "model": "m",
                "temperature": 0.2, "max_tokens": 256}


def _stub_transport(_payload):
    inner = json.dumps({"互动类型": "SI", "认知投入": "高"}, ensure_ascii=False)
    body = {"choices": [{"message": {"content": inner}}],
            "usage": {"total_tokens": 42, "prompt_tokens": 30, "completion_tokens": 12}}
    yield json.dumps(body, ensure_ascii=False)


def _wait_idle(dialog, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if dialog._worker is None and not dialog._queue:
            return True
        time.sleep(0.01)
    return False


def _ac_dialog_roundtrip():
    _prep_encoder()
    de.continue_()                                   # 一行 t=0
    de.timepoint_edit.setText("00:00:20")
    de._on_timepoint_text_edited("00:00:20")
    dlg = ac.AutoCodingDialog(de, transport=_stub_transport, settings=_FakeSettings())
    # 无抽帧后端时降级说明写在框里（不静默）
    assert "抽帧后端" in dlg.backend_notice.text(), dlg.backend_notice.text()
    dlg._refresh_time_points()
    assert dlg._checked_times() == [20.0], dlg._checked_times()
    dlg._on_run()
    assert _wait_idle(dlg), "worker 未在超时内完成"
    assert dlg.review_table.rowCount() == 2, dlg.review_table.rowCount()
    grouped = dlg._checked_review()
    assert 20.0 in grouped and len(grouped[20.0]) == 2, grouped
    rows_before, hist_before = len(de.selected_data), len(de.history)
    dlg._on_apply()
    assert len(de.selected_data) == rows_before + 1, de.selected_data
    times = de.computed_row_times()
    assert 20.0 in times, times
    row = de.selected_data[times.index(20.0)]
    assert row[:2] == ["SI", "高"] and ac.SUGGEST_MARKER in row[-1], row
    assert len(de.history) == hist_before + 1, de.history
    dlg._on_stop()


check("auto_coding 对话框闭环（桩传输→复核→写回）", _ac_dialog_roundtrip)


def _ac_dialog_out_of_vocab_not_checkable():
    """越界码进复核表但不可勾选、不能写回（人在环只放行词表内的）。"""
    _prep_encoder()
    de.continue_()
    de.timepoint_edit.setText("00:00:10")
    de._on_timepoint_text_edited("00:00:10")

    def bad_transport(_payload):
        inner = json.dumps({"互动类型": "乱码X", "认知投入": "高"}, ensure_ascii=False)
        yield json.dumps({"choices": [{"message": {"content": inner}}],
                          "usage": {"total_tokens": 9}}, ensure_ascii=False)

    dlg = ac.AutoCodingDialog(de, transport=bad_transport, settings=_FakeSettings())
    dlg._refresh_time_points()
    dlg._on_run()
    assert _wait_idle(dlg), "worker 未在超时内完成"
    assert dlg.review_table.rowCount() == 2, dlg.review_table.rowCount()
    grouped = dlg._checked_review()
    # 越界的「乱码X」不在可写回集合里，只有「高」被放行
    codes = [c for _i, c in grouped.get(10.0, [])]
    assert "乱码X" not in codes and "高" in codes, grouped


check("auto_coding 越界码不可写回", _ac_dialog_out_of_vocab_not_checkable)


def _encoder_button_guard():
    """编码页有「AI 辅助编码」按钮；无维度时 _open_auto_coding 走守卫（不开模态框、不崩）。"""
    assert hasattr(de, "auto_code_btn") and de.auto_code_btn.text(), "缺 AI 辅助编码入口"
    de.clear()
    assert not de.dimension_cards
    de._open_auto_coding()          # 无维度 → 只提示后返回，不会 exec_ 阻塞
    app.processEvents()


check("encoder AI 辅助编码入口与守卫", _encoder_button_guard)


check("码本释义 对话框导入/失败/清除", _cb_dialog_import)

print()
if fails:
    print("AUTOCODE_FAILED: %s" % fails)
    sys.exit(1)
print("AUTOCODE_OK")
