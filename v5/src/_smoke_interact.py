"""离屏交互冒烟测试：逐个触发被美化改动过的按钮回调，捕捉槽函数里的静默异常。

PySide6 槽函数中未捕获的异常会直接终止进程，所以这里逐个调用并单独捕获。
"""
import glob
import os
import sys
import tempfile
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QPointF, Qt, QEvent
from PySide6.QtGui import QMouseEvent, QKeyEvent
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (QApplication, QAbstractItemDelegate, QComboBox, QLineEdit,
                             QStyleOptionViewItem)

import theme

app = QApplication(sys.argv)
theme.load_fonts()
theme.apply_theme(app, "light")

from main_window import MainWindow

fails = []


def run(label, fn):
    try:
        fn()
        app.processEvents()
        print("PASS  " + label)
    except Exception:
        print("FAIL  " + label)
        traceback.print_exc(file=sys.stdout)
        fails.append(label)


def _click(x, y):
    p = QPointF(x, y)
    return QMouseEvent(QMouseEvent.MouseButtonPress, p, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)


def _click_at(widget):
    c = widget.geometry().center()
    return _click(c.x(), c.y())


w = MainWindow()
w.show()
app.processEvents()

ind, grp, de, ge = (w.individual_analysis, w.group_analysis,
                    w.data_encoder, w.group_encoder)
FIXED_COL = 2  # data_encoder.FIXED_COL_COUNT：时间 + 计时

run("main_window.add_new_tab", w.add_new_tab)
run("main_window.move_summary_to_end", w.move_summary_to_end)

def _reset_records():
    """只清表格数据：编码页的「清空」现在连框架一起清，用例之间造场景要留住维度"""
    de.selected_data = []
    de.row_times = []
    de.row_positions = []
    de.history = []
    de.result_table.setRowCount(0)
    de.update_table_columns()
    de.update_action_state()


def _timeline(unit="10", start=0, rows=0):
    """造一张干净时间轴：单维度框架 + 起始时间/间隔 + rows 条默认追加的顺序行"""
    _reset_records()
    de.dim_panel.apply_framework([("互动类型", ["ST", "SI"])])
    de.time_unit_entry.setText(unit)
    de.set_start_seconds(start)
    for _ in range(rows):
        de.continue_()


def _record_at(text):
    """手动改「当前时间」再点记录：走插入/覆盖那条路径"""
    de.timepoint_edit.setText(text)
    de._on_timepoint_text_edited(text)
    de.continue_()


# ── 个体 / 小组工作台：装载 → 选方法 → 生成 → 出表出图 → 导出 ──────────────
import analysis_methods as am                    # noqa: E402
import interaction_model as im                   # noqa: E402

_TESTDATA = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "_testdata", "*.xlsx")))


def _load_workbench(module):
    """有合成数据走批量上传，没有就退回当前会话取数：两条装载路都要跑过。"""
    ctx = module.load_files(_TESTDATA) if _TESTDATA else module.load_from_session()
    assert ctx is not None and ctx.usable, module.data_hint.text()
    return ctx


run("individual.load_data", lambda: _load_workbench(ind))
run("group.load_data", lambda: _load_workbench(grp))


def _workbench_flow(module, method_id):
    """切方法 → 生成 → 表页与预览图都要到位；不可用则必须写明阻断原因。"""
    assert module.set_method(method_id) == method_id, method_id
    result = module.generate()
    assert result is not None and result.method_id == method_id, method_id
    assert module.qa_list.count() > 0, "%s 质检卡为空" % method_id
    if result.blocked:
        assert result.blocked_reason, method_id
        return result
    assert result.tables, "%s 无结果表" % method_id
    assert module.result_tabs.count() > 0, "%s 未填表页" % method_id
    assert module.figures and module._canvases, "%s 无预览图" % method_id
    return result


for _module, _ids in ((ind, ("IND01", "IND02", "IND07")), (grp, ("GRP01", "GRP03", "GRP09"))):
    for _id in _ids:
        run("%s 工作台完整流程" % _id,
            lambda mod=_module, mid=_id: _workbench_flow(mod, mid))


def _unavailable_methods_explain_themselves():
    """不可用方法不隐藏：下拉项内附原因，选中后参数卡给出降级说明。"""
    for module in (ind, grp):
        for index in range(module.method_combo.count()):
            method_id = module.method_combo.itemData(index)
            if not method_id:
                continue
            spec = am.method_by_id(method_id)
            ok, note = spec.availability(module.ctx)
            if ok:
                continue
            assert note and note in module.method_combo.itemText(index), (method_id, note)
            module.set_method(method_id)
            result = module.generate()
            assert result is not None and result.blocked and result.blocked_reason, method_id


run("不可用方法降级说明", _unavailable_methods_explain_themselves)


def _group_workbench_table():
    """组清单：一行一组且小组模式多一列「事件来源」，只有组号可编辑。"""
    assert grp.group_table.rowCount() == grp.ctx.n_groups, grp.group_table.rowCount()
    headers = [grp.group_table.horizontalHeaderItem(col).text()
               for col in range(grp.group_table.columnCount())]
    assert "事件来源" in headers, headers
    assert grp.group_table.item(0, 1).flags() & Qt.ItemIsEditable
    assert not (grp.group_table.item(0, 0).flags() & Qt.ItemIsEditable)


run("group 组清单列与可编辑性", _group_workbench_table)


def _workbench_export():
    """一键导出：表 / 图 / config / audit / 模型卡全部落盘，同一 analysis_id 可追溯。"""
    ind.set_method("IND01")
    ind.generate()
    directory = os.path.join(tempfile.mkdtemp(prefix="learnlysis_wb_"), "out")
    written = ind.export_all_to(directory)
    assert written and os.path.isdir(directory), written
    names = sorted(os.listdir(directory))
    assert any(name.startswith("tables_") for name in names), names
    assert any(name.startswith("figure_") for name in names), names
    assert any(name.startswith("model_card_") for name in names), names
    assert any(name.startswith("run_config_") for name in names), names
    assert any(name.startswith("run_audit_") for name in names), names


run("workbench 一键导出", _workbench_export)


def _workbench_archive_roundtrip():
    """工作台只存方法/参数/批次名，不存数据本体。"""
    ind.set_method("IND02")
    state = ind.archive_state()
    assert state["method"] == "IND02" and state["mode"] == am.MODE_INDIVIDUAL
    ind.set_method("IND01")
    ind.restore_state(state)
    assert ind.method_id() == "IND02", ind.method_id()


run("workbench 存档往返", _workbench_archive_roundtrip)


# ── 小组交互页：记录 / 撤销 / 删行 / 校验拦截 ────────────────────────────
def _group_tab_mounted():
    """编码个体页 ≥ 2 才挂出「小组交互」，成员候选跟着编码页长。"""
    w._ensure_group_tab()
    inner = w.tabmodule_widget
    assert len(w.encoder_pages()) >= 2, w.encoder_pages()
    assert inner.indexOf(ge) >= 0, "小组交互页未挂出"
    assert inner.tabText(inner.indexOf(ge)) == "小组交互", inner.tabText(inner.indexOf(ge))
    assert ge.members == [page.student_id() for _t, page in w.encoder_pages()], ge.members


run("小组交互页随编码页数量出现", _group_tab_mounted)


def _set_moment(text):
    ge.time_ctrl.timepoint_edit.setText(text)
    ge.time_ctrl.mark_timepoint_edited()


def _pick(combo, value):
    """下拉只对已有项生效：设完回读确认，不要把「没选中」当成「选中了」。"""
    combo.setCurrentText(value)
    assert combo.currentText() == value, \
        (value, [combo.itemText(i) for i in range(combo.count())])


def _fill_event(**fields):
    for name, value in fields.items():
        widget = getattr(ge, name)
        if isinstance(widget, QComboBox):
            _pick(widget, str(value))
        else:
            widget.setText(str(value))


def _record(text=None, **fields):
    """填表 + 点记录，返回（前行数，后行数，状态栏提示）：拒收时行数不变。"""
    if text is not None:
        _set_moment(text)
    _fill_event(**fields)
    before = len(ge.event_rows)
    w.statusBar().showMessage("")        # 清掉上一条，下面的提示就是本次的结果
    ge.record_btn.click()
    return before, len(ge.event_rows), w.statusBar().currentMessage()


def _refused(message):
    return "拒绝写入" in message or "不能早于" in message, message


def _group_record_first_event():
    ge.clear_events()
    ge.time_ctrl.set_start_seconds(0)
    ge.time_ctrl.unit_edit.setText("10")
    before, after, hint = _record("00:00:10", actor_combo="S1", target_type_combo="TASK",
                                  target_id_combo="T1", interaction_combo="SPEAK",
                                  utterance_edit="我们先分工", turn_edit="T1")
    assert after == before + 1 == 1, (before, after, hint)
    row = ge.event_rows[0]
    assert row["actor_id"] == "S1" and row["timestamp"] == 10.0, row
    assert row["event_id"] == "%s_E001" % ge.group_id, row
    assert row[im.UTTERANCE_COLUMN] == "我们先分工", row
    assert ge.table.rowCount() == 1
    assert not ge.utterance_edit.text(), "记录后话语框应清空"


run("group_encoder 记录事件", _group_record_first_event)


def _group_allows_same_moment_events():
    """V09：同一时刻允许多条事件，不得去重。"""
    before, after, hint = _record("00:00:10", actor_combo="S2", target_type_combo="PERSON",
                                  target_id_combo="S1", interaction_combo="ASK")
    assert after == before + 1, (before, after, hint)
    assert [row["timestamp"] for row in ge.event_rows] == [10.0, 10.0], ge.event_rows
    assert len({row["event_id"] for row in ge.event_rows}) == 2, ge.event_rows


run("group_encoder 同窗多事件不去重", _group_allows_same_moment_events)


def _group_record_rejected_by_v02():
    """未登记主体 → V02 阻断：拒收不脏写，状态栏说明原因。"""
    members = list(ge.members)
    ge.members = []                 # 不刷下拉：表单仍停在 S1，但 S1 此刻已不是登记主体
    try:
        before, after, hint = _record("00:00:30")
    finally:
        ge.members = members
    assert after == before, (before, after, hint)
    assert not any(row["timestamp"] == 30.0 for row in ge.event_rows), ge.event_rows
    refused, hint = _refused(hint)
    assert refused, hint
    assert "V02" in hint, hint


run("group_encoder V02 拦截写入", _group_record_rejected_by_v02)


def _group_record_rejected_by_v10():
    """interaction_type 在词表外 → V10 阻断。"""
    before, after, hint = _record("00:00:40", actor_combo="S1", target_type_combo="ANSWER",
                                  interaction_combo="SHOUTING")
    assert after == before, (before, after, hint)
    refused, hint = _refused(hint)
    assert refused, hint
    assert "V10" in hint, hint


run("group_encoder V10 拦截写入", _group_record_rejected_by_v10)


def _group_record_rejected_outside_window():
    """超出观察窗口上界 → V03 阻断（窗口没定就上不封顶）。"""
    ge.observation_end = 45.0
    try:
        before, after, hint = _record("00:01:00", actor_combo="S1", target_type_combo="TASK",
                                      target_id_combo="T1", interaction_combo="SPEAK")
    finally:
        ge.observation_end = None
    assert after == before, (before, after, hint)
    refused, hint = _refused(hint)
    assert refused, hint
    assert "V03" in hint, hint


run("group_encoder 超出观察窗口被拒", _group_record_rejected_outside_window)


def _group_reply_candidates_and_undo_delete():
    """reply_to 只列更早事件；记录可撤销，删行可再撤销。"""
    _record("00:00:40", actor_combo="S1", target_type_combo="MATERIAL",
            target_id_combo="M1", interaction_combo="EXPLAIN")
    assert len(ge.event_rows) == 3, [row["timestamp"] for row in ge.event_rows]
    earlier, later = ge.event_rows[0]["event_id"], ge.event_rows[2]["event_id"]
    _set_moment("00:00:20")
    choices = [ge.reply_combo.itemText(i) for i in range(ge.reply_combo.count())]
    assert earlier in choices and later not in choices, choices
    # 指向更晚事件 → V07 拒收
    before, after, hint = _record(actor_combo="S1", target_type_combo="ANSWER",
                                  target_id_combo="A1", interaction_combo="ANSWER",
                                  reply_combo=later)
    assert after == before, (before, after, hint)
    assert "V07" in hint, hint
    # 指向更早事件 → 收下
    before, after, hint = _record(actor_combo="S1", target_type_combo="ANSWER",
                                  target_id_combo="A1", interaction_combo="ANSWER",
                                  reply_combo=earlier)
    assert after == before + 1 == 4, (before, after, hint)
    # 插到 20 s：时间序插入，不是接到表尾
    assert [row["timestamp"] for row in ge.event_rows] == [10.0, 10.0, 20.0, 40.0], \
        ge.event_rows
    inserted = [row for row in ge.event_rows if row["timestamp"] == 20.0]
    assert len(inserted) == 1 and inserted[0]["reply_to"] == earlier, ge.event_rows
    ge.undo_btn.click()
    assert len(ge.event_rows) == 3, ge.event_rows
    assert [row["timestamp"] for row in ge.event_rows] == [10.0, 10.0, 40.0], ge.event_rows
    ge.table.selectRow(0)
    key = QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)
    assert ge.eventFilter(ge.table, key) is True
    assert len(ge.event_rows) == 2, ge.event_rows
    ge.undo()
    assert len(ge.event_rows) == 3, ge.event_rows


run("group_encoder reply 候选/撤销/删行", _group_reply_candidates_and_undo_delete)


def _group_quality_report():
    """质量报告：现场放行过的警告 + 整表重跑 V01–V10；阻断类违例不该留在表里。"""
    ge.clear_events()
    ge.history = []
    ge.register_object("TASK", "T1")          # 先有字典，才能查「不在字典内」（否则 V05 无据可查）
    _record("00:00:10", actor_combo="S1", target_type_combo="TASK",
            target_id_combo="NOT-REGISTERED", interaction_combo="CONFIRM")
    before, after, hint = _record("00:00:10", actor_combo="S2", target_type_combo="NO_TARGET",
                                  target_id_combo="", interaction_combo="DIRECT")
    assert after == before + 1, (before, after, hint)
    items = ge.quality_items()
    assert items, "质量报告为空"
    codes = {item.code for item in items}
    assert "V05" in codes, codes                       # 对象不在字典 → 警告不拒收
    assert "V09" in codes, codes                       # 同窗多事件的并存说明
    assert not [item for item in items if item.level == "error"], \
        [item.message for item in items if item.level == "error"]


run("group_encoder 质量报告", _group_quality_report)


def _group_target_type_switches_candidates():
    """target_type 换类型 → target_id 候选跟着换，NO_TARGET 直接置灰。"""
    ge.register_object("TASK", "T9")
    ge.target_type_combo.setCurrentText("TASK")
    task_choices = [ge.target_id_combo.itemText(i)
                    for i in range(ge.target_id_combo.count())]
    assert "T9" in task_choices, task_choices
    ge.target_type_combo.setCurrentText("PERSON")
    person_choices = [ge.target_id_combo.itemText(i)
                      for i in range(ge.target_id_combo.count())]
    assert set(ge.members) <= set(person_choices), person_choices
    ge.target_type_combo.setCurrentText(im.NO_TARGET)
    assert not ge.target_id_combo.isEnabled()
    ge.target_type_combo.setCurrentText("TASK")
    assert ge.target_id_combo.isEnabled()


run("group_encoder 对象候选联动", _group_target_type_switches_candidates)


def _group_page_hidden_with_one_encoder():
    """只剩一页编码时摘掉小组页，但内存里的事件不能丢。"""
    kept = [row["event_id"] for row in ge.event_rows]
    extra = [index for index in range(w.tabmodule_widget.count())
             if isinstance(w.tabmodule_widget.widget(index), type(de))]
    if len(extra) < 2:
        return
    w.tabmodule_widget.removeTab(extra[-1])
    w._ensure_group_tab()
    assert w.tabmodule_widget.indexOf(ge) < 0, "编码页不足 2 页时小组页应摘掉"
    assert [row["event_id"] for row in ge.event_rows] == kept, "摘页不得丢事件"
    w.add_new_tab()
    w._ensure_group_tab()
    assert w.tabmodule_widget.indexOf(ge) >= 0, "回到 ≥2 页应原样挂回"


run("小组页按编码页数量增删", _group_page_hidden_with_one_encoder)


def _framework_hosts_are_encoder_only():
    """IND/GRP 不再有维度面板：框架复制只在编码页之间发生。"""
    hosts = w.framework_hosts()
    assert hosts, hosts
    from data_encoder import DataEncoderModule
    assert all(isinstance(widget, DataEncoderModule) for _label, widget in hosts), hosts
    assert not any(hasattr(widget, "dim_panel")
                   for widget in (ind, grp)), "新工作台不该带维度面板"


run("框架宿主只剩编码页", _framework_hosts_are_encoder_only)


def _framework_copy_between_encoder_pages():
    """两个编码页之间复制框架：维度名与编码原样过去，不带空编码。"""
    pages = [page for _title, page in w.encoder_pages()]
    assert len(pages) >= 2, len(pages)
    src, dst = pages[0], pages[1]
    src.dim_panel.apply_framework([("互动类型", ["ST", "SI"]), ("认知投入", ["A1", "A2", "A3"])])
    items = src.dim_panel.framework()
    assert len(items) == 2 and all(codes for _name, codes in items), items
    w.copy_framework(dst, items, w._framework_label(src))
    assert [card.get_codes() for card in dst.dimension_cards] == [codes for _n, codes in items], \
        dst.dim_panel.framework()
    assert all(entry["entry"].text().strip() for card in dst.dimension_cards for entry in card.blocks)
    menu = w._copy_framework_menu(dst, w.outer_tab_bar)
    assert menu is not None and menu.actions(), menu
    labels = [action.text() for action in menu.actions()]
    assert any("数据编码" in text or w._framework_label(src) in text for text in labels), labels
    assert not any("个体模式" in text or "小组模式" in text for text in labels), labels


run("编码页之间复制框架", _framework_copy_between_encoder_pages)

run("encoder.dim_panel add_dimension", de.dim_panel.add_dimension)
run("encoder.remove_module via x", lambda: de.dimension_cards[-1].remove_dim_btn.click())
if not de.dimension_cards:
    de.dim_panel.add_dimension()
card = de.dimension_cards[0]
run("encoder card add_block", card.add_block)
run("encoder card add_block x2", card.add_block)
run("encoder card edit/save_label", lambda: (card.edit_label(None), card.save_label()))
run("encoder card select block0", lambda: card.blocks[0]['checkbox'].click())
run("encoder card select block0 again", lambda: card.blocks[0]['checkbox'].click())
run("encoder card get_selected_entry", lambda: card.get_selected_entry())
run("encoder card reset", card.reset)
run("encoder card delete block via x", lambda: card.blocks[-1]['delete_btn'].click())
run("encoder card remove_block", card.remove_block)
run("encoder card relayout", card.relayout)
run("encoder card update_block_layout", card.update_block_layout)
run("encoder card toggle_blocks", card.toggle_blocks)
run("encoder card toggle_blocks again", card.toggle_blocks)
run("encoder card header click (blank area)", lambda: card.handle_header_click(_click(2, 2)))
run("encoder card header click (on x button)",
    lambda: card.handle_header_click(_click_at(card.remove_dim_btn)))

vp = w.video_player
run("video.play_segment (no media)", vp.play_segment)
run("video.replay_segment (no media)", vp.replay_segment)
run("video.seek_by(-5)", lambda: vp.seek_by(-5))
run("video.seek_by(+5)", lambda: vp.seek_by(5))
run("video.update_play_icon playing/paused",
    lambda: (vp.update_play_icon(QMediaPlayer.PlaybackState.PlayingState),
             vp.update_play_icon(QMediaPlayer.PlaybackState.PausedState)))
run("video speed combo change", lambda: vp.speed_combo.setCurrentIndex(5))
run("video duration slider change", lambda: vp.duration_slider.setValue(20))
run("video playlist normalize", lambda: vp.file_list.addItems([r"D:\fake\课堂互动_01.avi"]))
def _playlist_basename_ok():
    assert vp.file_list.item(0).text() == "课堂互动_01.avi", vp.file_list.item(0).text()
    assert vp.file_list.item(0).data(Qt.UserRole) == r"D:\fake\课堂互动_01.avi"


run("video playlist shows basename", _playlist_basename_ok)
run("video.highlight_current_file", vp.highlight_current_file)
run("video.load_video_from_list", lambda: vp.load_video_from_list(vp.file_list.item(0)))
run("video.set_position", lambda: vp.set_position(1500))
run("video.update_progress", vp.update_progress)
run("video.update_progress_bar", lambda: vp.update_progress_bar(100))
run("video.update_duration_bar", lambda: vp.update_duration_bar(1000))
def _timepoint_synced():
    vp.current_position = 262  # 00:04:22，应广播到编码区当前时间胶囊
    assert de.timepoint_edit.text() == "00:04:22", de.timepoint_edit.text()
    # 播放头推进（positionChanged）也要动态刷新，含超过一小时的进位
    vp.media_player.positionChanged.emit(3725000)
    assert de.timepoint_edit.text() == "01:02:05", de.timepoint_edit.text()
    vp.media_player.positionChanged.emit(262000)
    assert de.timepoint_edit.text() == "00:04:22", de.timepoint_edit.text()


run("video segment_start → encoder timepoint", _timepoint_synced)
def _record_at_time_logic():
    _reset_records()
    de.time_unit_entry.setText("10")
    de.set_start_seconds(0)
    de.selected_data = [["a", "c1"], ["b", "c2"]]
    de.row_times = [None, None]
    de.row_positions = [0, 1]
    de._record_at_time(["x", "ins"], 265)  # 不踩格点 → 末尾插入固定时间点行（不占格）
    assert de.row_times[-1] == 265 and de.row_positions[-1] is None
    assert de.computed_row_times() == [0, 10, 265], de.computed_row_times()
    de._record_at_time(["y", "ovr"], 0)  # 与首行 0s 重合 → 覆盖，行性质不变
    assert de.selected_data[0] == ["y", "ovr"] and de.row_times[0] is None
    de._record_at_time(["z", "tail"], 30)  # 表尾延长线上的格点 → 也是顺序行，接进骨架
    assert de.selected_data[2] == ["z", "tail"] and de.row_times[2] is None \
        and de.row_positions[2] == 3, (de.row_times, de.row_positions)
    de.time_unit_entry.setText("2")  # 改间隔：骨架上的行全重算，固定时间点行跳过
    assert de.computed_row_times() == [0, 2, 6, 265], de.computed_row_times()
    de.update_table()
    # 序号列已取消：第 0 列时间 HH:MM:SS，第 1 列计时（距起始时间的秒数）
    assert [de.result_table.horizontalHeaderItem(c).text() for c in range(2)] == ["时间", "计时"]
    assert [de.result_table.item(r, 0).text() for r in range(4)] == \
        ["00:00:00", "00:00:02", "00:00:06", "00:04:25"], \
        [de.result_table.item(r, 0).text() for r in range(4)]
    assert [de.result_table.item(r, 1).text() for r in range(4)] == ["0", "2", "6", "265"]
    # 起始时间变大后固定时间点行要按时间重新排位
    de.set_start_seconds(3600)
    assert de.computed_row_times() == [265, 3600, 3602, 3606], de.computed_row_times()
    assert de.result_table.item(0, 0).text() == "00:04:25"
    assert de.result_table.item(0, 1).text() == str(265 - 3600)
    de.set_start_seconds(0)
    assert de.computed_row_times() == [0, 2, 6, 265], de.computed_row_times()
    # 间隔放宽到 3600s 仍可（旧上限 30）
    de.time_unit_entry.setText("120")
    assert de.time_unit_entry.property("invalid") in (None, False)
    de.time_unit_entry.setText("10")
    assert de.computed_row_times() == [0, 10, 30, 265], de.computed_row_times()
    # 间隔 0 非法：时间轴沿用上一次的有效间隔，不会把所有行挤到同一时刻
    de.time_unit_entry.setText("0")
    assert de.time_unit_entry.property("invalid") is True
    assert de.computed_row_times() == [0, 10, 30, 265], de.computed_row_times()
    de._record_at_time(["w", "keep"], 45)  # 45 不踩 10s 的格点 → 固定时间点行
    assert de.computed_row_times() == [0, 10, 30, 45, 265], de.computed_row_times()
    assert de.row_times.count(45) == 1, de.row_times
    # 撞车：同一时刻上既有顺序行又有固定时间点行 → 优先覆盖固定行，格位留给时间轴骨架
    de.set_start_seconds(0)
    de.time_unit_entry.setText("5")
    de.selected_data = [["s1", ""], ["s2", ""], ["f", ""]]
    de.row_times = [None, None, 7]
    de.row_positions = [0, 1, None]
    de.time_unit_entry.setText("7")  # 骨架成 0/7，跟固定行 7 撞在同一时刻
    assert de.computed_row_times() == [0, 7, 7], de.computed_row_times()
    assert de._record_at_time(["hit", "ovr"], 7) is True
    assert de.selected_data[de.row_times.index(7)] == ["hit", "ovr"], de.selected_data
    assert de.row_times.count(None) == 2 and de.row_positions.count(None) == 1, \
        (de.row_times, de.row_positions)
    # 早于起始时间的记录一律拒收：起始时间是时间轴原点，格位不往负方向走
    de.set_start_seconds(100)
    assert de._record_at_time(["no", "?"], 90) is False
    assert len(de.selected_data) == 3, de.selected_data
    de.set_start_seconds(0)
    de.time_unit_entry.setText("10")
    _reset_records()


run("encoder record-at-time insert/overwrite", _record_at_time_logic)
def _delete_rows_in_table():
    """删行入口已内嵌到表格：选中行 + Delete/右键菜单，删后其余行时间不前移"""
    _timeline(rows=3)          # 0 10 20
    assert de.result_table.rowCount() == 3
    assert de.computed_row_times() == [0, 10, 20], de.computed_row_times()
    de.result_table.selectRow(1)
    assert de.selected_table_rows() == [1]
    # Delete 键路径（走事件过滤器）
    key = QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)
    run("encoder table Delete key", lambda: de.eventFilter(de.result_table, key))
    assert de.result_table.rowCount() == 2 and len(de.selected_data) == 2
    # 删掉中间行：其余行的时间/计时保持删除前的值，不随序号前移
    assert de.computed_row_times() == [0, 20], de.computed_row_times()
    assert [de.result_table.item(r, 0).text() for r in range(2)] == \
        ["00:00:00", "00:00:20"], [de.result_table.item(r, 0).text() for r in range(2)]
    assert [de.result_table.item(r, 1).text() for r in range(2)] == ["0", "20"]
    # 格位留着空档：新记录接在末行的下一格，不会被时间排序丢回表格中间
    de.continue_()
    assert de.computed_row_times() == [0, 20, 30], de.computed_row_times()
    # 改间隔对全部顺序行生效（被删行仍占一格，其余行按原始格距展开）
    de.time_unit_entry.setText("20")
    assert de.computed_row_times() == [0, 40, 60], de.computed_row_times()
    de.time_unit_entry.setText("10")
    assert de.computed_row_times() == [0, 20, 30], de.computed_row_times()
    de.undo()
    assert de.result_table.rowCount() == 2
    de.undo()
    assert de.result_table.rowCount() == 3
    assert de.computed_row_times() == [0, 10, 20], de.computed_row_times()
    # 未选中行时不删
    de.result_table.clearSelection()
    de.delete_selected_rows()
    assert de.result_table.rowCount() == 3
    # 多选删除
    de.result_table.selectAll()
    assert de.selected_table_rows() == [0, 1, 2]
    de.delete_selected_rows()
    assert de.result_table.rowCount() == 0


run("encoder delete rows in table", _delete_rows_in_table)
def _header_row_delete_locked():
    """起始时间行不可单独删（全选删除 = 清空表格数据则放行），时间轴原点不会漂"""
    _timeline(rows=3)          # 0 10 20
    assert de.header_row_index() == 0
    de.result_table.selectRow(0)
    de.delete_selected_rows()
    assert de.result_table.rowCount() == 3, "起始时间行被删了"  # 受保护，整批不删
    assert de.computed_row_times() == [0, 10, 20], de.computed_row_times()
    assert de.header_row_locked() is True and de.header_row_locked([1, 2]) is False
    de.result_table.clearSelection()
    de.result_table.selectRow(1)  # 其余行照旧可删（隐藏格）
    de.delete_selected_rows()
    assert de.computed_row_times() == [0, 20], de.computed_row_times()
    # 改大起始时间：固定的早行排到表头之前是允许的，起始时间仍跟着表头行
    de.time_unit_entry.setText("20")
    _record_at("00:00:05")     # 5 不踩 20s 的格点 → 固定时间点行
    assert de.computed_row_times() == [0, 5, 40], de.computed_row_times()
    de.set_start_seconds(10)
    assert de.computed_row_times() == [5, 10, 50], de.computed_row_times()
    assert de.header_row_index() == 1, de.row_positions  # 表头行排在固定行之后
    de.timepoint_edit.setText("00:00:03")
    de._on_timepoint_text_edited("00:00:03")
    pushed = len(de.history)
    de.continue_()               # 早于起始时间 → 被拒，行数不变、编辑态保留重试
    assert len(de.selected_data) == 3, de.selected_data
    assert de._timepoint_edited is True
    assert len(de.history) == pushed  # 被拒的记录不进撤销栈，不会多一步空回退
    de._reset_timepoint()
    de.undo()                    # 回退的是上一次真改动（5s 那条固定时间点行）
    assert de.computed_row_times() == [10, 50], de.computed_row_times()
    de.result_table.selectAll()
    de.delete_selected_rows()    # 全选删 = 清空表格数据，放行
    assert de.result_table.rowCount() == 0 and not de.selected_data
    de.continue_()               # 空表重新记录：又从起始时间起步
    assert de.computed_row_times() == [10], de.computed_row_times()
    assert de.row_positions == [0], de.row_positions
    de.set_start_seconds(0)
    de.time_unit_entry.setText("10")
    _reset_records()


run("encoder start row delete protection", _header_row_delete_locked)


def _record_refills_deleted_slot():
    """踩格点的记录一律入时间轴骨架（补回隐藏格或接在尾部），不踩格点的才是固定时间点行"""
    _timeline(rows=4)          # 0 10 20 30
    assert de.computed_row_times() == [0, 10, 20, 30], de.computed_row_times()
    de.result_table.selectRow(1)
    de.delete_selected_rows()  # 删掉 10s 行，格位 1 留成空档
    assert de.computed_row_times() == [0, 20, 30], de.computed_row_times()
    _record_at("00:00:10")     # 踩回空档 → 覆盖式补回，整个时间轴回到最原始的状态
    assert de.computed_row_times() == [0, 10, 20, 30], de.computed_row_times()
    assert de.row_times == [None, None, None, None], de.row_times
    assert de.row_positions == [0, 1, 2, 3], de.row_positions
    de.time_unit_entry.setText("5")  # 补回的行是顺序行，跟着间隔一起走
    assert de.computed_row_times() == [0, 5, 10, 15], de.computed_row_times()

    _timeline(rows=4)          # 0 10 20 30
    de.result_table.selectRow(1)
    de.delete_selected_rows()  # 0 20 30，格位 [0,2,3]
    de.time_unit_entry.setText("5")
    assert de.computed_row_times() == [0, 10, 15], de.computed_row_times()
    _record_at("00:00:06")     # 6 不踩格点 → 普通插入（固定时间行）
    assert de.computed_row_times() == [0, 6, 10, 15], de.computed_row_times()
    assert de.row_times.count(6) == 1, de.row_times
    de.undo()
    assert de.computed_row_times() == [0, 10, 15], de.computed_row_times()
    _record_at("00:00:05")     # 5 = 被删行现在的格点时间 → 补回，回到 0 5 10 15
    assert de.computed_row_times() == [0, 5, 10, 15], de.computed_row_times()
    assert de.row_times == [None, None, None, None], de.row_times
    assert de.row_positions == [0, 1, 2, 3], de.row_positions
    _record_at("00:00:20")     # 表尾延长线上的格点 → 也当顺序行接在骨架尾部
    assert de.computed_row_times() == [0, 5, 10, 15, 20], de.computed_row_times()
    assert de.row_positions == [0, 1, 2, 3, 4], de.row_positions
    de.time_unit_entry.setText("10")  # 新行照样跟着间隔走
    assert de.computed_row_times() == [0, 10, 20, 30, 40], de.computed_row_times()

    _timeline("5", rows=3)     # 0 5 10
    _record_at("00:00:06")     # → 0 5 6 10（6 是固定时间点行）
    assert de.computed_row_times() == [0, 5, 6, 10], de.computed_row_times()
    de.result_table.selectRow(2)
    de.delete_selected_rows()  # 删掉 6：它本来就不在骨架上，抹掉即回到原样
    assert de.computed_row_times() == [0, 5, 10], de.computed_row_times()
    assert de.row_positions == [0, 1, 2], de.row_positions
    de.time_unit_entry.setText("10")  # 改间隔：跟没插过 6 一样
    assert de.computed_row_times() == [0, 10, 20], de.computed_row_times()
    # 骨架没建立时（表里只有固定时间点行）：默认追加钉在末行之后，不会抢表头的起始时间
    _timeline()                # 空表，间隔 10s
    _record_at("00:00:20")     # 空表无从推格距 → 固定时间点行
    assert de.computed_row_times() == [20], de.computed_row_times()
    de.continue_()             # 默认追加接在末行之后，不会掉到表格最前面
    assert de.computed_row_times() == [20, 30], de.computed_row_times()
    assert de.row_times == [20, 30], de.row_times
    _reset_records()


run("encoder record refills deleted grid slot", _record_refills_deleted_slot)


def _archive_roundtrip_keeps_timeline():
    """存档往返要保住时间轴：隐藏格照旧隐藏、固定时间点行照旧固定、旧存档按行号补骨架"""
    _timeline(rows=4)          # 0 10 20 30
    de.result_table.selectRow(1)
    de.delete_selected_rows()          # 骨架上留一格隐藏：[0,2,3]
    _record_at("00:00:05")             # 5 不踩格点 → 固定时间点行
    assert de.computed_row_times() == [0, 5, 20, 30], de.computed_row_times()
    import copy as _copy
    import json as _json
    packed = _json.loads(_json.dumps(de.to_dict()))  # 走一遍 JSON：None 格位存取自如
    _reset_records()
    de.from_dict(_copy.deepcopy(packed))
    assert de.computed_row_times() == [0, 5, 20, 30], de.computed_row_times()
    assert de.row_times == [None, 5, None, None], de.row_times
    assert de.row_positions == [0, None, 2, 3], de.row_positions
    de.time_unit_entry.setText("20")   # 往返后间隔照样按原始格距展开
    assert de.computed_row_times() == [0, 5, 40, 60], de.computed_row_times()
    # 旧存档：只记了 row_times、没有格位 → 按行号还原骨架（中间那行仍是固定时间点行）
    legacy = {"dimension_cards": [{"name": "互动类型", "codes": ["ST", "SI"]}],
              "selected_data": [["a", ""], ["b", ""], ["c", ""]],
              "row_times": [None, 5, None], "start_seconds": 0, "time_unit": "10"}
    _reset_records()
    de.from_dict(legacy)
    assert de.computed_row_times() == [0, 5, 20], de.computed_row_times()
    assert de.row_positions == [0, None, 2], de.row_positions
    _reset_records()


run("encoder archive roundtrip keeps timeline", _archive_roundtrip_keeps_timeline)


def _time_column_fits_hhmmss():
    """时间列宽按字体量：HH:MM:SS 不能被省略号截断，字体变大时列要跟着长"""
    from timeline_axis import (CELL_H_PADDING, ELAPSED_COL_MIN_WIDTH, TIME_COL_MIN_WIDTH,
                               TIME_COL_SAMPLE, ELAPSED_COL_SAMPLE)
    probe = de.result_table
    fm = probe.fontMetrics()
    assert probe.columnWidth(0) >= fm.horizontalAdvance(TIME_COL_SAMPLE) + CELL_H_PADDING, \
        (probe.columnWidth(0), fm.horizontalAdvance(TIME_COL_SAMPLE))
    assert probe.columnWidth(1) >= fm.horizontalAdvance(ELAPSED_COL_SAMPLE) + CELL_H_PADDING
    assert probe.columnWidth(0) >= TIME_COL_MIN_WIDTH and probe.columnWidth(1) >= ELAPSED_COL_MIN_WIDTH
    # 模拟高分屏/字体回退：字体放大一档，重算出来的列宽必须跟着变宽
    font = probe.font()
    base_px = font.pixelSize()
    font.setPixelSize(base_px + 8)
    probe.setFont(font)
    try:
        grown = probe.fontMetrics().horizontalAdvance(TIME_COL_SAMPLE)
        if grown > fm.horizontalAdvance(TIME_COL_SAMPLE):  # QSS 抢了字体时 setFont 无效，那就只验前段
            de.apply_column_widths()
            assert probe.columnWidth(0) >= grown + CELL_H_PADDING, (probe.columnWidth(0), grown)
    finally:
        font.setPixelSize(base_px)
        probe.setFont(font)
        de.apply_column_widths()


run("encoder time column wide enough", _time_column_fits_hhmmss)
def _column_widths_single_strategy():
    """列宽只有一个出口：均分是地板、内容宽封顶到 CONTENT_COL_MAX，不会再两套策略互跳"""
    from timeline_axis import CODE_COL_MIN_WIDTH, CONTENT_COL_MAX
    probe = de.result_table
    _timeline(rows=3)
    flex = list(range(FIXED_COL, probe.columnCount()))
    # 旧的内容自适应函数已并入 apply_column_widths，留着它就是第二次覆盖的来源
    assert not hasattr(de, "adjust_column_widths")
    time_w, elapsed_w = de.fixed_column_widths()
    # ① 宽表不留右侧空白：弹性列之和要铺满视口减两定宽列后的余额（含尾差），且每列不低于可读下限
    assert sum(probe.columnWidth(c) for c in flex) >= probe.viewport().width() - time_w - elapsed_w, \
        [probe.columnWidth(c) for c in flex] + [probe.viewport().width()]
    assert all(probe.columnWidth(c) >= CODE_COL_MIN_WIDTH for c in flex)
    # ② 重建后重算（即拖完分割器那条路径）值不变 —— “记录一下跳一次”的互打已消除
    before = [probe.columnWidth(c) for c in range(probe.columnCount())]
    de.update_table()
    de.apply_column_widths()
    assert [probe.columnWidth(c) for c in range(probe.columnCount())] == before, \
        (before, [probe.columnWidth(c) for c in range(probe.columnCount())])
    # ③ 窄视口下长内容要撑宽，但绝不越封顶
    geo = w.size()
    try:
        w.resize(760, 520)
        app.processEvents()
        probe.item(0, flex[-1]).setText("很长的中文注释" * 10)  # 走 itemChanged 回写再重建
        de.update_table()
        assert probe.columnWidth(flex[-1]) == CONTENT_COL_MAX, probe.columnWidth(flex[-1])
        assert all(probe.columnWidth(c) >= CODE_COL_MIN_WIDTH for c in flex)
    finally:
        w.resize(geo)
        app.processEvents()
        de.update_table()


run("encoder column widths single strategy", _column_widths_single_strategy)
def _cell_edit_reflows_width_now():
    """双击改单元格：提交那一瞬就按新内容重量列宽，不等下一次记录或拖分割器"""
    from timeline_axis import CODE_COL_MIN_WIDTH, CONTENT_COL_MAX
    probe = de.result_table
    _timeline(rows=2)
    last = probe.columnCount() - 1
    geo = w.size()
    try:
        w.resize(760, 520)  # 窄视口：弹性列正贴在下限上，撑没撑开一量便知
        app.processEvents()
        de.update_table()
        base = probe.columnWidth(last)
        assert base == CODE_COL_MIN_WIDTH, base
        probe.editItem(probe.item(0, last))
        editor = probe.focusWidget()
        assert isinstance(editor, QLineEdit), type(editor).__name__
        editor.setText("很长的一段中文注释内容，用来观察列宽是否当场跟着变")
        probe.commitData(editor)  # 编辑器还开着时重量列宽：Qt 顺带重算编辑器几何，不能出错
        probe.closeEditor(editor, QAbstractItemDelegate.SubmitModelCache)
        app.processEvents()
        assert probe.item(0, last).text().startswith("很长的一段"), probe.item(0, last).text()
        assert base < probe.columnWidth(last) <= CONTENT_COL_MAX, probe.columnWidth(last)
        # 内容收回去也要跟着收窄：同一个出口重算，不会留着上一轮的宽
        probe.item(0, last).setText("")
        assert probe.columnWidth(last) == base, probe.columnWidth(last)
    finally:
        w.resize(geo)
        app.processEvents()
        de.update_table()


run("encoder cell edit reflows column width at once", _cell_edit_reflows_width_now)
def _timepoint_edit_flow():
    """编辑当前时间 → 记录插入行 → 当前时间自动回到跟随视频的默认态"""
    _timeline(rows=2)          # 两条顺序行：0s、10s
    de.timepoint_edit.setText("00:00:05")
    de._on_timepoint_text_edited("00:00:05")
    assert de.timepoint_edit.property("edited") is True
    de.continue_()  # 5s 不与任何行重合 → 插入到两行之间
    assert de.computed_row_times() == [0, 5, 10], de.computed_row_times()
    assert de._timepoint_edited is False  # 记录后回到默认态
    assert de.timepoint_edit.property("edited") is False
    de.time_unit_entry.setText("20")  # 插入行时间不受影响，顺序行重算
    assert de.computed_row_times() == [0, 5, 20], de.computed_row_times()


run("encoder timepoint edit → record → reset", _timepoint_edit_flow)

def _table_rebuild_no_stale_cells():
    """重建表格只调 setRowCount：行数收缩、行数据缺列都不能留下上一轮的残值"""
    _timeline(rows=3)
    probe = de.result_table
    de.comment_entry.setText("注A")
    de.continue_()                       # 第 4 行带注释
    assert probe.item(3, probe.columnCount() - 1).text() == "注A"
    de.selected_data[0] = ["ST"]         # 行比列短一格 → 尾列要补空
    del de.selected_data[2:]             # 行数收缩 → 尾部整行不能残留
    de.update_table()
    assert probe.rowCount() == 2 and probe.item(0, probe.columnCount() - 1).text() == ""
    assert [probe.item(r, 0).text() for r in range(2)] == ["00:00:00", "00:00:10"]
    _reset_records()


run("encoder table rebuild keeps no stale cells", _table_rebuild_no_stale_cells)

def _no_confirm_workflow():
    """无确认步骤：有维度就能记录，维度增删同步调整表格列与行宽"""
    _reset_records()
    assert bool(de.dimension_cards), "前面用例应已留下维度"
    assert de.continue_btn.isEnabled() is True
    assert de.output_btn.isEnabled() is False  # 无数据时不能输出
    before = len(de.dimension_cards)
    de.dim_panel.add_dimension()
    de.continue_()
    assert len(de.selected_data[0]) == before + 2, de.selected_data[0]  # 维度数 + 注释
    assert de.result_table.columnCount() == before + 4  # 时间 + 计时 + 维度 + 注释
    assert de.output_btn.isEnabled() is True and de.clear_btn.isEnabled() is True
    de.dim_panel.remove_dimension(de.dimension_cards[-1])
    assert len(de.dimension_cards) == before
    assert len(de.selected_data[0]) == before + 1, de.selected_data[0]
    assert de.result_table.columnCount() == before + 3


run("encoder record without confirm", _no_confirm_workflow)
def _cell_edit_in_table():
    """表格内直接修改：维度列弹编码下拉，注释列弹文本框，时间/计时列只读"""
    _reset_records()
    card = de.dimension_cards[0]
    card.blocks[0]['entry'].setText("ST")
    card.add_block()
    card.blocks[-1]['entry'].setText("SI")
    de.continue_()
    opt = QStyleOptionViewItem()
    delegate = de.result_table.itemDelegate()
    model = de.result_table.model()
    combo = delegate.createEditor(de.result_table.viewport(), opt, model.index(0, FIXED_COL))
    comment_col = len(de.dimension_cards) + FIXED_COL
    line = delegate.createEditor(de.result_table.viewport(), opt, model.index(0, comment_col))
    assert isinstance(combo, QComboBox), type(combo)
    assert [combo.itemText(i) for i in range(combo.count())] == ["ST", "SI"]
    assert isinstance(line, QLineEdit), type(line)
    # 时间/计时列不带 ItemIsEditable，不交给编辑器
    for col in range(FIXED_COL):
        assert not (int(de.result_table.item(0, col).flags()) & int(Qt.ItemIsEditable)), col
    assert int(de.result_table.item(0, FIXED_COL).flags()) & int(Qt.ItemIsEditable)
    combo.deleteLater()
    line.deleteLater()
    app.processEvents()
    # 直改后回写行数据，改时间间隔触发的重建不会丢改动
    de.result_table.item(0, FIXED_COL).setText("SI")
    de.result_table.item(0, comment_col).setText("备注")
    assert de.selected_data[0][0] == "SI", de.selected_data[0]
    assert de.selected_data[0][-1] == "备注", de.selected_data[0]
    de.time_unit_entry.setText("7")
    assert de.result_table.item(0, FIXED_COL).text() == "SI"
    assert de.result_table.item(0, comment_col).text() == "备注"


run("encoder cell editors + write-back", _cell_edit_in_table)
def _dblclick_cell_opens_combo():
    """打开维度单元格的下拉编辑器→选值提交，整条链路要能回写行数据并收起编辑器"""
    vp = de.result_table.viewport()
    index = de.result_table.model().index(0, FIXED_COL)
    expected = de.result_table.item(0, FIXED_COL).text()
    known = set(vp.findChildren(QComboBox))
    de.result_table.edit(index)  # 与双击走同一个 edit 入口（离屏平台收不到合成鼠标事件）
    app.processEvents()
    fresh = [c for c in vp.findChildren(QComboBox) if c not in known]
    assert fresh, "未创建下拉编辑器"
    editor = fresh[-1]
    assert editor.currentText() == expected, (editor.currentText(), expected)
    editor.setCurrentText("ST")
    editor.activated.emit(0)  # 模拟从下拉里选一项
    app.processEvents()
    assert de.result_table.item(0, FIXED_COL).text() == "ST", de.result_table.item(0, FIXED_COL).text()
    assert de.selected_data[0][0] == "ST", de.selected_data[0]
    assert not editor.isVisible(), "提交后编辑器未收起"


run("encoder combo editor → commit → write-back", _dblclick_cell_opens_combo)

def _clear_wipes_framework_and_table():
    """编码页「清空」：框架和表格一起清，但不得隔行碰到分析工作台"""
    _reset_records()
    de.dim_panel.apply_framework([("互动类型", ["ST", "SI"]), ("认知层次", ["高", "低"])])
    de.continue_()
    assert de.selected_data and len(de.dimension_cards) == 2
    assert de.clear_btn.isEnabled() is True
    kept = ind.method_id()
    de.clear_btn.click()
    assert not de.dimension_cards and not de.selected_data, (de.dimension_cards, de.selected_data)
    assert de.result_table.rowCount() == 0
    # 框架没了就没得再清，也没数据可输出
    assert de.clear_btn.isEnabled() is False and de.output_btn.isEnabled() is False
    assert de.continue_btn.isEnabled() is False and de.undo_btn.isEnabled() is True
    # 工作台不跟随编码页清空：它的数据是自己装载的那一份
    assert ind.ctx is not None and ind.ctx.usable and ind.method_id() == kept


run("encoder 清空 = 框架 + 表格", _clear_wipes_framework_and_table)
def _undo_covers_every_edit():
    """撤销逐步回退：记录 / 改单元格 / 删行 / 清空（含框架）都回到改动前的现场"""
    _reset_records()
    de.dim_panel.apply_framework([("互动类型", ["ST", "SI"])])
    de.time_unit_entry.setText("10")
    de.set_start_seconds(0)
    assert de.undo_btn.isEnabled() is False  # 还没改过任何东西
    de.continue_()
    de.continue_()
    assert len(de.selected_data) == 2 and len(de.history) == 2
    before = de.result_table.item(0, FIXED_COL).text()
    de.result_table.item(0, FIXED_COL).setText("XX")  # 直改单元格
    assert len(de.history) == 3 and de.selected_data[0][0] == "XX"
    de.undo()
    assert de.selected_data[0][0] == before, de.selected_data[0]
    assert de.result_table.item(0, FIXED_COL).text() == before
    de.undo()
    assert len(de.selected_data) == 1
    de.continue_()
    de.result_table.selectRow(1)  # 起始时间行受保护，删另一行
    de.delete_selected_rows()
    assert len(de.selected_data) == 1
    de.undo()
    assert len(de.selected_data) == 2, de.selected_data
    de.clear()  # 清空：框架 + 表格一起走
    assert not de.dimension_cards and not de.selected_data
    de.undo()   # 清空也能撤销，框架随行数据一起回来
    assert [c.name for c in de.dimension_cards] == ["互动类型"], [c.name for c in de.dimension_cards]
    assert de.dimension_cards[0].code_values() == ["ST", "SI"]
    assert len(de.selected_data) == 2, de.selected_data
    assert de.result_table.columnCount() == FIXED_COL + 2  # 时间/计时 + 一维度 + 注释


run("encoder 撤销覆盖记录/改格/删行/清空", _undo_covers_every_edit)


# ── Phase 1 编码辅助：一阶转移建议 + 注入不破坏撤销栈/脏标记/时间轴不变式 ──────
def _coding_assist_transition_logic():
    """转移模型三种口径：有出边走 transition、退回频次走 frequency、无证据 none。"""
    import coding_assist as ca
    framework = [("互动类型", ["ST", "SI", "SQ"])]
    # ST→SI 反复出现：last=ST 有转移证据，候选只有 SI
    rows = [["ST", ""], ["SI", ""], ["ST", ""], ["SI", ""], ["ST", ""], ["SI", ""], ["ST", ""]]
    item = ca.suggest_next(rows, framework)[0]
    assert item["last_code"] == "ST", item
    assert item["mode"] == "transition", item
    assert [c for c, _p in item["candidates"]] == ["SI"], item
    # 无历史 → none，候选为空
    empty = ca.suggest_next([], framework)[0]
    assert empty["mode"] == "none" and empty["candidates"] == [], empty
    # 无维度 → 空列表
    assert ca.suggest_next(rows, []) == []
    # last=SQ 无出边 → 退回整体频次（frequency），且候选都落在词表内
    rows2 = [["ST", ""], ["SI", ""], ["SQ", ""]]
    item2 = ca.suggest_next(rows2, framework)[0]
    assert item2["mode"] == "frequency", item2
    assert item2["candidates"] and all(c in ("ST", "SI", "SQ") for c, _ in item2["candidates"]), item2


run("coding_assist 转移/频次/无证据三口径", _coding_assist_transition_logic)


def _coding_assist_confidence_and_multidim():
    """多维度各出一条；证据不足诚实降级为「仅供参考」，充足才 confident。"""
    import coding_assist as ca
    framework = [("互动类型", ["ST", "SI"]), ("认知", ["高", "低"])]
    few = ca.suggest_next([["ST", "高", ""], ["SI", "低", ""]], framework)
    assert len(few) == 2 and [i["dimension"] for i in few] == ["互动类型", "认知"], few
    assert all(not i["confident"] for i in few), few          # support 远小于阈值
    many = []
    for _ in range(8):
        many.append(["ST", "高", ""])
        many.append(["SI", "低", ""])
    got = ca.suggest_next(many, framework)
    assert got[0]["last_code"] == "SI" and got[0]["confident"] is True, got[0]
    assert [c for c, _ in got[0]["candidates"]] == ["ST"], got[0]   # SI 后恒为 ST
    assert got[1]["confident"] is True and [c for c, _ in got[1]["candidates"]] == ["高"], got[1]


run("coding_assist 多维度与置信降级", _coding_assist_confidence_and_multidim)


def _coding_assist_encoder_integration():
    """apply_suggestion 只填勾选态：不写表、不动撤销栈/脏标记/时间轴；记录后才入表。"""
    import copy as _copy
    _timeline(unit="10", start=0, rows=0)      # 框架：互动类型 [ST, SI]
    de.selected_data = [["ST", ""], ["SI", ""], ["ST", ""], ["SI", ""]]
    de.row_times = [None, None, None, None]
    de.row_positions = [0, 1, 2, 3]
    de.update_table()
    de.suggest_bar.refresh()
    assert de.suggest_bar.body_layout.count() >= 1
    marker = []
    handler = lambda: marker.append(1)
    de.content_changed.connect(handler)
    try:
        hist_before = len(de.history)
        data_before = _copy.deepcopy(de.selected_data)
        times_before = de.computed_row_times()
        assert de.apply_suggestion(0, "ST") is True
        assert de.dimension_cards[0].get_selected_entry() == "ST"
        # 注入不碰任何数据面：撤销栈、行数据、时间轴、脏标记全部原样
        assert len(de.history) == hist_before, (len(de.history), hist_before)
        assert de.selected_data == data_before, de.selected_data
        assert de.computed_row_times() == times_before, de.computed_row_times()
        assert marker == [], "建议码不该触发 content_changed"
        # 越界维度 / 词表外的码 → False，不改勾选态
        assert de.apply_suggestion(99, "ST") is False
        assert de.apply_suggestion(0, "不存在的码") is False
        assert de.dimension_cards[0].get_selected_entry() == "ST"
    finally:
        de.content_changed.disconnect(handler)
    # 用户亲手点「记录」才真正入表：此刻才进撤销栈、才发脏标记
    de.continue_()
    assert len(de.history) == hist_before + 1, len(de.history)
    assert len(de.selected_data) == len(data_before) + 1, de.selected_data


run("coding_assist 注入不破坏撤销栈/脏标记/时间轴", _coding_assist_encoder_integration)


def _coding_assist_bar_pick_fills_card():
    """点建议条上的候选码按钮 → 对应维度被勾选，且不写表、不进撤销栈。"""
    from PySide6.QtWidgets import QPushButton
    _timeline(rows=0)
    de.selected_data = [["ST", ""], ["SI", ""], ["ST", ""], ["SI", ""]]
    de.row_times = [None, None, None, None]
    de.row_positions = [0, 1, 2, 3]
    de.update_table()
    bar = de.suggest_bar
    bar.refresh()
    buttons = [b for b in bar.body.findChildren(QPushButton) if b is not bar.refresh_btn]
    assert buttons, "建议条没有候选按钮"
    code = buttons[0].text()
    hist_before, rows_before = len(de.history), len(de.selected_data)
    buttons[0].click()
    app.processEvents()
    assert de.dimension_cards[0].get_selected_entry() == code, de.dimension_cards[0].get_selected_entry()
    assert len(de.history) == hist_before and len(de.selected_data) == rows_before


run("coding_assist 建议条点选填入", _coding_assist_bar_pick_fills_card)


def _coding_assist_select_value_no_double():
    """select_value 单选语义：换码时前一个自动取消，词表外返回 False 不改勾选态。"""
    _timeline(rows=0)
    card = de.dimension_cards[0]
    assert card.select_value("SI") is True and card.get_selected_entry() == "SI"
    assert card.select_value("ST") is True and card.get_selected_entry() == "ST"
    assert len([b for b in card.blocks if b['checkbox'].isChecked()]) == 1, "同时勾了多个块"
    assert card.select_value("XX") is False and card.get_selected_entry() == "ST"


run("coding_assist select_value 单选不重复", _coding_assist_select_value_no_double)

# 深色模式下再跑一遍增删，确认图标重绑定后控件仍可用
w.dark_action.setChecked(True)
app.processEvents()
run("individual 深色下可算可画", lambda: _workbench_flow(ind, "IND01"))
run("group 深色下可算可画", lambda: _workbench_flow(grp, "GRP01"))
run("encoder.add_dimension (dark)", de.dim_panel.add_dimension)
run("encoder card add_block (dark)", de.dimension_cards[0].add_block)
w.dark_action.setChecked(False)
app.processEvents()

print()
if fails:
    print("INTERACT_FAILED: %s" % fails)
    sys.exit(1)
print("INTERACT_OK")
