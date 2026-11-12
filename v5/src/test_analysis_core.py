"""数据契约与校验层测试：analysis_core + interaction_model。

覆盖计划 §9 的四个面：
- group 边界（末→首不得连接）、group_id 解析、session 取数
- V01–V10 逐条（每条都要有"触发"与"放行"两个方向）
- 合成数据可装载、质检只报故意留的违例

运行方式：
    cd version/v5
    python _gen_testdata.py      # 先生成 _testdata/（缺失时本文件自动跳过相关用例）
    pytest test_analysis_core.py -v
"""

import os
import sys

import pandas as pd
import pytest

_V5_DIR = os.path.dirname(os.path.abspath(__file__))
if _V5_DIR not in sys.path:
    sys.path.insert(0, _V5_DIR)

import analysis_core as ac                                    # noqa: E402
import interaction_model as im                                # noqa: E402

TESTDATA = os.path.join(_V5_DIR, "_testdata")
HAS_TESTDATA = os.path.isdir(TESTDATA) and bool(
    [name for name in os.listdir(TESTDATA) if name.endswith(".xlsx")])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 组号与学生主键
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIdentifiers:
    def test_group_id_from_path_strips_extension(self):
        assert ac.group_id_from_path(r"C:\data\group_1.xlsx") == "group_1"
        assert ac.group_id_from_path("/tmp/第三组.csv") == "第三组"

    def test_group_id_override_wins(self):
        assert ac.group_id_from_path("group_1.xlsx", "G7") == "G7"

    def test_group_id_sanitized_for_separator_collision(self):
        """group_id 会拼进 event_id 与 student_uid，带分隔符就会撞主键。"""
        assert ac.sanitize_group_id("a b__c") == "a_b__c"
        assert ac.sanitize_group_id("   ") == "group"

    def test_student_uid_round_trip(self):
        uid = ac.student_uid("group_1", "S2")
        assert uid == "group_1__S2"
        assert ac.split_uid(uid) == ("group_1", "S2")

    def test_split_uid_without_separator_keeps_group_empty(self):
        assert ac.split_uid("S1") == ("", "S1")

    def test_members_from_columns_keeps_first_appearance_order(self):
        columns = ["Time", "S2Encoding_1", "S1Encoding_2", "S2Encoding_3", "备注"]
        assert ac.members_from_columns(columns) == ["S2", "S1"]

    def test_duplicate_group_ids_renamed_not_dropped(self):
        a = ac.GroupBundle("group_1", wide=pd.DataFrame({"Time": [0, 1]}))
        b = ac.GroupBundle("group_1", wide=pd.DataFrame({"Time": [0, 1]}))
        out = ac.dedupe_group_ids([a, b])
        assert [x.group_id for x in out] == ["group_1", "group_1_2"]
        assert out[1].meta["group_id_renamed_from"] == "group_1"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# group 边界：跨组绝不拼接
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _bundle(group_id, student, codes, start=0.0, step=10.0):
    """造一个单学生单维度的 bundle：时间从 start 起等距。"""
    times = [start + index * step for index in range(len(codes))]
    frame = pd.DataFrame({"Time": times, "%sEncoding_1" % student: list(codes)})
    return ac.GroupBundle(group_id, wide=frame)


class TestGroupBoundary:
    def test_concat_bundles_keeps_one_sequence_per_group(self):
        """两组的 S1 各自成链：末→首不得连接（否则转移矩阵会多出一条跨组边）。"""
        left = _bundle("g1", "S1", "ABC")
        right = _bundle("g2", "S1", "CAB")
        seqs = ac.concat_bundles([left, right], dimension="D1")
        assert len(seqs) == 2
        assert {tuple(s.codes) for s in seqs} == {("A", "B", "C"), ("C", "A", "B")}
        assert sorted(s.group_id for s in seqs) == ["g1", "g2"]

    def test_no_sequence_carries_codes_from_two_groups(self):
        seqs = ac.concat_bundles([_bundle("g1", "S1", "AB"), _bundle("g2", "S1", "CD")])
        for seq in seqs:
            assert seq.codes in (["A", "B"], ["C", "D"])
            assert seq.uid.count("__") == 1

    def test_same_student_in_two_groups_yields_two_uids(self):
        """四组各有 S1 时特征表里必须是四个样本，而不是一个被合并的 S1。"""
        bundles = [_bundle("g%d" % i, "S1", "AB") for i in range(1, 5)]
        uids = {seq.uid for seq in ac.concat_bundles(bundles)}
        assert uids == {"g1__S1", "g2__S1", "g3__S1", "g4__S1"}

    def test_concat_frames_tags_group_and_uid(self):
        merged = ac.concat_frames([_bundle("g1", "S1", "AB"), _bundle("g2", "S1", "CD")])
        assert set(merged["group_id"]) == {"g1", "g2"}
        assert set(merged["uid"]) == {"g1__S1", "g2__S1"}

    def test_sequence_helpers(self):
        seq = ac.Sequence("g1", "S1", "D1", [0, 1, 2, 3], ["A", "A", "B", ""])
        assert seq.length == 4
        assert seq.valid == ["A", "A", "B"]
        assert seq.compressed() == ["A", "B", ""]
        assert seq.ngram_support(2) == [("A", "A"), ("A", "B"), ("B", "")]


class TestBundleViews:
    def test_time_points_ascending_and_deduped(self):
        frame = pd.DataFrame({"Time": [20, 0, 10, 0], "S1Encoding_1": ["A", "B", "C", "D"]})
        bundle = ac.GroupBundle("g1", wide=frame)
        assert bundle.time_points() == [0.0, 10.0, 20.0]
        assert bundle.n_times == 3

    def test_non_string_column_labels_are_skipped(self):
        """Excel 空表头/重复表头会把列名读成 int / None：只能跳过，不能 TypeError 崩掉装载。

        坐实的是真实数据 `Results/testfile/group_1 - 副本.xlsx`：它第 9 列的表头就是一个数字。
        """
        frame = pd.DataFrame([["x", 0, "A"], ["y", 10, "B"]],
                             columns=[None, "Time", "S1Encoding_1"])
        mapping, unparsed = ac.parse_columns(frame.columns, time_col="Time")
        assert mapping == {"S1Encoding_1": ("S1", "D1")} and unparsed == []
        bundle = ac.GroupBundle("g1", wide=frame)
        assert bundle.members == ["S1"]
        assert bundle.sequences()[0].codes == ["A", "B"]

    def test_consensus_sequence_takes_mode_code(self):
        # 宽表一行一个时间点：三人在 t=0 的 D1 全是 A，众数必为 A
        frame = pd.DataFrame({
            "Time": [0, 10],
            "S1Encoding_1": ["A", "A"], "S2Encoding_1": ["A", "B"],
            "S3Encoding_1": ["A", "B"],
            "S1Encoding_2": ["x", "x"], "S2Encoding_2": ["y", "y"],
            "S3Encoding_2": ["y", "y"],
        })
        bundle = ac.GroupBundle("g1", wide=frame)
        consensus = bundle.consensus_sequence("D1")
        assert consensus.codes[0] == "A"
        assert consensus.student == "GROUP"
        # 第二维 t=10 是 y/y/y → y
        assert bundle.consensus_sequence("D2").codes[1] == "y"

    def test_member_matrix_is_points_plus_dict(self):
        bundle = _bundle("g1", "S1", "ABC")
        points, matrix = bundle.member_matrix("D1")
        assert points == [0.0, 10.0, 20.0]
        assert matrix["S1"] == ["A", "B", "C"]

    def test_outcome_none_when_absent(self):
        assert _bundle("g1", "S1", "AB").outcome is None

    def test_bundle_dict_round_trip(self):
        bundle = _bundle("g1", "S1", "ABC")
        restored = ac.GroupBundle.from_dict(bundle.to_dict())
        assert restored.group_id == "g1"
        assert restored.sequences()[0].codes == ["A", "B", "C"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 取数：批量上传 + 当前会话
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FakeSummaryTab:
    """只实现 build_analysis_data 的假汇总页 —— 契约就这一个方法。"""

    def __init__(self, columns, rows):
        self._columns, self._rows = columns, rows

    def build_analysis_data(self, with_comments=False, with_group=False):
        return list(self._columns), [list(row) for row in self._rows]


class FakeGroupEncoder:
    def __init__(self, frame):
        self._frame = frame

    def events_frame(self):
        return self._frame

    def collect_meta(self):
        return {"actors": ["T"], "object_dict": {"TASK": ["T1"]},
                "interaction_types": list(im.INTERACTION_TYPES), "group_name": "三班一组"}


@pytest.mark.skipif(not HAS_TESTDATA, reason="先运行 python _gen_testdata.py 生成 _testdata/")
class TestLoadBundles:
    def paths(self):
        return sorted(os.path.join(TESTDATA, name) for name in os.listdir(TESTDATA)
                      if name.endswith(".xlsx"))

    def test_load_bundles_one_file_one_group(self):
        bundles = ac.load_bundles(self.paths())
        assert len(bundles) == len(self.paths())
        assert {b.group_id for b in bundles} == {"group_1", "group_2", "group_3",
                                                 "group_4", "group_legacy"}

    def test_group_id_override_applied_by_position(self):
        paths = self.paths()
        bundles = ac.load_bundles(paths[:1], group_ids=["A组"])
        assert bundles[0].group_id == "A组"
        # 事件表主键的组号一并改写，否则 event_id 会留着旧组号
        if bundles[0].has_events:
            assert all(str(value).startswith("A组_E")
                       for value in bundles[0].events["event_id"])

    def test_legacy_single_sheet_file_still_loads(self):
        """老格式只有 Sheet1：列名契约在，就应读得出 240 个时间点、认作"无事件"。"""
        bundle = ac.load_bundle(os.path.join(TESTDATA, "group_legacy.xlsx"))
        assert bundle.n_times == 240
        assert bundle.members == ["S1", "S2", "S3"]
        assert not bundle.has_events
        assert bundle.codebook == []

    def test_rich_file_reads_all_sheets(self):
        bundle = ac.load_bundle(os.path.join(TESTDATA, "group_1.xlsx"))
        assert bundle.has_events and len(bundle.events) == 50
        assert bundle.meta["group_name"] == "group_1"
        assert dict(bundle.codebook)["认知投入"] == ["A", "B", "C"]

    def test_embedded_structure_three_phase_transition(self):
        """生成器内嵌的三段式必须在 S1/D1 上原样读出来 —— 否则方法层断言没有依据。"""
        bundle = ac.load_bundle(os.path.join(TESTDATA, "group_1.xlsx"))
        seq = next(s for s in bundle.sequences(dimension="D1", student="S1"))
        assert seq.codes[:20] == ["A"] * 20
        assert seq.codes[20:40] == ["B"] * 20
        assert seq.codes[40:60] == ["C"] * 20

    def test_quality_reports_only_planted_violations(self):
        """V05 / V08 各 2 条（两个带事件的组各一条），且不得冒出阻断级的意外错误。"""
        bundles = ac.load_bundles(self.paths())
        items = ac.bundle_quality(bundles)
        codes = pd.Series([item.code for item in items]).value_counts().to_dict()
        assert codes.get("V05") == 2 and codes.get("V08") == 2
        assert codes.get("V01") is None and codes.get("V07") is None
        assert codes.get("V09")            # 同窗多事件按规格只报计数，不阻断
        assert all(not item.blocking or item.code in ("V05", "V08") for item in items)

    def test_single_group_downgrade_note(self):
        """只有一组时不隐藏功能，但必须给出降级文案；≥ 2 组时清空。"""
        only = ac.load_bundles([os.path.join(TESTDATA, "group_1.xlsx")])
        assert "仅 1 组" in ac.group_count_note(only)
        assert any(item.code == "core.single_group" for item in ac.bundle_quality(only))
        assert ac.group_count_note(ac.load_bundles(self.paths())) == ""


class TestBundleFromSession:
    def test_session_data_builds_bundle(self):
        columns = ["Time", "S1Encoding_1", "S2Encoding_1"]
        rows = [[0, "A", "B"], [10, "B", "A"], [20, "C", "C"]]
        events = im.normalize_events(pd.DataFrame([{
            "group_id": "g1", "event_id": "g1_E001", "timestamp": 0,
            "actor_id": "S1", "target_id": "S2", "target_type": "PERSON",
            "interaction_type": "ASK"}]), "g1")
        bundle = ac.bundle_from_session(
            "g1", summary_tab=FakeSummaryTab(columns, rows),
            group_encoder=FakeGroupEncoder(events))
        assert bundle.members == ["S1", "S2"]
        assert bundle.n_times == 3
        assert bundle.has_events
        assert {s.uid for s in bundle.sequences()} == {"g1__S1", "g1__S2"}
        assert bundle.meta["actors"] == ["T"]

    def test_empty_session_yields_empty_bundle(self):
        bundle = ac.bundle_from_session("g1", summary_tab=FakeSummaryTab([], []))
        assert bundle.members == []
        assert bundle.sequences() == []

    def test_members_argument_overrides_column_parse(self):
        bundle = ac.bundle_from_session(
            "g1", summary_tab=FakeSummaryTab(["Time", "S1Encoding_1"], [[0, "A"]]),
            members=["甲", "乙"])
        assert bundle.members == ["甲", "乙"]


class TestResolveCodebook:
    def test_framework_argument_has_top_priority(self):
        bundle = _bundle("g1", "S1", "ABC")
        assert ac.resolve_codebook(bundle, framework=[("投入", ["高", "低"])]) == \
            [("投入", ["高", "低"])]

    def test_bundle_codebook_beats_inference(self):
        bundle = ac.GroupBundle("g1", wide=pd.DataFrame({"Time": [0], "S1Encoding_1": ["A"]}),
                                codebook=[("维度", ["A", "B"])])
        assert ac.resolve_codebook(bundle) == [("维度", ["A", "B"])]

    def test_inference_falls_back_to_data(self):
        bundle = _bundle("g1", "S1", "AAB")
        assert ac.resolve_codebook(bundle) == [("D1", ["A", "B"])]

    def test_framework_excel_requires_exact_two_columns(self, tmp_path):
        path = str(tmp_path / "framework.xlsx")
        pd.DataFrame({"维度": ["投入", "投入"], "类型": ["高", "低"]}).to_excel(path, index=False)
        assert ac.read_framework_excel(path) == [("投入", ["高", "低"])]
        bad = str(tmp_path / "bad.xlsx")
        pd.DataFrame({"维度": ["投入"], "编码": ["高"]}).to_excel(bad, index=False)
        with pytest.raises(ValueError):
            ac.read_framework_excel(bad)

    def test_framework_from_pages_merges_same_dimension(self):
        class Page:
            class dim_panel:
                @staticmethod
                def framework():
                    return [("投入", ["高", "低"])]

        assert ac.framework_from_pages([Page, Page]) == [("投入", ["高", "低"])]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# V01–V10 逐条
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _ctx(**kwargs):
    base = dict(group_id="g1", members=["S1", "S2"], actors=["T1"],
                object_dict={"TASK": ["TK1"], "QUESTION": ["Q1"]},
                t_min=0.0, t_max=600.0,
                events=[{"event_id": "g1_E001", "timestamp": 10.0}])
    base.update(kwargs)
    return im.EventContext(**base)


def _row(**kwargs):
    base = dict(group_id="g1", event_id="g1_E002", timestamp=20.0, actor_id="S1",
                target_id="TK1", target_type="TASK", interaction_type="SPEAK",
                reply_to="", turn_id="t1", duration=3.0, source="CODING",
                confidence=0.9, notes="")
    base.update(kwargs)
    return im.blank_event(**base)


def _codes(row, ctx=None):
    return [item.code for item in im.validate_row(row, ctx or _ctx())]


class TestValidationRules:
    def test_clean_row_passes(self):
        assert _codes(_row()) == []

    @pytest.mark.parametrize("column", ["group_id", "event_id", "actor_id"])
    def test_v01_blocks_missing_required(self, column):
        assert "V01" in _codes(_row(**{column: ""}))

    def test_v01_accepts_timestamp_zero(self):
        """0 秒就是观察起点，不能被当成"没填"。"""
        assert "V01" not in _codes(_row(timestamp=0))

    def test_v02_unknown_actor(self):
        assert "V02" in _codes(_row(actor_id="S9"))

    def test_v02_accepts_registered_external_actor(self):
        assert "V02" not in _codes(_row(actor_id="T1"))

    def test_v03_negative_and_unparseable_and_out_of_window(self):
        assert "V03" in _codes(_row(timestamp=-5))
        assert "V03" in _codes(_row(timestamp="abc"))
        assert "V03" in _codes(_row(timestamp=9999))

    def test_v03_accepts_hhmmss(self):
        assert "V03" not in _codes(_row(timestamp="00:01:00"))

    def test_v04_unknown_target_type(self):
        assert "V04" in _codes(_row(target_type="STUFF"))

    @pytest.mark.parametrize("target_type", im.TARGET_TYPE_CHOICES)
    def test_v04_accepts_every_enum_value(self, target_type):
        """枚举 11 个值都得能过 V04，包括 NO_TARGET / SELF。"""
        row = _row(target_type=target_type,
                   target_id="S1" if target_type in ("SELF", "PERSON") else "")
        if target_type == im.NO_TARGET:
            row["target_id"] = ""
        elif target_type == "SELF":
            row["target_id"] = row["actor_id"]
        assert "V04" not in _codes(row)

    def test_v05_object_type_needs_id(self):
        assert "V05" in _codes(_row(target_type="TASK", target_id=""))

    def test_v05_object_not_in_registered_dict(self):
        assert "V05" in _codes(_row(target_id="TK99"))

    def test_v05_silent_when_no_dict_registered_for_that_type(self):
        """文件没带对象字典时无从判一致性，不该满屏假告警。"""
        ctx = _ctx(object_dict={})
        assert "V05" not in _codes(_row(target_id="anything"), ctx)

    def test_v05_no_target_must_leave_id_empty(self):
        assert "V05" in _codes(_row(target_type=im.NO_TARGET, target_id="TK1"))

    def test_v05_self_target_must_equal_actor(self):
        assert "V05" in _codes(_row(target_type="SELF", target_id="S2"))
        assert "V05" not in _codes(_row(target_type="SELF", target_id="S1"))

    def test_v06_id_shape_and_group_prefix(self):
        assert "V06" in _codes(_row(event_id="E001"))
        assert "V06" in _codes(_row(event_id="g2_E001"))
        assert "V06" not in _codes(_row(event_id="g1_E0123"))   # 位数可扩展

    def test_v07_reply_to_unknown_target(self):
        assert "V07" in _codes(_row(reply_to="g1_E999"))

    def test_v07_reply_to_later_event_rejected(self):
        assert "V07" in _codes(_row(reply_to="g1_E001", timestamp=5.0))

    def test_v07_reply_to_earlier_event_passes(self):
        assert "V07" not in _codes(_row(reply_to="g1_E001", timestamp=20.0))

    def test_v07_blank_reply_passes(self):
        """空 reply_to 是"不是应答"，读表回来的空串不能变成 'nan' 去查引用。"""
        for blank in ("", None, float("nan")):
            assert "V07" not in _codes(_row(reply_to=blank))

    def test_v08_negative_duration_and_out_of_range_confidence(self):
        assert "V08" in _codes(_row(duration=-1))
        assert "V08" in _codes(_row(confidence=1.7))
        assert "V08" in _codes(_row(confidence=-0.1))

    def test_v08_accepts_boundary_confidence_and_zero_duration(self):
        assert "V08" not in _codes(_row(confidence=1.0, duration=0))
        assert "V08" not in _codes(_row(confidence=0.0, duration=0.0))

    def test_v10_interaction_type_outside_current_vocabulary(self):
        assert "V10" in _codes(_row(interaction_type="TEACH"))
        assert "V10" not in _codes(_row(interaction_type="CHALLENGE"))

    def test_v10_respects_edited_vocabulary(self):
        ctx = _ctx(interaction_types=["SPEAK", "REPAIR"])
        assert "V10" not in _codes(_row(interaction_type="REPAIR"), ctx)
        assert "V10" in _codes(_row(interaction_type="CHALLENGE"), ctx)

    def test_warning_vs_blocking_split(self):
        """阻断级违例必须能拦下写入，警告级只进质量报告。"""
        items = im.validate_row(_row(actor_id="S9", confidence=1.7), _ctx())
        assert [i.code for i in im.blocking_items(items)] == ["V02"]
        assert [i.code for i in im.warning_items(items)] == ["V08"]

    def test_items_sorted_by_rule_number(self):
        items = im.validate_row(_row(actor_id="S9", target_type="STUFF",
                                     timestamp=9999, confidence=2.0), _ctx())
        assert [i.code for i in items] == sorted([i.code for i in items])

    def test_v09_same_timestamp_multiple_events_allowed(self):
        frame = im.normalize_events(pd.DataFrame([
            _row(event_id="g1_E001", timestamp=10.0),
            _row(event_id="g1_E002", timestamp=10.0, target_id="Q1",
                 target_type="QUESTION"),
        ]), "g1")
        items = im.validate_events(frame, _ctx())
        assert len(frame) == 2
        assert any(item.code == "V09" and item.level == "info" for item in items)
        assert not im.blocking_items(items)

    def test_v06_duplicate_event_id_detected_on_frame(self):
        frame = pd.DataFrame([_row(event_id="g1_E002"), _row(event_id="g1_E002")])
        items = im.validate_events(im.ensure_event_columns(frame), _ctx())
        assert any(item.code == "V06" for item in items)

    def test_empty_frame_reports_info_not_error(self):
        items = im.validate_events(pd.DataFrame(), _ctx())
        assert len(items) == 1 and items[0].level == "info" and not items[0].blocking


class TestEventIdGeneration:
    def test_format_is_group_plus_three_digits(self):
        assert im.new_event_id("g1", 1) == "g1_E001"
        assert im.new_event_id("g1", 42) == "g1_E042"
        assert im.new_event_id("g1", 1200) == "g1_E1200"     # 位数自然扩展

    def test_next_number_skips_used_ids(self):
        existing = ["g1_E001", "g1_E002", "g1_E007", "", "other_E003"]
        assert im.next_event_number(existing, "g1") == 8

    def test_reindex_restarts_from_one_in_time_order(self):
        frame = im.normalize_events(pd.DataFrame([
            _row(event_id="g1_E050", timestamp=30.0),
            _row(event_id="g1_E003", timestamp=10.0)]), "g1")
        out = im.reindex_event_ids(frame, "g1")
        assert list(out["event_id"]) == ["g1_E001", "g1_E002"]
        assert list(out["timestamp"]) == [10.0, 30.0]


class TestEventTableContract:
    def test_event_column_order_is_the_data_dictionary(self):
        """列序 = 规格说明书 §3 总表；导出、R 参考层、批量上传都按它对齐。"""
        assert im.EVENT_COLUMNS == [
            "group_id", "event_id", "timestamp", "actor_id", "target_id", "target_type",
            "interaction_type", "reply_to", "turn_id", "duration", "source",
            "confidence", "notes"]

    def test_utterance_is_a_separate_column_from_notes(self):
        assert im.UTTERANCE_COLUMN == "utterance"
        assert im.UTTERANCE_COLUMN not in im.EVENT_COLUMNS

    def test_blank_event_has_every_column(self):
        row = im.blank_event("g1")
        assert set(im.EVENT_COLUMNS) <= set(row)
        assert row["group_id"] == "g1" and row["source"] == "CODING"

    def test_ensure_event_columns_drops_extras_and_fills_missing(self):
        frame = pd.DataFrame([{"group_id": "g1", "junk": 1}])
        out = im.ensure_event_columns(frame)
        assert list(out.columns) == im.EVENT_COLUMNS
        assert out["actor_id"].iloc[0] == ""

    def test_ensure_event_columns_washes_nan_into_empty_string(self):
        frame = pd.DataFrame([{"event_id": "g1_E001", "reply_to": float("nan"),
                               "notes": None, "timestamp": float("nan")}])
        out = im.ensure_event_columns(frame)
        assert out["reply_to"].iloc[0] == "" and out["notes"].iloc[0] == ""

    def test_blocking_and_warning_sets_match_rule_table(self):
        assert im.BLOCKING_RULES == {"V01", "V02", "V03", "V04", "V06", "V07", "V10"}
        assert im.WARNING_RULES == {"V05", "V08"}
        assert im.BLOCKING_RULES | im.WARNING_RULES == set(im.RULES) - {"V09"}

    def test_codebook_rows_flattens_encoder_framework(self):
        rows = im.codebook_rows([("投入", ["高", "低"]), ("阶段", ["计划"])])
        assert rows == [{"dimension": "投入", "code": "高", "label": ""},
                        {"dimension": "投入", "code": "低", "label": ""},
                        {"dimension": "阶段", "code": "计划", "label": ""}]

    def test_codebook_rows_accepts_already_flat_dicts(self):
        assert im.codebook_rows([{"dimension": "D", "code": "A", "label": "x"}]) == \
            [{"dimension": "D", "code": "A", "label": "x"}]


class TestWorkbookTemplates:
    def test_template_has_six_named_sheets_in_order(self):
        sheets = im.template_sheets(group_id="g1", members=["S1", "S2"])
        assert list(sheets) == ["Group_Info", "Students", "Behavior_State",
                                "Interaction_Event", "Codebook", "Enums"]

    def test_students_sheet_carries_uid_per_group(self):
        students = im.template_sheets(group_id="g1", members=["S1", "S2"])["Students"]
        assert list(students["student_uid"]) == ["g1__S1", "g1__S2"]

    def test_enums_sheet_lists_target_types_and_rules(self):
        enums = im.template_sheets()["Enums"]
        values = set(zip(enums["field"], enums["value"]))
        assert ("target_type", "NO_TARGET") in values
        assert ("interaction_type", "SPEAK") in values
        assert ("rule", "V09") in values

    def test_write_and_reload_round_trip(self, tmp_path):
        sheets = im.template_sheets(
            group_id="g1", group_name="一组", members=["S1", "S2"],
            codebook=im.codebook_rows([("投入", ["高"])]),
            behavior_state=pd.DataFrame({"Time": [0, 10],
                                         "S1Encoding_1": ["高", "低"],
                                         "S2Encoding_1": ["低", "高"]}),
            events=im.normalize_events(pd.DataFrame([_row()]), "g1"))
        path = str(tmp_path / "pack.xlsx")
        im.write_workbook(sheets, path)
        assert os.path.exists(path)
        state = im.load_group_sheet(path)
        assert list(state.columns) == ["Time", "S1Encoding_1", "S2Encoding_1"]
        events = im.load_event_sheet(path)
        assert list(events["event_id"]) == ["g1_E002"]

    def test_load_event_sheet_returns_empty_when_sheet_absent(self, tmp_path):
        path = str(tmp_path / "plain.xlsx")
        pd.DataFrame({"Time": [0], "S1Encoding_1": ["A"]}).to_excel(path, index=False)
        assert im.load_event_sheet(path).empty
        assert list(im.load_group_sheet(path).columns) == ["Time", "S1Encoding_1"]
