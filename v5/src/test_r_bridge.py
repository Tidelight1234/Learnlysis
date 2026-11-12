"""R 参考层（`analysis_r_bridge`）离线单测（计划 §9：两路互证）。

这些测试**不要求本机装了 R**：`run_reference` / `probe` 一律用假实现顶掉，只验桥接层
自己的契约 —— 载荷落盘、键对齐、双门槛判定、缺失计数、降级说明、meta 回写。
「两侧数出来的值真的一样」这件事由 `r_refs/*.R` + `_smoke_modes.py` 的 R 通道负责，
它需要 Rscript 在场；这里断的是「R 撒谎时校验卡必须当场发现」。

运行方式：
    cd version/v5
    pytest test_r_bridge.py -v
"""

import json
import os
import sys

import pandas as pd
import pytest

_V5_DIR = os.path.dirname(os.path.abspath(__file__))
if _V5_DIR not in sys.path:
    sys.path.insert(0, _V5_DIR)

import analysis_core as ac                                    # noqa: E402
import analysis_methods as am                                 # noqa: E402
import analysis_r_bridge as rb                                # noqa: E402


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 夹具：两组 × 两人 × 两维的小数据（够 IND02 出转移矩阵）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _bundle(group_id, codes_by_member, dimensions=1):
    """宽表一行一个时间点：列名契约 `Time + S{i}Encoding_{j}` 不可改。"""
    rows = len(list(codes_by_member.values())[0])
    data = {"Time": [step * 10.0 for step in range(rows)]}
    for member, codes in codes_by_member.items():
        data["%sEncoding_1" % member] = list(codes)
    if dimensions > 1:
        for member, codes in codes_by_member.items():
            data["%sEncoding_2" % member] = [str(code).lower() for code in codes]
    return ac.GroupBundle(group_id, wide=pd.DataFrame(data))


def _ctx(mode=am.MODE_INDIVIDUAL):
    """两组 × 两人（g2 只一人）× 两维：长度 ≥ MIN_SEQUENCE_LENGTH，否则统计层按“太短”丢弃。"""
    bundles = [_bundle("g1", {"S1": "AABBAABBCCAA", "S2": "BBAABBAACCBB"}, dimensions=2),
               _bundle("g2", {"S1": "ABABABCCCABB", "S2": "CCAABBCCABAB"}, dimensions=2)]
    return am.AnalysisContext(bundles, mode=mode)


def _params(spec, ctx):
    out = {item.key: item.default for item in spec.params}
    out.setdefault("students", list(ctx.students))
    out.setdefault("dimensions", list(ctx.dimensions))
    return out


def _numeric_column(frame):
    for column in frame.columns:
        if pd.to_numeric(frame[column], errors="coerce").notna().any():
            return column
    raise AssertionError("表里没有数值列：%s" % list(frame.columns))


def _fake_run_reference(tables, engine=None, status="ok"):
    """造一个「R 已跑完」的返回：tables 由调用方给，可以是被篡改过的。"""
    def runner(script, payload, params=None, timeout=None, rscript=None):
        return {"available": True, "ok": status == "ok", "status": status,
                "error": "" if status == "ok" else "假失败",
                "stderr": "", "tables": {name: frame.copy() for name, frame in tables.items()},
                "values": {}, "engine": dict(engine or {"r": "fake"}),
                "elapsed_ms": 1, "workdir": ""}
    return runner


AVAILABLE = {"available": True, "version": "R 4.5.0", "rscript": "Rscript",
             "packages": {"TraMineR": True}, "found": ["TraMineR"], "missing": []}
UNAVAILABLE = {"available": False, "version": "", "rscript": "",
               "packages": {}, "found": [], "missing": ["TraMineR"]}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 参数展平 / 输入落盘
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestInputs:
    def test_flatten_params_writes_scalars_as_text(self):
        flat = rb._flatten_params({"flag": True, "off": False, "picks": ["A", "B"],
                                   "none": None, "n": 3, "map": {"k": 1}})
        assert flat["flag"] == "TRUE" and flat["off"] == "FALSE"
        assert flat["picks"] == "A,B"                      # R 侧 ly_list() 再拆回来
        assert flat["none"] == "" and flat["n"] == "3"
        assert json.loads(flat["map"]) == {"k": 1}

    def test_write_inputs_writes_every_nonempty_table(self, tmp_path):
        workdir = str(tmp_path / "r")
        payload = {"sequences": pd.DataFrame([{"label": "g1__S1 / D1", "codes": "A|B"}]),
                   "state order 表!": pd.DataFrame([{"dimension": "D1", "state": "A"}]),
                   "codebook": pd.DataFrame()}          # 空表：不该落盘
        rb.write_inputs(workdir, payload, {"max_lag": 2, "note": "有\t制表"})
        files = sorted(os.listdir(workdir))
        assert "sequences.csv" in files
        assert "state_order.csv" in files                # 表名洗成 R 读得到的文件名
        assert "codebook.csv" not in files
        with open(os.path.join(workdir, "params.tsv"), encoding="utf-8") as handle:
            text = handle.read()
        assert text.splitlines()[0] == "key\tvalue"
        assert "note\t有 制表" in text                  # 制表符被压成空格，一参数一行
        with open(os.path.join(workdir, "payload.json"), encoding="utf-8") as handle:
            blob = json.load(handle)
        assert blob["params"]["max_lag"] == "2"
        assert blob["tables"]["sequences"][0]["codes"] == "A|B"

    def test_sequences_frame_labels_match_distance_matrix(self):
        ctx = _ctx()
        frame = rb.sequences_frame(ctx.sequences(dimension="D1"))
        assert list(frame.columns) == ["label", "group_id", "uid", "student",
                                       "dimension", "codes", "stream", "length"]
        assert set(frame["label"]) == {"%s / D1" % uid for uid in ctx.students}
        assert set(frame["length"]) == {12}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 差异表：双门槛判定 / 键对齐 / 缺失 / 折叠
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCrossCheck:
    def test_identical_numbers_all_agree(self):
        left = pd.DataFrame([{"state": "A", "n": 3}, {"state": "B", "n": 4}])
        rows = rb.cross_check(left, left.copy(), keys=("state",))
        assert [row["判定"] for row in rows] == ["一致", "一致"]
        assert all(row["绝对差"] == 0.0 for row in rows)

    def test_absolute_and_relative_thresholds(self):
        left = pd.DataFrame([{"state": "A", "v": 1.0}, {"state": "B", "v": 100000.0}])
        right = pd.DataFrame([{"state": "A", "v": 1.0 + 5e-7},      # 绝对差 ≤ 1e-6 → 一致
                              {"state": "B", "v": 100000.0 + 1e-3}])  # 相对差 ≤ 1e-4 → 一致
        rows = rb.cross_check(left, right, keys=("state",))
        assert [row["判定"] for row in rows] == ["一致", "一致"]
        bad = pd.DataFrame([{"state": "A", "v": 1.001}, {"state": "B", "v": 100000.0 + 30.0}])
        verdicts = [row["判定"] for row in rb.cross_check(left, bad, keys=("state",))]
        assert verdicts == ["超差", "超差"]

    def test_unaligned_keys_count_as_missing(self):
        left = pd.DataFrame([{"state": "A", "v": 1.0}, {"state": "B", "v": 2.0}])
        right = pd.DataFrame([{"state": "A", "v": 1.0}, {"state": "C", "v": 9.0}])
        rows = rb.cross_check(left, right, keys=("state",))
        agreed, failed, missing, _worst = rb.summarize_rows(rows)
        assert agreed == 1 and failed == 0 and missing == 2   # B 只 Python 有、C 只 R 有
        sides = {row["Python"] for row in rows if row["判定"] == "缺失"}
        assert sides == {"仅 Python 侧有", rb.MISSING_TEXT}

    def test_row_count_mismatch_in_positional_mode_is_missing(self):
        left = pd.DataFrame([{"x": 1.0}, {"x": 2.0}, {"x": 3.0}])
        right = pd.DataFrame([{"x": 1.0}, {"x": 2.0}])
        rows = rb.cross_check(left, right)
        assert [row["判定"] for row in rows] == ["一致", "一致", "缺失"]
        assert "仅 Python 侧多 1 行" in rows[-1]["量"]

    def test_missing_table_is_reported_not_swallowed(self):
        rows = rb.cross_check({"lsa_cells": pd.DataFrame([{"a": 1}])}, {}, keys=None)
        assert len(rows) == 1 and rows[0]["判定"] == "缺失"

    def test_positional_compare_when_no_shared_key(self):
        left = pd.DataFrame([{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}])
        right = pd.DataFrame([{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 5.0}])
        rows = rb.cross_check(left, right)
        assert len(rows) == 4
        assert [row["判定"] for row in rows].count("超差") == 1

    def test_blank_and_placeholder_values_count_as_missing(self):
        left = pd.DataFrame([{"state": "A", "v": 1.0}, {"state": "B", "v": ""}])
        right = pd.DataFrame([{"state": "A", "v": 1.0}, {"state": "B", "v": "-"}])
        rows = rb.cross_check(left, right, keys=("state",))
        assert [row["判定"] for row in rows] == ["一致", "缺失"]
        assert rb._clean("nan") is None and rb._clean(float("nan")) is None
        assert rb._text_of(None) == rb.MISSING_TEXT

    def test_trim_rows_collapses_but_keeps_verdicts(self):
        rows = [{"量": str(i), "Python": "1", "R": "1", "绝对差": 0.0,
                 "相对差": 0.0, "判定": "一致"} for i in range(15)]
        rows.append({"量": "坏", "Python": "1", "R": "9", "绝对差": 8.0,
                     "相对差": 0.9, "判定": "超差"})
        trimmed = rb.trim_rows(rows, limit=10)
        assert len(trimmed) == 11
        assert trimmed[-1]["判定"] == "超差"              # 折叠行仍把结论带出来
        assert trimmed[-1]["Python"] == "一致 5 / 超差 1 / 缺失 0"
        assert len(rb.trim_rows(rows[:4], limit=10)) == 4   # 不超限就原样返回


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 对照计划与脚本契约
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPlans:
    def test_every_declared_script_ships_with_the_package(self):
        for method_id, plan in rb.CHECK_PLANS.items():
            scripts = [plan.get("script")] + list((plan.get("scripts") or {}).values())
            for script in filter(None, scripts):
                assert os.path.exists(os.path.join(rb.R_REFS_DIR, script)), \
                    "%s 的 R 参照脚本缺席：%s" % (method_id, script)

    def test_plan_table_keys_are_documented(self):
        for method_id, plan in rb.CHECK_PLANS.items():
            assert plan.get("kind"), method_id
            assert plan.get("tables"), "%s 没声明对照表键" % method_id
            for name, keys in plan["tables"].items():
                assert isinstance(keys, tuple) and keys, (method_id, name)

    def test_registry_and_plans_stay_in_agreement(self):
        """注册表写了 r_reference 就必须有对照计划；反向允许（阶段 6/7 先钉契约）。"""
        declared = {spec.id: (spec.r_reference or "") for spec in am.all_methods()}
        for method_id, reference in declared.items():
            if reference and reference not in ("—", "-"):
                assert method_id in rb.CHECK_PLANS, \
                    "%s 声明了 %s 却没有对照计划" % (method_id, reference)
        for method_id in rb.CHECK_PLANS:
            if method_id in declared:
                assert declared[method_id], "%s 有对照计划却把 r_reference 写空了" % method_id

    def test_multichannel_method_routes_to_its_own_script(self):
        plan = rb.CHECK_PLANS["IND05"]
        assert rb._plan_script(plan, {"method": "OM"}) == "traminer_om.R"
        assert rb._plan_script(plan, {"method": "mcm"}) == "traminer_multichannel.R"
        assert rb._plan_script(plan, {}) == "traminer_om.R"

    def test_lsa_max_lag_caps_match_python_side(self):
        assert rb._lsa_max_lag({"max_lag": 9}, "dimension") == 5      # IND01 上限 5
        assert rb._lsa_max_lag({"max_lag": 9}, "group+dimension") == 3  # GRP05 上限 3
        assert rb._lsa_max_lag({}, "dimension") == 3                  # 没给 → IND01 默认 3
        assert rb._lsa_max_lag({}, "group+dimension") == 1            # 没给 → GRP05 默认 1

    def test_status_note_templates_take_at_most_one_placeholder(self):
        for key, template in rb.STATUS_NOTES.items():
            assert template.count("%s") <= 1, key


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# reference_check：真假 R 与降级路径
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestReferenceCheck:
    def test_no_data_honest_note(self):
        rows, note = rb.reference_check(am.method_by_id("IND02"), None, None, {})
        assert rows == [] and "尚未装载数据" in note

    def test_method_without_plan_explains_itself(self):
        spec = am.method_by_id("GRP01")
        assert spec is not None and spec.id not in rb.CHECK_PLANS
        rows, note = rb.reference_check(spec, None, _ctx(), {})
        assert rows == [] and "R 参考实现" in note

    def test_missing_rscript_degrades_without_raising(self, monkeypatch):
        spec = am.method_by_id("IND02")
        ctx = _ctx()
        params = _params(spec, ctx)
        result = spec.compute(ctx, params)
        monkeypatch.setattr(rb, "probe", lambda force=False: dict(UNAVAILABLE))
        rows, note = rb.reference_check(spec, result, ctx, params)
        assert rows == [] and "Rscript" in note

    def test_cells_match_when_r_agrees(self, monkeypatch):
        spec = am.method_by_id("IND02")
        ctx = _ctx()
        params = _params(spec, ctx)
        result = spec.compute(ctx, params)
        tables = rb.python_reference(spec, ctx, params, result)
        assert any(len(frame) for frame in tables.values())
        monkeypatch.setattr(rb, "probe", lambda force=False: dict(AVAILABLE))
        monkeypatch.setattr(rb, "run_reference", _fake_run_reference(tables))
        rows, note = rb.reference_check(spec, result, ctx, params)
        full = result.meta["r_crosscheck"]
        agreed, failed, missing, worst = rb.summarize_rows(full)
        assert failed == 0 and missing == 0 and worst == 0.0
        assert agreed == len(full) and len(rows) <= rb.MAX_ROWS + 1
        assert "markov_ref.R" in note and "R 4.5.0" in note
        assert result.meta["r_status"] == "ok"
        assert result.meta["r_engine"] == {"r": "fake"}

    def test_tampered_r_table_is_caught(self, monkeypatch):
        """R 撒谎（哪怕只改一格）时必须被判超差 —— 这才是「两路互证」的意义。"""
        spec = am.method_by_id("IND02")
        ctx = _ctx()
        params = _params(spec, ctx)
        result = spec.compute(ctx, params)
        tables = {name: frame.copy() for name, frame in
                  rb.python_reference(spec, ctx, params, result).items()}
        victim = tables["markov_N"]
        column = _numeric_column(victim)
        victim[column] = victim[column].astype(float)
        victim.loc[victim.index[-1], column] = float(victim[column].iloc[-1]) + 1.0
        monkeypatch.setattr(rb, "probe", lambda force=False: dict(AVAILABLE))
        monkeypatch.setattr(rb, "run_reference", _fake_run_reference(tables))
        rows, note = rb.reference_check(spec, result, ctx, params)
        _agreed, failed, missing, worst = rb.summarize_rows(result.meta["r_crosscheck"])
        assert failed == 1 and missing == 0 and worst == pytest.approx(1.0)
        assert "超差 1" in note
        assert rows and any(row["判定"] == "超差" for row in rows)

    def test_failed_r_run_reports_status_text(self, monkeypatch):
        spec = am.method_by_id("IND02")
        ctx = _ctx()
        params = _params(spec, ctx)
        result = spec.compute(ctx, params)
        monkeypatch.setattr(rb, "probe", lambda force=False: dict(AVAILABLE))
        monkeypatch.setattr(rb, "run_reference",
                            _fake_run_reference({}, status="package-missing"))
        rows, note = rb.reference_check(spec, result, ctx, params)
        assert rows == [] and "package-missing" not in note   # 翻成人话，不甩状态码
        assert "setup_r_reference.R" in note

    def test_skip_plan_covers_only_the_other_aggregate(self):
        """IND01 只勾「跨维」时 R 参照不覆盖：必须实话，不拿同维的数冒充。"""
        spec = am.method_by_id("IND01")
        ctx = _ctx()
        params = _params(spec, ctx)
        params["scope"] = "cross"
        rows, note = rb.reference_check(spec, spec.compute(ctx, params), ctx, params)
        assert rows == [] and "不做逐格对照" in note


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 探测与安装：无 R 的机器上也要安静
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestProbeAndInstall:
    def test_probe_shape_is_stable(self):
        status = rb.probe(force=True)
        for key in ("available", "version", "packages", "found", "missing", "rscript"):
            assert key in status, key
        assert isinstance(status["packages"], dict)
        assert bool(status["available"]) == bool(status["rscript"])

    def test_find_rscript_never_raises(self):
        assert rb.find_rscript() is None or os.path.isfile(rb.find_rscript())

    def test_run_reference_without_rscript_degrades(self, monkeypatch):
        monkeypatch.setattr(rb, "find_rscript", lambda: None)
        out = rb.run_reference("markov_ref.R", {}, {})
        assert out["ok"] is False and out["status"] == "no-r"
        assert "r_settings.json" in out["error"]

    def test_run_reference_without_script_file_degrades(self, monkeypatch):
        monkeypatch.setattr(rb, "find_rscript", lambda: "Rscript")
        out = rb.run_reference("没有这个脚本.R", {}, {})
        assert out["status"] == "no-script" and out["available"] is True

    def test_install_command_names_script_and_library(self, monkeypatch):
        monkeypatch.setattr(rb, "probe", lambda force=False: {"rscript": "Rscript"})
        command = rb.install_command("https://mirrors.tuna.tsinghua.edu.cn/CRAN/")
        assert "--vanilla" in command
        assert "setup_r_reference.R" in command
        assert "Rlibrary" in command
        assert "CRAN" in command

    def test_install_references_without_r_is_a_note_not_a_crash(self, monkeypatch):
        monkeypatch.setattr(rb, "find_rscript", lambda: None)
        monkeypatch.setattr(rb, "probe", lambda force=False: dict(UNAVAILABLE))
        report = rb.install_references()
        assert report["ok"] is False and report["status"] == "no-r"
