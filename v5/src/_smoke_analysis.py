"""无 GUI 的统计口径回归测试：逐条核对 V1.1 方案 §17 的验收基准。

运行：D:\\miniconda\\miniconda\\python.exe _smoke_analysis.py
不依赖 Qt，只用 pandas / matplotlib(Agg)。
"""

import os
import sys
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
warnings.simplefilter("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import theme                       # noqa: E402  仅用 sync_figure / CATEGORY_COLORS
from statistic_visualization_data import (CodingDataset, load_dataframe,
                                          parse_bin_bounds)   # noqa: E402
import statistic_visualization_methods as AM      # noqa: E402
import statistic_visualization_plots as AP        # noqa: E402

GROUP1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "Results", "testfile", "group_1.xlsx")

failures = []


def check(label, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + label + ("  " + str(extra) if extra else ""))
    if not cond:
        failures.append(label)


def counts_of(result, student, dimension):
    frame = result.table()
    part = frame[(frame["student_id"] == student) & (frame["dimension"] == dimension)]
    return dict(zip(part["code"].tolist(), part["count"].tolist()))


def unit_total(result, student, dimension):
    frame = result.table()
    part = frame[(frame["student_id"] == student) & (frame["dimension"] == dimension)]
    return int(part["count"].sum())


def run(spec_id, ds, **override):
    spec = AM.method_spec(spec_id)
    params = dict(AM.DEFAULT_PARAMS[spec_id])
    params.update(override)
    return spec, params, spec.compute(ds, params)


PREVIEW_W, PREVIEW_H = 9.0, 5.2      # 预览区典型尺寸（英寸），与模块内一致


def render(spec, result, ds, params):
    """离屏渲染，走模块同一套“先量内容需要多高、再在最终尺寸上重画”的流程。"""
    fig = Figure(figsize=(PREVIEW_W, PREVIEW_H), dpi=100)

    def paint():
        fig.clear()
        theme.sync_figure(fig)
        AP.reset_size(fig)
        spec.render(fig, result, ds, params)

    paint()
    size = AP.fit_figure(fig, PREVIEW_W, PREVIEW_H)
    if abs(size[1] - PREVIEW_H) > 0.05 or abs(size[0] - PREVIEW_W) > 0.05:
        paint()                       # 尺寸变了：版式必须按新画布重排
        AP.fit_figure(fig, PREVIEW_W, PREVIEW_H)
    return fig


# ------------------------------------------------------------------ 基准数据
print("=" * 78)
print("1) group_1.xlsx 时间口径")
print("=" * 78)

if not os.path.exists(GROUP1):
    print("跳过：找不到测试文件 %s" % GROUP1)
    sys.exit(0)

base = CodingDataset.build(load_dataframe(GROUP1), source=os.path.basename(GROUP1))
check("解析出 6 个编码列", len(base.column_map) == 6, len(base.column_map))
check("学生 = S1/S2", base.students == ["S1", "S2"], base.students)
check("维度 = D1/D2/D3", base.dimensions == ["D1", "D2", "D3"], base.dimensions)
check("188 个唯一时间点", base.time_info.n_positions == 188, base.time_info.n_positions)
check("median Δt = 10", abs(base.time_info.median_dt - 10.0) < 1e-9, base.time_info.median_dt)
check("判定为规则采样", bool(base.time_info.is_regular))
check("长表 1128 行 = 6 单元 × 188", len(base.long_df) == 1128, len(base.long_df))
check("无缺失单元格", int(base.long_df["is_missing"].sum()) == 0)
check("时间跨度 0–1870", (base.time_info.t_min, base.time_info.t_max) == (0.0, 1870.0),
      (base.time_info.t_min, base.time_info.t_max))

print()
print("=" * 78)
print("2) M01 逐单元频次 = 文档基准")
print("=" * 78)

spec, params, m01 = run("M01", base)
expected = {
    ("S1", "D1"): {"ST": 185, "SO": 3},
    ("S2", "D1"): {"ST": 182, "SI": 3, "SO": 3},
    ("S1", "D2"): {"MN": 104, "ML": 71, "MS": 5, "ME": 3, "MB": 3, "MA": 2},
    ("S2", "D2"): {"MN": 95, "ML": 55, "ME": 23, "MS": 9, "MB": 3, "MI": 2, "MC": 1},
    ("S1", "D3"): {"PO": 91, "PW": 58, "PD": 16, "PR": 12, "PA": 8, "PS": 3},
    ("S2", "D3"): {"PO": 107, "PW": 34, "PR": 26, "PA": 9, "PD": 9, "PS": 3},
}
for (student, dimension), table in expected.items():
    got = counts_of(m01, student, dimension)
    check("%s×%s 频次" % (student, dimension), got == table, got)
    check("%s×%s 合计 188" % (student, dimension), unit_total(m01, student, dimension) == 188)

frame = m01.table()
check("missing_N 全为 0", int(frame["missing_N"].sum()) == 0)
check("valid_N 每单元 188", set(frame["valid_N"].tolist()) == {188})
check("比例每单元合计 100%", bool(
    (frame.groupby(["student_id", "dimension"])["percentage"].sum() - 100.0)
    .abs().le(0.5).all()))

print()
print("=" * 78)
print("3) M03 时长 / M08 分箱 / M14 spell / M15 共现")
print("=" * 78)

_, _, m03 = run("M03", base)
st = m03.table()
row = st[(st["student_id"] == "S1") & (st["dimension"] == "D1") & (st["code"] == "ST")]
check("S1×D1 ST 时长 1850 s", float(row["duration_s"].iloc[0]) == 1850.0,
      float(row["duration_s"].iloc[0]))
check("S1×D1 ST 时长 30.83 min", abs(float(row["duration_min"].iloc[0]) - 30.8333) < 0.01)
check("规则采样下 D_c = F_c × Δt", bool(
    np.allclose(st["duration_s"].to_numpy(dtype=float),
                st["count"].to_numpy(dtype=float) * base.time_info.median_dt)))

_, _, m08 = run("M08", base, bin_mode="count", bin_value=6)
check("六分箱产生 6 个窗口", len(base.bin_order) == 6, base.bin_order)
check("窗口边界对齐采样格", base.bin_bounds[0][1] == 0.0 and
      base.bin_bounds[1][1] == 320.0, base.bin_bounds[:2])
for name, table in m08.tables.items():
    student, dimension = name.split("×")
    columns = [c for c in table.columns if c != "code"]
    totals = {c: int(table[c].sum()) for c in columns}
    window = base.slice([student], [dimension])
    window = window[window["time_bin"].isin(columns)]
    per_bin = window.groupby("time_bin").size()
    ok = all(totals[c] == int(per_bin.get(c, 0)) for c in columns)
    check("%s 各列合计 = 该窗口有效编码数" % name, ok and sum(totals.values()) == 188,
          (totals, sum(totals.values())))

_, _, m15 = run("M15", base, scope="within", student="S1", dim_a="D2", dim_b="D3")
matrix = m15.table()
numeric = [c for c in matrix.columns if c not in ("D2 编码", "合计")]
check("S1 D2×D3 共现矩阵总和 = 188",
      int(matrix[numeric].to_numpy(dtype=float).sum()) == 188,
      int(matrix[numeric].to_numpy(dtype=float).sum()))
check("共现矩阵行合计列 = 188", int(matrix["合计"].sum()) == 188)
check("长表总和 = 188", int(m15.tables["长表"]["count"].sum()) == 188)

_, _, m08c = run("M08", base, bin_mode="custom", bin_custom="0, 600, 1200, 1880")
check("自定义边界文本拆成数值列表",
      parse_bin_bounds("0, 600，1200；1880 x") == [0.0, 600.0, 1200.0, 1880.0],
      parse_bin_bounds("0, 600，1200；1880 x"))
check("自定义边界产生 3 个窗口", len(base.bin_order) == 3, base.bin_order)
for name, table in m08c.tables.items():
    cols = [c for c in table.columns if c != "code"]
    total = int(table[cols].to_numpy(dtype=float).sum())
    check("%s 自定义分箱合计 = 188" % name, total == 188, total)

# 固定窗长允许小数（亚秒级采样时整数窗长表达不了）
_, _, m08f = run("M08", base, bin_mode="seconds", bin_value=12.5)
check("小数窗长 12.5 s 产生 151 个窗口", len(base.bin_order) == 151, len(base.bin_order))
for name, table in m08f.tables.items():
    student, dimension = name.split("×")
    cols = [c for c in table.columns if c != "code"]
    window = base.slice([student], [dimension])
    per_bin = window[window["time_bin"].isin(cols)].groupby("time_bin").size()
    ok = all(int(table[c].sum()) == int(per_bin.get(c, 0)) for c in cols)
    check("%s 小数窗长各列合计仍等于窗口有效数" % name,
          ok and int(table[cols].to_numpy(dtype=float).sum()) == 188)
check("小数边界写进元数据",
      any(abs(float(low) % 1) > 0 for _l, low, _h in base.bin_bounds), base.bin_bounds[:2])

_, _, m15x = run("M15", base, scope="cross", student_a="S1", student_b="S2", dimension="D2")
check("cross 口径不叫 co-occurrence",
      "Cross-student" in m15x.method_name and "Co-occurrence" not in m15x.method_name,
      m15x.method_name)

_, _, m14 = run("M14", base)
detail = m14.tables["Spell 明细"]
key = ((detail["student_id"] == "S1") & (detail["dimension"] == "D1")
       & (detail["code"] == "ST"))
check("S1×D1 ST spell 总时长 1850 s", float(detail[key]["duration_s"].sum()) == 1850.0,
      float(detail[key]["duration_s"].sum()))
check("spell 明细含 spell_index / start_time",
      {"spell_index", "start_time", "run_length", "duration_s"} <= set(detail.columns))
summary = m14.tables["Spell 汇总"]
check("spell 总时长 = M03 时长", bool(np.allclose(
    summary["total_s"].to_numpy(dtype=float).sum(),
    m03.table()["duration_s"].to_numpy(dtype=float).sum())))

print()
print("=" * 78)
print("4) M15 未确认 scope 必须阻断")
print("=" * 78)

_, _, blocked = run("M15", base)
check("未选 scope → blocked", bool(blocked.blocked), blocked.blocked_reason)
check("给出 cooccurrence_scope_unconfirmed",
      any(item.code == "cooccurrence_scope_unconfirmed" for item in blocked.qa))
check("阻断项标记 blocking",
      any(item.blocking for item in blocked.qa))

print()
print("=" * 78)
print("5) 确定性：同输入同参数跑两遍")
print("=" * 78)

for method_id, override in (("M01", {}), ("M05", {"bin_mode": "count", "bin_value": 6}),
                            ("M11", {"sort_by": "first_code"}), ("M16", {"dimension": "D2"})):
    spec_a, pa, ra = run(method_id, base, **override)
    _, _, rb = run(method_id, base, **override)
    same = set(ra.tables) == set(rb.tables) and all(
        ra.tables[name].equals(rb.tables[name]) for name in ra.tables)
    check("%s 两次结果逐单元格一致" % method_id, same)
    check("%s 两次 axes 数一致" % method_id,
          len(render(spec_a, ra, base, pa).axes) == len(render(spec_a, rb, base, pa).axes))

print()
print("=" * 78)
print("6) 合成不规则采样数据")
print("=" * 78)

irregular = pd.DataFrame({
    "Time": [0, 10, 25, 60, 61, 120, 121, 200],
    "S1Encoding_1": list("AABBBAAA")[:8],
    "S2Encoding_1": list("BBAAABAB")[:8],
    "S1Encoding_2": list("XYXYXYXY")[:8],
    "S2Encoding_2": list("YXYXYXYX")[:8],
})
ir_ds = CodingDataset.build(irregular, source="synthetic-irregular")
check("判定为不规则采样", not ir_ds.time_info.is_regular)
check("Δt 取相邻差的中位数", abs(ir_ds.time_info.median_dt - 15.0) < 1e-9,
      ir_ds.time_info.median_dt)
check("QA 含 irregular_sampling",
      any(item.code == "irregular_sampling" for item in ir_ds.qa))
durations = ir_ds.long_df[ir_ds.long_df["student_id"] == "S1"].drop_duplicates("time")
check("duration 走相邻时间差",
      durations["duration"].tolist()[:4] == [10.0, 15.0, 35.0, 1.0],
      durations["duration"].tolist())
check("末点沿用前一段差", abs(float(durations["duration"].iloc[-1]) - 79.0) < 1e-9,
      float(durations["duration"].iloc[-1]))
_, _, ir_m03 = run("M03", ir_ds)
check("不规则下 D_c ≠ F_c × Δt", bool(
    not np.allclose(ir_m03.table()["duration_s"].to_numpy(dtype=float),
                    ir_m03.table()["count"].to_numpy(dtype=float) * ir_ds.time_info.median_dt)))
per_unit = ir_m03.table().query("student_id=='S1' and dimension=='D1'")["duration_s"].sum()
check("每单元时长合计 = 时间跨度 + 末段沿用差值",
      abs(float(per_unit) - (200.0 + 79.0)) < 1e-6, float(per_unit))
check("不规则时长不再等于 F_c × Δt", bool(
    np.abs(ir_m03.table()["duration_s"].to_numpy(dtype=float)
           - ir_m03.table()["count"].to_numpy(dtype=float) * 15.0).max() > 1e-6))

print()
print("=" * 78)
print("7) 合成缺失数据：分母口径")
print("=" * 78)

sparse = pd.DataFrame({
    "Time": ["00:00", "00:10", "00:20", "00:30", "00:40", "00:50"],
    "S1Encoding_1": ["A", "A", None, "B", "", "B"],
    "S2Encoding_1": ["A", None, "B", "B", "A", "A"],
    "S1Encoding_2": [None, None, None, None, None, None],
    "S2Encoding_2": [None, None, None, None, None, None],
})
sp_ds = CodingDataset.build(sparse, source="synthetic-missing")
check("HH:MM:SS 文本被折算成秒", sp_ds.time_info.t_max == 50.0, sp_ds.time_info.t_max)
_, _, sp_m01 = run("M01", sp_ds)
s1d1 = counts_of(sp_m01, "S1", "D1")
unit_rows = sp_m01.table().query("student_id=='S1' and dimension=='D1'")
check("S1×D1 有效数 4（None 与空串都不计）",
      int(unit_rows["valid_N"].iloc[0]) == 4, int(unit_rows["valid_N"].iloc[0]))
check("S1×D1 A=2 B=2", s1d1 == {"A": 2, "B": 2}, s1d1)
check("S1×D1 missing_N=2", int(unit_rows["missing_N"].iloc[0]) == 2,
      int(unit_rows["missing_N"].iloc[0]))
_, _, sp_m02 = run("M02", sp_ds)
part = sp_m02.table().query("student_id=='S1' and dimension=='D1'")
check("比例分母是有效数 4 → 各 50%",
      sorted(part["percentage"].tolist()) == [50.0, 50.0], part["percentage"].tolist())
check("全空维度 D2 被从维度列表剔除", "D2" not in sp_ds.dimensions, sp_ds.dimensions)
check("QA 含 dimension_all_empty",
      any(item.code == "dimension_all_empty" for item in sp_ds.qa),
      [item.code for item in sp_ds.qa])
check("QA 含 partial_missing",
      any(item.code == "partial_missing" for item in sp_ds.qa))

print()
print("=" * 78)
print("8) 全部 19 个方法 × 三种数据都能跑通并出图")
print("=" * 78)

CASES = {
    "group_1": (base, {
        "M06": {"student": "S1", "dimension": "D1"},
        "M09": {"dimension": "D1"},
        "M10": {"dimension": "D1"},
        "M12": {"dimension": "D1"},
        "M15": {"scope": "within", "student": "S1", "dim_a": "D2", "dim_b": "D3"},
        "M16": {"dimension": "D1"},
    }),
    "irregular": (ir_ds, {
        "M06": {"student": "S1", "dimension": "D1"},
        "M09": {"dimension": "D1"}, "M10": {"dimension": "D1"},
        "M12": {"dimension": "D1"},
        "M15": {"scope": "cross", "student_a": "S1", "student_b": "S2", "dimension": "D1"},
        "M16": {"dimension": "D1"},
    }),
    "missing": (sp_ds, {
        "M06": {"student": "S1", "dimension": "D1"},
        "M09": {"dimension": "D1"}, "M10": {"dimension": "D1"},
        "M12": {"dimension": "D1"},
        "M15": {"scope": "within", "student": "S1", "dim_a": "D1", "dim_b": "D1"},
        "M16": {"dimension": "D1"},
    }),
}

for case, (ds, overrides) in CASES.items():
    crashed = []
    blocked_ids = []
    for spec in AM.METHODS:
        params = dict(AM.DEFAULT_PARAMS[spec.id])
        params.update(overrides.get(spec.id, {}))
        try:
            result = spec.compute(ds, params)
            if result.blocked:
                blocked_ids.append(spec.id)
                continue
            fig = render(spec, result, ds, params)
            if not fig.axes:
                crashed.append("%s 无坐标轴" % spec.id)
            if not AP.MIN_FIG_H_IN - 0.01 <= fig.get_figheight() <= AP.MAX_FIG_H_IN + 0.01:
                crashed.append("%s 画布高度越界 %.2f" % (spec.id, fig.get_figheight()))
            need_h = float(getattr(fig, "_sv_need", (0.0, 0.0))[1])
            if need_h > PREVIEW_H and abs(fig.get_figheight()
                                          - min(need_h, AP.MAX_FIG_H_IN)) > 0.06:
                crashed.append("%s 未按内容声明撑高：%.2f ≠ %.2f"
                               % (spec.id, fig.get_figheight(), need_h))
            for name, table in result.tables.items():
                if not isinstance(table, pd.DataFrame):
                    crashed.append("%s/%s 不是 DataFrame" % (spec.id, name))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            crashed.append("%s: %s" % (spec.id, exc))
    check("%s：19 个方法无异常" % case, not crashed, crashed)
    check("%s：仅 M15 未确认口径一类被阻断" % case,
          set(blocked_ids) <= {"M15"}, blocked_ids)

print()
print("=" * 78)
print("9) 自动质检：每一类问题都要能真的被检出")
print("=" * 78)


def qa_codes_of(ds_or_result):
    return [item.code for item in ds_or_result.qa]


# Time 重复 + 非递增：同一 student×time 出现两次，且原始行序乱序
messy = pd.DataFrame({
    "Time": [0, 20, 10, 10, 30, 40],
    "S1Encoding_1": list("AABABB")[:6],
    "S2Encoding_1": list("BBAABA")[:6],
})
md_ds = CodingDataset.build(messy, source="synthetic-messy")
check("检出 time_duplicate", "time_duplicate" in qa_codes_of(md_ds), qa_codes_of(md_ds))
check("time_duplicate 标记为阻断",
      any(item.code == "time_duplicate" and item.blocking for item in md_ds.qa))
check("重复计数 = 2（两名学生各一条）", md_ds.duplicate_rows == 2, md_ds.duplicate_rows)
check("重复行只保留首条，长表 10 行", len(md_ds.long_df) == 10, len(md_ds.long_df))
check("检出 time_non_monotonic", "time_non_monotonic" in qa_codes_of(md_ds),
      qa_codes_of(md_ds))
ordered = md_ds.long_df[md_ds.long_df["student_id"] == "S1"]["time"].tolist()
check("非递增已自动排序", ordered == sorted(ordered), ordered)
check("乱序数据仍按 10 s 判为规则采样",
      bool(md_ds.time_info.is_regular) and abs(md_ds.time_info.median_dt - 10.0) < 1e-9,
      (md_ds.time_info.is_regular, md_ds.time_info.median_dt))

# Time 有效值不足 2 个
bad_time = CodingDataset.build(pd.DataFrame({
    "Time": ["", "not-a-time", "10"],
    "S1Encoding_1": ["A", "A", "B"],
}), source="synthetic-bad-time")
check("检出 time_missing", "time_missing" in qa_codes_of(bad_time), qa_codes_of(bad_time))
check("time_missing 标记为阻断",
      any(item.code == "time_missing" and item.blocking for item in bad_time.qa))
check("不可解析的 Time 被丢弃", len(bad_time.long_df) == 1, len(bad_time.long_df))
_, _, bt_m04 = run("M04", bad_time)
check("Time 不足时时间窗口方法被阻断", bool(bt_m04.blocked), bt_m04.blocked_reason)

# 状态数过多（>25 种编码）
many = pd.DataFrame({
    "Time": [i * 10 for i in range(30)],
    "S1Encoding_1": ["C%02d" % i for i in range(30)],
})
mn_ds = CodingDataset.build(many, source="synthetic-many-states")
check("检出 too_many_states", "too_many_states" in qa_codes_of(mn_ds), qa_codes_of(mn_ds))
check("状态数阈值取 25", len(mn_ds.codes_by_dim["D1"]) == 30)

# 序列唯一性过高：group_1 两名学生两条完全不同的序列
_, _, m12 = run("M12", base, dimension="D1")
check("M12 检出 high_sequence_uniqueness",
      "high_sequence_uniqueness" in qa_codes_of(m12), qa_codes_of(m12))
check("唯一率判定符合实测（2 学生 2 序列）",
      abs(base.uniqueness_ratio("D1") - 1.0) < 1e-9, base.uniqueness_ratio("D1"))
top_table = m12.tables["Top-10 序列"]
check("两条唯一序列各计 1 人 / 50%",
      top_table["count"].tolist() == [1, 1]
      and top_table["percentage"].tolist() == [50.0, 50.0],
      top_table[["count", "percentage"]].to_dict("records"))
check("未超出 Top-N 时不另出「全部序列」表", "全部序列" not in m12.tables,
      list(m12.tables))
check("相位序列表逐学生给出原始序列", len(m12.tables["相位序列"]) == len(base.students))
_, _, m12_top1 = run("M12", base, dimension="D1", top_n=1)
check("Top-N 满额截断时补「全部序列」表",
      len(m12_top1.tables["Top-1 序列"]) == 1 and len(m12_top1.tables["全部序列"]) == 2,
      {k: len(v) for k, v in m12_top1.tables.items()})

print()
print("=" * 78)
print("10) M17–M19 熵族：手算基准与内部一致性")
print("=" * 78)

# sp_ds 的 S1×D1 有效序列 = [A, A, B, B]（4 格），S2×D1 = [A, B, B, A, A]（5 格）
import math                                                      # noqa: E402
_, _, m17 = run("M17", sp_ds)
e17 = m17.tables["状态熵"]
s1_row = e17.query("student_id=='S1' and dimension=='D1'").iloc[0]
check("M17 S1×D1 平衡二态 → H = ln2", abs(s1_row["entropy"] - math.log(2)) < 1e-4,
      s1_row["entropy"])
check("M17 H_norm = 1.0（完全均匀）", abs(s1_row["H_norm"] - 1.0) < 1e-9, s1_row["H_norm"])
check("M17 N=4 且 missing_N=2（缺失不入分母）",
      int(s1_row["N"]) == 4 and int(s1_row["missing_N"]) == 2, dict(s1_row))
s2_p = [3 / 5, 2 / 5]
hand_s2 = -sum(q * math.log(q) for q in s2_p)
s2_row = e17.query("student_id=='S2' and dimension=='D1'").iloc[0]
check("M17 S2×D1 H 与手算一致", abs(s2_row["entropy"] - hand_s2) < 1e-4,
      (s2_row["entropy"], hand_s2))
check("M17 n_states 取实际出现状态数", int(s1_row["n_states"]) == 2)
contrib = m17.tables["各状态概率贡献"].query("student_id=='S1' and dimension=='D1'")
check("M17 逐状态贡献相加 = H", abs(float(contrib["contribution"].sum()) - math.log(2)) < 2e-4,
      float(contrib["contribution"].sum()))

_, _, m18 = run("M18", sp_ds)
e18 = m18.tables["切换与保持"]
r18 = e18.query("student_id=='S1' and dimension=='D1'").iloc[0]
check("M18 S1×D1 转移对数 = T-1 = 3", int(r18["n_transitions"]) == 3, int(r18["n_transitions"]))
check("M18 switch_rate = 1/3", abs(r18["switch_rate"] - 1 / 3) < 1e-4, r18["switch_rate"])
check("M18 self_transition = 2/3", abs(r18["self_transition"] - 2 / 3) < 1e-4,
      r18["self_transition"])
check("M18 runs=2、平均 run 长 = 2.0",
      int(r18["n_runs"]) == 2 and abs(r18["mean_run_length"] - 2.0) < 1e-9,
      (r18["n_runs"], r18["mean_run_length"]))
r18b = e18.query("student_id=='S2' and dimension=='D1'").iloc[0]
check("M18 S2×D1 切换 2/4 = 0.5、runs=3",
      abs(r18b["switch_rate"] - 0.5) < 1e-9 and int(r18b["n_runs"]) == 3,
      (r18b["switch_rate"], r18b["n_runs"]))
check("M18 全体单元 switch + stay = 1", bool(
    (e18["switch_rate"] + e18["self_transition"]).round(4).eq(1.0).all()))
check("M18 逐单元 n_switch = n_runs - 1", bool(
    (e18["n_switch"] == e18["n_runs"] - 1).all()))
check("M18 逐单元 mean_run_length × runs = T_valid", bool(
    (e18["mean_run_length"] * e18["n_runs"] - e18["T_valid"]).abs().le(0.02).all()))

_, _, m19 = run("M19", sp_ds)
e19 = m19.tables["条件熵"]
r19 = e19.query("student_id=='S1' and dimension=='D1'").iloc[0]
hand_hyx = (2 / 3) * math.log(2)
hand_hy = -(1 / 3) * math.log(1 / 3) - (2 / 3) * math.log(2 / 3)
check("M19 S1×D1 H(Y|X) 手算对照", abs(r19["H_Y_given_X"] - hand_hyx) < 2e-4,
      (r19["H_Y_given_X"], hand_hyx))
check("M19 H(Y) 手算对照", abs(r19["H_Y"] - hand_hy) < 2e-4, (r19["H_Y"], hand_hy))
check("M19 reduction = H(Y) - H(Y|X)", abs(r19["reduction"] - (hand_hy - hand_hyx)) < 3e-4,
      r19["reduction"])
check("M19 H_norm = H(Y|X)/ln2", abs(r19["H_norm"] - hand_hyx / math.log(2)) < 3e-4,
      r19["H_norm"])
check("M19 逐单元 0 ≤ reduction ≤ H(Y) 且 H(Y|X) ≤ H(Y)", bool(
    ((e19.reduction >= 0) & (e19.reduction <= e19.H_Y + 1e-9)
     & (e19.H_Y_given_X <= e19.H_Y + 1e-9)).all()))
b19 = m19.tables["前态熵分解"].query("student_id=='S1' and dimension=='D1'")
check("M19 前态贡献相加 = H(Y|X)",
      abs(float(b19["contribution"].sum()) - r19["H_Y_given_X"]) < 3e-4,
      float(b19["contribution"].sum()))
d19 = m19.tables["转移条件分布"].query("student_id=='S1' and dimension=='D1'")
check("M19 条件分布每行相加 = 1", bool(
    d19.groupby("from_code")["p_to_given_from"].sum().round(4).eq(1.0).all()),
    d19.groupby("from_code")["p_to_given_from"].sum().to_dict())

# group_1 实测一致性：切换数 = runs - 1，M17 与 M01 同分母
_, _, g18 = run("M18", base)
check("M18 group_1 逐单元 n_switch = n_runs - 1", bool(
    (g18.tables["切换与保持"]["n_switch"] == g18.tables["切换与保持"]["n_runs"] - 1).all()))
_, _, g17 = run("M17", base)
check("M17 group_1 的 N 合计 = 单元数 × 188",
      int(g17.tables["状态熵"]["N"].sum()) == 188 * len(base.units()),
      int(g17.tables["状态熵"]["N"].sum()))

# 单一状态与空数据：边界不能抛异常
flat = CodingDataset.build(pd.DataFrame({
    "Time": [0, 10, 20, 30], "S1Encoding_1": ["A", "A", "A", "A"],
    "S2Encoding_1": ["A", "A", None, "A"]}), source="synthetic-flat")
_, _, f17 = run("M17", flat)
f17_row = f17.tables["状态熵"].query("student_id=='S1'").iloc[0]
check("单一状态：H=0 且 H_norm 记 0（不是 0/0）",
      float(f17_row["entropy"]) == 0.0 and float(f17_row["H_norm"]) == 0.0, dict(f17_row))
_, _, f18 = run("M18", flat)
check("单一状态：switch_rate=0、self_transition=1、runs=1", bool(
    (f18.tables["切换与保持"]["switch_rate"] == 0.0).all()
    and (f18.tables["切换与保持"]["self_transition"] == 1.0).all()
    and (f18.tables["切换与保持"]["n_runs"] == 1).all()))
_, _, f19 = run("M19", flat)
check("只有一种状态时条件熵无定义→单元被跳过并进 QA",
      bool(f19.blocked) or "insufficient_transitions" in qa_codes_of(f19),
      f19.blocked_reason or [i.code for i in f19.qa])
empty_ds = CodingDataset.build(pd.DataFrame({"Time": [0, 10], "S1D1": ["A", "B"]}),
                               source="synthetic-unmapped")
for method_id in ("M01", "M02", "M03", "M17", "M18", "M19"):
    _spec, _params, _res = run(method_id, empty_ds)
    check("%s 在无可解析编码列时走阻断而非报错" % method_id,
          bool(_res.blocked) and not _res.tables, getattr(_res, "blocked_reason", ""))

# 三种视图与两个主题都得出图
for mode in ("light", "dark"):
    theme.theme.mode = mode
    for method_id, override in (("M17", {}), ("M17", {"value": "H_norm"}),
                                ("M18", {}), ("M19", {}),
                                ("M19", {"view": "decompose"}), ("M19", {"view": "matrix"}),
                                ("M19", {"view": "matrix", "show_values": False})):
        spec, params, result = run(method_id, base, **override)
        fig = render(spec, result, base, params)
        check("%s %s 出图（%s）" % (method_id, params.get("view", "-"), mode),
              bool(fig.axes), len(fig.axes))
theme.theme.mode = "light"

print()
print("=" * 78)
print("失败 %d 项" % len(failures))
for name in failures:
    print("  -", name)
sys.exit(1 if failures else 0)
