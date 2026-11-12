# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""M01–M19 方法注册表。

每个方法用一份 `MethodSpec` 声明：编号、名称、分组、回答什么问题、需要哪些参数、
绑定哪个统计函数与绘图函数、需要哪些分析单元。UI 只读这张表，不写任何方法专有分支；
新增方法只需在此追加一条声明。

参数的 `kind` 决定 UI 生成什么控件：
    checklist  多选（学生 / 维度）
    combo      单选下拉
    segmented  互斥分段按钮（口径选择，必须显式确认时使用）
    spin       整数输入
    dspin      浮点输入（窗长可以是亚秒级）
    check      勾选框
    text       单行文本（逗号分隔的边界列表）
`shows_for` 形如 ("scope", "within")：只有该参数取该值时控件才出现。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import statistic_visualization_plots as plots
import statistic_visualization_stats as stats

GROUP_BASIC = "A 基础频次"
GROUP_TIME = "B 时间分布"
GROUP_TRACE = "C 时间轨迹"
GROUP_SEQUENCE = "D 序列描述"
GROUP_COOC = "E 共现描述"
GROUP_PATH = "F 路径呈现"
GROUP_ENTROPY = "G 熵与多样性"

SORT_OPTIONS = (("count", "频次降序"), ("alpha", "字母序"), ("time", "时间序"))
BIN_MODE_OPTIONS = (("count", "等分窗数"), ("seconds", "固定窗长（秒）"),
                    ("custom", "自定义边界"))
# 图例位置：图外两个选项会单独预留边距，图内选项由 tight_layout 给面板多留行高
LEGEND_OPTIONS = (("below", "图形正下方"), ("right", "图形右侧"),
                  ("best", "图内空白处"), ("upper right", "图内右上"),
                  ("lower left", "图内左下"), ("none", "不显示"))


@dataclass
class ParamSpec:
    key: str
    label: str
    kind: str = "combo"
    options: tuple = ()
    default: object = None
    tooltip: str = ""
    shows_for: tuple = ()
    minimum: float = 0
    maximum: float = 100000


@dataclass
class MethodSpec:
    id: str
    name: str
    group: str
    question: str
    compute: object
    render: object
    params: tuple = ()
    units: tuple = ("students", "dimensions")
    requires_time: bool = False
    needs_dimension: bool = True
    min_students: int = 1
    student_hint: str = ""

    def title(self):
        return "%s %s" % (self.id, self.name)

    def availability(self, ds):
        """(是否可计算, 说明)。不可计算时 UI 在下拉里标注，但仍允许选中并查看质检原因。"""
        if ds is None or not getattr(ds, "usable", False):
            return False, "尚未装载可解析的编码数据"
        if self.requires_time and not ds.time_info.valid:
            return False, ds.time_info.describe()
        if self.needs_dimension and not ds.dimensions:
            return False, "没有解析出任何编码维度列"
        if len(ds.students) < self.min_students:
            return False, "该方法至少需要 %d 名学生，当前只有 %d 名" % (
                self.min_students, len(ds.students))
        if self.student_hint and len(ds.students) < 3:
            return True, self.student_hint
        return True, ""


# ------------------------------------------------------------------ 公共参数

P_STUDENTS = ParamSpec("students", "分析学生", "checklist", default=None,
                       tooltip="留空表示全部学生")
P_DIMENSIONS = ParamSpec("dimensions", "编码维度", "checklist", default=None,
                         tooltip="留空表示全部维度")
P_SORT = ParamSpec("sort", "编码排序", "combo", SORT_OPTIONS, default="count")
P_VALUES = ParamSpec("show_values", "显示数值标签", "check", default=True)
P_LEGEND = ParamSpec("legend_loc", "图例位置", "combo", LEGEND_OPTIONS, default="below",
                     tooltip="选“图形正下方 / 右侧”时会在图上单独留位，不会压住图形")
P_MERGE = ParamSpec("merge_students", "另附跨学生合计表", "check", default=False,
                    tooltip="该口径把 S1/S2 当作同类观测合并计数，不代表任何单个学生")
P_BIN_MODE = ParamSpec("bin_mode", "时间窗口划分", "combo", BIN_MODE_OPTIONS, default="count")
P_BIN_COUNT = ParamSpec("bin_value", "窗口数量", "spin", default=6, minimum=2, maximum=60,
                        shows_for=("bin_mode", "count"))
P_BIN_SECONDS = ParamSpec("bin_value", "窗口长度（秒）", "dspin", default=300.0, minimum=0.01,
                          maximum=100000.0, shows_for=("bin_mode", "seconds"),
                          tooltip="支持小数，亚秒级采样可填 0.5 这样的窗长")
P_BIN_CUSTOM = ParamSpec("bin_custom", "自定义边界（秒，逗号分隔）", "text", default="",
                         shows_for=("bin_mode", "custom"),
                         tooltip="例如 0,600,1200,1880，左闭右开")


def _bin_params():
    return (P_BIN_MODE, P_BIN_COUNT, P_BIN_SECONDS, P_BIN_CUSTOM)


# ------------------------------------------------------------------ M01–M03

M01 = MethodSpec(
    id="M01", name="Frequency 频次", group=GROUP_BASIC,
    question="每个编码在该学生该维度下一共出现了多少次？",
    compute=stats.frequency, render=plots.plot_frequency,
    params=(P_STUDENTS, P_DIMENSIONS, P_SORT, P_VALUES, P_MERGE))

M02 = MethodSpec(
    id="M02", name="Percentage 比例", group=GROUP_BASIC,
    question="每个编码占该单元有效观测数的比例是多少？",
    compute=stats.percentage, render=plots.plot_percentage,
    params=(P_STUDENTS, P_DIMENSIONS, P_SORT, P_VALUES))

M03 = MethodSpec(
    id="M03", name="Duration 持续时长", group=GROUP_BASIC,
    question="每个编码累计占据多长时间？",
    compute=stats.duration, render=plots.plot_duration,
    params=(P_STUDENTS, P_DIMENSIONS, P_SORT, P_VALUES),
    requires_time=True)

# ------------------------------------------------------------------ M04–M05

M04 = MethodSpec(
    id="M04", name="Time-bin Frequency 时间窗口频次", group=GROUP_TIME,
    question="每个时间窗口内各编码出现了多少次？",
    compute=stats.bin_frequency, render=plots.plot_bin_frequency,
    params=(P_STUDENTS, P_DIMENSIONS) + _bin_params() + (P_VALUES, P_LEGEND),
    requires_time=True)

M05 = MethodSpec(
    id="M05", name="Time-bin Percentage 时间窗口构成", group=GROUP_TIME,
    question="每个时间窗口内各编码的构成比例如何变化？",
    compute=stats.bin_percentage, render=plots.plot_bin_percentage,
    params=(P_STUDENTS, P_DIMENSIONS) + _bin_params()
            + (ParamSpec("chart_style", "图形形式", "combo",
                         (("bar", "100% 堆叠条"), ("area", "堆叠面积")), default="bar"),
               P_VALUES, P_LEGEND),
    requires_time=True)

# ------------------------------------------------------------------ M06–M08

P_STUDENT = ParamSpec("student", "学生", "combo", default=None,
                      tooltip="只分析一个 student × dimension 单元")
P_DIMENSION = ParamSpec("dimension", "编码维度", "combo", default=None)

M06 = MethodSpec(
    id="M06", name="Timeline 时间轴", group=GROUP_TRACE,
    question="单个学生在单个维度上的编码随时间如何排列？",
    compute=stats.timeline, render=plots.plot_timeline,
    params=(P_STUDENT, P_DIMENSION,
            ParamSpec("show_missing", "显示缺失色块", "check", default=True), P_LEGEND),
    units=("student", "dimension"), requires_time=True)

M07 = MethodSpec(
    id="M07", name="Multi-dimensional Timeline 多维同步时间轴", group=GROUP_TRACE,
    question="同一学生的 D1/D2/D3 三个维度在同一时刻是否同步变化？",
    compute=stats.multidim_timeline, render=plots.plot_multidim_timeline,
    params=(P_STUDENTS, P_DIMENSIONS,
            ParamSpec("show_missing", "显示缺失色块", "check", default=True), P_LEGEND),
    requires_time=True)

M08 = MethodSpec(
    id="M08", name="Heatmap 时间窗口热图", group=GROUP_TRACE,
    question="哪些编码集中在哪些时间段？",
    compute=stats.heatmap, render=plots.plot_heatmap,
    params=(P_STUDENTS, P_DIMENSIONS) + _bin_params()
            + (ParamSpec("value", "数值口径", "combo",
                         (("count", "计数"), ("percentage", "窗口内比例 (%)")), default="count"),
               P_VALUES),
    requires_time=True)

# ------------------------------------------------------------------ M09–M11

M09 = MethodSpec(
    id="M09", name="State Distribution 状态分布", group=GROUP_TRACE,
    question="固定一个维度，各时间位置上所有学生的状态构成如何？",
    compute=stats.state_distribution, render=plots.plot_state_distribution,
    params=(ParamSpec("dimension", "固定维度", "combo", default=None,
                      tooltip="跨学生统计，只能固定一个维度"), P_STUDENTS, P_LEGEND),
    units=("students", "dimension"), requires_time=True,
    student_hint="学生数 <3 时每个位置的比例只能是 0/50/100%，只能作个体比较")

M10 = MethodSpec(
    id="M10", name="Modal State 主导状态", group=GROUP_TRACE,
    question="每个时间位置上最常见的状态是什么？哪些位置存在并列？",
    compute=stats.modal_state, render=plots.plot_modal_state,
    params=(ParamSpec("dimension", "固定维度", "combo", default=None), P_STUDENTS, P_LEGEND),
    units=("students", "dimension"), requires_time=True,
    student_hint="主导状态只代表最常见状态，不表示所有学生都处于该状态")

M11 = MethodSpec(
    id="M11", name="Sequence Index 序列索引图", group=GROUP_TRACE,
    question="每学生的编码序列长什么样？个体之间的结构差异在哪里？",
    compute=stats.sequence_index, render=plots.plot_sequence_index,
    params=(P_DIMENSIONS, P_STUDENTS,
            ParamSpec("sort_by", "学生排序", "combo",
                      (("student", "按学生 ID"), ("first_code", "按起始状态"),
                       ("code_frequency", "按主要状态累计时长")), default="student"),
            P_LEGEND),
    requires_time=True,
    student_hint="学生数 <3 时索引图只能作个体展示")

# ------------------------------------------------------------------ M12–M14

M12 = MethodSpec(
    id="M12", name="Sequence Frequency 序列频次", group=GROUP_SEQUENCE,
    question="最常见的完整序列（或压缩后的相位序列）是哪几条？",
    compute=stats.sequence_frequency, render=plots.plot_sequence_frequency,
    params=(ParamSpec("dimension", "固定维度", "combo", default=None), P_STUDENTS,
            ParamSpec("top_n", "Top N（0 表示全部）", "spin", default=10, minimum=0, maximum=200),
            ParamSpec("compress", "序列压缩", "combo",
                      (("none", "不压缩（完整序列）"),
                       ("bin", "按时间窗口取相位众数")), default="none"),
            ParamSpec("compress_bins", "相位窗口数", "spin", default=6, minimum=2, maximum=24,
                      shows_for=("compress", "bin"))),
    units=("students", "dimension"), requires_time=True,
    student_hint="学生数 <3 时序列类型几乎必然唯一，请配合压缩使用")

M13 = MethodSpec(
    id="M13", name="Mean Time 平均状态占用时间", group=GROUP_SEQUENCE,
    question="各状态的人均累计占用时长是多少？个体差异有多大？",
    compute=stats.mean_time, render=plots.plot_mean_time,
    params=(P_STUDENTS, P_DIMENSIONS,
            ParamSpec("error_bar", "误差条", "combo",
                      (("sd", "SD"), ("se", "SE")), default="sd",
                      tooltip="仅作描述性展示，不做 t 检验 / ANOVA"), P_VALUES),
    requires_time=True,
    student_hint="学生数 <3 时均值与标准差不具群体稳定性")

M14 = MethodSpec(
    id="M14", name="Spell / Run 连续段", group=GROUP_SEQUENCE,
    question="每次连续进入某一状态能维持多久？",
    compute=stats.spells, render=plots.plot_spells,
    params=(P_STUDENTS, P_DIMENSIONS,
            ParamSpec("plot_style", "分布图形", "combo",
                      (("box", "箱线图"), ("violin", "小提琴图"), ("dot", "点分布")),
                      default="box")),
    requires_time=True)

# ------------------------------------------------------------------ M15

M15 = MethodSpec(
    id="M15", name="Co-occurrence 共现 / 同步", group=GROUP_COOC,
    question="两个编码在同一时间点上同时出现的频率有多高？",
    compute=stats.cooccurrence, render=plots.plot_cooccurrence,
    params=(ParamSpec("scope", "共现口径（必须先确认）", "segmented",
                      (("within", "同一学生跨维度共现"),
                       ("cross", "跨学生同维同步比较")), default=None,
                      tooltip="两者统计含义完全不同，未选择时不产出结果"),
            ParamSpec("student", "学生", "combo", default=None, shows_for=("scope", "within")),
            ParamSpec("dim_a", "维度 A（行）", "combo", default=None, shows_for=("scope", "within")),
            ParamSpec("dim_b", "维度 B（列）", "combo", default=None, shows_for=("scope", "within")),
            ParamSpec("student_a", "学生 A（行）", "combo", default=None,
                      shows_for=("scope", "cross")),
            ParamSpec("student_b", "学生 B（列）", "combo", default=None,
                      shows_for=("scope", "cross")),
            ParamSpec("dimension", "固定维度", "combo", default=None,
                      shows_for=("scope", "cross")),
            ParamSpec("normalize", "归一化", "combo",
                      (("none", "原始计数"), ("row", "按行 %"), ("col", "按列 %"),
                       ("total", "总比例 %")), default="none"), P_VALUES),
    units=("pair",), requires_time=True, min_students=1)

# ------------------------------------------------------------------ M16

M16 = MethodSpec(
    id="M16", name="Alluvial / Sankey 阶段路径", group=GROUP_PATH,
    question="学生在各阶段之间如何流转（阶段性路径，不是转移推断）？",
    compute=stats.alluvial, render=plots.plot_alluvial,
    params=(ParamSpec("dimension", "固定维度", "combo", default=None), P_STUDENTS,
            ParamSpec("phase_count", "阶段数（上限 6）", "spin", default=4, minimum=2, maximum=6),
            ParamSpec("min_flow", "最小流量（学生数）", "spin", default=1, minimum=1, maximum=50),
            P_LEGEND),
    units=("students", "dimension"), requires_time=True,
    student_hint="学生数过少时流带宽度差异不明显，仅示意阶段构成")

# ------------------------------------------------------------------ M17–M19（熵族）

M17 = MethodSpec(
    id="M17", name="Entropy 状态熵", group=GROUP_ENTROPY,
    question="编码分布有多分散？不同学生／维度的多样性怎么比？",
    compute=stats.state_entropy, render=plots.plot_state_entropy,
    params=(P_STUDENTS, P_DIMENSIONS,
            ParamSpec("value", "条高指标", "combo",
                      (("entropy", "熵 H（ln）"), ("H_norm", "归一熵 H/ln K")),
                      default="entropy",
                      tooltip="两者都是逐单元指标：熵受状态数影响，归一熵只看均匀程度"),
            P_VALUES),
    needs_dimension=True,
    student_hint="熵是逐单元描述量，学生数 <3 时只能当个体画像看")

M18 = MethodSpec(
    id="M18", name="Switch & Retention 切换率/保持率", group=GROUP_ENTROPY,
    question="序列在相邻有效观测之间到底切了多少次？平均一次状态能维持多久？",
    compute=stats.switch_retention, render=plots.plot_switch_retention,
    params=(P_STUDENTS, P_DIMENSIONS, P_VALUES),
    requires_time=True,
    student_hint="逐单元比率，不按学生平均；学生少时只能看个体差异")

M19 = MethodSpec(
    id="M19", name="Transition Entropy 条件熵", group=GROUP_ENTROPY,
    question="知道上一状态后，下一状态还剩多少不确定性（转移可预测性）？",
    compute=stats.transition_entropy, render=plots.plot_transition_entropy,
    params=(P_STUDENTS, P_DIMENSIONS,
            ParamSpec("view", "图形视图", "combo",
                      (("compare", "H(Y) vs H(Y|X) 成对条"),
                       ("decompose", "逐前态贡献分解"),
                       ("matrix", "条件转移概率热图")), default="compare",
                      tooltip="三者同源于一张转移矩阵：对比看单元差异 / 分解看谁贡献大 / 热图看具体去向"),
            P_VALUES),
    requires_time=True,
    student_hint="条件熵至少需要两种状态与若干转移对，不满足时该单元直接跳过")

METHODS: tuple = (M01, M02, M03, M04, M05, M06, M07, M08, M09, M10, M11,
                  M12, M13, M14, M15, M16, M17, M18, M19)

METHOD_BY_ID: dict = {spec.id: spec for spec in METHODS}

GROUP_ORDER: tuple = (GROUP_BASIC, GROUP_TIME, GROUP_TRACE, GROUP_SEQUENCE,
                      GROUP_COOC, GROUP_PATH, GROUP_ENTROPY)

# 各方法的默认参数：UI 首次选中时用它初始化，之后按 method_id 缓存用户改动
DEFAULT_PARAMS: dict = {}
for _spec in METHODS:
    _defaults = {}
    for _param in _spec.params:
        if _param.default is not None:
            _defaults[_param.key] = _param.default
    _defaults.setdefault("sort", "count")
    DEFAULT_PARAMS[_spec.id] = _defaults


def method_spec(method_id):
    return METHOD_BY_ID.get(method_id)


def run(ds, spec, params):
    """执行统计并返回 AnalysisResult；绘图由 UI 另外调 spec.render。"""
    return spec.compute(ds, dict(params))
