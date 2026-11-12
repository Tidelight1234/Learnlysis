"""合成测试数据生成器（`_` 前缀 = 非产品代码，只服务于本地验证）。

固定种子产出 4 组 × 3 学生 × 240 时间步（Δt=10 s）的三维类别序列，
并内嵌可验证结构，使阶段 3+ 的每个方法都有"已知答案"可对照：

- **三段式转移**：S1 严格按 A→B→C 循环，每段 20 步 → LSA/Markov 的一阶充分性必然通过；
- **两名高同步成员**：S2 复制 S1，仅在第 3 维错开 2 步 → 同步度 / lead-lag 有正解；
- **一名跟随型角色**：S3 随机游走，且总在第 5 维跟随 S1 → 角色聚类应把它单独分出来；
- **事件流**：带 ``reply_to`` 链、PERSON/TASK/QUESTION 混合、同窗多事件，
  并故意留 V05（缺 interaction_type）与 V08（confidence 越界）违例样本。
- **组结果**：Group_Info 里写 ``outcome``（两类交替），监督型方法（IND12 outcome 路径 /
  GRP10 组预测 / IND15 GAT）才有真值可拟合，老格式文件故意不给——缺值路径也要跑。

用法：``python _gen_testdata.py``，产物写 ``_testdata/``（已在 .gitignore）。
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import interaction_model as im

SEED = 20240517
GROUPS = 4
STUDENTS = 3
STEPS = 240
INTERVAL = 10.0
PHASE = 20                                   # 三段式每段长度（步）
DIM_NAMES = ["认知投入", "互动类型", "任务阶段"]
DIM_CODES = [("A", "B", "C"),               # 主维：三段式转移的载体
             ("L", "H"),                    # 参与 / 游离
             ("计划", "执行", "汇报")]        # 任务阶段
#: 组结果（Group_Info.outcome）：两类交替，保证“带 outcome 的组 ≥ 3 且 ≥ 2 类”，
#: IND12 的 outcome 路径与 GRP10 / IND15 的监督分支以此处为唯一来源。
OUTCOMES = ("高绩效", "低绩效", "高绩效", "低绩效")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_testdata")


# ------------------------------------------------------------------ 序列构造

def _rng():
    import numpy as np
    return np.random.default_rng(SEED)


def phase_of(step):
    """三段式骨架：0..PHASE-1 → A，PHASE..2*PHASE-1 → B，其余 → C，按 3*PHASE 循环。"""
    cycle = step % (3 * PHASE)
    return "ABC"[cycle // PHASE]


def state_rows(rng, group_id, students):
    """造 `(student_id, time, dimension, code)` 记录 + 宽表。

    结构是刻意留下的"答案"：S1 全维规律、S2 与 S1 高同步（第 3 维滞后 2 步）、
    S3 随机且滞后跟随 S1（第 5 个时间点起跟），跨组绝不共用行，方便断言边界。
    """
    records = {}
    for index, student in enumerate(students):
        for step in range(STEPS):
            moment = step * INTERVAL
            lagged = max(step - 2, 0) if index == 1 else step          # S2：滞后 2 步
            follower = index == 2 and step >= 5                        # S3：滞后跟随
            for dim, codes in enumerate(DIM_CODES, start=1):
                if dim == 1:
                    code = phase_of(lagged)
                elif dim == 2:
                    if follower and index == 2:
                        code = "H" if rng.random() < 0.3 else "L"
                    else:
                        # 高投入与 C 阶段相关：让"主导态"有可读结论
                        code = "H" if phase_of(lagged) == "C" or rng.random() < 0.35 else "L"
                else:
                    if follower:
                        code = records.get((students[0], moment, 3), ("计划",))[0]
                    else:
                        code = DIM_CODES[2][min(lagged // 80, 2)]
                records[(student, moment, dim)] = code
    return records


def to_wide(records, students):
    """(student, time, dim) → code 的记录字典 → `Time + S{i}Encoding_{j}` 宽表。"""
    times = sorted({moment for _, moment, _ in records})
    columns = ["Time"] + ["%sEncoding_%d" % (student, dim)
                          for student in students for dim in range(1, len(DIM_CODES) + 1)]
    rows = []
    for moment in times:
        row = {"Time": moment}
        for student in students:
            for dim in range(1, len(DIM_CODES) + 1):
                row["%sEncoding_%d" % (student, dim)] = records.get((student, moment, dim), "")
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


# ------------------------------------------------------------------ 事件流构造

def event_rows(group_id, students, rng):
    """带 ``reply_to`` 链的事件流：同窗多事件、混合 target_type、含 V05/V08 违例。

    违例是故意留的 —— 质检卡必须能抓到它们，否则校验器等于没写。
    """
    rows = []
    number = 1
    turns = {}
    for step in range(0, STEPS, 6):
        moment = step * INTERVAL
        actor = students[step % len(students)]
        reply = ""
        # 每 4 条挂一条应答链：reply_to 指向本组更早的一条 event_id
        if number > 1 and step % 24 == 0:
            reply = rows[-1]["event_id"]
            turn_key = reply
        else:
            turn_key = "%s_T%d" % (group_id, number)
        turns.setdefault(turn_key, len(turns) + 1)
        target_type = ["PERSON", "TASK", "QUESTION"][step % 3]
        target_id = (students[(step // len(students) + 1) % len(students)]
                     if target_type == "PERSON"
                     else "%s%d" % ({"TASK": "T", "QUESTION": "Q"}[target_type], step % 5))
        rows.append(im.blank_event(
            group_id,
            event_id=im.new_event_id(group_id, number),
            timestamp=moment,
            actor_id=actor,
            target_id=target_id,
            target_type=target_type,
            interaction_type=im.INTERACTION_TYPES[step % len(im.INTERACTION_TYPES)],
            reply_to=reply,
            turn_id="%s_TN%d" % (group_id, turns[turn_key]),
            duration=float(rng.integers(1, 6)),
            source="CODING",
            confidence=round(float(rng.uniform(0.6, 1.0)), 2),
            notes="合成话语 %d" % number,
        ))
        number += 1
        # 同窗多事件（V09 不去重）：同一时刻再记一条不同对象的事件
        if step % 30 == 0:
            rows.append(im.blank_event(
                group_id,
                event_id=im.new_event_id(group_id, number),
                timestamp=moment,
                actor_id=students[(step // len(students)) % len(students)],
                target_id="M%d" % (step % 3),
                target_type="MATERIAL",
                interaction_type="EXPLAIN",
                reply_to=reply,
                turn_id="%s_TN%d" % (group_id, turns[turn_key]),
                duration=2.0,
                source="TRANSCRIPT",
                confidence=0.9,
                notes="同窗第二条",
            ))
            number += 1
    # V05 违例：缺 interaction_type（警告，不阻断）
    rows.append(im.blank_event(group_id, event_id=im.new_event_id(group_id, number),
                               timestamp=60.0, actor_id=students[0], target_id="T1",
                               target_type="TASK", interaction_type="", source="MANUAL",
                               confidence=0.8))
    number += 1
    # V08 违例：confidence 越界（警告，不阻断）
    rows.append(im.blank_event(group_id, event_id=im.new_event_id(group_id, number),
                               timestamp=120.0, actor_id=students[1], target_id="SELF",
                               target_type="SELF", interaction_type="SPEAK", source="MANUAL",
                               confidence=1.7))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 落盘

def write_group(path, group_id, members, wide, events=None, with_info=True, outcome=None):
    """写一个组文件；``with_info`` 决定是否带 Group_Info / Students / Codebook sheet。

    留一份"裸表"版本（只有 Sheet1）用于验证老格式兼容路径；``outcome`` 写进
    Group_Info，供 IND12 / GRP10 / IND15 当监督目标（老格式没这一列，正好测缺值路径）。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        wide.to_excel(writer, sheet_name="Behavior_State" if with_info else "Sheet1",
                      index=False)
        if events is not None:
            events.to_excel(writer, sheet_name="Interaction_Event", index=False)
        if with_info:
            pd.DataFrame([{
                "group_id": group_id, "group_name": group_id, "member_count": len(members),
                "members": ",".join(members), "observation_start": 0,
                "observation_end": (STEPS - 1) * INTERVAL, "source_file": os.path.basename(path),
                "outcome": outcome or "",
            }]).to_excel(writer, sheet_name="Group_Info", index=False)
            pd.DataFrame({
                "group_id": [group_id] * len(members),
                "student_id": members,
                "student_uid": ["%s__%s" % (group_id, member) for member in members],
                "role": ["规律型", "同步型", "跟随型"],
            }).to_excel(writer, sheet_name="Students", index=False)
            pd.DataFrame(
                [{"dimension": name, "code": code, "label": ""}
                 for name, codes in zip(DIM_NAMES, DIM_CODES) for code in codes],
                columns=["dimension", "code", "label"],
            ).to_excel(writer, sheet_name="Codebook", index=False)
    return path


def generate(out_dir=OUT_DIR):
    """产出 4 个完整组文件 + 1 个老格式（单 sheet、无事件）文件。"""
    rng = _rng()
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for g in range(1, GROUPS + 1):
        group_id = "group_%d" % g
        students = ["S%d" % s for s in range(1, STUDENTS + 1)]
        records = state_rows(rng, group_id, students)
        wide = to_wide(records, students)
        events = event_rows(group_id, students, rng) if g % 2 == 1 else None
        written.append(write_group(os.path.join(out_dir, "%s.xlsx" % group_id),
                                   group_id, students, wide, events,
                                   outcome=OUTCOMES[(g - 1) % len(OUTCOMES)]))
    # 老格式：只有 Sheet1、无 group 前缀 sheet，列名契约仍在
    students = ["S%d" % s for s in range(1, STUDENTS + 1)]
    wide = to_wide(state_rows(rng, "group_legacy", students), students)
    written.append(write_group(os.path.join(out_dir, "group_legacy.xlsx"),
                               "group_legacy", students, wide, None, with_info=False))
    return written


def main():
    for path in generate():
        print("wrote", path)
    print("OK ->", OUT_DIR)


if __name__ == "__main__":
    main()
