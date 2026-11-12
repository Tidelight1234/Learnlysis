# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""框架草稿的确定性质检；不依赖模型，离线也能运行。"""

from __future__ import annotations

import re

from framework_schema import active_version

INFERENCE_WORDS = re.compile(r"理解|相信|意识到|感到|认为|动机|焦虑|投入|掌握|内化")


def item(level, code, message, path=""):
    return {"level": level, "code": code, "message": message, "path": path}


def inspect(project, known_reference_ids=()):
    issues = []
    brief = (project or {}).get("research_brief") or {}
    if not str(brief.get("research_question") or "").strip():
        issues.append(item("block", "RQ_MISSING", "尚未填写研究问题。", "research_brief"))
    if not str(brief.get("task_context") or "").strip():
        issues.append(item("warn", "CONTEXT_MISSING", "尚未说明协作任务或研究情境。", "research_brief"))
    if not str(brief.get("unit") or "").strip():
        issues.append(item("block", "UNIT_MISSING", "尚未定义分析/切分单位。", "research_brief"))

    version = active_version(project)
    if not version:
        issues.append(item("block", "DRAFT_MISSING", "尚未生成或保存框架草稿。", "framework"))
        return issues

    dim_ids, code_ids = set(), set()
    for dpos, dim in enumerate(version.get("dimensions") or []):
        dpath = "dimensions[%d]" % dpos
        did = dim.get("dimension_id")
        if did in dim_ids:
            issues.append(item("block", "DIM_ID_DUP", "维度 ID 重复：%s" % did, dpath))
        dim_ids.add(did)
        codes = dim.get("codes") or []
        if dim.get("kind") == "state" and not codes:
            issues.append(item("block", "STATE_EMPTY", "状态维度“%s”没有编码。" % dim.get("name"), dpath))
        if len(codes) > 12:
            issues.append(item("warn", "TOO_MANY_CODES", "维度“%s”包含 %d 个编码，编码负担可能过高。" % (dim.get("name"), len(codes)), dpath))
        for cpos, code in enumerate(codes):
            cpath = "%s.codes[%d]" % (dpath, cpos)
            cid = "%s/%s" % (did, code.get("code_id"))
            if cid in code_ids:
                issues.append(item("block", "CODE_ID_DUP", "同一维度内编码 ID 重复：%s" % code.get("code_id"), cpath))
            code_ids.add(cid)
            label = str(code.get("label") or "").strip()
            definition = str(code.get("definition") or "").strip()
            if not label:
                issues.append(item("block", "LABEL_MISSING", "存在没有名称的编码。", cpath))
            if not definition:
                issues.append(item("block", "DEFINITION_MISSING", "编码“%s”缺少操作定义。" % label, cpath))
            if not code.get("observable_evidence"):
                issues.append(item("block", "EVIDENCE_MISSING", "编码“%s”缺少可观察证据。" % label, cpath))
            if definition and INFERENCE_WORDS.search(definition) and not code.get("include_when"):
                issues.append(item("warn", "INFERENCE_RISK", "编码“%s”包含心理推断词，但没有纳入判据。" % label, cpath))
            if not code.get("exclude_when"):
                issues.append(item("warn", "EXCLUSION_MISSING", "编码“%s”没有排除规则，边界可能含混。" % label, cpath))
            if code.get("decision") == "proposed":
                issues.append(item("block", "DECISION_PENDING", "编码“%s”仍待研究者确认。" % label, cpath))
            for source in code.get("source_links") or []:
                ref = source.get("reference_id")
                if ref and known_reference_ids and ref not in known_reference_ids:
                    issues.append(item("block", "REFERENCE_UNKNOWN", "编码“%s”引用了未知文献 ID：%s" % (label, ref), cpath))
    if len(version.get("dimensions") or []) > 8:
        issues.append(item("warn", "TOO_MANY_DIMS", "框架超过 8 个维度，建议先以核心研究问题收缩。", "framework"))
    return issues


def summary(issues):
    blocks = sum(value.get("level") == "block" for value in issues or [])
    warns = sum(value.get("level") == "warn" for value in issues or [])
    return {"blocks": blocks, "warnings": warns, "passed": blocks == 0}
