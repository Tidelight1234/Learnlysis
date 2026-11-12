# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""框架设计师的两段式 prompt 与结构化结果解析。"""

from __future__ import annotations

import json
import re

from ai_client import system_message, user_message
from framework_schema import clean_dimension

SYSTEM = """你是协作学习观察研究的编码框架设计顾问。你只能引用输入中列出的 reference_id，
不得发明文献、作者、DOI 或原文定义。明确区分文献概念、为本研究做的改编和 AI 新建议。
优先使用可观察证据，不把理解、动机、情绪等心理状态当作已经观测到的事实。
所有结果都是待研究者确认的草稿，不得声称已经验证。"""


def _payload(project, references, current_framework=None):
    return {
        "research_brief": project.get("research_brief") or {},
        "selected_references": [{key: value.get(key) for key in
                                 ("id", "title_zh", "year", "doi", "summary",
                                  "unit_guidance", "limitations")}
                                for value in references or []],
        "current_framework": current_framework or [],
    }


def diagnosis_messages(project, references, current_framework=None):
    task = """请进行设计诊断，不要生成完整码本。输出中文，依次包含：
1. 研究问题是否可由当前证据回答；2. 仍缺的关键信息；3. 每个候选框架的适用与不适用理由；
4. 建议的状态/事件分流与分析单位；5. 主要效度、推断和可靠性风险。
输入如下：\n""" + json.dumps(_payload(project, references, current_framework),
                                      ensure_ascii=False, indent=2)
    return [system_message(SYSTEM), user_message(task)]


def draft_messages(project, references, current_framework=None):
    allowed = [value.get("id") for value in references or []]
    task = """基于输入生成编码框架草稿。只返回一个 JSON 对象，不要 Markdown 围栏和解释文字。
顶层格式：{"dimensions":[...] }。
每个 dimension 必须含 name、kind(state/event/annotation)、level(individual/dyad/group)、
multi_label、unit、codes。每个 code 必须含 label、definition、include_when、exclude_when、
positive_examples、negative_examples、observable_evidence、uncertainty_rule、source_links、decision。
decision 一律 proposed。source_links 中 relation 只能是 adopted/renamed/adapted/merged/split/
new_by_researcher/new_by_ai/removed；引用文献时 reference_id 只能取白名单 %s；AI 自创内容用
reference_id 空串和 relation=new_by_ai。定义必须能由视频、话语、操作或作品证据判断。
输入如下：\n%s""" % (allowed, json.dumps(_payload(project, references, current_framework),
                                                ensure_ascii=False, indent=2))
    return [system_message(SYSTEM), user_message(task)]


def repair_messages(raw_text, allowed_reference_ids=()):
    task = """下面是一份未通过本地 JSON 解析的框架草稿。只修复 JSON 格式和缺失的必需键，
不要增加新文献或改变研究含义；只返回一个含 dimensions 数组的 JSON 对象。
reference_id 白名单：%s\n\n原始输出：\n%s""" % (
        list(allowed_reference_ids or []), str(raw_text or "")[:16000])
    return [system_message(SYSTEM), user_message(task)]


def extract_json(text):
    value = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", value, re.S | re.I)
    if fenced:
        value = fenced.group(1)
    else:
        start, end = value.find("{"), value.rfind("}")
        if start >= 0 and end > start:
            value = value[start:end + 1]
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("dimensions"), list):
        raise ValueError("模型返回的 JSON 缺少 dimensions 数组")
    return parsed


def parse_draft(text, allowed_reference_ids=()):
    parsed = extract_json(text)
    dimensions = [clean_dimension(value, index + 1)
                  for index, value in enumerate(parsed["dimensions"])]
    allowed = set(allowed_reference_ids or [])
    for dimension in dimensions:
        for code in dimension["codes"]:
            for source in code["source_links"]:
                reference_id = source.get("reference_id")
                if reference_id and reference_id not in allowed:
                    raise ValueError("模型引用了未提供的文献 ID：%s" % reference_id)
    return dimensions
