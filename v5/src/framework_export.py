# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""框架项目的 Markdown / JSON / Excel / BibTeX 导出。"""

from __future__ import annotations

import json
from copy import copy

from framework_schema import active_version, project_to_state_framework


def markdown_text(project, references, issues=()):
    brief = project.get("research_brief") or {}
    version = active_version(project) or {"dimensions": []}
    lines = ["# %s" % project.get("title", "编码框架设计"), "",
             "- 项目 ID：%s" % project.get("project_id", ""),
             "- 框架版本：%s" % version.get("version_id", "尚未生成"),
             "- 状态：%s" % project.get("status", "draft"), "",
             "## 研究简报", ""]
    labels = (("研究问题", "research_question"), ("主要目标", "primary_goal"),
              ("分析层级", "analysis_level"), ("参与者", "participants"),
              ("任务与情境", "task_context"), ("数据模态", "data_modalities"),
              ("分析单位", "unit"), ("时间规则", "temporal_rule"),
              ("理论关注", "theory_focus"), ("可靠性计划", "reliability_plan"),
              ("伦理与隐私", "ethics_constraints"))
    for label, key in labels:
        value = brief.get(key, "")
        if isinstance(value, list):
            value = "、".join(value)
        lines.append("- %s：%s" % (label, value or "（未填写）"))
    lines += ["", "## 文献依据", ""]
    for ref in references or []:
        lines.append("- %s（%s）. %s. DOI: %s" % (
            "、".join(ref.get("authors") or []), ref.get("year", ""),
            ref.get("title_zh", ""), ref.get("doi", "")))
    lines += ["", "## 编码框架", ""]
    for dim in version.get("dimensions") or []:
        lines += ["### %s" % dim.get("name", "维度"), "",
                  "类型：%s；层级：%s；单位：%s" % (
                      dim.get("kind"), dim.get("level"), dim.get("unit")), ""]
        for code in dim.get("codes") or []:
            lines += ["#### %s [%s]" % (code.get("label", "编码"), code.get("decision")), "",
                      code.get("definition") or "（无定义）", "",
                      "- 纳入：%s" % ("；".join(code.get("include_when") or []) or "（待补）"),
                      "- 排除：%s" % ("；".join(code.get("exclude_when") or []) or "（待补）"),
                      "- 可观察证据：%s" % ("；".join(code.get("observable_evidence") or []) or "（待补）"),
                      "- 不确定规则：%s" % (code.get("uncertainty_rule") or "（待补）")]
            sources = ["%s/%s" % (source.get("reference_id") or "AI/研究者新增",
                                    source.get("relation")) for source in code.get("source_links") or []]
            lines += ["- 来源谱系：%s" % ("；".join(sources) or "（待补）"), ""]
    lines += ["## 质检", ""]
    if issues:
        for issue in issues:
            lines.append("- [%s] %s" % (issue.get("level", "info").upper(), issue.get("message", "")))
    else:
        lines.append("- 未发现阻断或警告。")
    lines += ["", "> 本文档记录的是研究者确认前后的设计依据，不等同于已经完成信效度验证。", ""]
    return "\n".join(lines)


def write_json(path, project):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(project, handle, ensure_ascii=False, indent=2)


def write_markdown(path, project, references, issues=()):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(markdown_text(project, references, issues))


def write_bibtex(path, references):
    blocks = []
    for ref in references or []:
        key = "%s%s" % ((ref.get("authors") or ["source"])[0].split()[-1], ref.get("year", ""))
        blocks.append("@article{%s,\n  author = {%s},\n  title = {%s},\n  year = {%s},\n  doi = {%s},\n  url = {%s}\n}" % (
            key, " and ".join(ref.get("authors") or []), ref.get("title_original") or ref.get("title_zh"),
            ref.get("year", ""), ref.get("doi", ""), ref.get("source_url", "")))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(blocks) + ("\n" if blocks else ""))


def write_excel(path, project, references):
    from openpyxl import Workbook
    wb = Workbook()
    brief_ws = wb.active
    brief_ws.title = "Research_Brief"
    brief_ws.append(["字段", "内容"])
    for key, value in (project.get("research_brief") or {}).items():
        brief_ws.append([key, "；".join(value) if isinstance(value, list) else value])
    dim_ws = wb.create_sheet("Dimensions")
    dim_ws.append(["dimension_id", "name", "kind", "level", "multi_label", "unit"])
    code_ws = wb.create_sheet("Codes")
    code_ws.append(["dimension_id", "code_id", "label", "definition", "include_when",
                    "exclude_when", "observable_evidence", "uncertainty_rule", "decision", "sources"])
    version = active_version(project) or {"dimensions": []}
    for dim in version.get("dimensions") or []:
        dim_ws.append([dim.get(key, "") for key in
                       ("dimension_id", "name", "kind", "level", "multi_label", "unit")])
        for code in dim.get("codes") or []:
            sources = "; ".join("%s:%s" % (value.get("reference_id", ""), value.get("relation", ""))
                                for value in code.get("source_links") or [])
            code_ws.append([dim.get("dimension_id"), code.get("code_id"), code.get("label"),
                            code.get("definition"), "；".join(code.get("include_when") or []),
                            "；".join(code.get("exclude_when") or []),
                            "；".join(code.get("observable_evidence") or []),
                            code.get("uncertainty_rule"), code.get("decision"), sources])
    ref_ws = wb.create_sheet("Sources")
    ref_ws.append(["id", "authors", "year", "title", "doi", "url"])
    for ref in references or []:
        ref_ws.append([ref.get("id"), "; ".join(ref.get("authors") or []), ref.get("year"),
                       ref.get("title_original"), ref.get("doi"), ref.get("source_url")])
    light_ws = wb.create_sheet("Learnlysis_Import")
    light_ws.append(["维度", "类型"])
    for name, codes in project_to_state_framework(project):
        for code in codes:
            light_ws.append([name, code])
    for sheet in wb.worksheets:
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            font = copy(cell.font)
            font.bold = True
            cell.font = font
        for column in sheet.columns:
            letter = column[0].column_letter
            sheet.column_dimensions[letter].width = min(max(len(str(cell.value or "")) for cell in column) + 2, 48)
    wb.save(path)
