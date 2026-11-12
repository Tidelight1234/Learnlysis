# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""AI 框架设计工作台的纯数据契约、清洗、版本与投影。"""

from __future__ import annotations

import copy
import re
import time
import uuid

SCHEMA_VERSION = 1
DIMENSION_KINDS = ("state", "event", "annotation")
LEVELS = ("individual", "dyad", "group")
DECISIONS = ("proposed", "accepted", "rejected")
SOURCE_RELATIONS = ("adopted", "renamed", "adapted", "merged", "split",
                    "new_by_researcher", "new_by_ai", "removed")


def _id(prefix):
    return "%s_%s" % (prefix, uuid.uuid4().hex[:12])


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def default_brief():
    return {
        "research_question": "", "primary_goal": "描述", "analysis_level": "小组",
        "participants": "", "task_context": "", "setting": "",
        "data_modalities": [], "observable_evidence": "", "unit": "语义单元",
        "temporal_rule": "", "coding_mode": "状态与事件并用",
        "theory_focus": [], "required_theory": "", "comparison": "",
        "out_of_scope": "", "uncertainty_rule": "",
        "reliability_plan": "", "ethics_constraints": "",
    }


def new_project(title="未命名框架项目"):
    stamp = now()
    return {
        "schema_version": SCHEMA_VERSION,
        "project_id": _id("fw"), "title": str(title or "未命名框架项目"),
        "status": "draft", "research_brief": default_brief(),
        "selected_references": [], "framework_versions": [],
        "active_version_id": "", "pilot_log": [],
        "created_at": stamp, "updated_at": stamp,
    }


def slug(text, fallback="item"):
    value = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fff]+", "_", str(text or "").strip())
    return value.strip("_")[:80] or fallback


def _strings(values):
    return [str(value).strip() for value in (values or []) if str(value).strip()]


def clean_source(raw):
    raw = raw if isinstance(raw, dict) else {}
    relation = str(raw.get("relation") or "new_by_researcher")
    if relation not in SOURCE_RELATIONS:
        relation = "new_by_researcher"
    return {"reference_id": str(raw.get("reference_id") or ""),
            "relation": relation,
            "source_concept": str(raw.get("source_concept") or ""),
            "adaptation_note": str(raw.get("adaptation_note") or "")}


def clean_code(raw, fallback_index=1):
    raw = raw if isinstance(raw, dict) else {}
    label = str(raw.get("label") or "编码%d" % fallback_index).strip()
    decision = str(raw.get("decision") or "proposed")
    if decision not in DECISIONS:
        decision = "proposed"
    return {
        "code_id": slug(raw.get("code_id") or label, "code_%d" % fallback_index),
        "label": label, "definition": str(raw.get("definition") or "").strip(),
        "include_when": _strings(raw.get("include_when")),
        "exclude_when": _strings(raw.get("exclude_when")),
        "positive_examples": _strings(raw.get("positive_examples")),
        "negative_examples": _strings(raw.get("negative_examples")),
        "observable_evidence": _strings(raw.get("observable_evidence")),
        "uncertainty_rule": str(raw.get("uncertainty_rule") or "").strip(),
        "source_links": [clean_source(value) for value in (raw.get("source_links") or [])],
        "decision": decision,
    }


def clean_dimension(raw, fallback_index=1):
    raw = raw if isinstance(raw, dict) else {}
    name = str(raw.get("name") or "维度%d" % fallback_index).strip()
    kind = str(raw.get("kind") or "state")
    level = str(raw.get("level") or "individual")
    return {
        "dimension_id": slug(raw.get("dimension_id") or name, "dim_%d" % fallback_index),
        "name": name, "kind": kind if kind in DIMENSION_KINDS else "state",
        "level": level if level in LEVELS else "individual",
        "multi_label": bool(raw.get("multi_label", False)),
        "unit": str(raw.get("unit") or "语义单元").strip(),
        "codes": [clean_code(value, i + 1)
                  for i, value in enumerate(raw.get("codes") or [])],
    }


def new_version(dimensions, label="框架草稿", reason="", parent_id="", created_by="researcher"):
    return {
        "version_id": _id("v"), "parent_version_id": str(parent_id or ""),
        "label": str(label or "框架草稿"), "change_reason": str(reason or ""),
        "created_by": str(created_by or "researcher"),
        "dimensions": [clean_dimension(value, i + 1)
                       for i, value in enumerate(dimensions or [])],
        "qa_report": {}, "created_at": now(),
    }


def active_version(project):
    wanted = str((project or {}).get("active_version_id") or "")
    versions = (project or {}).get("framework_versions") or []
    for version in versions:
        if version.get("version_id") == wanted:
            return version
    return versions[-1] if versions else None


def add_version(project, dimensions, label="框架草稿", reason="", created_by="researcher"):
    out = copy.deepcopy(project)
    parent = active_version(out)
    version = new_version(dimensions, label, reason,
                          parent.get("version_id") if parent else "", created_by)
    out.setdefault("framework_versions", []).append(version)
    out["active_version_id"] = version["version_id"]
    out["updated_at"] = now()
    return out, version


def clean_project(raw):
    raw = raw if isinstance(raw, dict) else {}
    project = new_project(raw.get("title") or "未命名框架项目")
    for key in ("project_id", "status", "created_at", "updated_at"):
        if raw.get(key):
            project[key] = str(raw[key])
    brief = default_brief()
    if isinstance(raw.get("research_brief"), dict):
        for key in brief:
            value = raw["research_brief"].get(key, brief[key])
            brief[key] = _strings(value) if isinstance(brief[key], list) else str(value or "")
    project["research_brief"] = brief
    project["selected_references"] = _strings(raw.get("selected_references"))
    versions = []
    for item in raw.get("framework_versions") or []:
        if not isinstance(item, dict):
            continue
        version = new_version(item.get("dimensions") or [], item.get("label"),
                              item.get("change_reason"), item.get("parent_version_id"),
                              item.get("created_by"))
        for key in ("version_id", "created_at"):
            if item.get(key):
                version[key] = str(item[key])
        version["qa_report"] = dict(item.get("qa_report") or {})
        versions.append(version)
    project["framework_versions"] = versions
    wanted = str(raw.get("active_version_id") or "")
    project["active_version_id"] = wanted if any(
        value["version_id"] == wanted for value in versions) else (
            versions[-1]["version_id"] if versions else "")
    project["pilot_log"] = list(raw.get("pilot_log") or [])
    return project


def project_to_state_framework(project, accepted_only=True):
    version = active_version(project)
    if not version:
        return []
    out = []
    for dimension in version.get("dimensions") or []:
        if dimension.get("kind") != "state":
            continue
        codes = []
        for code in dimension.get("codes") or []:
            if accepted_only and code.get("decision") != "accepted":
                continue
            label = str(code.get("label") or "").strip()
            if label:
                codes.append(label)
        if codes:
            out.append((str(dimension.get("name") or "维度"), codes))
    return out


def compare_versions(old, new):
    def flat(version):
        return {"%s/%s" % (dim.get("dimension_id"), code.get("code_id")): code
                for dim in (version or {}).get("dimensions") or []
                for code in dim.get("codes") or []}
    left, right = flat(old), flat(new)
    return {
        "added": sorted(set(right) - set(left)),
        "removed": sorted(set(left) - set(right)),
        "changed": sorted(key for key in set(left) & set(right) if left[key] != right[key]),
    }
