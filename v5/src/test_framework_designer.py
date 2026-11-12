# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.

import json

import framework_export
import framework_library
import framework_prompt
import framework_quality
from framework_schema import add_version, new_project, project_to_state_framework
from framework_store import FrameworkProjectStore


def _project_with_mixed_framework():
    project = new_project("测试项目")
    project["research_brief"].update({
        "research_question": "小组如何共同监控任务进度？",
        "task_context": "四人组科学探究",
        "unit": "语义单元",
    })
    dimensions = [
        {"name": "调节状态", "kind": "state", "level": "group", "codes": [{
            "label": "共同监控", "definition": "成员共同检查小组进度或理解状态。",
            "include_when": ["至少两名成员明确回应"],
            "exclude_when": ["仅个人自我检查"],
            "observable_evidence": ["话语", "指向共同产物的动作"],
            "uncertainty_rule": "共同对象不明时记为不确定。",
            "source_links": [{"reference_id": "ssrl_family", "relation": "adapted"}],
            "decision": "accepted",
        }]},
        {"name": "调节事件", "kind": "event", "level": "group", "codes": [{
            "label": "调整策略", "definition": "小组明确改变原有策略。",
            "include_when": ["出现改变及其共同对象"], "exclude_when": ["仅提出想法"],
            "observable_evidence": ["话语"], "uncertainty_rule": "不明时不强判。",
            "source_links": [{"reference_id": "ssrl_family", "relation": "adapted"}],
            "decision": "accepted",
        }]},
    ]
    project, _ = add_version(project, dimensions, "v1")
    project["selected_references"] = ["ssrl_family"]
    return project


def test_reference_library_has_verified_seed_metadata():
    cards = framework_library.all_cards()
    assert len(cards) >= 6
    assert len({card["id"] for card in cards}) == len(cards)
    assert all(card["doi"] and card["source_url"] for card in cards)
    assert framework_library.card("icap_2014")["year"] == 2014


def test_state_projection_never_merges_events():
    project = _project_with_mixed_framework()
    assert project_to_state_framework(project) == [("调节状态", ["共同监控"])]


def test_quality_passes_complete_confirmed_framework():
    project = _project_with_mixed_framework()
    issues = framework_quality.inspect(project, framework_library.ids())
    assert not [item for item in issues if item["level"] == "block"]


def test_quality_blocks_pending_decision_and_unknown_reference():
    project = _project_with_mixed_framework()
    code = project["framework_versions"][0]["dimensions"][0]["codes"][0]
    code["decision"] = "proposed"
    code["source_links"][0]["reference_id"] = "invented_paper"
    codes = {item["code"] for item in framework_quality.inspect(project, framework_library.ids())}
    assert {"DECISION_PENDING", "REFERENCE_UNKNOWN"}.issubset(codes)


def test_prompt_parser_rejects_hallucinated_reference():
    raw = json.dumps({"dimensions": [{"name": "D", "kind": "state", "codes": [{
        "label": "C", "definition": "可见行为", "observable_evidence": ["动作"],
        "source_links": [{"reference_id": "fake", "relation": "adopted"}]
    }]}]}, ensure_ascii=False)
    try:
        framework_prompt.parse_draft(raw, ["icap_2014"])
        assert False, "unknown reference should be rejected"
    except ValueError as exc:
        assert "未提供" in str(exc)


def test_store_round_trip_and_exports(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNLYSIS_AUTOSAVE", "1")
    project = _project_with_mixed_framework()
    path = tmp_path / "projects.json"
    project_store = FrameworkProjectStore(str(path))
    assert project_store.upsert(project)
    loaded = FrameworkProjectStore(str(path)).get(project["project_id"])
    assert loaded["title"] == "测试项目"

    refs = [framework_library.card("ssrl_family")]
    framework_export.write_json(str(tmp_path / "framework.json"), loaded)
    framework_export.write_markdown(str(tmp_path / "framework.md"), loaded, refs)
    framework_export.write_bibtex(str(tmp_path / "references.bib"), refs)
    framework_export.write_excel(str(tmp_path / "codebook.xlsx"), loaded, refs)
    assert (tmp_path / "codebook.xlsx").stat().st_size > 0
    assert "共同监控" in (tmp_path / "framework.md").read_text(encoding="utf-8")
