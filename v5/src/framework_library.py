# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""人工整理的协作学习框架参考卡；只含转述摘要与可核查元数据。"""

from __future__ import annotations

import copy

CARDS = (
    {
        "id": "icap_2014", "title_zh": "ICAP 认知投入框架",
        "title_original": "The ICAP Framework", "authors": ["Michelene T. H. Chi", "Ruth Wylie"],
        "year": 2014, "doi": "10.1080/00461520.2014.965823",
        "source_url": "https://doi.org/10.1080/00461520.2014.965823",
        "focus": ["认知投入"], "modalities": ["视频", "观察", "学习产出"],
        "summary": "以可观察学习行为区分被动、主动、建构和交互投入。",
        "unit_guidance": "按行为事件、学习产出或研究者定义的固定时窗编码。",
        "strengths": ["分类直观", "可连接学习活动与投入方式"],
        "limitations": ["不是协作过程的完整框架", "互动行为需要共同生成而非仅仅交谈"],
        "starter_dimensions": [
            {"name": "认知投入方式", "kind": "state", "level": "individual", "codes": [
                ("被动投入", "以注意和接收为主，缺少可见的操作或生成。"),
                ("主动投入", "对材料进行可观察的操作，但没有生成超出给定信息的新内容。"),
                ("建构投入", "生成超出给定材料的新推论、解释、联系或产出。"),
                ("交互投入", "成员围绕彼此贡献共同生成、修正或推进理解。")]}],
    },
    {
        "id": "akcf_2006", "title_zh": "论证性知识建构框架",
        "title_original": "A Framework to Analyze Argumentative Knowledge Construction in CSCL",
        "authors": ["Armin Weinberger", "Frank Fischer"], "year": 2006,
        "doi": "10.1016/j.compedu.2005.04.003",
        "source_url": "https://doi.org/10.1016/j.compedu.2005.04.003",
        "focus": ["知识建构", "论证"], "modalities": ["话语", "转写文本", "在线讨论"],
        "summary": "从参与、认识活动、论证结构和社会共建等维度分析协作论证。",
        "unit_guidance": "通常使用命题或语义单元；领域知识维度需按任务重做。",
        "strengths": ["多维解释协作论证过程"],
        "limitations": ["不能压成单一高低等级", "领域认识维度不可直接跨任务照搬"],
        "starter_dimensions": [
            {"name": "论证功能", "kind": "event", "level": "individual", "codes": [
                ("主张", "提出需要支持或讨论的判断。"), ("证据", "为主张提供事实、数据或案例依据。"),
                ("论证理由", "说明证据为何能够支持主张。"), ("反驳或修正", "针对既有主张提出反例、限制或修改。")]},
            {"name": "社会共建方式", "kind": "event", "level": "dyad", "codes": [
                ("外化观点", "表达自己的观点但未直接处理同伴推理。"),
                ("引出观点", "主动邀请同伴给出观点或理由。"),
                ("快速一致", "表示同意但没有展开理由或整合。"),
                ("冲突导向共识", "通过质疑、修正或整合差异形成共识。")]}],
    },
    {
        "id": "iam_1997", "title_zh": "知识社会建构交互分析模型（IAM）",
        "title_original": "Interaction Analysis Model", "authors": ["Charlotte N. Gunawardena", "Constance A. Lowe", "Terry Anderson"],
        "year": 1997, "doi": "10.2190/7MQV-X9UJ-C7Q3-NRAG",
        "source_url": "https://doi.org/10.2190/7MQV-X9UJ-C7Q3-NRAG",
        "focus": ["知识建构", "协商"], "modalities": ["在线讨论", "转写文本"],
        "summary": "关注讨论从信息分享、分歧、意义协商、检验到应用的知识共建过程。",
        "unit_guidance": "消息或意义单元；阶段不应被假设为每次讨论都线性经历。",
        "strengths": ["适合观察知识协商的推进"],
        "limitations": ["源自异步计算机会议", "迁移到面对面场景需要重做切分规则"],
        "starter_dimensions": [{"name": "知识共建阶段", "kind": "event", "level": "group", "codes": [
            ("分享与比较", "分享观察、意见或已有信息并进行比较。"),
            ("发现分歧", "明确识别观点、概念或证据之间的不一致。"),
            ("协商与共建", "澄清术语、权衡理由并共同建构意义。"),
            ("检验与修订", "用证据、经验或逻辑检验并修改共同建构。"),
            ("达成与应用", "形成可陈述的共识并将其应用或反思。")]}],
    },
    {
        "id": "coi_2001", "title_zh": "探究共同体（Community of Inquiry）",
        "title_original": "Community of Inquiry", "authors": ["D. Randy Garrison", "Terry Anderson", "Walter Archer"],
        "year": 2001, "doi": "10.1080/08923640109527071",
        "source_url": "https://doi.org/10.1080/08923640109527071",
        "focus": ["认知临场", "社会临场", "教学临场"], "modalities": ["在线讨论", "转写文本"],
        "summary": "以认知、社会和教学临场感理解在线学习共同体中的探究过程。",
        "unit_guidance": "消息或意义单元；短时面对面任务需谨慎使用。",
        "strengths": ["兼顾认知、社会和教学支持"],
        "limitations": ["原场景偏课程型在线讨论", "不等同于所有面对面协作"],
        "starter_dimensions": [{"name": "认知临场阶段", "kind": "event", "level": "individual", "codes": [
            ("触发", "识别问题、困惑或需要探究的事件。"), ("探索", "交换信息、发散想法或搜索解释。"),
            ("整合", "连接观点并形成较连贯的解释或方案。"), ("解决", "检验、应用或评价形成的解释与方案。")]}],
    },
    {
        "id": "ssrl_family", "title_zh": "社会共享调节学习（SSRL）理论族",
        "title_original": "Socially Shared Regulation of Learning", "authors": ["Sanna Järvelä", "Allyson F. Hadwin"],
        "year": 2013, "doi": "10.1080/00461520.2012.748006",
        "source_url": "https://doi.org/10.1080/00461520.2012.748006",
        "focus": ["共同调节", "元认知", "动机情绪"], "modalities": ["视频", "话语", "日志"],
        "summary": "区分自我调节、共调节与社会共享调节，并关注小组共同计划、监控和调整。",
        "unit_guidance": "以调节事件及其主体、对象和目标为单位，通常需要序列上下文。",
        "strengths": ["适合过程与时序研究", "能够区分调节的社会层级"],
        "limitations": ["是理论族而非唯一固定码本", "心理过程必须转成可观察证据"],
        "starter_dimensions": [
            {"name": "调节社会层级", "kind": "event", "level": "group", "codes": [
                ("自我调节外显", "成员明确调节自己的目标、策略或状态。"),
                ("共调节", "一名或部分成员引导、支持另一成员进行调节。"),
                ("社会共享调节", "多名成员共同协商并调节小组层面的目标、策略或状态。")]},
            {"name": "调节功能", "kind": "event", "level": "group", "codes": [
                ("计划", "共同设定目标、分工、策略或时间安排。"), ("监控", "检查理解、进度、参与或当前状态。"),
                ("调整", "基于监控改变策略、分工、努力或互动方式。"), ("评价", "评价过程、策略或结果并形成后续判断。")]}],
    },
    {
        "id": "oecd_cps_2015", "title_zh": "OECD PISA 协作问题解决框架",
        "title_original": "PISA 2015 Collaborative Problem Solving Framework", "authors": ["OECD"],
        "year": 2017, "doi": "10.1787/9789264281820-en",
        "source_url": "https://doi.org/10.1787/9789264281820-en",
        "focus": ["协作问题解决", "团队组织"], "modalities": ["操作日志", "对话", "任务行为"],
        "summary": "将建立共享理解、采取适当行动和组织团队，与问题解决过程相结合。",
        "unit_guidance": "任务中的交流或行动事件；真实同伴协作不能直接沿用标准化测评分数解释。",
        "strengths": ["协作能力与问题解决过程结合紧密"],
        "limitations": ["原设计含人机协作测评", "迁移到自然小组需重新验证"],
        "starter_dimensions": [{"name": "协作功能", "kind": "event", "level": "group", "codes": [
            ("建立共享理解", "发现视角与能力，并协商问题和共同目标。"),
            ("采取协同行动", "沟通行动、实施计划并遵守协作规则。"),
            ("组织与维持团队", "理解角色、管理组织并监控或修复团队过程。")]}],
    },
)


def all_cards():
    return copy.deepcopy(list(CARDS))


def card(reference_id):
    for value in CARDS:
        if value["id"] == str(reference_id):
            return copy.deepcopy(value)
    return None


def ids():
    return {value["id"] for value in CARDS}


def search(text="", focus=()):
    needle = str(text or "").strip().lower()
    wanted = {str(value) for value in (focus or [])}
    out = []
    for value in CARDS:
        haystack = " ".join([value["id"], value["title_zh"], value["title_original"],
                             value["summary"], " ".join(value["focus"])]).lower()
        if needle and needle not in haystack:
            continue
        if wanted and not wanted.intersection(value["focus"]):
            continue
        out.append(copy.deepcopy(value))
    return out


def starter_dimensions(reference_ids):
    dimensions = []
    used = set()
    for ref_id in reference_ids or []:
        value = card(ref_id)
        if not value:
            continue
        for source_dim in value.get("starter_dimensions") or []:
            name = source_dim["name"]
            key = (name, source_dim.get("kind"))
            if key in used:
                continue
            used.add(key)
            codes = []
            for label, definition in source_dim.get("codes") or []:
                codes.append({
                    "label": label, "definition": definition,
                    "include_when": [], "exclude_when": [],
                    "positive_examples": [], "negative_examples": [],
                    "observable_evidence": ["话语或可见行为"],
                    "uncertainty_rule": "证据不足时记为不确定，不强行归类。",
                    "source_links": [{"reference_id": ref_id, "relation": "adapted",
                                      "source_concept": label,
                                      "adaptation_note": "参考卡起始定义，需按当前研究情境进一步操作化。"}],
                    "decision": "proposed",
                })
            dimensions.append({"name": name, "kind": source_dim.get("kind", "state"),
                               "level": source_dim.get("level", "individual"),
                               "multi_label": source_dim.get("kind") == "event",
                               "unit": "语义单元", "codes": codes})
    return dimensions
