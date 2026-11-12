# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""把「当前工作现场」序列化成给 LLM 的上下文。

主窗口不把内部控件结构摊开给 AI 模块：它注入一个 ``context_provider`` 回调，回调返回
一份**会话快照**（普通 dict），本模块只认这份快照的键，不认任何控件类型。快照约定：

    {
        "summary_tab":   数据汇总页对象（有 build_analysis_data / table_dataframe）或 None,
        "group_encoder": 小组交互页对象（有 events_frame / collect_meta）或 None,
        "encoder_pages": [编码页, ...] 或 [(标题, 编码页), ...]（两种都收）,
        "group_id":      可选，当前组号（缺省 "group_1"）,
        "bundles":       可选，[GroupBundle, ...]（批量上传/工作台的现成组，用于报组数）,
        "result_text":   可选，用户当前正在看的某张结果表的纯文本,
    }

数据面复用 analysis_core 的 ``bundle_from_session()`` + ``resolve_codebook()``（拿到组数 /
学生数 / 维度 / 时间点 / 框架）；方法面把 statistic_visualization_methods 的 M01–M19 与
analysis_methods 的 IND/GRP ``MethodSpec`` 序列化成「可推荐方法清单」。智能体档案的
``context`` 开关（data / framework / methods / results）决定哪几段进上下文。
"""

from __future__ import annotations

import logging

import analysis_core

#: 结果表正文的长度上限：够放一张表，又不至于把 prompt 撑爆（BYO key 的成本考量）
_RESULT_CHAR_CAP = 4000


# ------------------------------------------------------------------ 快照规整

def _unwrap_pages(encoder_pages):
    """``encoder_pages`` 可能是 ``[页]`` 也可能是 ``[(标题, 页)]``；统一成 ``[页]``。

    主窗口的 ``encoder_pages()`` 返回 ``(标题, 编码页)`` 元组，而 analysis_core 的
    ``framework_from_pages`` 只认带 ``dim_panel`` 的页对象，这里把元组拆出页再交下去。
    """
    out = []
    for item in encoder_pages or []:
        if isinstance(item, (tuple, list)):
            page = item[-1] if item else None
            if page is not None:
                out.append(page)
        elif item is not None:
            out.append(item)
    return out


def build_bundle(snapshot):
    """从会话快照取一个 ``GroupBundle``（数据面）。取不到就返回 None，绝不抛给上层。"""
    snapshot = snapshot or {}
    pages = _unwrap_pages(snapshot.get("encoder_pages"))
    try:
        return analysis_core.bundle_from_session(
            group_id=snapshot.get("group_id") or "group_1",
            summary_tab=snapshot.get("summary_tab"),
            group_encoder=snapshot.get("group_encoder"),
            encoder_pages=pages or None,
        )
    except Exception as exc:                    # noqa: BLE001 取数失败不该打断对话
        logging.warning("AI 上下文取数失败：%s", exc)
        return None


def collect_facts(snapshot):
    """把快照算成一份派生事实：``{"bundle", "codebook", "bundles"}``。"""
    snapshot = snapshot or {}
    bundle = build_bundle(snapshot)
    codebook = []
    if bundle is not None:
        try:
            codebook = analysis_core.resolve_codebook(bundle)
        except Exception as exc:                # noqa: BLE001
            logging.warning("AI 上下文解析框架失败：%s", exc)
    bundles = [item for item in (snapshot.get("bundles") or []) if item is not None]
    if not bundles and bundle is not None and (bundle.members or bundle.n_times):
        bundles = [bundle]
    return {"bundle": bundle, "codebook": codebook, "bundles": bundles}


# ------------------------------------------------------------------ 各段文本

def data_section(facts):
    """数据概况：组数 / 学生数 / 维度数 / 时间点 / 事件表有无。"""
    bundle = facts.get("bundle")
    bundles = facts.get("bundles") or []
    codebook = facts.get("codebook") or []
    lines = ["## 数据概况"]
    if bundle is None and not bundles:
        lines.append("- 尚未装载可解析的编码数据（用户还没在编码页录入或上传文件）")
        return "\n".join(lines)
    if len(bundles) > 1:
        lines.append("- 组数：%d（批量上传 / 工作台现场）" % len(bundles))
    else:
        lines.append("- 组数：1（当前编码会话）")
    if bundle is not None:
        members = list(bundle.members or [])
        times = bundle.time_points()
        shown = "、".join(str(item) for item in members[:12])
        if len(members) > 12:
            shown += " …"
        lines.append("- 学生数：%d%s" % (len(members), "（%s）" % shown if shown else ""))
        lines.append("- 编码维度数：%d" % len(codebook))
        lines.append("- 时间点数：%d" % len(times))
        if times:
            lines.append("- 时间范围：%g ~ %g 秒" % (times[0], times[-1]))
        if bundle.has_events:
            lines.append("- 交互事件表：有（%d 条）" % len(bundle.events))
        else:
            lines.append("- 交互事件表：无")
    return "\n".join(lines)


def _codebook_notes():
    """取已导入的码本释义（全局设置）；不在场/无 Qt 脚本下就当没导入。"""
    try:
        import ai_settings
        return ai_settings.settings.codebook_notes() or {}
    except Exception:                            # noqa: BLE001
        return {}


def framework_section(facts):
    """编码框架：维度名 → 该维度下的编码词表（已导入码本释义时逐码附注）。"""
    codebook = facts.get("codebook") or []
    if not codebook:
        return "## 编码框架\n- 尚未解析出编码框架（维度 / 编码）"
    try:
        from auto_coding import format_code_tokens
    except Exception:                            # noqa: BLE001 无 auto_coding 时退化为裸码
        format_code_tokens = None
    notes = _codebook_notes()
    lines = ["## 编码框架（维度 → 编码）"]
    if notes:
        lines.append("（以下编码后括号内为其释义/示例，请据此理解每个码的适用边界）")
    for name, codes in codebook:
        if codes and format_code_tokens is not None:
            body = "、".join(format_code_tokens(codes, notes.get(str(name))))
        elif codes:
            body = "、".join(str(code) for code in codes)
        else:
            body = "（无编码）"
        lines.append("- %s：%s" % (name, body))
    return "\n".join(lines)


def methods_section():
    """可推荐方法清单：M01–M19 描述统计 + IND/GRP 进阶分析，只读注册表。"""
    lines = ["## 本软件可用的分析方法（推荐时请引用编号与名称）"]
    try:
        import statistic_visualization_methods as statviz
        buckets, order = {}, []
        for spec in statviz.METHODS:
            if spec.group not in buckets:
                buckets[spec.group] = []
                order.append(spec.group)
            buckets[spec.group].append(spec)
        ordered = [group for group in statviz.GROUP_ORDER if group in buckets]
        ordered += [group for group in order if group not in ordered]
        lines.append("### 描述统计与可视化")
        for group in ordered:
            lines.append("〔%s〕" % group)
            for spec in buckets[group]:
                lines.append("- %s %s：%s" % (spec.id, spec.name, spec.question))
    except Exception as exc:                    # noqa: BLE001 注册表没落地就跳过
        logging.warning("AI 上下文读取描述统计方法失败：%s", exc)
    try:
        import analysis_methods
        adv = analysis_methods.all_methods()
        if adv:
            lines.append("### 进阶分析（序列 / 网络 / 概率模型 / 预测 / 组间）")
            buckets, order = {}, []
            for spec in adv:
                if spec.group not in buckets:
                    buckets[spec.group] = []
                    order.append(spec.group)
                buckets[spec.group].append(spec)
            for group in order:
                lines.append("〔%s〕" % group)
                for spec in buckets[group]:
                    req = "（需 %s）" % "/".join(spec.requires) if spec.requires else ""
                    lines.append("- %s %s：%s%s" % (spec.id, spec.name, spec.question, req))
    except Exception as exc:                    # noqa: BLE001
        logging.warning("AI 上下文读取进阶分析方法失败：%s", exc)
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def results_section(snapshot):
    """用户当前正在看的某张结果表（可选）。"""
    text = str((snapshot or {}).get("result_text") or "").strip()
    if not text:
        return ""
    if len(text) > _RESULT_CHAR_CAP:
        text = text[:_RESULT_CHAR_CAP] + "\n…（结果过长已截断）"
    return "## 用户当前正在查看的结果\n%s" % text


# ------------------------------------------------------------------ 组装

def context_block(agent, snapshot):
    """按智能体档案的 ``context`` 开关拼出上下文文本；全关或无现场时返回空串。"""
    flags = (agent or {}).get("context") or {}
    need_data = bool(flags.get("data"))
    need_framework = bool(flags.get("framework"))
    need_methods = bool(flags.get("methods"))
    need_results = bool(flags.get("results"))
    if not (need_data or need_framework or need_methods or need_results):
        return ""
    facts = collect_facts(snapshot) if (need_data or need_framework) else \
        {"bundle": None, "codebook": [], "bundles": []}
    parts = []
    if need_data:
        parts.append(data_section(facts))
    if need_framework:
        parts.append(framework_section(facts))
    if need_methods:
        parts.append(methods_section())
    if need_results:
        parts.append(results_section(snapshot))
    return "\n\n".join(part for part in parts if part)


def build_messages(agent, user_text, snapshot=None, history=None):
    """拼出发给 LLM 的 messages：系统提示（角色 + 上下文）+ 历史 + 本轮用户消息。

    ``agent`` 是 ai_settings 里的一条档案（含 ``role`` / ``context``）；``history`` 是
    ``[{"role": "user"|"assistant", "content": str}, ...]``，只保留这两类角色。
    """
    from ai_client import system_message, user_message, assistant_message

    role = str((agent or {}).get("role") or "").strip()
    block = context_block(agent, snapshot)
    system_text = "\n\n".join(part for part in (role, block) if part)
    messages = []
    if system_text:
        messages.append(system_message(system_text))
    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        who = turn.get("role")
        content = turn.get("content")
        if who == "user" and content:
            messages.append(user_message(content))
        elif who == "assistant" and content:
            messages.append(assistant_message(content))
    messages.append(user_message(user_text))
    return messages
