# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""AI 助手设置：data/ai_settings.json 的内存镜像 + 智能体档案。

与 theme.py 的 ui_settings.json、user_state.py 的 user_state.json 分开存：三份各写各的
整文件，混用会互相覆盖。本模块只装「AI 助手页的配置」——OpenAI 兼容端点、模型、
API key、采样参数，以及用户自建的智能体档案（角色 / 附带上下文 / 任务模板）。

安全口径（照 AGENTS.md 与本项目一贯风格）：
- api_key 落盘前先过 Windows DPAPI（见 secure_store.py）：密文带 `dpapi:` 前缀存进
  data/ai_settings.json，同机其他 Windows 账户 / 拷走的配置文件都读不出明文。
  非 Windows 或加密失败时退回明文（UI 警示照旧留着），不因此丢用户的 key。
  内存里始终是明文——ai_client 直接拿它发请求，解密只发生在读盘那一刻。
  勾选「本次不保存 key」时写盘前把 api_key 抹成空串，只留在内存里用一次。
- 升级前就存过明文 key 的老配置文件照常能读（不带前缀即按明文原样返回），
  不需要用户重填。
- api_key 绝不进 archive.json、任何导出、日志与关于弹窗；本文件的键名也不作为对外
  数据契约（见 AGENTS.md「data/ 不作为数据契约」）。
- offscreen（冒烟/渲染脚本）下默认不写盘：那类脚本会真的改设置，不挡一道就把工程里
  的 data/ai_settings.json 盖了。要在那种环境验写盘，置 LEARNLYSIS_AUTOSAVE=1。
"""

import os
import json
import time
import logging

import secure_store
from config import data_dir, AI_SETTINGS_FILE

#: 采样参数默认值：温度低一点，规划/编码建议要的是稳定而非发散
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1024
DEFAULT_ACTIVE_AGENT = "planner"

#: 智能体档案要附带的上下文维度（ai_context 按这几个开关取数）
CONTEXT_KEYS = ("data", "framework", "methods", "results")

#: 预置智能体档案：第一次启动时种进去，用户可改可删可另建
DEFAULT_AGENTS = (
    {
        "id": "framework_designer",
        "name": "框架设计师",
        "role": "你是协作学习观察研究的编码框架设计顾问。先检查研究问题、研究现场、"
                "可用证据与分析单位，再建议适用的文献框架和操作化方案。每条建议必须区分"
                "文献概念、针对当前研究的改编与 AI 新建议；优先给出可观察定义、纳入/排除"
                "规则和例证，不得杜撰文献、DOI 或原文定义。所有输出都是待研究者确认的草稿，"
                "不得声称已经验证，也不得直接覆盖编码数据。",
        "context": {"data": True, "framework": True, "methods": True, "results": False},
        "templates": ["从研究问题开始设计新的编码框架",
                      "以现有编码框架为基础进行适配",
                      "审查当前框架的重叠、缺口和可观察性"],
    },
    {
        "id": "planner",
        "name": "分析规划师",
        "role": "你是协作学习行为分析的方法学顾问。基于用户当前的编码数据规模、编码框架"
                "与本软件可用的分析方法清单，帮用户规划一套数据分析方案：先做什么描述统计、"
                "再做哪些序列/网络/预测分析，每一步用哪个具体方法（给出方法编号与名称）、"
                "为什么、需要注意的口径与前提。回答要具体、可执行，贴合用户的数据现状，"
                "不要泛泛而谈。",
        "context": {"data": True, "framework": True, "methods": True, "results": False},
        "templates": ["请根据我当前的数据，规划一套完整的分析方案",
                      "我想研究协作中的交互模式，该选哪些方法？",
                      "我的样本量偏小，哪些方法仍然可靠？"],
    },
    {
        "id": "coding",
        "name": "编码助手",
        "role": "你是协作学习行为编码的助手。用户会给出编码框架（维度与每个维度下的编码）"
                "以及一段观察描述或话语转写。请严格按框架，把观察到的行为/话语归入最贴切的"
                "编码，逐维度给出建议码与简短理由；框架里没有合适编码时如实说明，不要杜撰。",
        "context": {"data": False, "framework": True, "methods": False, "results": False},
        "templates": ["这段话语应该编成什么？", "帮我核对这一时刻各维度的编码",
                      "这个行为算不算框架里的某一类？"],
    },
    {
        "id": "interpret",
        "name": "结果解读",
        "role": "你是数据分析结果的解读者。用户会贴出一张结果表或指标（可能附带方法说明）。"
                "请用通俗但严谨的语言解释这些数字意味着什么、能支持什么结论、有哪些局限与"
                "不能过度解读的地方；涉及统计口径时点明前提。",
        "context": {"data": True, "framework": True, "methods": True, "results": True},
        "templates": ["帮我解读这张表", "这个指标偏高/偏低说明什么？",
                      "这些结果能写进结论吗，有什么要注意的？"],
    },
    {
        "id": "report",
        "name": "报告撰写",
        "role": "你是学术写作助手。基于用户提供的数据概况、分析方法与结果，帮其组织论文/"
                "报告的「方法」与「结果」段落：结构清晰、口径准确、语言规范。不得编造未在"
                "结果中出现的数字；缺信息时先向用户提问。",
        "context": {"data": True, "framework": True, "methods": True, "results": True},
        "templates": ["帮我把这些结果写成一段「结果」", "起草「数据分析方法」小节",
                      "给这张图配一段说明文字"],
    },
)


def _now():
    return time.time()


def _reveal(stored):
    """盘上的 api_key → 内存明文：DPAPI 密文解开，历史明文原样返回。

    解不开只可能是「配置文件被从别的 Windows 账户/别的机器搬来」——DPAPI 绑登录用户，
    这是设计而非故障。此时宁可按空处理，也不能把一串解不开的东西当 key 发给端点。
    """
    text = str(stored or "")
    if not secure_store.is_protected(text):
        return text
    plain = secure_store.unprotect(text)
    if plain is None:
        logging.warning("AI 设置里的 api_key 无法解密（换了 Windows 账户或配置文件被搬动），按空处理")
        return ""
    return plain


def _seal(plain):
    """内存明文 → 盘上密文；DPAPI 不可用或调用失败时降级存明文（不静默丢 key）。"""
    text = str(plain or "")
    if not text:
        return ""
    token = secure_store.protect(text)
    if token is None:
        logging.warning("DPAPI 不可用，api_key 将按明文保存（非 Windows 或加密调用失败）")
        return text
    return token


def _disk_writes_blocked():
    """offscreen 且未显式开 LEARNLYSIS_AUTOSAVE 时，不往工程 data/ 里写设置。"""
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None and app.platformName() == "offscreen":
            return os.environ.get("LEARNLYSIS_AUTOSAVE") != "1"
    except Exception:                          # noqa: BLE001  无 Qt 的脚本照常写
        pass
    return False


def _default_agent(agent_id, name, role, context=None, templates=None):
    return {
        "id": str(agent_id),
        "name": str(name),
        "role": str(role),
        "context": {key: bool((context or {}).get(key, False)) for key in CONTEXT_KEYS},
        "templates": [str(item) for item in (templates or [])],
    }


def _clean_agent(raw):
    """把存档里的一条档案洗成规整字典：缺字段补默认，脏类型挡掉。"""
    if not isinstance(raw, dict):
        return None
    agent_id = str(raw.get("id") or "").strip()
    name = str(raw.get("name") or "").strip()
    if not agent_id or not name:
        return None
    return _default_agent(agent_id, name, raw.get("role") or "",
                          raw.get("context"), raw.get("templates"))


def count_notes(notes):
    """码本释义的 (维度数, 编码数)；模块级，UI 与测试共用。"""
    notes = notes if isinstance(notes, dict) else {}
    return len(notes), sum(len(codes) for codes in notes.values() if isinstance(codes, dict))


def _clean_codebook(raw):
    """把盘上的码本释义洗成 ``{维度: {编码: {"desc", "example"}}}``；脏类型整条丢掉了。"""
    if not isinstance(raw, dict):
        return {}
    out = {}
    for dim, codes in raw.items():
        if not isinstance(codes, dict):
            continue
        dim_clean = str(dim).strip()
        if not dim_clean:
            continue
        entry = {}
        for code, note in codes.items():
            code_clean = str(code).strip()
            if not code_clean or not isinstance(note, dict):
                continue
            item = {"desc": str(note.get("desc") or "").strip(),
                    "example": str(note.get("example") or "").strip()}
            if item["desc"] or item["example"]:
                entry[code_clean] = item
        if entry:
            out[dim_clean] = entry
    return out


class AiSettings:
    """data/ai_settings.json 的内存镜像；写盘一律走原子替换。"""

    def __init__(self):
        self._payload = None

    # ---------------------------------------------------------------- 读写
    def payload(self):
        if self._payload is None:
            self._payload = self._read()
        return self._payload

    def _read(self):
        data = {
            "provider": {
                "base_url": DEFAULT_BASE_URL, "model": DEFAULT_MODEL, "api_key": "",
                "temperature": DEFAULT_TEMPERATURE, "max_tokens": DEFAULT_MAX_TOKENS,
                "persist_key": True,
            },
            "agents": [_default_agent(item["id"], item["name"], item["role"],
                                      item["context"], item["templates"])
                       for item in DEFAULT_AGENTS],
            "active_agent": DEFAULT_ACTIVE_AGENT,
            "codebook": {},
        }
        try:
            with open(AI_SETTINGS_FILE, encoding="utf-8") as handle:
                stored = json.load(handle)
        except FileNotFoundError:
            return data
        except Exception as exc:                # noqa: BLE001 坏文件按没存过处理
            logging.warning("AI 设置读取失败，按默认处理：%s", exc)
            return data
        if isinstance(stored, dict):
            provider = stored.get("provider")
            if isinstance(provider, dict):
                data["provider"].update({
                    key: provider[key] for key in
                    ("base_url", "model", "api_key", "temperature", "max_tokens",
                     "persist_key") if key in provider})
                # 内存镜像一律持明文，ai_client / 设置页取的就是这个值
                data["provider"]["api_key"] = _reveal(data["provider"].get("api_key"))
            agents = [_clean_agent(item) for item in (stored.get("agents") or [])]
            agents = [item for item in agents if item]
            if agents:
                data["agents"] = agents
                # 新增系统预置采用按固定 id 补种的迁移方式：不覆盖用户改过的旧档案，
                # 也不改变其当前选中的智能体。
                existing_ids = {item["id"] for item in data["agents"]}
                for seed in DEFAULT_AGENTS:
                    if seed["id"] not in existing_ids:
                        data["agents"].append(_default_agent(
                            seed["id"], seed["name"], seed["role"],
                            seed["context"], seed["templates"]))
            active = str(stored.get("active_agent") or "")
            if active and any(item["id"] == active for item in data["agents"]):
                data["active_agent"] = active
            data["codebook"] = _clean_codebook(stored.get("codebook"))
        return data

    def save(self):
        """原子写：先写 .tmp 再替换；persist_key=False 时抹掉 api_key 再落盘。

        写盘的是深拷贝的副本，api_key 在那份上被 DPAPI 封起来——内存里的明文不受影响，
        所以「保存 → 立刻发一次请求」不需要先回读文件。
        """
        if _disk_writes_blocked():
            return True                         # 测试环境：只改内存，不碰工程 data/
        payload = self.payload()
        disk = json.loads(json.dumps(payload, ensure_ascii=False))   # 深拷贝再脱敏
        if not disk.get("provider", {}).get("persist_key", True):
            disk["provider"]["api_key"] = ""
        else:
            disk["provider"]["api_key"] = _seal(disk["provider"].get("api_key"))
        try:
            os.makedirs(data_dir, exist_ok=True)
            tmp = AI_SETTINGS_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(disk, handle, ensure_ascii=False, indent=2)
            os.replace(tmp, AI_SETTINGS_FILE)
        except Exception as exc:                # noqa: BLE001 设置写失败不该打扰用户
            logging.warning("AI 设置保存失败：%s", exc)
            return False
        return True

    # ---------------------------------------------------------------- provider
    def provider(self):
        return dict(self.payload()["provider"])

    def set_provider(self, **kwargs):
        provider = self.payload()["provider"]
        for key in ("base_url", "model", "api_key", "temperature", "max_tokens",
                    "persist_key"):
            if key in kwargs:
                provider[key] = kwargs[key]
        self.save()

    def api_key(self):
        return str(self.payload()["provider"].get("api_key") or "")

    def base_url(self):
        return str(self.payload()["provider"].get("base_url") or DEFAULT_BASE_URL).rstrip("/")

    def model(self):
        return str(self.payload()["provider"].get("model") or DEFAULT_MODEL)

    def temperature(self):
        try:
            return float(self.payload()["provider"].get("temperature", DEFAULT_TEMPERATURE))
        except (TypeError, ValueError):
            return DEFAULT_TEMPERATURE

    def max_tokens(self):
        try:
            return int(self.payload()["provider"].get("max_tokens", DEFAULT_MAX_TOKENS))
        except (TypeError, ValueError):
            return DEFAULT_MAX_TOKENS

    # ---------------------------------------------------------------- 档案
    def agents(self):
        return [dict(item) for item in self.payload()["agents"]]

    def agent(self, agent_id):
        for item in self.payload()["agents"]:
            if item["id"] == str(agent_id):
                return dict(item)
        return None

    def active_agent_id(self):
        return str(self.payload().get("active_agent") or "")

    def active_agent(self):
        return (self.agent(self.active_agent_id()) or self.agent(DEFAULT_ACTIVE_AGENT)
                or (self.agents()[0] if self.agents() else None))

    def set_active_agent(self, agent_id):
        if self.agent(agent_id) is not None:
            self.payload()["active_agent"] = str(agent_id)
            self.save()

    def add_agent(self, name, role, context=None, templates=None, agent_id=None):
        """新建档案：id 缺省时按时间戳生成，撞名不拦（id 唯一即可）。"""
        agent_id = str(agent_id or ("agent_%d" % int(_now() * 1000)))
        if self.agent(agent_id) is not None:
            agent_id = "%s_%d" % (agent_id, int(_now() * 1000))
        agent = _default_agent(agent_id, name or "新智能体", role or "", context, templates)
        self.payload()["agents"].append(agent)
        self.payload()["active_agent"] = agent_id
        self.save()
        return dict(agent)

    def update_agent(self, agent_id, **kwargs):
        agent = self.agent(agent_id)
        if agent is None:
            return None
        for item in self.payload()["agents"]:
            if item["id"] == str(agent_id):
                if "name" in kwargs:
                    item["name"] = str(kwargs["name"])
                if "role" in kwargs:
                    item["role"] = str(kwargs["role"])
                if "context" in kwargs and isinstance(kwargs["context"], dict):
                    item["context"] = {key: bool(kwargs["context"].get(key, False))
                                       for key in CONTEXT_KEYS}
                if "templates" in kwargs:
                    item["templates"] = [str(value) for value in (kwargs["templates"] or [])]
                self.save()
                return dict(item)
        return None

    # ---------------------------------------------------------------- 码本释义
    def codebook_notes(self):
        """AI 码本释义：``{维度名: {编码: {"desc": 释义, "example": 示例}}}``，全局一份。

        由「AI 辅助编码」对话框从富释义 Excel 导入；自动编码 prompt 与对话上下文共用。
        """
        return dict(self.payload().get("codebook") or {})

    def set_codebook_notes(self, notes):
        """整体替换码本释义（导入语义：一次导入对应一份完整码本）；传空即清除。"""
        self.payload()["codebook"] = _clean_codebook(notes)
        self.save()

    def codebook_notes_counts(self):
        """(维度数, 码数)：给 UI 状态标签用。"""
        return count_notes(self.codebook_notes())

    def remove_agent(self, agent_id):
        agents = self.payload()["agents"]
        kept = [item for item in agents if item["id"] != str(agent_id)]
        if len(kept) == len(agents):
            return False
        # 至少留一个档案：删光时补回默认的第一个，UI 下拉不至于空
        if not kept:
            seed = next((item for item in DEFAULT_AGENTS
                         if item["id"] == DEFAULT_ACTIVE_AGENT), DEFAULT_AGENTS[0])
            kept = [_default_agent(seed["id"], seed["name"], seed["role"],
                                   seed["context"], seed["templates"])]
        self.payload()["agents"] = kept
        if not any(item["id"] == self.payload().get("active_agent") for item in kept):
            self.payload()["active_agent"] = kept[0]["id"]
        self.save()
        return True


settings = AiSettings()
