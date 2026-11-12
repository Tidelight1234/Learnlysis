# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""OpenAI 兼容的对话客户端：标准库 urllib 直连 /v1/chat/completions，支持 SSE 流式。

刻意零新增依赖：HTTP 用 urllib.request、TLS 用 ssl、SSE 自己按行解析。探测到 requests
时可用它作增强后端（见 optional_backends），但**不装也全功能可用**——照 analysis_ml 对
torch/xgboost 的「可选后端」范式。

设计要点：
- ``transport`` 可注入：默认走 urllib，冒烟测试注入一个产出假 SSE 行的桩，就不连真网。
  transport 收 payload(dict)、返回「逐行原始文本」的可迭代对象，解析逻辑因此对真假同源。
- ``cancel`` 可注入：每收到一个 token 检查一次，取消即收摊（镜像 ctx.cancelled 单通路）。
- 错误一律抛 ``AiError``，message 是人话（无效 key / 限流 / 端点不存在 / 断网 / 超时分开说）。
- 多模态：``user_multimodal`` 把关键帧拼成 ``image_url`` 内容块，供 Phase 2 视频自动编码复用。
"""

from __future__ import annotations

import json
import ssl
import importlib.util
import urllib.request
import urllib.error

DEFAULT_TIMEOUT = 90          # 秒：多模态/长回答留足余量
_DONE = "[DONE]"


class AiError(Exception):
    """带人话 message 的客户端错误；上层只 str(exc) 展示，不 traceback 给用户。"""


def optional_backends():
    """可选依赖在场判定：requests（增强后端，缺席不影响任何功能）。"""
    return {"requests": importlib.util.find_spec("requests") is not None}


def rough_tokens(text):
    """粗略 token 估算：CJK 约 1 字 1 token，ASCII 约 4 字符 1 token。

    仅用于 API 未回传 usage 时给用户一个量级参考（BYO key 的成本透明），不作计费依据。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
    other = len(text) - cjk
    return cjk + max(0, other // 4)


# ------------------------------------------------------------------ 消息构造

def system_message(text):
    return {"role": "system", "content": str(text or "")}


def user_message(text):
    return {"role": "user", "content": str(text or "")}


def assistant_message(text):
    return {"role": "assistant", "content": str(text or "")}


def user_multimodal(text, images=None):
    """文本 + 若干 data URI 图片的用户消息（OpenAI 兼容多模态内容块）。

    ``images`` 是形如 ``data:image/jpeg;base64,....`` 的字符串列表；空则退化为纯文本消息。
    """
    images = [item for item in (images or []) if item]
    if not images:
        return user_message(text)
    content = [{"type": "text", "text": str(text or "")}]
    for uri in images:
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return {"role": "user", "content": content}


# ------------------------------------------------------------------ 客户端

class OpenAICompatClient:
    """一个端点 + 一把 key + 一个模型的对话客户端。"""

    def __init__(self, base_url, api_key, model, temperature=0.3, max_tokens=1024,
                 timeout=DEFAULT_TIMEOUT, transport=None):
        self.base_url = str(base_url or "").rstrip("/")
        self.api_key = str(api_key or "")
        self.model = str(model or "")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = timeout
        self._transport = transport            # 注入的桩：payload -> Iterable[str]

    # ---------- 校验 ----------
    def ready(self):
        """(是否可用, 原因)。缺 key / 缺 base_url / 缺 model 时给人话。"""
        if not self.base_url:
            return False, "未填写 API base_url（如 https://api.openai.com/v1）"
        if not self.api_key:
            return False, "未填写 API key"
        if not self.model:
            return False, "未填写模型名（如 gpt-4o-mini / deepseek-chat / qwen-plus）"
        return True, ""

    def endpoint(self):
        base = self.base_url
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    # ---------- 传输 ----------
    def _urllib_lines(self, payload):
        """默认传输：POST 后逐行产出原始响应文本（SSE 或整包 JSON 都按行给）。"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % self.api_key,
            "Accept": "text/event-stream",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.endpoint(), data=body, headers=headers,
                                         method="POST")
        context = ssl.create_default_context()
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout, context=context)
        except urllib.error.HTTPError as exc:
            raise AiError(self._explain_http(exc)) from None
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if "timed out" in str(reason).lower():
                raise AiError("请求超时（%d 秒）：网络较慢或端点无响应" % self.timeout) from None
            raise AiError("网络不可达：%s（检查 base_url、代理与联网状态）" % reason) from None
        except TimeoutError:
            raise AiError("请求超时（%d 秒）" % self.timeout) from None
        except Exception as exc:                # noqa: BLE001
            raise AiError("请求失败：%s" % exc) from None
        with response:
            for raw in response:
                yield raw.decode("utf-8", "replace")

    def _explain_http(self, exc):
        """把 HTTPError 翻成人话，并尽量带上服务端返回的 error.message。"""
        code = getattr(exc, "code", None)
        detail = ""
        try:
            raw = exc.read().decode("utf-8", "replace")
            parsed = json.loads(raw) if raw.strip() else {}
            detail = str((parsed.get("error") or {}).get("message") or "")[:300]
        except Exception:                        # noqa: BLE001 读不到正文就用状态码说话
            detail = ""
        suffix = ("：%s" % detail) if detail else ""
        if code in (401, 403):
            return "API key 无效或无权限（%d）%s" % (code, suffix)
        if code == 404:
            return "端点或模型不存在（404）：核对 base_url 与 model%s" % suffix
        if code == 429:
            return "触发限流或额度不足（429）：稍后再试或检查账户余额%s" % suffix
        if code is not None and 500 <= code < 600:
            return "服务端错误（%d）：对方服务暂时不可用%s" % (code, suffix)
        if code is not None:
            return "请求被拒（HTTP %d）%s" % (code, suffix)
        return "请求失败%s" % (suffix or "：未知 HTTP 错误")

    def _lines(self, payload):
        transport = self._transport or self._urllib_lines
        for line in transport(payload):
            yield line

    # ---------- 解析 ----------
    @staticmethod
    def _parse_line(line):
        """一行原始响应 → (增量文本 or None, usage or None, done?)。"""
        text = (line or "").strip()
        if not text:
            return None, None, False
        if text.startswith("data:"):
            text = text[len("data:"):].strip()
        if text == _DONE:
            return None, None, True
        if not text.startswith("{"):
            return None, None, False            # 心跳/注释行（: ping 等）忽略
        try:
            obj = json.loads(text)
        except ValueError:
            return None, None, False
        usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
        choices = obj.get("choices") or []
        if not choices:
            # 有些实现把错误塞进 200 响应的 body
            err = obj.get("error")
            if isinstance(err, dict) and err.get("message"):
                raise AiError("服务端返回错误：%s" % err["message"])
            return None, usage, False
        first = choices[0] or {}
        delta = (first.get("delta") or {}).get("content")
        if delta is None:
            delta = (first.get("message") or {}).get("content")
        return delta, usage, False

    # ---------- 对话 ----------
    def chat(self, messages, on_token=None, cancel=None, stream=True, **overrides):
        """发一轮对话，返回 ``{"text": 全文, "usage": {...}}``。

        ``on_token(piece)`` 逐块回调（流式）；``cancel()`` 返回真值时提前收摊并保留已收文本。
        """
        ok, note = self.ready()
        if not ok:
            raise AiError(note)
        payload = {
            "model": overrides.get("model") or self.model,
            "messages": list(messages or []),
            "temperature": float(overrides.get("temperature", self.temperature)),
            "max_tokens": int(overrides.get("max_tokens", self.max_tokens)),
            "stream": bool(stream),
        }
        if stream:
            # 让兼容端在流里回传用量统计（OpenAI 及多数国产兼容实现支持）
            payload["stream_options"] = {"include_usage": True}
        collected, usage, cancelled = [], None, False
        for line in self._lines(payload):
            if cancel is not None and cancel():
                cancelled = True
                break
            piece, chunk_usage, done = self._parse_line(line)
            if chunk_usage:
                usage = chunk_usage
            if piece:
                collected.append(piece)
                if on_token is not None:
                    on_token(piece)
            if done:
                break
        text = "".join(collected)
        if usage is None:
            usage = {"estimated": True,
                     "prompt_tokens": rough_tokens(json.dumps(messages, ensure_ascii=False)),
                     "completion_tokens": rough_tokens(text),
                     "total_tokens": rough_tokens(json.dumps(messages, ensure_ascii=False))
                     + rough_tokens(text)}
        return {"text": text, "usage": usage, "cancelled": cancelled}

    def test_connection(self):
        """最小 ping：发一句 "ping"、只要一个 token，验证 key/端点/模型是否通。"""
        result = self.chat([user_message("ping")], stream=False, max_tokens=1)
        return result.get("text", "")
