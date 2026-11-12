# Copyright © 2026 Learnlysis. Licensed under the MIT License.
# Licensed under the MIT License; see the repository root LICENSE file.
"""把一轮 AI 对话搬到子线程：网络调用绝不堵 UI 线程。

镜像 analysis_workbench.ComputeWorker 的范式——计算/IO 在 QThread 里跑，信号回主线程交货，
取消只有一条通路（``cancel()`` 挂标志，客户端在下一个 token 检查点自己收摊）。
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from ai_client import AiError


class AiWorker(QThread):
    """一次对话（流式或非流式）的后台执行体。

    信号：``token`` 逐块文本（流式追加）、``done`` 回传结果字典
    ``{"text", "usage", "cancelled"}``、``error`` 回传人话错误串。三者都在主线程交货，
    工作线程绝不碰任何控件。
    """

    token = Signal(str)
    done = Signal(object)
    error = Signal(str)

    def __init__(self, client, messages, parent=None, stream=True, **overrides):
        super().__init__(parent)
        self.client = client
        self.messages = list(messages or [])
        self.stream = bool(stream)
        self.overrides = dict(overrides)
        self._cancelled = False

    # ---------- 取消：唯一通路 ----------
    def cancel(self):
        """请求取消：给客户端的 cancel 探针挂标志，它在下一个 token 处收摊。"""
        self._cancelled = True

    def _is_cancelled(self):
        return self._cancelled

    def _emit_token(self, piece):
        if piece:
            self.token.emit(piece)

    def run(self):
        try:
            result = self.client.chat(self.messages, on_token=self._emit_token,
                                      cancel=self._is_cancelled, stream=self.stream,
                                      **self.overrides)
        except AiError as exc:
            self.error.emit(str(exc))
            return
        except Exception as exc:                # noqa: BLE001 线程里抛不出去，只能回传
            self.error.emit("对话失败：%s" % exc)
            return
        self.done.emit(result)
